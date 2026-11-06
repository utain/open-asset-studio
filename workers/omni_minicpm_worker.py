"""MiniCPM-o backend (envs/omni_minicpm: transformers 4.51.0, minicpmo-utils, torch 2.9 cu130).

  minicpm-o-4_5   tasks s2s, tts, voice_clone   SigLIP2 (not loaded: no vision here) + Whisper-medium encoder +
                  Qwen3-8B + CosyVoice2-style TTS head + Step-Audio-2 token2wav (24 kHz).
                  s2s: spoken question in, spoken answer out (voice = a reference clip; bundled one by default).
                  tts / voice_clone: "read this text aloud in the voice of this reference clip".
Half-duplex, offline (model.chat); the full-duplex live-streaming mode is out of scope.
"""
from __future__ import annotations

import gc
import importlib.util
import math
import random
import re
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

REPO = "openbmb/MiniCPM-o-4_5"
SUFFIX_EN = ("Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a "
             "high quality. Please chat with the user in a highly human-like and oral style. You are a helpful assistant "
             "developed by ModelBest: MiniCPM-Omni.")
SUFFIX_ZH = "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。"
CJK = re.compile(r"[一-鿿]")


def seed0(p):
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def save_wav(path, wav, sr):
    import soundfile as sf

    w = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1))
    if len(w) < int(0.05 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    meta = {"sample_rate": int(sr)}
    peak = float(np.abs(w).max())
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_16")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def name_of(text, i, n):
    t = re.sub(r"\s+", " ", text).strip()
    return f"{t[:48]}{'…' if len(t) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def patch_torchaudio():
    import soundfile as sf
    import torch
    import torchaudio

    def load(path, *a, **k):
        y, sr = sf.read(str(path) if not hasattr(path, "read") else path, dtype="float32", always_2d=True)
        return torch.from_numpy(y.T.copy()), sr

    def save(path, wav, sr=None, *a, **k):
        sr = sr or k.get("sample_rate")
        w = wav.detach().cpu().numpy().T
        sf.write(path if hasattr(path, "write") else str(path), w, sr, **({"format": "WAV"} if hasattr(path, "write") else {}))

    torchaudio.load, torchaudio.save = load, save


class MiniCPMBackend(Backend):
    name = "omni_minicpm"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("transformers", "minicpmo", "stepaudio2", "s3tokenizer", "onnxruntime", "hyperpyyaml", "librosa", "torch")
                   if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        patch_torchaudio()
        local = snapshot_download(REPO, local_files_only=True)
        m = AutoModel.from_pretrained(local, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
                                      init_vision=False, init_audio=True, init_tts=True)
        m.eval().cuda()
        m.init_tts()
        return {"model": m, "dir": local}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import librosa
        import torch

        h = self.handle(job.model_id)
        m, local = h["model"], Path(h["dir"])
        p = job.params
        tts_mode = job.task in ("tts", "voice_clone")
        typed = (p.get("text") or "").strip()
        instruction = (p.get("instruction") or "").strip()
        src = None if tts_mode else (job.input(role="audio") or next(
            (i for i in job.inputs if i.get("type") in ("audio", "voice") and i.get("role") not in ("voice_ref", "ref")), None))
        vref = job.input(role="voice_ref") or (job.input(role="ref") if tts_mode else None)
        if tts_mode and not typed:
            raise InputError("Write the line to speak first.")
        if not tts_mode and not src and not typed:
            raise InputError("Pick an audio clip to answer (or type a message).")
        ref_path = vref["path"] if vref else str(local / "assets" / "HT_ref_audio.wav")
        ref, _ = librosa.load(ref_path, sr=16000, mono=True, duration=20)
        zh = bool(CJK.search(typed + instruction))
        if tts_mode:
            sys_msg = {"role": "system", "content": ["模仿音频样本的音色并生成新的内容。", ref, "请用这种声音风格来为用户提供帮助。 直接作答，不要有冗余内容"]}
            user = {"role": "user", "content": ["请朗读以下内容。" + " " + typed]}
            shown = typed
        else:
            pre = "模仿音频样本的音色并生成新的内容。" if zh else "Clone the voice in the provided audio prompt."
            suf = (SUFFIX_ZH if zh else SUFFIX_EN) + ((" " + instruction) if instruction else "")
            sys_msg = {"role": "system", "content": [pre, ref, suf]}
            parts = []
            if src:
                a, _ = librosa.load(src["path"], sr=16000, mono=True, duration=60)
                parts.append(a)
            if typed:
                parts.append(typed)
            user = {"role": "user", "content": parts}
            shown = "Reply"
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        max_new = max(64, int(float(p.get("seconds") or 20) * 6))
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            f = job.out_dir / f"take_{i + 1}.wav"
            if f.exists():
                f.unlink()
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}: thinking + speaking", interruptible=False)
            reply = m.chat(msgs=[sys_msg, user], do_sample=True, max_new_tokens=max_new, use_tts_template=True,
                           generate_audio=True, temperature=float(p.get("temperature") or (0.1 if tts_mode else 0.7)),
                           output_audio_path=str(f))
            if not f.exists():
                raise RuntimeError("MiniCPM-o produced no audio file (see the worker log for the traceback)")
            import soundfile as sf
            y, sr = sf.read(str(f), dtype="float32")
            fm = save_wav(f, y, sr)
            reply = (reply or "").strip()
            if tts_mode:
                reply = typed
            (job.out_dir / f"take_{i + 1}.txt").write_text(reply, encoding="utf-8")
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(reply or shown, i, n), final_prompt=reply,
                       reply_text=reply, input_asset=(src or {}).get("asset_id"), attach=[f"take_{i + 1}.txt"], **fm)
            gc.collect()
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(MiniCPMBackend())
