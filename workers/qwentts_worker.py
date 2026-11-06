"""Qwen3-TTS backend (envs/qwentts: qwen-tts 0.1.1, transformers 4.57.3).

  qwen3-tts-customvoice   task tts          9 preset speakers + free-text `instruct` (emotion / tone / pace)
  qwen3-tts-base          task voice_clone  zero-shot clone from a reference clip (+ its transcript; without it
                                            x-vector-only mode, lower quality)
10 languages, no Thai (Thai text is refused with a pointer to OmniVoice). Recipe: audio_snippets/qwen3_tts.py.
"""
from __future__ import annotations

import importlib.util
import math
import random
import re

import numpy as np

from studio_worker import Backend, InputError, serve

THAI = re.compile(r"[฀-๿]")


def seed0(p):
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def tidy_voice(w, sr, pad_s=0.12, floor_db=-50.0):
    mono = np.abs(w)
    idx = np.flatnonzero(mono > 10 ** (floor_db / 20))
    if idx.size == 0:
        return w
    pad = int(pad_s * sr)
    return w[max(0, idx[0] - pad): min(len(w), idx[-1] + pad + 1)]


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
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def name_of(text, i, n):
    t = re.sub(r"\s+", " ", text).strip()
    return f"{t[:48]}{'…' if len(t) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


class QwenTTSBackend(Backend):
    name = "qwentts"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("qwen_tts", "torch", "soundfile") if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel

        # transformers 4.57 calls model_info() for hub ids even offline: pass the local snapshot path
        local = snapshot_download(model["weights"][0]["hf"], local_files_only=True)
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        return Qwen3TTSModel.from_pretrained(local, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=attn)

    def run(self, job, ctx):
        import soundfile as sf
        import torch

        tts = self.handle(job.model_id)
        p = job.params
        text = (p.get("text") or p.get("prompt") or "").strip()
        if not text:
            raise InputError("Write the line to speak first.")
        if THAI.search(text):
            raise InputError("Qwen3-TTS has no Thai. Use OmniVoice or OmniVoice Thai for Thai lines.")
        lang = p.get("language") or "Auto"
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        gen = {}
        if p.get("temperature") is not None:
            gen["temperature"] = float(p["temperature"])
        clone_prompt, extra = None, {}
        if job.task == "voice_clone":
            ref = job.input(role="ref") or next((i for i in job.inputs if i.get("type") in ("voice", "audio")), None)
            if not ref:
                raise InputError("Pick a reference clip to clone.")
            rt = (p.get("ref_text") or "").strip() or (((ref.get("meta") or {}).get("x_voice") or {}).get("ref_text") or "").strip()
            y, sr = sf.read(ref["path"], dtype="float32", always_2d=True)
            y = y.mean(axis=1)[: sr * 20]
            ctx.progress(0, n, "speaking", "Encoding the reference voice")
            clone_prompt = tts.create_voice_clone_prompt(ref_audio=(y, sr), ref_text=rt or None, x_vector_only_mode=not rt)
            if not rt:
                ctx.log("no reference transcript: x-vector-only clone (lower similarity); add the transcript for best results")
            extra = {"ref_asset": ref.get("asset_id"), "clone_mode": "icl" if rt else "x_vector"}
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}")
            if clone_prompt is not None:
                wavs, sr = tts.generate_voice_clone(text=text, language=lang, voice_clone_prompt=clone_prompt, **gen)
            else:
                wavs, sr = tts.generate_custom_voice(text=text, language=lang, speaker=p.get("speaker") or "Ryan",
                                                     instruct=(p.get("instruct") or "").strip() or None, **gen)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, tidy_voice(np.asarray(wavs[0], dtype=np.float32), sr), sr)
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(text, i, n), final_prompt=text, **extra, **fm)
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(QwenTTSBackend())
