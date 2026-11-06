"""Step-Audio 2 mini backend (envs/omni_step: transformers 4.49.0, torch 2.9 cu130; upstream code in vendor/Step-Audio2).

  step-audio-2-mini   task s2s   audio encoder + Qwen2.5-7B LM that emits text and 25 Hz CosyVoice2 speech tokens;
                                 flow-matching + HiFT token2wav (24 kHz) with a voice prompt wav (default female / male).
One turn per job. English and Chinese (also dialects).
"""
from __future__ import annotations

import gc
import importlib.util
import io
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "Step-Audio2"
REPO = "stepfun-ai/Step-Audio-2-mini"
VOICES = {"female": "default_female.wav", "male": "default_male.wav"}


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
    """torchaudio 2.9 load/save need torchcodec (no aarch64 wheel): route them through soundfile."""
    import soundfile as sf
    import torch
    import torchaudio

    def load(path, *a, **k):
        y, sr = sf.read(str(path) if not hasattr(path, "read") else path, dtype="float32", always_2d=True)
        return torch.from_numpy(y.T.copy()), sr

    def save(path, wav, sr=None, *a, **k):
        sr = sr or k.get("sample_rate")
        w = wav.detach().cpu().numpy().T
        if hasattr(path, "write"):
            sf.write(path, w, sr, format="WAV")
        else:
            sf.write(str(path), w, sr)

    torchaudio.load, torchaudio.save = load, save


class StepBackend(Backend):
    name = "omni_step"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("transformers", "s3tokenizer", "onnxruntime", "hyperpyyaml", "torch") if importlib.util.find_spec(m) is None]
        if not VENDOR.exists():
            missing.append(f"vendor/Step-Audio2 ({VENDOR})")
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        from huggingface_hub import snapshot_download

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        patch_torchaudio()
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from stepaudio2 import StepAudio2
        from token2wav import Token2wav

        local = snapshot_download(REPO, local_files_only=True)
        lm = StepAudio2(local)
        t2w = Token2wav(f"{local}/token2wav")
        return {"lm": lm, "t2w": t2w}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import soundfile as sf
        import torch

        h = self.handle(job.model_id)
        lm, t2w = h["lm"], h["t2w"]
        p = job.params
        src = job.input(role="audio") or next((i for i in job.inputs if i.get("type") in ("audio", "voice") and i.get("role") != "voice_ref"), None)
        vref = job.input(role="voice_ref")
        typed = (p.get("text") or "").strip()
        if not src and not typed:
            raise InputError("Pick an audio clip to answer (or type a message).")
        system = (p.get("instruction") or "").strip() or "You are a helpful assistant."
        if src:
            user = [{"type": "audio", "audio": src["path"]}]
            if typed:
                user.insert(0, {"type": "text", "text": typed})
        else:
            user = typed
        msgs = [{"role": "system", "content": system},
                {"role": "human", "content": user},
                {"role": "assistant", "content": "<tts_start>", "eot": False}]
        prompt_wav = vref["path"] if vref else str(VENDOR / "assets" / VOICES.get(p.get("voice") or "female", VOICES["female"]))
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        max_tokens = max(256, int(float(p.get("seconds") or 20) * 25 * 1.6))  # 25 audio tokens/s plus text tokens
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}: generating speech tokens", interruptible=False)
            with torch.inference_mode():
                _, reply, atoks = lm(msgs, max_new_tokens=max_tokens, temperature=float(p.get("temperature") or 0.7),
                                     repetition_penalty=1.05, top_p=0.9, do_sample=True)
                atoks = [t for t in atoks if t < 6561]
                if not atoks:
                    raise RuntimeError("the model produced no speech tokens (try again or a different seed)")
                ctx.progress(i, n, "decoding", "flow-matching + vocoder", interruptible=False)
                wav_bytes = t2w(atoks, prompt_wav=prompt_wav)
            y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            reply = reply.strip()
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, y, sr)
            (job.out_dir / f"take_{i + 1}.txt").write_text(reply, encoding="utf-8")
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(reply or "Reply", i, n), final_prompt=reply,
                       reply_text=reply, voice=p.get("voice") or "female", input_asset=(src or {}).get("asset_id"),
                       attach=[f"take_{i + 1}.txt"], **fm)
            gc.collect()
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(StepBackend())
