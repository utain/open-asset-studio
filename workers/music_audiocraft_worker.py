"""Music backend `music_audiocraft` (envs/music_audiocraft): Meta AudioCraft models.

MusicGen (medium, melody, stereo-medium), AudioGen (medium, sound effects) and MAGNeT (medium, 10 s, non-autoregressive).
All weights are CC-BY-NC-4.0. The code is the vendored `audiocraft` package (vendor/audiocraft, installed with
--no-deps) with three shims for this machine (aarch64, torch 2.9, offline):
  * `xformers` is replaced by a tiny stub whose memory_efficient_attention runs on torch SDPA (MusicGen / AudioGen use
    audiocraft's own `torch` backend; MAGNeT checkpoints force the xformers code path, which the stub serves).
  * torch >= 2.6 defaults to `weights_only=True`, which cannot unpickle the OmegaConf config inside the checkpoints.
  * the T5 conditioner asks for the legacy ids `t5-base` (MusicGen, MAGNeT) / `t5-large` (AudioGen), which cannot be resolved
    offline: mapped to `google-t5/t5-base` / `google-t5/t5-large`.
"""
from __future__ import annotations

import math
import os
import random
import re
import sys
import types
from pathlib import Path

import numpy as np

from studio_worker import Backend, Cancelled, InputError, serve


# ------------------------------------------------------------------ shims (must run before `import audiocraft`)
def _install_shims():
    import torch

    if "xformers" not in sys.modules:
        x = types.ModuleType("xformers")
        ops = types.ModuleType("xformers.ops")
        ops.unbind = lambda t, dim=0: torch.unbind(t, dim=dim)

        class LowerTriangularMask:  # noqa: D401
            pass

        def _mea(q, k, v, attn_bias=None, p=0.0, **_):
            """xformers.memory_efficient_attention on torch SDPA. Layout (B, T, H, D); bias None / causal / additive tensor.
            (MAGNeT checkpoints force the xformers code path, so this is what actually runs for them.)"""
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            causal = isinstance(attn_bias, LowerTriangularMask)
            mask = None if (causal or attn_bias is None) else attn_bias.to(q.dtype)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, dropout_p=p)
            return out.transpose(1, 2)

        ops.memory_efficient_attention = _mea
        ops.LowerTriangularMask = LowerTriangularMask
        x.ops = ops
        sys.modules["xformers"] = x
        sys.modules["xformers.ops"] = ops

    if not getattr(torch.load, "_studio_patched", False):
        _orig = torch.load

        def _load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig(*a, **k)

        _load._studio_patched = True
        torch.load = _load

    import transformers

    for cls in (transformers.T5EncoderModel, transformers.T5Tokenizer):
        if getattr(cls.from_pretrained, "_studio_patched", False):
            continue
        orig = cls.from_pretrained.__func__

        def _fp(klass, name, *a, _orig=orig, **k):
            if name in ("t5-base", "t5-large"):
                name = "google-t5/" + name
            return _orig(klass, name, *a, **k)

        _fp._studio_patched = True
        cls.from_pretrained = classmethod(_fp)


# ------------------------------------------------------------------ helpers
def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def seed_all(s: int):
    import torch

    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def save_wav(path: Path, wav, sr: int) -> dict:
    """Float audio (C, T) or (T,) -> 24-bit WAV; brings overs to -1 dBFS. Returns file meta."""
    import soundfile as sf

    w = np.asarray(wav, dtype=np.float32)
    if w.ndim == 2:
        w = w.T if w.shape[0] <= 8 and w.shape[0] < w.shape[1] else w
        if w.shape[1] == 1:
            w = w[:, 0]
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if w.size == 0 or len(w) < int(0.02 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    peak = float(np.abs(w).max())
    meta = {"sample_rate": int(sr)}
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def take_name(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def read_audio(path: str):
    """-> torch float tensor (C, T), sample rate"""
    import soundfile as sf
    import torch

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(y.T.copy()), sr


REPOS = {
    "musicgen-medium": ("musicgen", "facebook/musicgen-medium"),
    "musicgen-melody": ("musicgen", "facebook/musicgen-melody"),
    "musicgen-stereo-medium": ("musicgen", "facebook/musicgen-stereo-medium"),
    "audiogen-medium": ("audiogen", "facebook/audiogen-medium"),
    "magnet-medium-10secs": ("magnet", "facebook/magnet-medium-10secs"),
}


class MusicAudiocraft(Backend):
    name = "music_audiocraft"
    max_loaded = 2

    def probe(self, model):
        _install_shims()
        import audiocraft.models  # noqa: F401

        return True, ""

    def load(self, model, ctx):
        _install_shims()
        import torch
        from audiocraft.models import AudioGen, MAGNeT, MusicGen

        fam, repo = REPOS[model["id"]]
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        cls = {"musicgen": MusicGen, "audiogen": AudioGen, "magnet": MAGNeT}[fam]
        m = cls.get_pretrained(repo, device="cuda")
        torch.cuda.synchronize()
        return {"fam": fam, "model": m}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    # ------------------------------------------------------------------ run
    def run(self, job, ctx):
        import torch

        torch.cuda.reset_peak_memory_stats()
        h = self.handle(job.model_id)
        fam, gm = h["fam"], h["model"]
        p = job.params
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Write a description of the sound first.")
        n = int(p.get("batch", 1))
        s0 = seed0(p)
        sr = int(gm.sample_rate)
        secs = float(p.get("seconds", 10))

        melody = job.input(role="melody") or None
        cont = job.input(role="source") or job.input(role="continue") or None
        if melody and job.model_id != "musicgen-melody":
            ctx.log("A melody clip is only used by MusicGen Melody; ignored.")
            melody = None
        if cont and fam == "magnet":
            ctx.log("MAGNeT cannot continue a clip; ignored.")
            cont = None
        if job.model_id == "musicgen-melody" and not melody and not cont:
            ctx.log("No melody clip given: MusicGen Melody runs text-only.")

        if fam == "magnet":
            secs = 10.0
            gm.set_generation_params(
                use_sampling=True, top_k=0, top_p=float(p.get("top_p", 0.9)),
                temperature=float(p.get("temperature", 3.0)),
                max_cfg_coef=float(p.get("cfg_coef", 10.0)), min_cfg_coef=1.0,
                decoding_steps=[int(p.get("steps", 20)), 10, 10, 10], span_arrangement="nonoverlap")
        else:
            gm.set_generation_params(
                use_sampling=True, top_k=int(p.get("top_k", 250)), top_p=float(p.get("top_p", 0.0)),
                temperature=float(p.get("temperature", 1.0)), duration=secs, cfg_coef=float(p.get("cfg_coef", 3.0)))

        state = {"i": 0}

        def cb(done, total):
            ctx.check()
            ctx.progress(done, max(total, 1), "sampling", f"take {state['i'] + 1}/{n}: {done}/{total} tokens")

        gm.set_custom_progress_callback(cb)
        cont_wav = cont_sr = None
        if cont:
            cont_wav, cont_sr = read_audio(cont["path"])
            keep = min(cont_wav.shape[1] / cont_sr, float(p.get("prompt_seconds", 8)))
            cont_wav = cont_wav[:, : int(keep * cont_sr)]
            secs = keep + float(p.get("extend_seconds", 10))      # total length = the kept lead-in + the continuation
            gm.set_generation_params(
                use_sampling=True, top_k=int(p.get("top_k", 250)), top_p=float(p.get("top_p", 0.0)),
                temperature=float(p.get("temperature", 1.0)), duration=secs, cfg_coef=float(p.get("cfg_coef", 3.0)))
        mel_wav = mel_sr = None
        if melody:
            mel_wav, mel_sr = read_audio(melody["path"])

        for i in range(n):
            ctx.check()
            state["i"] = i
            seed_all(s0 + i)
            with torch.inference_mode():
                if cont_wav is not None:
                    out = gm.generate_continuation(cont_wav[None].to("cuda"), cont_sr, [prompt], progress=True)
                elif mel_wav is not None:
                    out = gm.generate_with_chroma([prompt], mel_wav[None].to("cuda"), mel_sr, progress=True)
                else:
                    out = gm.generate([prompt], progress=True)
            wav = out[0].float().cpu().numpy()          # (C, T)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, sr)
            ctx.output(f, "audio", seed=s0 + i, index=i, name=take_name(prompt, i, n), final_prompt=prompt, **fm)
        return {"meta": {"family": fam, "sample_rate": sr, "channels": int(wav.shape[0]),
                         "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}}


if __name__ == "__main__":
    serve(MusicAudiocraft())
