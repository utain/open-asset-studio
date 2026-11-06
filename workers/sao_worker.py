"""sao backend (envs/sao: torch 2.9.1+cu130, stable-audio-tools 0.0.20, transformers 5): Stability AI's open text-to-audio models.

  stable-audio-open-1.0     47 s stereo 44.1 kHz, v-diffusion DiT, ~100 steps with the dpmpp-3m-sde sampler, CFG 7
  stable-audio-open-small   11 s stereo 44.1 kHz, ARC post-trained rectified-flow DiT, 8 steps with the ping-pong sampler

Both repos are gated on Hugging Face (Stability AI Community Licence): accept the terms and `hf auth login` first. The worker
loads the stable-audio-tools format (model.safetensors + model_config.json) straight from the local HF cache; the T5 text
encoder named in the config (google-t5/t5-base) must be cached too. Output is a 24-bit stereo WAV per take.
Recipes: docs/research/audio_more/sfx_more.md
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve


def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def snap(repo: str) -> Path:
    """Local snapshot dir of a cached repo (we download only model.safetensors + model_config.json, so no snapshot_download)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    root = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    ref = root / "refs" / "main"
    if not ref.exists():
        raise InputError(f"{repo} is not downloaded. It is gated: accept the terms on huggingface.co/{repo}, run `hf auth login`, "
                         f"then `hf download {repo} model.safetensors model_config.json`.")
    return root / "snapshots" / ref.read_text().strip()


def save_wav(path: Path, wav, sr: int) -> dict:
    """float (C, T) -> 24-bit stereo/mono WAV. Rejects silence; brings overs to -1 dBFS."""
    import soundfile as sf

    w = np.nan_to_num(np.asarray(wav, dtype=np.float32))
    if w.ndim == 2:
        w = w.T                                     # (C, T) -> (T, C)
    if w.ndim == 2 and w.shape[1] == 1:
        w = w[:, 0]
    if w.shape[0] < int(0.02 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    peak = float(np.abs(w).max())
    if peak < 1e-5:
        raise RuntimeError("the model returned silence")
    meta = {"sample_rate": int(sr)}
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(w.shape[0] / sr, 3)
    return meta


def take_name(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


class SaoBackend(Backend):
    name = "sao"
    max_loaded = 1

    def probe(self, model):
        import importlib.util

        missing = [m for m in ("torch", "stable_audio_tools", "transformers", "soundfile") if importlib.util.find_spec(m) is None]
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    # ---------------------------------------------------------------- load
    def load(self, model, ctx):
        import torch
        from safetensors.torch import load_file
        from stable_audio_tools.models.factory import create_model_from_config

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        root = snap(model["weights"][0]["hf"])
        cfg = json.loads((root / "model_config.json").read_text())
        for c in cfg["model"].get("conditioning", {}).get("configs", []):   # the hub alias "t5-base" does not resolve offline:
            t5 = c.get("config", {}).get("t5_model_name")                  # point the conditioner at the cached snapshot
            if c.get("type") == "t5" and t5:
                c["config"]["model_path"] = str(snap("google-t5/" + t5))
        net = create_model_from_config(cfg)
        missing, unexpected = net.load_state_dict(load_file(str(root / "model.safetensors")), strict=False)
        if len(missing) > 20:
            raise RuntimeError(f"checkpoint does not match its config ({len(missing)} missing tensors)")
        net = net.to("cuda").eval().requires_grad_(False)
        if model["id"] == "stable-audio-open-1.0":
            net = net.to(torch.float16)             # DiT + autoencoder in fp16: 4.8 GB -> 2.4 GB, no audible change
        return {"model": net, "cfg": cfg, "sr": int(cfg["sample_rate"]), "size": int(cfg["sample_size"])}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    # ----------------------------------------------------------------- run
    def run(self, job, ctx):
        import torch
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        prompt = (job.params.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Write a prompt first: describe the sound.")
        h = self.handle(job.model_id)
        small = job.model_id.endswith("small")
        p = job.params
        seconds = min(float(p.get("seconds", 5 if small else 10)), h["size"] / h["sr"])
        steps = int(p.get("steps", 8 if small else 100))
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        neg = (p.get("negative") or "").strip()
        if small:
            kw = dict(sampler_type="pingpong", cfg_scale=1.0)
        else:
            kw = dict(sampler_type="dpmpp-3m-sde", sigma_min=0.3, sigma_max=500, cfg_scale=float(p.get("guidance", 7)))
            if neg:
                kw["negative_conditioning"] = [{"prompt": neg, "seconds_start": 0, "seconds_total": seconds}]
        torch.cuda.reset_peak_memory_stats()
        for i in range(n):
            ctx.check()
            seed = s0 + i

            def cb(d, i=i):
                ctx.check()
                ctx.progress(i * steps + int(d["i"]) + 1, n * steps, "sampling", f"take {i + 1}/{n}: step {int(d['i']) + 1}/{steps}")
            cond = [{"prompt": prompt, "seconds_total": seconds}] if small else [{"prompt": prompt, "seconds_start": 0, "seconds_total": seconds}]
            with torch.inference_mode():
                out = generate_diffusion_cond(h["model"], steps=steps, conditioning=cond, sample_size=h["size"], seed=seed,
                                              device="cuda", callback=cb, disable_tqdm=True, **kw)
            wav = out[0].float().cpu().numpy()[:, : int(seconds * h["sr"])]
            wav = wav - wav.mean(axis=1, keepdims=True)
            f = job.out_dir / f"take_{i + 1}.wav"
            meta = save_wav(f, wav, h["sr"])
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(prompt, i, n), final_prompt=prompt, **meta)
        return {"meta": {"seed": s0, "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}}


if __name__ == "__main__":
    serve(SaoBackend())
