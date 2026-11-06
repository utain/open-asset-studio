"""Music backend `music_riffusion` (shares envs/music_audiocraft): Riffusion v1 (Stable Diffusion 1.5 fine-tuned
on spectrogram images). Text -> 512x512 mel-spectrogram image -> Griffin-Lim -> ~5 s mono clip at 44.1 kHz.

The image <-> audio conversion follows riffusion-hobby's SpectrogramParams defaults (44.1 kHz, 10 ms step, 100 ms
window, 400 ms padded FFT, 512 mel bins 0-10 kHz, power curve 0.25, max volume 50, Griffin-Lim 32 iterations).
Weights: riffusion/riffusion-model-v1 (diffusers layout; the 14 GB .ckpt and the safety checker are not needed).
"""
from __future__ import annotations

import math
import random
import re
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

SR = 44100
STEP_MS, WIN_MS, PAD_MS = 10, 100, 400
N_MELS, F_MIN, F_MAX = 512, 0, 10000
POWER, MAX_VOLUME, GL_ITERS, MEL_ITERS = 0.25, 50.0, 32, 200


def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def take_name(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def spectrogram_from_image(image) -> np.ndarray:
    data = np.array(image.convert("RGB")).astype(np.float32)
    data = data[::-1, :, 0]                      # flip Y, one channel
    data = 255.0 - data                          # invert
    data = np.power(data / 255.0, 1.0 / POWER)   # undo the power curve
    return data * MAX_VOLUME                     # (n_mels, T)


def waveform_from_spectrogram(spec: np.ndarray, device) -> np.ndarray:
    import torch
    import torchaudio

    n_fft = int(PAD_MS / 1000 * SR)
    hop = int(STEP_MS / 1000 * SR)
    win = int(WIN_MS / 1000 * SR)
    inv = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=N_MELS, sample_rate=SR, f_min=F_MIN, f_max=F_MAX, norm=None, mel_scale="htk",
    )   # torchaudio >= 2.9 inverts the mel scale by least squares; kept on the CPU (its filter bank is not a buffer)
    gl = torchaudio.transforms.GriffinLim(n_fft=n_fft, n_iter=GL_ITERS, win_length=win, hop_length=hop, power=1.0).to(device)
    s = torch.from_numpy(spec.copy()).float()[None]    # (1, mel, T)
    with torch.no_grad():
        lin = inv(s).to(device)
        wav = gl(lin)
    return wav[0].cpu().numpy()


class MusicRiffusion(Backend):
    name = "music_riffusion"
    max_loaded = 1

    def probe(self, model):
        import diffusers  # noqa: F401
        import torchaudio  # noqa: F401

        return True, ""

    def load(self, model, ctx):
        import torch
        from diffusers import StableDiffusionPipeline
        from huggingface_hub import snapshot_download

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        path = snapshot_download("riffusion/riffusion-model-v1", local_files_only=True,
                                 allow_patterns=["unet/*", "vae/*", "text_encoder/*", "tokenizer/*", "scheduler/*", "model_index.json"])
        pipe = StableDiffusionPipeline.from_pretrained(
            path, torch_dtype=torch.float16, safety_checker=None, feature_extractor=None, requires_safety_checker=False,
            local_files_only=True)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import soundfile as sf
        import torch

        pipe = self.handle(job.model_id)
        p = job.params
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Write a description of the music first.")
        neg = (p.get("negative") or "").strip() or None
        n = int(p.get("batch", 1))
        steps = int(p.get("steps", 30))
        s0 = seed0(p)
        for i in range(n):
            ctx.check()
            g = torch.Generator("cuda").manual_seed(s0 + i)
            img = pipe(prompt, negative_prompt=neg, num_inference_steps=steps, guidance_scale=float(p.get("guidance", 7.0)),
                       width=512, height=512, generator=g,
                       callback_on_step_end=ctx.step_callback(steps, "denoising", f"take {i + 1}/{n} ")).images[0]
            ctx.check()
            ctx.progress(steps, steps, "decoding", "Spectrogram to audio (Griffin-Lim)")
            wav = waveform_from_spectrogram(spectrogram_from_image(img), "cuda")
            wav = np.nan_to_num(wav.astype(np.float32))
            peak = float(np.abs(wav).max())
            if peak < 1e-6:
                raise RuntimeError("the spectrogram decoded to silence")
            wav = wav * (10 ** (-1 / 20) / peak)                 # peak-normalise to -1 dBFS
            f = job.out_dir / f"take_{i + 1}.wav"
            sf.write(str(f), wav, SR, subtype="PCM_24")
            img.save(job.out_dir / f"take_{i + 1}_spectrogram.png")
            ctx.output(f, "audio", seed=s0 + i, index=i, name=take_name(prompt, i, n), final_prompt=prompt,
                       sample_rate=SR, duration=round(len(wav) / SR, 3), attach=[f"take_{i + 1}_spectrogram.png"])
        return {"meta": {"clip_seconds": round(512 * STEP_MS / 1000, 2)}}


if __name__ == "__main__":
    serve(MusicRiffusion())
