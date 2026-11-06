"""Stable Audio 3 (medium = music/ambience/sfx up to ~380 s; small-sfx = sfx up to 120 s).
venv: envs/audio  (stable-audio-tools @ git 3241adb, torch 2.9.1 cu130, transformers 5.x)
usage: python stable_audio_3.py OUT.wav [medium|small-sfx] "prompt" SECONDS [SEED]"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _bench import Bench
import torch
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

out = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else "small-sfx"
prompt = sys.argv[3] if len(sys.argv) > 3 else "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style"
seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

b = Bench(f"stabilityai/stable-audio-3-{variant}")
model, cfg = get_pretrained_model(f"stabilityai/stable-audio-3-{variant}")
model = model.to("cuda").eval().requires_grad_(False).to(torch.float16)  # gradio default: model_half
sr, max_samples = cfg["sample_rate"], cfg["sample_size"]
b.loaded()

with torch.inference_mode():
    audio = generate_diffusion_cond_inpaint(
        model,
        steps=8,                 # ARC-distilled rf_denoiser: 8 steps, cfg 1 (UI defaults)
        cfg_scale=1.0,
        conditioning=[{"prompt": prompt, "seconds_total": seconds}],
        sample_size=max_samples,  # adapt_duration (auto) shrinks it to seconds+6 s
        sampler_type="pingpong",
        seed=seed,
        device="cuda",
    )
wav = audio[0, :, : int(seconds * sr)].float().clamp(-1, 1).cpu().numpy().T  # (T, 2)
b.done(wav, sr, out)
