"""MOSS-SoundEffect-v2.0: text-to-SFX/ambience (diffusers-style pipeline). Adapted from the earlier standalone SFX worker.
venv: envs/audio (tested; pkg installed --no-deps) or the original SFX env (transformers 4.57.1, diffusers 0.37.1)
usage: python moss_soundeffect.py OUT.wav "prompt" SECONDS"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
from _bench import Bench
import torch
from moss_soundeffect_v2 import MossSoundEffectPipeline

out, prompt, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
b = Bench("OpenMOSS-Team/MOSS-SoundEffect-v2.0")
pipe = MossSoundEffectPipeline.from_pretrained("OpenMOSS-Team/MOSS-SoundEffect-v2.0", torch_dtype=torch.bfloat16, device="cuda")
b.loaded()
audio = pipe(prompt=prompt, seconds=secs, num_inference_steps=100, cfg_scale=4.0, negative_prompt="", seed=42)
b.done(audio[0].detach().float().cpu().numpy().T, pipe.sample_rate, out)
