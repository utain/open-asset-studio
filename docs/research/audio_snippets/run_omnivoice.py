"""OmniVoice (k2-fsa, 600+ langs incl. Thai) and hotdogs/omnivoice-thai (Thai fine-tune).
Modes: clone (ref_audio+ref_text), design (instruct = comma-separated attributes), auto (neither).
venv: envs/audio (omnivoice==0.2.1, transformers>=5.3 needed for clone; torch 2.9.1 cu130). 24 kHz mono.
usage: python run_omnivoice.py OUT.wav [k2-fsa/OmniVoice|hotdogs/omnivoice-thai] clone|design|auto TEXT [REF.wav "ref text"]"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _bench import Bench
import torch
from omnivoice import OmniVoice

out, repo, mode, text = sys.argv[1:5]
b = Bench(repo)
model = OmniVoice.from_pretrained(repo, device_map="cuda:0", dtype=torch.float16)
b.loaded()
torch.manual_seed(42)
kw = dict(num_step=32, guidance_scale=2.0)          # defaults; num_step=16 is ~2x faster
if mode == "clone":
    kw.update(ref_audio=sys.argv[5], ref_text=sys.argv[6])
elif mode == "design":
    kw.update(instruct="male, middle-aged, low pitch")  # gender/age/pitch/style(whisper)/accent/dialect
audio = model.generate(text=text, **kw)               # also: language="th" (name or code), speed=1.0, duration=secs
b.done(audio[0], model.sampling_rate, out)
