"""FLUX.1-Kontext-dev (unsloth/FLUX.1-Kontext-dev, full bf16 diffusers repo, 32 GB) -- instruction edit (1 image) + t2i.
venv: envs/qwen. License: FLUX.1 [dev] Non-Commercial (outputs may be used commercially per BFL terms, the
model itself not for commercial service without a BFL license) -- check before shipping.
Measured @1024px: load 36 s fast path; edit 7.8 s/step (image tokens double the sequence), t2i 2.3 s/step;
peak torch alloc 34 GB. No negative prompt by default (true_cfg_scale>1 enables one at 2x cost).
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from PIL import Image
from diffusers import FluxKontextPipeline, FluxTransformer2DModel
from transformers import CLIPTextModel, T5EncoderModel
from loaders import snap, fast_diffusers, fast_te

S = snap("unsloth/FLUX.1-Kontext-dev")
pipe = FluxKontextPipeline.from_pretrained(
    S, dtype=torch.bfloat16,
    transformer=fast_diffusers(FluxTransformer2DModel, S, "transformer"),
    text_encoder=fast_te(CLIPTextModel, S + "/text_encoder"),
    text_encoder_2=fast_te(T5EncoderModel, S + "/text_encoder_2")).to("cuda")

src = Image.open(sys.argv[1]).convert("RGB")
out = pipe(image=src,  # auto-resized to ~1 MP at a Kontext-preferred aspect ratio (_auto_resize=True)
           prompt="Change the chest to red lacquered wood with silver trim while keeping the same shape, pose, "
                  "plaque text and white background",
           guidance_scale=2.5, num_inference_steps=24, generator=torch.Generator("cuda").manual_seed(42)).images[0]
out.save("kontext_edit.png")
