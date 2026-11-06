"""Qwen-Image-2.1 (Qwen/Qwen-Image-2.1, bf16, 31 GB) -- t2i, edit, multi-reference (1-10 images), native RGBA.
venv: envs/qwen   (HF_HUB_OFFLINE=1 python qwen_image_21.py)
Measured @1024px (idle GPU): load 28.5 s fast path vs ~3 min stock; 1.93 s/step t2i. Contended: 4.3-4.6 s/step t2i, 4.9 s/step edit (+~30 s to first step
for the vision encode); peak torch alloc 37-39 GB.
Optional --int8: unsloth/Qwen-Image-2.1-FP8 INT8 transformer dequantised to bf16 (same speed/memory as bf16, see doc).
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from transformers import Qwen3VLForConditionalGeneration
from loaders import snap, fast_diffusers, fast_te, load_unsloth_safetensors, transformer_from_sd

S = snap("Qwen/Qwen-Image-2.1")
if "--int8" in sys.argv:
    sd, _ = load_unsloth_safetensors(snap("unsloth/Qwen-Image-2.1-FP8") + "/Qwen-Image-2.1-INT8.safetensors")
    tr = transformer_from_sd(QwenImage21Transformer2DModel, S, sd); del sd
else:
    tr = fast_diffusers(QwenImage21Transformer2DModel, S, "transformer")
te = fast_te(Qwen3VLForConditionalGeneration, S + "/text_encoder")
pipe = QwenImage21Pipeline.from_pretrained(S, dtype=torch.bfloat16, transformer=tr, text_encoder=te).to("cuda")
g = lambda s=42: torch.Generator("cuda").manual_seed(s)

# 1) text-to-image (output PIL is RGBA; alpha is opaque unless asked for transparency)
img = pipe(prompt="A stylized fantasy treasure chest, gold trim, wooden plaque reading \"LOOT\", game asset, 3/4 view, "
                  "plain white background", width=1024, height=1024, num_inference_steps=25, true_cfg_scale=1.0,
           generator=g()).images[0]
img.save("qwen21_t2i.png")

# 2) transparent asset: start the prompt with the trained trigger sentence
rgba = pipe(prompt="This is an RGBA image with transparency. A cute low-poly fox character, full body, isolated.",
            width=1024, height=1024, num_inference_steps=25, generator=g()).images[0]
rgba.save("qwen21_rgba.png")  # real alpha channel, no matting needed

# 3) edit / multi-reference: image=[1..10 PIL images]; refer to them as "image 1", "image 2" in the prompt
edit = pipe(prompt="Change the chest to red lacquered wood with silver trim, keep everything else identical.",
            image=[img], output_resolution=1024, width=1024, height=1024, num_inference_steps=25,
            generator=g()).images[0]
edit.save("qwen21_edit.png")

# negative prompt only has an effect with true_cfg_scale > 1 (2x cost per step):
#   pipe(prompt=..., negative_prompt="blurry, watermark", true_cfg_scale=4.0, ...)
