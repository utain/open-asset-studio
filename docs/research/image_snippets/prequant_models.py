"""Models whose cached diffusers repo has NO transformer weights, paired with the unsloth INT8 transformer file.
The INT8 weights are dequantised to bf16 at load (loaders.py) -> runs in envs/qwen, no torchao needed.

  python prequant_models.py zimage  "prompt"     # Z-Image-Turbo  : unsloth/Z-Image-Turbo + unsloth/Z-Image-Turbo-FP8
  python prequant_models.py lumina  "prompt"     # Lumina-Image-2 : unsloth/Lumina-Image-2.0 + unsloth/Lumina-Image-2.0-FP8 (.pt)
  python prequant_models.py hunyuan "prompt"     # HunyuanImage-2.1: hunyuanvideo-community/HunyuanImage-2.1-Diffusers
                                                 #                   + unsloth/HunyuanImage-2.1-FP8 (.pt)
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
import diffusers
import transformers
from loaders import snap, fast_te, load_unsloth_safetensors, load_unsloth_legacy_pt, transformer_from_sd


def load(model: str):
    if model == "zimage":
        S = snap("unsloth/Z-Image-Turbo")
        sd, _ = load_unsloth_safetensors(snap("unsloth/Z-Image-Turbo-FP8") + "/Z-Image-Turbo-INT8.safetensors")
        tr = transformer_from_sd(diffusers.ZImageTransformer2DModel, S, sd); del sd
        te = fast_te(transformers.Qwen3Model, S + "/text_encoder")
        pipe = diffusers.ZImagePipeline.from_pretrained(S, dtype=torch.bfloat16, transformer=tr, text_encoder=te)
        defaults = dict(num_inference_steps=9, guidance_scale=0.0)  # 9 steps = 8 NFE (distilled); no CFG/negative
    elif model == "lumina":
        S = snap("unsloth/Lumina-Image-2.0")
        sd, _ = load_unsloth_legacy_pt(snap("unsloth/Lumina-Image-2.0-FP8") + "/Lumina-Image-2.0-INT8.pt")
        tr = transformer_from_sd(diffusers.Lumina2Transformer2DModel, S, sd); del sd
        te = fast_te(transformers.Gemma2Model, S + "/text_encoder")
        pipe = diffusers.Lumina2Pipeline.from_pretrained(S, dtype=torch.bfloat16, transformer=tr, text_encoder=te)
        defaults = dict(num_inference_steps=30, guidance_scale=4.0, cfg_trunc_ratio=0.25, cfg_normalization=True,
                        negative_prompt="blurry, low quality, watermark")
    elif model == "hunyuan":
        S = snap("hunyuanvideo-community/HunyuanImage-2.1-Diffusers")
        sd, _ = load_unsloth_legacy_pt(snap("unsloth/HunyuanImage-2.1-FP8") + "/HunyuanImage-2.1-INT8.pt")
        tr = transformer_from_sd(diffusers.HunyuanImageTransformer2DModel, S, sd); del sd
        pipe = diffusers.HunyuanImagePipeline.from_pretrained(
            S, dtype=torch.bfloat16, transformer=tr,
            text_encoder=fast_te(transformers.Qwen2_5_VLForConditionalGeneration, S + "/text_encoder"),
            text_encoder_2=fast_te(transformers.T5EncoderModel, S + "/text_encoder_2"))
        defaults = dict(num_inference_steps=50, distilled_guidance_scale=3.25, negative_prompt="")
    else:
        raise SystemExit(__doc__)
    return pipe.to("cuda"), defaults


if __name__ == "__main__":
    model, prompt = sys.argv[1], sys.argv[2]
    res = int(sys.argv[3]) if len(sys.argv) > 3 else (2048 if model == "hunyuan" else 1024)
    pipe, kw = load(model)
    img = pipe(prompt=prompt, width=res, height=res, generator=torch.Generator("cuda").manual_seed(42), **kw).images[0]
    img.save(f"{model}.png")
