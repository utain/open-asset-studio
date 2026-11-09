"""Diffusers image worker: every diffusers image model of the studio (runs in envs/qwen, read-only).

Families (picked from the registry id):
  qwen21   Qwen-Image-2.1           t2i, edit / multi-ref (1-10 images), native RGBA, extract (edit + RGBA prompt)
  zimage   Z-Image-Turbo            t2i (unsloth INT8 DiT dequantised to bf16)
  kontext  FLUX.1-Kontext-dev       edit (1 image), t2i
  hunyuan  HunyuanImage-2.1         t2i at 2K (unsloth legacy INT8 .pt dequantised to bf16), real CFG
  lumina   Lumina-Image-2.0         t2i (unsloth legacy INT8 .pt), CFG truncation
  krea     FLUX.1-Krea-dev          t2i (only once its transformer weights are downloaded)

Proven pieces reused (docs/research/image_models.md):
  * fast direct-to-GPU loading: diffusers `device_map="cuda", disable_mmap=True`; transformers models are built as a
    lazy mmap view and their tensors swapped for `safetensors.load_file(device="cuda")` reads (stock .to("cuda") of
    mmapped weights runs at ~170 MB/s here: 3 min instead of 15-30 s). Recipe from engines/hy3d/workers/qwen_worker.py and
    docs/research/image_snippets/loaders.py (copied below with attribution, unchanged in substance).
  * unsloth prequant INT8 (v1 + ConvRot v2 safetensors, legacy torchao .pt) -> bf16 without torchao (loaders.py).
  * Qwen RGBA prompt format, 3D-ready framing, extract prompt: engines/hy3d/app/image_params.py (imported).
  * RGBA alpha clean-up (Qwen leaves a 1-2/255 haze: default threshold 8) and near-opaque alpha -> RGB:
    engines/hy3d/app/imgtools.py (imported).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # the venv is read-only for the studio: no __pycache__ writes into engines/hy3d

import enum
import glob
import inspect
import json
import os
import random
import struct
import time
from pathlib import Path

from studio_worker import Backend, InputError, log, serve, size_from_aspect

STUDIO = Path(os.environ.get("STUDIO_ROOT") or Path(__file__).resolve().parent.parent)
HY3D_APP = Path(os.environ.get("STUDIO_HY3D_APP") or STUDIO / "engines" / "hy3d" / "app")
if str(HY3D_APP) not in sys.path:
    sys.path.append(str(HY3D_APP))   # appended: never shadows this venv's own packages

HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")

FAMILIES = {"qwen-image-2.1": "qwen21", "z-image-turbo": "zimage", "flux1-kontext-dev": "kontext",
            "hunyuanimage-2.1": "hunyuan", "lumina-image-2.0": "lumina", "flux1-krea-dev": "krea"}
GRID = {"qwen21": 32, "hunyuan": 32, "zimage": 16, "kontext": 16, "lumina": 16, "krea": 16}
PIPELINE_CLASS = {"qwen21": "QwenImage21Pipeline", "zimage": "ZImagePipeline", "kontext": "FluxKontextPipeline",
                  "hunyuan": "HunyuanImagePipeline", "lumina": "Lumina2Pipeline", "krea": "FluxPipeline"}


def family_of(model: dict) -> str:
    fam = model.get("family") or FAMILIES.get(model.get("id", ""))
    if fam:
        return fam
    raise InputError(f"The diffusers worker does not know the model '{model.get('id')}'.")


# =========================================================== loaders (from docs/research/image_snippets/loaders.py)
def snap(repo: str) -> str:
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"))
    ref = os.path.join(d, "refs", "main")
    if os.path.exists(ref):
        s = os.path.join(d, "snapshots", open(ref).read().strip())
        if os.path.isdir(s):
            return s
    snaps = sorted(glob.glob(d + "/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"{repo} is not in the local Hugging Face cache ({HUB})")
    return snaps[-1]


def fast_diffusers(cls, path: str, subfolder: str):
    import torch
    return cls.from_pretrained(path, subfolder=subfolder, dtype=torch.bfloat16, device_map="cuda", disable_mmap=True)


def fast_te(cls, path: str, **kw):
    """transformers model straight to the GPU: lazy mmap model, then tensors swapped for device="cuda" reads."""
    import torch
    from safetensors.torch import load_file
    m = cls.from_pretrained(path, dtype=torch.bfloat16, **kw)
    params = dict(m.named_parameters())
    params.update(dict(m.named_buffers()))
    hit = 0
    for f in sorted(glob.glob(path + "/*.safetensors")):
        sd = load_file(f, device="cuda")
        for k, v in sd.items():
            for kk in (k, k.removeprefix("model."), "model." + k):
                t = params.get(kk)
                if t is not None and t.shape == v.shape:
                    t.data = v if v.dtype == t.dtype else v.to(t.dtype)
                    hit += 1
                    break
        del sd
    if hit < len(dict(m.named_parameters())):
        log(f"fast_te {os.path.basename(path)}: {len(dict(m.named_parameters())) - hit} tensors not swapped (copied the slow way)")
    return m.to("cuda")


def _convrot_h256():
    import torch
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
    h = h4
    for _ in range(3):
        h = torch.kron(h, h4)
    return (h / 16.0).float()


def load_unsloth_safetensors(path: str, device="cuda"):
    """unsloth prequant v1 / v2 (ConvRot) safetensors -> bf16 state dict (W = (q - zp) * scale [@ H])."""
    import torch
    from safetensors import safe_open
    with open(path, "rb") as f:
        meta = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))["__metadata__"]
    um = json.loads(meta.get("unsloth_metadata", "{}"))
    rot = set(um.get("activation_rotation_fqns") or [])
    g = int(um.get("activation_rotation_group", 256))
    H = _convrot_h256().to(device) if rot else None
    if rot and g != 256:
        raise RuntimeError("only group-256 ConvRot is implemented")
    sd = {}
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with safe_open(path, "pt", device=device) as f:
            keys = set(f.keys())
            for k in keys:
                if k.startswith("unsloth_root::"):
                    sd[k.removeprefix("unsloth_root::")] = f.get_tensor(k)
            for name, desc in meta.items():
                if not desc.startswith('{"_type"'):
                    continue
                d = json.loads(desc)
                if d["_type"] == "Tensor":
                    if name in keys:
                        sd[name] = f.get_tensor(name)
                    continue
                mod, leaf = name.rsplit(".", 1)
                q = f.get_tensor(f"{mod}._{leaf}_qdata").float()
                if f"{mod}._{leaf}_zero_point" in keys:
                    q = q - f.get_tensor(f"{mod}._{leaf}_zero_point").float()
                w = q * f.get_tensor(f"{mod}._{leaf}_scale").float().reshape(q.shape[0], -1)
                if mod in rot:
                    n, kk = w.shape
                    w = (w.reshape(n, kk // g, g) @ H).reshape(n, kk)
                sd[name] = w.to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return sd, um


def load_unsloth_legacy_pt(path: str, device="cuda"):
    """Legacy torchao .pt (HunyuanImage / Lumina INT8) -> bf16 state dict, with inert stand-ins (weights_only stays on)."""
    import torch

    class _Inert(torch.Tensor):
        __torch_function__ = torch._C._disabled_torch_function_impl

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise RuntimeError(f"stand-in cannot run {func}")

    LAQ = type("LinearActivationQuantizedTensor", (_Inert,), {})
    AQT = type("AffineQuantizedTensor", (_Inert,), {})
    IMP = type("PlainAQTTensorImpl", (_Inert,), {})
    PL = type("PlainLayout", (), {})

    def _act(*a, **k):
        raise RuntimeError("never called")

    ZPD = enum.Enum("ZeroPointDomain", {"INT": 1, "FLOAT": 2, "NONE": 3})
    from torch.torch_version import TorchVersion
    safe = [(LAQ, "torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor"),
            (AQT, "torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor"),
            (IMP, "torchao.dtypes.uintx.plain_layout.PlainAQTTensorImpl"),
            (PL, "torchao.dtypes.utils.PlainLayout"),
            (_act, "torchao.quantization.quant_api._int8_symm_per_token_reduced_range_quant"),
            (ZPD, "torchao.quantization.quant_primitives.ZeroPointDomain"), TorchVersion]
    with torch.serialization.safe_globals(safe):
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    sdq = obj.get("state_dict", obj)
    sd = {}
    for k, v in sdq.items():
        if isinstance(v, _Inert):
            impl = v.__dict__["original_weight_tensor"].__dict__["tensor_impl"].__dict__
            q = impl["int_data"].to(device).float()
            zp = impl.get("zero_point")
            if zp is not None and torch.count_nonzero(zp):
                q = q - zp.to(device).float().reshape(q.shape[0], -1)
            sd[k] = (q * impl["scale"].to(device).float().reshape(q.shape[0], -1)).to(torch.bfloat16)
        else:
            sd[k] = v.to(device)
    del obj, sdq
    return sd


def transformer_from_sd(cls, base_path: str, sd: dict):
    import torch
    from accelerate import init_empty_weights
    with init_empty_weights():
        m = cls.from_config(cls.load_config(base_path, subfolder="transformer"))
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return m.to(device="cuda", dtype=torch.bfloat16).eval()


# ====================================================================== tileable generation (Z-Image)
class TileMode:
    """Seamless (wrap-around) generation for the Z-Image family: latent rolling + a circular-padding VAE (docs/research/materials.md).

    * latent rolling: `transformer.forward` is wrapped. Every call rolls each latent of the list by a random EVEN offset (whole 2x2 patches)
      drawn from random.Random(seed), calls the original, and un-rolls the predictions. The transformer never sees a fixed border, so no
      seam can form there.
    * circular VAE: padding_mode="circular" on every Conv2d with padding in vae.decoder (and vae.encoder for the img2img task), so the
      decoder does not invent an edge.
    The two must always be switched together: rolling alone is WORSE than plain generation (seam p95 1.96-3.16 vs 1.32-2.63), roll +
    circular VAE gives 0.83-1.17. The pipeline is shared with plain jobs, so __exit__ restores forward and zero padding, also on errors."""

    def __init__(self, pipe, seed, encoder: bool = False):
        self.pipe, self.seed, self.encoder = pipe, int(seed), encoder
        self._orig = None
        self._convs: list = []

    @staticmethod
    def circular_convs(module):
        return [m for m in module.modules() if m.__class__.__name__ == "Conv2d"
                and any(int(x) > 0 for x in (m.padding if isinstance(m.padding, tuple) else (1,)))]

    def __enter__(self):
        import torch
        tr = self.pipe.transformer
        self._orig = tr.forward
        rng = random.Random(self.seed)
        orig = self._orig

        def rolled(x, *a, **kw):
            H, W = x[0].shape[-2:]
            sy, sx = rng.randrange(0, H, 2), rng.randrange(0, W, 2)
            xs = [torch.roll(xi, (sy, sx), (-2, -1)) for xi in x]
            out = orig(xs, *a, **kw)
            lst = out[0] if isinstance(out, tuple) else out.sample
            lst = [torch.roll(o, (-sy, -sx), (-2, -1)) for o in lst]
            return (lst,) + tuple(out[1:]) if isinstance(out, tuple) else type(out)(sample=lst)

        tr.forward = rolled
        self._convs = self.circular_convs(self.pipe.vae.decoder) + (self.circular_convs(self.pipe.vae.encoder) if self.encoder else [])
        for m in self._convs:
            m.padding_mode = "circular"
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            self.pipe.transformer.forward = self._orig
        for m in self._convs:
            m.padding_mode = "zeros"
        return False


def tile_meta(img) -> dict:
    import numpy as np
    import material_common as mc
    return {"tileable": True, "tile_method": "latent_roll+circular_vae", "seam": mc.seam_score(np.asarray(img.convert("RGB")))["score"]}


# ====================================================================== image helpers
def _imgtools():
    import imgtools  # engines/hy3d/app/imgtools.py (alpha stats / clean-up)
    return imgtools


def _ip():
    import image_params  # engines/hy3d/app/image_params.py (RGBA / 3D-ready / extract prompt templates)
    return image_params


def compose_prompt(task: str, fam: str, p: dict) -> tuple[str, bool]:
    """(exact text sent to the model, transparent?)."""
    ip = _ip()
    prompt = (p.get("prompt") or "").strip()
    rgba_ok = fam == "qwen21"
    if task == "extract":
        what = prompt or ip.EXTRACT_WHAT_DEFAULT
        body = ip.EXTRACT_PROMPT.format(what=what)
        return f"{ip.RGBA_PREFIX}{body}{ip.RGBA_SUFFIX}", True
    transparent = rgba_ok and (task == "rgba" or bool(p.get("transparent")))
    if prompt.startswith(ip.RGBA_PREFIX.strip()):  # the improver (qwen_rgba style) already wrote the trigger
        prompt = prompt[len(ip.RGBA_PREFIX.strip()):].strip()
        transparent = transparent or rgba_ok
    if task in ("edit", "multi_ref"):
        if not prompt:
            raise InputError("Write the edit instruction first.")
        return ip.compose_edit_prompt(prompt, transparent, "none"), transparent
    if not prompt:
        raise InputError("Write a prompt first.")
    threed = bool(p.get("threed_ready"))
    text = ip.compose_generate_prompt(prompt, transparent, threed, ip.THREED_DEFAULT)
    if threed and fam == "zimage":  # Z-Image turns negative phrasing into the thing itself: positive framing only
        text = text.replace("no cast shadow, no text, no watermark, no other objects, ", "")
    return text, transparent


def edit_size(w: int, h: int, side: int, grid: int) -> tuple[int, int]:
    """Aspect ratio of image 1 at the area of a side x side square (what Qwen does with output_resolution)."""
    ratio = w / max(1, h)
    ww = (side * side * ratio) ** 0.5
    r = lambda v: max(grid, int(round(v / grid)) * grid)  # noqa: E731
    return r(ww), r(ww / ratio)


# ============================================================================ backend
class DiffusersBackend(Backend):
    name = "qwen"
    max_loaded = 2   # e.g. Z-Image drafts + Qwen refine stay resident; the core evicts when memory is short

    def probe(self, model):
        fam = family_of(model)
        import diffusers
        cls = PIPELINE_CLASS[fam]
        if not hasattr(diffusers, cls):
            return False, f"diffusers {diffusers.__version__} has no {cls}"
        try:
            for w in model.get("weights") or []:
                if "hf" in w:
                    snap(w["hf"])
        except FileNotFoundError as e:
            return False, str(e)
        return True, ""

    # ------------------------------------------------------------------ load
    def load(self, model, ctx):
        import torch
        import diffusers
        import transformers
        fam = family_of(model)
        t = time.time()
        ctx.progress(0, 0, "loading", f"Loading {model.get('name')} (direct to GPU)", interruptible=False)
        extra = {}
        if fam == "qwen21":
            S = snap("Qwen/Qwen-Image-2.1")
            tr = fast_diffusers(diffusers.QwenImage21Transformer2DModel, S, "transformer")
            te = fast_te(transformers.Qwen3VLForConditionalGeneration, S + "/text_encoder")
            pipe = diffusers.QwenImage21Pipeline.from_pretrained(S, dtype=torch.bfloat16, transformer=tr, text_encoder=te)
        elif fam == "zimage":
            S = snap("unsloth/Z-Image-Turbo")
            sd, _ = load_unsloth_safetensors(snap("unsloth/Z-Image-Turbo-FP8") + "/Z-Image-Turbo-INT8.safetensors")
            tr = transformer_from_sd(diffusers.ZImageTransformer2DModel, S, sd)
            del sd
            te = fast_te(transformers.Qwen3Model, S + "/text_encoder")
            pipe = diffusers.ZImagePipeline.from_pretrained(S, dtype=torch.bfloat16, transformer=tr, text_encoder=te)
        elif fam in ("kontext", "krea"):
            repo = "unsloth/FLUX.1-Kontext-dev" if fam == "kontext" else "unsloth/FLUX.1-Krea-dev"
            S = snap(repo)
            cls = diffusers.FluxKontextPipeline if fam == "kontext" else diffusers.FluxPipeline
            pipe = cls.from_pretrained(
                S, dtype=torch.bfloat16,
                transformer=fast_diffusers(diffusers.FluxTransformer2DModel, S, "transformer"),
                text_encoder=fast_te(transformers.CLIPTextModel, S + "/text_encoder"),
                text_encoder_2=fast_te(transformers.T5EncoderModel, S + "/text_encoder_2"))
        elif fam == "hunyuan":
            S = snap("hunyuanvideo-community/HunyuanImage-2.1-Diffusers")
            ctx.progress(0, 0, "loading", "Loading HunyuanImage-2.1: dequantising the 17 B DiT (cold load ~4 min)", interruptible=False)
            sd = load_unsloth_legacy_pt(snap("unsloth/HunyuanImage-2.1-FP8") + "/HunyuanImage-2.1-INT8.pt")
            tr = transformer_from_sd(diffusers.HunyuanImageTransformer2DModel, S, sd)
            del sd
            pipe = diffusers.HunyuanImagePipeline.from_pretrained(
                S, dtype=torch.bfloat16, transformer=tr,
                text_encoder=fast_te(transformers.Qwen2_5_VLForConditionalGeneration, S + "/text_encoder"),
                text_encoder_2=fast_te(transformers.T5EncoderModel, S + "/text_encoder_2"))
            extra = {"guider": getattr(pipe, "guider", None), "ocr_guider": getattr(pipe, "ocr_guider", None)}
        elif fam == "lumina":
            S = snap("unsloth/Lumina-Image-2.0")
            sd = load_unsloth_legacy_pt(snap("unsloth/Lumina-Image-2.0-FP8") + "/Lumina-Image-2.0-INT8.pt")
            tr = transformer_from_sd(diffusers.Lumina2Transformer2DModel, S, sd)
            del sd
            te = fast_te(transformers.Gemma2Model, S + "/text_encoder")
            pipe = diffusers.Lumina2Pipeline.from_pretrained(S, dtype=torch.bfloat16, transformer=tr, text_encoder=te)
        else:
            raise InputError(f"unknown family {fam}")
        pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        torch.cuda.synchronize()
        accepts = set(inspect.signature(pipe.__call__).parameters)
        ctx.log(f"{model.get('name')} ready in {time.time() - t:.1f} s ({torch.cuda.memory_allocated() / 2**30:.1f} GB on the GPU)")
        return {"pipe": pipe, "family": fam, "accepts": accepts, **extra}

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        if isinstance(h, dict):
            h.clear()

    # ------------------------------------------------------------------- run
    def _inputs(self, job):
        """Image inputs in prompt order: role 'image' first (image 1), then references (image 2, 3 ...)."""
        it = _imgtools()
        ims = [i for i in job.inputs if i.get("type") == "image"]
        ims.sort(key=lambda i: 0 if i.get("role") in ("image", "input") else 1)
        out = []
        for i in ims:
            im = it.open_checked(i["path"])
            out.append((i, im))
        return out

    def run(self, job, ctx):
        import torch
        from PIL import Image
        it = _imgtools()
        h = self.handle(job.model_id)
        pipe, fam, accepts = h["pipe"], h["family"], h["accepts"]
        p = job.params
        task = job.task
        grid = GRID[fam]
        inputs = self._inputs(job)
        needs_image = task in ("edit", "multi_ref", "extract")
        if task == "tileable":
            if fam != "zimage":
                raise InputError(f"{job.model.get('name')} cannot make an image tileable (Z-Image Turbo only).")
            return self._tileable(job, ctx, h, inputs)
        if needs_image and not inputs:
            raise InputError("Add the image to edit first.")
        if task == "t2i" and fam == "kontext":
            inputs = []
        max_refs = int((job.model.get("features") or {}).get("max_refs") or 1)
        if needs_image and len(inputs) > max_refs:
            raise InputError(f"{job.model.get('name')} takes at most {max_refs} image(s); remove {len(inputs) - max_refs}.")
        tileable = bool(p.get("tileable")) and task == "t2i" and fam == "zimage"
        if tileable:
            p = dict(p, threed_ready=False)       # a seamless texture has no single object to frame
        final, transparent = compose_prompt(task, fam, p)
        n = 1 if task == "extract" else max(1, int(p.get("batch", 1)))
        steps = int(p.get("steps") or 25)
        seed0 = int(p.get("seed", -1))
        if seed0 < 0:
            seed0 = random.randrange(2**31)
        side = int(p.get("size") or 1024)
        if needs_image:
            first = inputs[0][1]
            w, hgt = edit_size(first.width, first.height, side, grid)
        else:
            w, hgt = size_from_aspect(p.get("aspect", "1:1"), side, grid)
        neg = (p.get("negative") or "").strip()
        kw = {"prompt": final, "num_inference_steps": steps}
        if fam == "qwen21":
            true_cfg = float(p.get("true_cfg", 1.0))
            kw.update(width=w, height=hgt, true_cfg_scale=true_cfg, use_kv_cache=bool(p.get("kv_cache", True)))
            if true_cfg > 1.0:
                kw["negative_prompt"] = neg or " "
            if inputs:
                # RGBA references keep their alpha; Qwen's VAE is 4-channel
                kw.update(image=[im.convert("RGBA") for _, im in inputs], output_resolution=int(p.get("output_resolution", 1024)))
        elif fam == "zimage":
            kw.update(width=w, height=hgt, guidance_scale=0.0)
        elif fam == "kontext":
            g = float(p.get("guidance", 2.5))
            kw.update(guidance_scale=g)
            if inputs:
                src = inputs[0][1]
                kw.update(image=it.flatten(src) if src.mode == "RGBA" else src.convert("RGB"), max_area=side * side)
            else:
                kw.update(width=w, height=hgt)
        elif fam == "krea":
            kw.update(width=w, height=hgt, guidance_scale=float(p.get("guidance", 4.0)))
        elif fam == "hunyuan":
            g = float(p.get("guidance", 3.5))
            for key in ("guider", "ocr_guider"):
                base = h.get(key)
                if base is not None:
                    setattr(pipe, key, base.new(guidance_scale=g) if hasattr(base, "new") else base)
            kw.update(width=w, height=hgt, negative_prompt=neg or None)
        elif fam == "lumina":
            kw.update(width=w, height=hgt, guidance_scale=float(p.get("guidance", 4.0)),
                      cfg_trunc_ratio=float(p.get("cfg_trunc_ratio", 0.25)), cfg_normalization=True,
                      negative_prompt=neg or "blurry, low quality, watermark")
        kw = {k: v for k, v in kw.items() if k in accepts}
        if neg and "negative_prompt" not in kw and fam not in ("zimage", "kontext", "krea"):
            ctx.log("negative prompt ignored (needs True CFG > 1 on Qwen-Image)")
        cb_ok = "callback_on_step_end" in accepts
        alpha_thr = int(p.get("alpha_clean", 8))
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(n):
            ctx.check()
            seed = (seed0 + i) % (2**32)
            t = time.time()
            label = f"image {i + 1}/{n}: " if n > 1 else ""
            ctx.progress(0, steps, "encoding", f"{label}encoding prompt" + (" and images" if inputs else ""))
            call = dict(kw, generator=torch.Generator("cuda").manual_seed(seed))
            if cb_ok:
                call["callback_on_step_end"] = ctx.step_callback(steps, "denoising", label)
            if tileable:
                with TileMode(pipe, seed):
                    out = pipe(**call).images[0]
            else:
                out = pipe(**call).images[0]
            if not isinstance(out, Image.Image):
                import numpy as np
                out = Image.fromarray(np.asarray(out))
            # Qwen always returns RGBA: keep the alpha only when it is real, then clean the faint haze
            if out.mode == "RGBA" and it.alpha_stats(out)[0]:
                out = it.clean_alpha(out, alpha_thr)
            else:
                out = out.convert("RGB")
            has_alpha, frac = it.alpha_stats(out)
            if transparent and not has_alpha:
                ctx.log(f"image {i + 1}: asked for transparency but the model returned an opaque image")
            f = job.out_dir / f"image_{i + 1}.png"
            out.save(f)
            dt = round(time.time() - t, 2)
            times.append(dt)
            name = None
            if task == "extract":
                name = f"Cut-out: {(p.get('prompt') or 'main subject').strip()[:50]}"
            extra = tile_meta(out) if tileable else {}
            ctx.output(f, "image", seed=seed, index=i, final_prompt=final, name=name, seconds=dt,
                       size=[out.width, out.height], transparent_fraction=round(frac, 4), **extra)
            ctx.progress(steps, steps, "saving", f"{label}saved ({dt:.0f} s)")
        peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        ctx.log(f"{n} image(s) at {w}x{hgt}: {', '.join(f'{x:.1f}s' for x in times)} (peak {peak} GB)")
        return {"meta": {"family": fam, "width": w, "height": hgt, "final_prompt": final, "seconds": times, "peak_gpu_gb": peak}}

    def _tileable(self, job, ctx, h, inputs):
        """Task `tileable`: Z-Image img2img at low strength with latent rolling + circular VAE (encoder and decoder) makes an image wrap
        around. The result is a redraw (mean abs difference to the source 0.05-0.14 at strength 0.35), not the original pixels."""
        import numpy as np
        import torch
        import diffusers
        import material_common as mc

        if not inputs:
            raise InputError("Add the image to make tileable first.")
        p = job.params
        pipe = h["pipe"]
        if "i2i" not in h:                      # shares every module with the t2i pipeline: no extra memory
            # from_pipe defaults to torch_dtype=float32 and casts the SHARED modules in place (4x slower, +20 GB, and it would slow every later
            # t2i job in this worker): pass the dtype the modules already have.
            h["i2i"] = diffusers.ZImageImg2ImgPipeline.from_pipe(pipe, torch_dtype=torch.bfloat16)
            h["i2i"].set_progress_bar_config(disable=True)
            h["i2i_accepts"] = set(inspect.signature(h["i2i"].__call__).parameters)
        i2i = h["i2i"]
        src = inputs[0][1].convert("RGB")
        side = int(p.get("size") or 1024)
        k = side / max(src.size)
        w, hgt = (max(16, int(round(v * k / 16)) * 16) for v in src.size)
        src = mc.resize_wrap(src, (w, hgt)) if (w, hgt) != src.size else src
        prompt = (p.get("prompt") or "").strip() or "seamless tileable texture"
        steps = int(p.get("steps") or 9)
        strength = float(p.get("strength") or 0.35)
        seed = int(p.get("seed", -1))
        if seed < 0:
            seed = random.randrange(2**31)
        kw = dict(prompt=prompt, image=src, strength=strength, width=w, height=hgt, num_inference_steps=steps, guidance_scale=0.0,
                  generator=torch.Generator("cuda").manual_seed(seed))
        if "callback_on_step_end" in h["i2i_accepts"]:
            kw["callback_on_step_end"] = ctx.step_callback(max(1, int(steps * strength)), "denoising", "")
        before = mc.seam_score(np.asarray(src))["score"]
        torch.cuda.reset_peak_memory_stats()
        ctx.check()
        ctx.progress(0, steps, "encoding", "encoding the image")
        t = time.time()
        phases: dict = {}
        hooks = []

        def watch(mod, name):
            def pre(*_):
                torch.cuda.synchronize()
                phases.setdefault(name, [0, 0.0, 0.0])[2] = time.time()

            def post(*_):
                torch.cuda.synchronize()
                ph = phases[name]
                ph[0] += 1
                ph[1] += time.time() - ph[2]
            hooks.append(mod.register_forward_pre_hook(pre))
            hooks.append(mod.register_forward_hook(post))
        for mod, name in ((pipe.vae.encoder, "vae_encode"), (pipe.transformer, "transformer"), (pipe.vae.decoder, "vae_decode")):
            watch(mod, name)
        try:
            with TileMode(pipe, seed, encoder=True):
                out = i2i(**kw).images[0]
        finally:
            for hk in hooks:
                hk.remove()
        torch.cuda.synchronize()
        dt = round(time.time() - t, 2)
        ctx.log("tileable phases: " + ", ".join(f"{k} {v[1]:.1f}s/{v[0]}" for k, v in phases.items()))
        out = out.convert("RGB")
        f = job.out_dir / "image_1.png"
        out.save(f)
        diff = float(np.abs(mc.to_float(np.asarray(out)) - mc.to_float(np.asarray(src.resize(out.size)))).mean())
        peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        ctx.output(f, "image", seed=seed, index=0, final_prompt=prompt, seconds=dt, size=[out.width, out.height],
                   tileable=True, tile_method="regenerate", strength=strength, seam_before=before,
                   seam=mc.seam_score(np.asarray(out))["score"], diff_from_source=round(diff, 4))
        ctx.log(f"tileable {w}x{hgt}, strength {strength}: {dt:.1f} s (peak {peak} GB)")
        return {"meta": {"family": "zimage", "width": w, "height": hgt, "seconds": [dt], "peak_gpu_gb": peak}}


if __name__ == "__main__":
    serve(DiffusersBackend())
