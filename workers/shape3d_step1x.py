"""Step1X-3D geometry (stepfun-ai/Step1X-3D, Apache-2.0) for the shape3d backend: probe / load / generate hooks.

Only the geometry pipeline is used (1.3B flow-matching DiT + Michelangelo VAE, DINOv2 image condition, skimage marching
cubes). The texture synthesis stage needs nvcc rasterizers, so meshes are white: paint them with Texture (Hunyuan3D 2.1).
Upstream code is imported from vendor/Step1X-3D, never edited. Weights are read from the HF cache (offline).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

STUDIO = Path(os.environ.get("STUDIO_ROOT") or Path(__file__).resolve().parent.parent)
SUB = {"step1x3d": "Step1X-3D-Geometry-1300m"}
REPO = "stepfun-ai/Step1X-3D"


def _path():
    p = str(STUDIO / "vendor" / "Step1X-3D")
    if p not in sys.path:
        sys.path.insert(0, p)


def probe(model):
    import m3d_common as C
    C.shim_missing("diso", "streaming")
    _path()
    from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline  # noqa: F401
    return True, ""


def load(model, ctx):
    import torch
    import m3d_common as C
    C.shim_missing("diso", "streaming")
    _path()
    from huggingface_hub import snapshot_download
    from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
    # the vendored DINOv2-with-registers _init_weights breaks under transformers >= 4.50 (init functions are no-ops that
    # return None); the weights come from the checkpoint anyway, so skip the random init
    from step1x3d_geometry.models.conditional_encoders.dinov2_with_registers import modeling_dinov2_with_registers as _dr
    _dr.Dinov2WithRegistersPreTrainedModel._init_weights = lambda self, module: None
    sub = SUB[model["id"]]
    root = snapshot_download(REPO, allow_patterns=[f"{sub}/*"])
    pipe = Step1X3DGeometryPipeline.from_pretrained(root, subfolder=sub, torch_dtype=torch.bfloat16).to("cuda")
    return {"kind": "step1x", "pipe": pipe}


def generate(h, rgba, p, seed, ctx):
    import torch
    pipe = h["pipe"]
    steps = int(p.get("steps", 30))
    state = {"n": 0}
    tr = pipe.transformer
    if not hasattr(tr, "_oas_forward"):
        tr._oas_forward = tr.forward

        def fwd(*a, **k):
            ctx.check()
            st = getattr(tr, "_oas_state", None)
            if st is not None:
                st["n"] += 1
                ctx.progress(min(st["n"], st["total"]), st["total"], "denoising", f"Step {st['n']}/{st['total']}")
            return tr._oas_forward(*a, **k)

        tr.forward = fwd
    tr._oas_state = {"n": 0, "total": steps}
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # upstream check_inputs rejects PIL / tensor inputs (inverted isinstance test), only a file path passes
    import tempfile
    with tempfile.TemporaryDirectory(prefix="step1x_") as td:
        src = os.path.join(td, "input.png")
        rgba.save(src)
        out = _call(pipe, src, p, steps, gen)
    tr._oas_state = None
    return out.mesh[0]


def _call(pipe, src, p, steps, gen):
    return pipe(src, guidance_scale=float(p.get("guidance", 7.5)), num_inference_steps=steps,
               generator=gen, octree_resolution=int(p.get("octree", 384)), force_remove_background=False,
               foreground_ratio=0.95, do_reduce_face=False, do_remove_floater=True, max_facenum=10**9)
