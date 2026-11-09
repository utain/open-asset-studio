"""Seamless PBR materials, GPU part (runs in envs/qwen, read-only; no new packages): the PBRify_Remix SPAN networks (CC0).

  task material_maps   inputs[0] = the 1024 px tileable base colour (PNG); params: size (1024 | 2048 | 4096)
      1. base colour at `size`: the 4x PBRify upscaler (circular padding, fp16) -> 4096, then the exact periodic 2x / 1x box filter
      2. map_res = min(size, 2048); the source at map_res (box-downsampled for 4096)
      3. NormalV3 / Height / RoughnessV2 (1x SPAN, circular, so the maps stay tileable; 1024 tiles with a 32 px wrap overlap above 2048)
      4. writes basecolor_up.png (8-bit), normal_dx.png (8-bit, DirectX Y-), height_raw.png (16-bit, luminance), rough_raw.png (8-bit)
  The worker does no CPU post-processing: delight, the OpenGL flip, levels, AO, ORM and the export packing stay in
  app/plugins/material.py + workers/material_common.py (the export-time repack reuses them without a GPU).

Weights: ~/.cache/pbrify/1.7.2/ (scripts/fetch_pbrify.py, ./install.sh model material-zimage). Re-parameterisation of the Conv3XC
blocks: workers/span_lite.py. docs/research/materials.md has the measurements.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os
import time
from pathlib import Path

from studio_worker import Backend, InputError, serve

PBRIFY_DIR = Path(os.environ.get("STUDIO_PBRIFY_DIR") or os.path.expanduser("~/.cache/pbrify/1.7.2"))
NETS = {"normal": "1x-PBRify_NormalV3.pth", "height": "1x-PBRify_Height.pth", "rough": "1x-PBRify_RoughnessV2.pth",
        "up4": "4x-PBRify_UpscalerSPANV4.pth"}
FULL_FRAME_MAX = 2048          # above this the 1x nets run in 1024 tiles
TILE = 1024


def weight_path(key: str) -> Path:
    return PBRIFY_DIR / NETS[key]


class MaterialBackend(Backend):
    name = "material"
    max_loaded = 4

    def probe(self, model):
        import torch  # noqa: F401

        missing = [n for n in NETS.values() if not (PBRIFY_DIR / n).is_file()]
        if missing:
            return False, f"PBRify weights missing in {PBRIFY_DIR}: {', '.join(missing)}"
        return True, ""

    # the four nets are ~9 MB each: load them lazily and keep them resident
    def load(self, model, ctx):
        ctx.progress(0, 0, "loading", "Loading the PBRify networks", interruptible=False)
        return {"nets": {}}

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        if isinstance(h, dict):
            h.get("nets", {}).clear()

    @staticmethod
    def _net(h, key, ctx):
        if key not in h["nets"]:
            from span_lite import load_span

            p = weight_path(key)
            if not p.is_file():
                raise InputError(f"PBRify weights missing: {p}. Run ./install.sh model material-zimage")
            t = time.time()
            h["nets"][key] = load_span(p, "cuda", circular=True)
            ctx.log(f"PBRify {key} loaded in {time.time() - t:.2f} s")
        return h["nets"][key]

    def run(self, job, ctx):
        if job.task != "material_maps":
            raise InputError(f"The material worker does not know the task '{job.task}'.")
        import numpy as np
        import torch
        from PIL import Image

        import material_common as mc
        from span_lite import run_span

        h = self.handle(job.model_id)
        inp = job.input(type="image")
        if inp is None:
            raise InputError("The material maps step needs the base colour image.")
        size = int(job.params.get("size") or 1024)
        if size not in (1024, 2048, 4096):
            raise InputError("size must be 1024, 2048 or 4096.")
        with Image.open(inp["path"]) as im:
            im = im.convert("RGB")
            if im.size != (1024, 1024):
                if im.width != im.height:
                    raise InputError("The base colour must be square.")
                ctx.log(f"base colour {im.width} px resized to 1024 (wrap-safe)")
                im = mc.resize_wrap(im, (1024, 1024))
            base = mc.to_float(np.asarray(im))
        torch.cuda.reset_peak_memory_stats()
        timings = {}
        t0 = time.time()

        def tick(name, t):
            torch.cuda.synchronize()
            timings[name] = round(time.time() - t, 3)

        # 1. base colour at `size`
        ctx.check()
        if size == 1024:
            alb = base
        else:
            t = time.time()
            ctx.progress(0, 5, "upscaling", f"Upscaling the base colour to {size} px")
            up = run_span(self._net(h, "up4", ctx), base, fp16=True)          # 4096, circular: exact wrap
            alb = up if size == 4096 else mc.box_down(up, 2)
            tick("upscale", t)
        # 2. map source
        map_res = min(size, FULL_FRAME_MAX)
        src = alb if alb.shape[0] == map_res else mc.box_down(alb, alb.shape[0] // map_res)
        fp16 = map_res > 1024
        tile = TILE if map_res > FULL_FRAME_MAX else 0
        # 3. the three maps
        raw = {}
        for i, key in enumerate(("normal", "height", "rough")):
            ctx.check()
            t = time.time()
            ctx.progress(1 + i, 5, "maps", f"Estimating {('normal', 'height', 'roughness')[i]}")
            raw[key] = run_span(self._net(h, key, ctx), src, fp16=fp16, tile=tile)
            tick(key, t)
        # 4. files
        out = job.out_dir
        t = time.time()
        files = {"basecolor_up.png": mc.to_pil(alb), "normal_dx.png": mc.to_pil(raw["normal"]),
                 "height_raw.png": mc.to_pil(mc.luminance(raw["height"]), 16), "rough_raw.png": mc.to_pil(mc.luminance(raw["rough"]))}
        for name, im in files.items():
            im.save(out / name)
        timings["write"] = round(time.time() - t, 3)
        peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        timings["total"] = round(time.time() - t0, 3)
        ctx.progress(5, 5, "saving", "maps ready")
        for name in files:
            ctx.output(out / name, "image", name=name.rsplit(".", 1)[0])
        ctx.log(f"maps at {map_res} px (base {size}): {timings}, peak {peak} GB")
        return {"meta": {"size": size, "map_res": map_res, "timings": timings, "peak_gb": peak, "fp16": fp16}}


if __name__ == "__main__":
    serve(MaterialBackend())
