"""Shape-only image -> 3D backend (envs/shape3d): TripoSG, Hunyuan3D-2mini (+ turbo), Step1X-3D geometry.

  task shape   input role `image`  -> mesh asset: `mesh.glb` (untextured, +Y up) + `input.png` (cleaned RGBA conditioning image)

These are geometry-only models (white mesh). They need no CUDA extensions, so they run here: the `diso` iso-surface kernel of
TripoSG (CUDA) is stubbed and its hierarchical marching-cubes decoder used instead; Hunyuan3D-2 uses its skimage marching
cubes.  To paint a texture, send the mesh to the Texture tab with hunyuan3d-2.1 (the asset keeps `input.png`, which that
worker uses as the reference image).  Upstream code is imported from vendor/<repo>, never edited.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import gc
import os
import time
from pathlib import Path

import m3d_common as C
from studio_worker import Backend, Cancelled, InputError, serve

STUDIO = Path(os.environ.get("STUDIO_ROOT") or Path(__file__).resolve().parent.parent)
VENDOR = STUDIO / "vendor"


def _add_path(*names):
    for n in names:
        p = str(VENDOR / n)
        if p not in sys.path:
            sys.path.insert(0, p)


HY_SUB = {  # registry id -> (dit subfolder, flashvdm)
    "hunyuan3d-2mini-turbo": ("hunyuan3d-dit-v2-mini-turbo", True),
    "hunyuan3d-2mini": ("hunyuan3d-dit-v2-mini", False),
}


class Shape3DBackend(Backend):
    name = "shape3d"
    max_loaded = 1

    def probe(self, model):
        mid = model["id"]
        if mid == "triposg":
            C.shim_missing("diso")
            _add_path("TripoSG")
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # noqa: F401
        elif mid in HY_SUB:
            _add_path("Hunyuan3D-2")
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        elif mid.startswith("step1x3d"):
            from shape3d_step1x import probe
            return probe(model)
        return True, ""

    def load(self, model, ctx):
        import torch
        mid = model["id"]
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        if mid == "triposg":
            C.shim_missing("diso")
            _add_path("TripoSG")
            from huggingface_hub import snapshot_download
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline
            d = snapshot_download("VAST-AI/TripoSG")
            pipe = TripoSGPipeline.from_pretrained(d).to("cuda", torch.float16)
            return {"kind": "triposg", "pipe": pipe}
        if mid in HY_SUB:
            _add_path("Hunyuan3D-2")
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            sub, turbo = HY_SUB[mid]
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mini", subfolder=sub, use_safetensors=True,
                                                                    device="cuda")
            if turbo:
                pipe.enable_flashvdm(mc_algo="mc")
            return {"kind": "hy2mini", "pipe": pipe, "turbo": turbo}
        from shape3d_step1x import load as step_load
        return step_load(model, ctx)

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        if isinstance(h, dict):
            h.clear()
        gc.collect()

    def run(self, job, ctx):
        if job.task != "shape":
            raise InputError(f"{job.model.get('name', job.model_id)} cannot do '{job.task}'.")
        import torch
        inp = job.input(role="image") or job.input(type="image")
        if inp is None:
            raise InputError("Add the image to turn into 3D.")
        p = job.params
        h = self.handle(job.model_id)
        kind = h["kind"]
        t = {}
        t0 = time.time()
        ctx.progress(0, 1, "image", "Preparing the image")
        rgba = C.prep_object_image(inp["path"], bool(p.get("remove_bg", True)), float(p.get("foreground_ratio", 0.85)))
        out = job.out_dir / "asset"
        out.mkdir(parents=True, exist_ok=True)
        rgba.save(out / "input.png")
        t["prep_s"] = round(time.time() - t0, 2)
        ctx.check()
        seed = C.seed_of(p)
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        if kind == "triposg":
            mesh = self._triposg(h["pipe"], rgba, p, seed, ctx)
        elif kind == "hy2mini":
            mesh = self._hy2(h["pipe"], rgba, p, seed, ctx)
        else:
            from shape3d_step1x import generate
            mesh = generate(h, rgba, p, seed, ctx)
        t["gen_s"] = round(time.time() - t1, 2)
        ctx.check()
        mesh = _postprocess(mesh, int(p.get("max_faces", 0) or 0), ctx)
        info = C.export_glb(out / "mesh.glb", mesh.vertices, mesh.faces)
        peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        ctx.log(f"timings {t}, peak GPU {peak} GB, {info}")
        ctx.output(out / "mesh.glb", "mesh", seed=seed, name=f"{(inp.get('name') or 'mesh')[:50]} ({job.model.get('name', kind)})",
                   attach=["input.png"], textured=False, stats=info)
        return {"meta": {"timings": t, "peak_gpu_gb": peak}}

    # --------------------------------------------------------------- TripoSG
    def _triposg(self, pipe, rgba, p, seed, ctx):
        import numpy as np
        import torch
        import trimesh
        from PIL import Image
        fg = C.crop_resize_foreground(rgba, float(p.get("foreground_ratio", 0.85)), 1024)
        img = Image.fromarray((np.asarray(C.composite_gray(fg, 1.0))))      # RGB over white, like upstream
        steps = int(p.get("steps", 50))
        depth = {256: 8, 512: 9, 768: 9, 1024: 10}.get(int(p.get("octree", 512)), 9)
        with torch.no_grad():
            out = pipe(image=img, generator=torch.Generator(device="cuda").manual_seed(seed), num_inference_steps=steps,
                       guidance_scale=float(p.get("guidance", 7.0)), use_flash_decoder=False,
                       dense_octree_depth=min(8, depth), hierarchical_octree_depth=depth,
                       callback_on_step_end=ctx.step_callback(steps, "denoising", "TripoSG "))
        ctx.progress(steps, steps, "mesh", "Marching cubes done")
        v, f = out.samples[0]
        return trimesh.Trimesh(np.asarray(v, dtype="float32"), np.ascontiguousarray(f), process=False)

    # ----------------------------------------------------------- Hunyuan3D-2mini
    def _hy2(self, pipe, rgba, p, seed, ctx):
        import torch
        steps = int(p.get("steps", 5))
        # Hunyuan3D-2 conditions on an RGBA image (its own preprocessor crops / pads)
        fg = C.crop_resize_foreground(rgba, float(p.get("foreground_ratio", 0.85)), 1024)
        with torch.no_grad():
            mesh = pipe(image=fg, num_inference_steps=steps, guidance_scale=float(p.get("guidance", 5.0)),
                        octree_resolution=int(p.get("octree", 384)), num_chunks=int(p.get("num_chunks", 20000)),
                        generator=torch.manual_seed(seed), enable_pbar=False,
                        callback=lambda i, t, o: (ctx.check(), ctx.progress(i + 1, steps, "denoising", f"step {i + 1}/{steps}")),
                        callback_steps=1)[0]
        return mesh


def _postprocess(mesh, max_faces, ctx):
    """Drop floating specks and (optionally) decimate. Pure trimesh / pymeshlab, CPU."""
    import numpy as np
    import trimesh
    try:
        mesh.merge_vertices()
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            big = max(len(x.faces) for x in parts)
            keep = [x for x in parts if len(x.faces) >= max(50, 0.02 * big)]
            if len(keep) != len(parts):
                ctx.log(f"removed {len(parts) - len(keep)} floating fragments")
                mesh = trimesh.util.concatenate(keep)
    except Exception as e:  # noqa: BLE001
        ctx.log(f"fragment clean-up skipped: {e}")
    if max_faces and len(mesh.faces) > max_faces:
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype="float64"), face_matrix=np.asarray(mesh.faces, dtype="int32")))
            ms.meshing_merge_close_vertices()
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=max_faces)
            m = ms.current_mesh()
            ctx.log(f"decimated {len(mesh.faces)} -> {m.face_number()} faces")
            mesh = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=False)
        except Exception as e:  # noqa: BLE001
            ctx.log(f"decimation skipped: {e}")
    return mesh


if __name__ == "__main__":
    serve(Shape3DBackend())
