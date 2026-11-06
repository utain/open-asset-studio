"""Stability AI 3D backend (envs/stab3d): image -> 3D mesh (TripoSR, Stable Fast 3D, SPAR3D) and multi-view / orbit
(Zero123++, Stable Zero123, SV3D).

  task shape      input role `image`   -> mesh asset: `mesh.glb` (+ `input.png`, the cleaned RGBA conditioning image)
  task multiview  input role `image`   -> image asset(s) (one per view) or an orbit video  (see the multiview section)

  triposr   stabilityai/TripoSR (MIT)                     NeRF triplane -> marching cubes; VERTEX COLOURS (no UV texture:
                                                          the upstream texture bake needs an OpenGL context)
  sf3d      stabilityai/stable-fast-3d (gated)            mesh + UVs + baked albedo/roughness/metallic/normal (PBR GLB);
                                                          texture_baker / uv_unwrapper are built CPU-only (no nvcc)
  spar3d    stabilityai/stable-point-aware-3d (gated)     point-cloud diffusion -> mesh (PBR GLB), same CPU baker

Remeshing options of SF3D / SPAR3D (quad / triangle remesh, vertex reduction) need `gpytoolbox` / `pynanoinstantmeshes`
which have no aarch64 wheel: they are stubbed (the code path is never called) and not offered.
Upstream code is never edited: it is imported from vendor/<repo> and wrapped here.
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
DATA = Path(os.environ.get("STUDIO_DATA") or STUDIO / "data")


def _add_path(*names):
    for n in names:
        p = str(VENDOR / n)
        if p not in sys.path:
            sys.path.insert(0, p)


def _patch_baker():
    """texture_baker is a CPU-only build here (no nvcc): hand it CPU tensors and move the results back to the caller's device."""
    import torch
    import texture_baker
    tb = texture_baker.TextureBaker
    if getattr(tb, "_oas_cpu", False):
        return

    def wrap(fn):
        def f(self, *args, **kw):
            dev = next((a.device for a in args if torch.is_tensor(a)), torch.device("cpu"))
            out = fn(self, *[a.cpu() if torch.is_tensor(a) else a for a in args], **kw)
            return out.to(dev) if torch.is_tensor(out) else out
        return f

    tb.rasterize = wrap(tb.rasterize)
    tb.interpolate = wrap(tb.interpolate)
    tb._oas_cpu = True


class Stab3DBackend(Backend):
    name = "stab3d"
    max_loaded = 1

    # ------------------------------------------------------------ probe / load
    def probe(self, model):
        import torch  # noqa: F401
        mid = model["id"]
        if mid == "triposr":
            C.shim_torchmcubes()
            _add_path("TripoSR")
            import tsr.system  # noqa: F401
        elif mid in ("sf3d", "spar3d"):
            import texture_baker  # noqa: F401
            import uv_unwrapper  # noqa: F401
            C.shim_missing("gpytoolbox", "pynanoinstantmeshes")
            _add_path("stable-fast-3d" if mid == "sf3d" else "stable-point-aware-3d")
            if mid == "sf3d":
                import sf3d.system  # noqa: F401
            else:
                import alpha_clip  # noqa: F401
                import spar3d.system  # noqa: F401
        else:
            from stab3d_multiview import probe as mv_probe
            return mv_probe(model)
        return True, ""

    def load(self, model, ctx):
        import torch
        mid = model["id"]
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        t0 = time.time()
        if mid == "triposr":
            C.shim_torchmcubes()
            _add_path("TripoSR")
            from tsr.system import TSR
            m = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
            m.renderer.set_chunk_size(8192)
            m.to("cuda").eval()
            return {"kind": "triposr", "model": m}
        if mid == "sf3d":
            C.shim_missing("gpytoolbox", "pynanoinstantmeshes")
            _patch_baker()
            _add_path("stable-fast-3d")
            from sf3d.system import SF3D
            m = SF3D.from_pretrained("stabilityai/stable-fast-3d", config_name="config.yaml", weight_name="model.safetensors")
            m.to("cuda").eval()
            return {"kind": "sf3d", "model": m}
        if mid == "spar3d":
            # transparent_background (SPAR3D's own background remover) is not used: the worker removes backgrounds with rembg
            C.shim_missing("gpytoolbox", "pynanoinstantmeshes", "transparent_background")
            _patch_baker()
            _add_path("stable-point-aware-3d")
            clip_dir = Path(os.path.expanduser("~/.cache/clip"))
            os.environ.setdefault("ALPHA_CLIP_PATH", str(clip_dir))
            from spar3d.system import SPAR3D
            m = SPAR3D.from_pretrained("stabilityai/stable-point-aware-3d", config_name="config.yaml",
                                       weight_name="model.safetensors", low_vram_mode=False)
            m.to("cuda").eval()
            return {"kind": "spar3d", "model": m}
        from stab3d_multiview import load as mv_load
        h = mv_load(model, ctx)
        ctx.log(f"loaded in {time.time() - t0:.1f}s")
        return h

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        if isinstance(h, dict):
            h.clear()
        gc.collect()

    # ------------------------------------------------------------------ run
    def run(self, job, ctx):
        if job.task == "multiview":
            from stab3d_multiview import run as mv_run
            return mv_run(self, job, ctx)
        if job.task != "shape":
            raise InputError(f"{job.model.get('name', job.model_id)} cannot do '{job.task}'.")
        return self._shape(job, ctx)

    def _shape(self, job, ctx):
        import torch
        inp = job.input(role="image") or job.input(type="image")
        if inp is None:
            raise InputError("Add the image to turn into 3D.")
        p = job.params
        h = self.handle(job.model_id)
        kind = h["kind"]
        t = {}
        t0 = time.time()
        ctx.progress(0, 4, "image", "Preparing the image")
        rgba = C.prep_object_image(inp["path"], bool(p.get("remove_bg", True)), float(p.get("foreground_ratio", 0.85)))
        t["prep_s"] = round(time.time() - t0, 2)
        ctx.check()
        out = job.out_dir / "asset"
        out.mkdir(parents=True, exist_ok=True)
        rgba.save(out / "input.png")
        seed = C.seed_of(p)
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        m = h["model"]
        t1 = time.time()
        if kind == "triposr":
            info = self._triposr(m, rgba, p, ctx, out / "mesh.glb")
        else:
            info = self._sf3d_like(kind, m, rgba, p, ctx, out / "mesh.glb")
        t["gen_s"] = round(time.time() - t1, 2)
        peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        ctx.log(f"timings {t}, peak GPU {peak} GB, {info}")
        ctx.output(out / "mesh.glb", "mesh", seed=seed, name=f"{(inp.get('name') or 'mesh')[:50]} ({job.model.get('name', kind)})",
                   attach=["input.png"], textured=kind != "triposr", vertex_colors=kind == "triposr", stats=info)
        return {"meta": {"timings": t, "peak_gpu_gb": peak}}

    # --------------------------------------------------------------- TripoSR
    def _triposr(self, m, rgba, p, ctx, dst):
        import numpy as np
        import torch
        import trimesh
        img = C.composite_gray(rgba, 0.5)
        ctx.progress(1, 4, "encode", "Encoding the image (DINO + triplane transformer)")
        with torch.no_grad():
            codes = m(img, device="cuda")
            ctx.check()
            ctx.progress(2, 4, "mesh", f"Marching cubes at {int(p.get('mc_resolution', 256))}^3")
            mesh = m.extract_mesh(codes, True, resolution=int(p.get("mc_resolution", 256)),
                                  threshold=float(p.get("threshold", 25.0)))[0]
        ctx.check()
        ctx.progress(3, 4, "export", "Writing GLB")
        from tsr.utils import to_gradio_3d_orientation   # TripoSR is z-up: glTF wants +Y up, front facing +Z
        mesh = to_gradio_3d_orientation(mesh)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))   # front of the object -> +Z (glTF)
        mesh = _orient(mesh)
        return C.export_glb(dst, mesh.vertices, mesh.faces, vertex_colors=np.asarray(mesh.visual.vertex_colors))

    # ------------------------------------------------------------- SF3D / SPAR3D
    def _sf3d_like(self, kind, m, rgba, p, ctx, dst):
        import torch
        ctx.progress(1, 4, "encode", f"{kind.upper()}: generating mesh + UV + material bake")
        res = int(p.get("texture_resolution", 1024))
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mesh, _g = m.run_image(rgba, bake_resolution=res, remesh="none", vertex_count=-1)
        ctx.check()
        ctx.progress(3, 4, "export", "Writing GLB")
        mesh.export(str(dst), include_normals=True)
        return C.mesh_info(dst)


def _orient(mesh):
    """Make sure the mesh has outward-facing triangles (the CPU marching-cubes shim may flip the winding)."""
    import trimesh
    if len(mesh.faces) and mesh.volume < 0:
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[:, ::-1], vertex_colors=mesh.visual.vertex_colors, process=False)
    return mesh


if __name__ == "__main__":
    serve(Stab3DBackend())
