"""TRELLIS backend (envs/trellis): microsoft/TRELLIS-image-large (MIT), image -> 3D mesh with vertex colours.

  task shape   input role `image`  -> mesh asset: `mesh.glb` (+Y up, front +Z, vertex colours) + `input.png`

The upstream pipeline (vendor/TRELLIS, never edited) needs spconv, flash-attn and, for GLB export, nvdiffrast + a CUDA
Gaussian rasterizer.  None of them exist for aarch64 / without nvcc, so:
  * spconv + flash_attn are pure-PyTorch stand-ins (workers/trellis_shims.py),
  * attention of the dense transformers uses torch SDPA (ATTN_BACKEND=sdpa),
  * only the *mesh* decoder is run (FlexiCubes, pure PyTorch) and its per-vertex colours are written to the GLB; the upstream
    texture bake (rendering the Gaussians through nvdiffrast) is skipped.
  * FlexiCubes (a git submodule of TRELLIS) comes from vendor/FlexiCubes (Apache-2.0).
  * DINOv2 (image conditioning) is loaded from vendor/dinov2 + the torch-hub cache (the runtime never downloads).
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


def _setup():
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_BACKEND", "spconv")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")
    import trellis_shims
    trellis_shims.install()
    C.shim_missing("utils3d", "utils3d.numpy", "utils3d.torch", "utils3d.io", "open3d")
    # FlexiCubes is a git submodule of TRELLIS (empty in the clone): serve it from vendor/FlexiCubes under its package name
    import types
    fc = "trellis.representations.mesh.flexicubes"
    if fc not in sys.modules:
        m = types.ModuleType(fc)
        m.__path__ = [str(VENDOR / "FlexiCubes")]
        sys.modules[fc] = m
    if "kaolin" not in sys.modules:                  # FlexiCubes only uses kaolin's debug-time shape checks
        import importlib.machinery
        for n in ("kaolin", "kaolin.utils", "kaolin.utils.testing"):
            km = types.ModuleType(n)
            km.__path__ = []
            km.__spec__ = importlib.machinery.ModuleSpec(n, None)
            sys.modules[n] = km
        sys.modules["kaolin.utils.testing"].check_tensor = lambda *a, **k: True
        sys.modules["kaolin"].utils = sys.modules["kaolin.utils"]
        sys.modules["kaolin.utils"].testing = sys.modules["kaolin.utils.testing"]
    p = str(VENDOR / "TRELLIS")
    if p not in sys.path:
        sys.path.insert(0, p)
    import torch
    if not getattr(torch.hub, "_oas_patched", False):
        orig = torch.hub.load

        def hub_load(repo, name, *a, **k):
            if repo == "facebookresearch/dinov2":
                return orig(str(VENDOR / "dinov2"), name, *a, source="local", **k)
            return orig(repo, name, *a, **k)

        torch.hub.load = hub_load
        torch.hub._oas_patched = True


class TrellisBackend(Backend):
    name = "trellis"
    max_loaded = 1

    def probe(self, model):
        _setup()
        from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: F401
        return True, ""

    def load(self, model, ctx):
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        _setup()
        from trellis.pipelines import TrellisImageTo3DPipeline
        pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipe.cuda()
        return {"pipe": pipe}

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        if isinstance(h, dict):
            h.clear()
        gc.collect()

    def run(self, job, ctx):
        if job.task != "shape":
            raise InputError(f"{job.model.get('name', job.model_id)} cannot do '{job.task}'.")
        import numpy as np
        import torch
        inp = job.input(role="image") or job.input(type="image")
        if inp is None:
            raise InputError("Add the image to turn into 3D.")
        p = job.params
        pipe = self.handle(job.model_id)["pipe"]
        out = job.out_dir / "asset"
        out.mkdir(parents=True, exist_ok=True)
        t = {}
        t0 = time.time()
        ctx.progress(0, 1, "image", "Preparing the image")
        rgba = C.prep_object_image(inp["path"], bool(p.get("remove_bg", True)), float(p.get("foreground_ratio", 0.83)))
        rgba.save(out / "input.png")
        t["prep_s"] = round(time.time() - t0, 2)
        ctx.check()
        seed = C.seed_of(p)
        ss_steps, slat_steps = int(p.get("ss_steps", 12)), int(p.get("slat_steps", 12))
        total = ss_steps + slat_steps
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        # progress: wrap the samplers' model call to count steps and honour cancel
        counter = {"n": 0}

        def hook(*a, **k):
            ctx.check()
            counter["n"] += 1
            stage = "sparse structure" if counter["n"] <= ss_steps else "structured latent"
            ctx.progress(min(counter["n"], total), total, "denoising", f"{stage}: step {counter['n']}/{total}")

        for s in (pipe.sparse_structure_sampler, pipe.slat_sampler):
            if not hasattr(s, "_oas_inf"):
                s._oas_inf = s._inference_model
                s._inference_model = (lambda orig: (lambda model, x_t, t_, cond=None, **kw: (hook(), orig(model, x_t, t_, cond=cond, **kw))[1]))(s._oas_inf)
        with torch.no_grad():
            outputs = pipe.run(
                rgba, seed=seed, formats=["mesh", "gaussian"], preprocess_image=True,
                sparse_structure_sampler_params={"steps": ss_steps, "cfg_strength": float(p.get("ss_cfg", 7.5))},
                slat_sampler_params={"steps": slat_steps, "cfg_strength": float(p.get("slat_cfg", 3.0))},
            )
        ctx.check()
        m = outputs["mesh"][0]
        v = m.vertices.detach().float().cpu().numpy()
        f = m.faces.detach().cpu().numpy()
        # the mesh decoder only predicts geometry: colour every vertex from the nearest Gaussians of the same latent (their DC
        # spherical-harmonics term is the albedo); no rasterizer needed
        g = outputs["gaussian"][0]
        col = _gaussian_colors(v, g)
        t["gen_s"] = round(time.time() - t1, 2)
        ctx.progress(total, total, "export", "Writing GLB")
        # TRELLIS is z-up (front -y?): rotate to glTF +Y up, front +Z
        v = v[:, [0, 2, 1]] * np.array([1, 1, -1], dtype=v.dtype)
        f = f[:, ::-1] if _flipped(v, f) else f
        mesh_v, mesh_f, mesh_c = C.simplify(v, f, col, int(p.get("max_faces", 0) or 0), ctx)
        info = C.export_glb(out / "mesh.glb", mesh_v, mesh_f, vertex_colors=mesh_c)
        peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        ctx.log(f"timings {t}, peak GPU {peak} GB, {info}")
        ctx.output(out / "mesh.glb", "mesh", seed=seed, name=f"{(inp.get('name') or 'mesh')[:50]} ({job.model.get('name', 'TRELLIS')})",
                   attach=["input.png"], textured=False, vertex_colors=col is not None, stats=info)
        return {"meta": {"timings": t, "peak_gpu_gb": peak}}


def _gaussian_colors(verts, g, k=8):
    """Vertex colours (N, 3) in [0, 1]: opacity- and distance-weighted mean of the k nearest Gaussians' base colour."""
    import numpy as np
    from scipy.spatial import cKDTree
    xyz = g.get_xyz.detach().float().cpu().numpy()
    dc = g._features_dc.detach().float().cpu().numpy().reshape(len(xyz), 3)
    rgb = np.clip(0.5 + 0.28209479177387814 * dc, 0.0, 1.0)
    op = g.get_opacity.detach().float().cpu().numpy().reshape(-1)
    d, idx = cKDTree(xyz).query(verts, k=k)
    w = op[idx] / (d + 1e-4)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    return np.clip((rgb[idx] * w[..., None]).sum(1), 0.0, 1.0).astype(np.float32)


def _flipped(v, f):
    """True when the triangles wind inward (negative signed volume)."""
    import numpy as np
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) < 0


if __name__ == "__main__":
    serve(TrellisBackend())
