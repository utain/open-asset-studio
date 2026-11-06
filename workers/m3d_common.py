"""Shared helpers of the extra 3D workers (stab3d, tripo3d, ...): conditioning-image prep, GLB export, shims.

Imported by workers/<backend>_worker.py of the 3D-more group (`import m3d_common`; workers/ is on PYTHONPATH).
Pure numpy / PIL / trimesh at import time (torch only inside functions that need it), so the unit tests run in envs/core
as long as numpy + PIL are available.

Conventions (same as the Hunyuan3D worker): every shape job emits ONE mesh asset whose main file is `mesh.glb`
(glTF 2.0, +Y up, metres-ish unit cube, vertex colours or a baked PBR texture), plus attachments in the same folder:
`input.png` (the cleaned RGBA conditioning image).  Textures that need nvcc-only rasterizers are skipped on this
machine: shape models that only have vertex colours say so in their registry summary.
"""
from __future__ import annotations

import os
import random
import sys
import types
from pathlib import Path

MAX_SEED = 2**31 - 1


def seed_of(params: dict) -> int:
    """Seed actually used (seed -1 / missing = random). Report it in the asset meta."""
    s = params.get("seed", -1)
    s = -1 if s is None else int(s)
    return s if s >= 0 else random.randrange(MAX_SEED)


# ------------------------------------------------------------------ conditioning image
def open_rgba(path: str):
    from PIL import Image, ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGBA")


def has_useful_alpha(im, min_transparent: float = 0.02) -> bool:
    import numpy as np
    a = np.asarray(im.getchannel("A"))
    return float((a < 250).mean()) >= min_transparent


_REMBG = {}


def remove_bg(im):
    """RGBA -> RGBA with the background removed (rembg u2net on CPU); an image that already has transparency is kept."""
    if has_useful_alpha(im):
        return im
    import rembg
    if "s" not in _REMBG:
        _REMBG["s"] = rembg.new_session()
    return rembg.remove(im, session=_REMBG["s"])


def crop_resize_foreground(im, ratio: float = 0.85, size: int | None = None):
    """Crop to the alpha bounding box and centre it on a transparent square so the object fills `ratio` of the side."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im.getchannel("A"))
    ys, xs = np.nonzero(a > 10)
    if len(xs) == 0:
        raise ValueError("The image has no visible object after background removal.")
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    obj = im.crop(box)
    w, h = obj.size
    side = int(round(max(w, h) / max(0.1, min(1.0, ratio))))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(obj, ((side - w) // 2, (side - h) // 2))
    if size:
        canvas = canvas.resize((size, size), Image.LANCZOS)
    return canvas


def composite_gray(im, gray: float = 0.5):
    """RGBA -> RGB over a flat grey background (TripoSR / SF3D style conditioning)."""
    import numpy as np
    from PIL import Image
    x = np.asarray(im.convert("RGBA")).astype("float32") / 255.0
    rgb = x[..., :3] * x[..., 3:4] + (1 - x[..., 3:4]) * gray
    return Image.fromarray((rgb * 255).round().astype("uint8"))


def prep_object_image(path: str, do_remove_bg: bool = True, fg_ratio: float = 0.85, size: int | None = None):
    """Open -> (optional) remove background -> centre on a square. Returns the RGBA image."""
    im = open_rgba(path)
    if do_remove_bg:
        im = remove_bg(im)
    return crop_resize_foreground(im, fg_ratio, size) if has_useful_alpha(im, 0.0005) else im


# ------------------------------------------------------------------ mesh output
def export_glb(path, vertices, faces, vertex_colors=None, vertex_normals=None, process=False) -> dict:
    """Write a GLB. vertices (N,3) float, faces (M,3) int, vertex_colors (N,3|4) uint8 or float 0..1."""
    import numpy as np
    import trimesh
    v = np.asarray(vertices, dtype="float64")
    f = np.asarray(faces, dtype="int64")
    col = None
    if vertex_colors is not None:
        col = np.asarray(vertex_colors)
        if col.dtype != np.uint8:
            col = (np.clip(col, 0, 1) * 255).round().astype("uint8")
        if col.shape[1] == 3:
            col = np.concatenate([col, np.full((len(col), 1), 255, "uint8")], 1)
    m = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=col, process=process)
    if vertex_normals is not None:
        m.vertex_normals = np.asarray(vertex_normals)
    m.export(str(path))
    return {"vertices": int(len(v)), "faces": int(len(f))}


def simplify(vertices, faces, colors=None, max_faces: int = 0, ctx=None):
    """Quadric edge-collapse decimation to `max_faces` (pymeshlab, CPU) keeping vertex colours. 0 = keep everything."""
    import numpy as np
    if not max_faces or len(faces) <= max_faces:
        return vertices, faces, colors
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        kw = {"vertex_matrix": np.asarray(vertices, dtype="float64"), "face_matrix": np.asarray(faces, dtype="int32")}
        if colors is not None:
            c = np.asarray(colors, dtype="float64")
            kw["v_color_matrix"] = np.concatenate([c, np.ones((len(c), 1))], 1)
        ms.add_mesh(pymeshlab.Mesh(**kw))
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(max_faces), preservenormal=True)
        m = ms.current_mesh()
        if ctx:
            ctx.log(f"decimated {len(faces)} -> {m.face_number()} faces")
        col = m.vertex_color_matrix()[:, :3] if colors is not None else None
        return m.vertex_matrix(), m.face_matrix(), col
    except Exception as e:  # noqa: BLE001
        if ctx:
            ctx.log(f"decimation skipped: {e}")
        return vertices, faces, colors


def mesh_info(path) -> dict:
    import trimesh
    sc = trimesh.load(str(path), force="scene")
    nv = nf = 0
    for g in sc.geometry.values():
        nv += len(g.vertices)
        nf += len(g.faces)
    return {"vertices": nv, "faces": nf, "bounds": [list(map(float, b)) for b in sc.bounds]}


# ------------------------------------------------------------------ shims for packages that need nvcc / have no aarch64 wheel
def shim_torchmcubes():
    """`torchmcubes` (TripoSR) is built with CUDA only; provide a CPU marching-cubes with the same call signature."""
    if "torchmcubes" in sys.modules:
        return
    import numpy as np
    import torch
    from skimage import measure

    def marching_cubes(vol, thresh):
        v = vol.detach().float().cpu().numpy()
        if not (v.min() < thresh < v.max()):
            return torch.zeros(0, 3), torch.zeros(0, 3, dtype=torch.long)
        verts, faces, _n, _ = measure.marching_cubes(v, level=thresh)
        # torchmcubes returns coordinates in (last axis, ..., first axis) order; skimage in (first, ..., last)
        verts = np.ascontiguousarray(verts[:, ::-1])
        faces = np.ascontiguousarray(faces[:, ::-1])
        return torch.from_numpy(verts.astype("float32")), torch.from_numpy(faces.astype("int64"))

    mod = types.ModuleType("torchmcubes")
    mod.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = mod


def shim_missing(*names):
    """Register stub modules for optional heavy dependencies that are only used by code paths we never call."""
    for n in names:
        if n in sys.modules:
            continue
        try:
            __import__(n)
            continue
        except Exception:  # noqa: BLE001
            pass
        mod = types.ModuleType(n)

        def _missing(*a, _n=n, **k):
            raise RuntimeError(f"{_n} is not installed in this environment (remeshing options are disabled)")

        mod.__getattr__ = lambda attr, _m=_missing: _m
        sys.modules[n] = mod


def vendor(name: str) -> Path:
    studio = Path(os.environ.get("STUDIO_ROOT") or Path(__file__).resolve().parent.parent)
    return studio / "vendor" / name
