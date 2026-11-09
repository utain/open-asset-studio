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

import json
import os
import random
import struct
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


class _Missing:
    """Placeholder for an attribute of a stubbed module: attribute access chains (so annotations such as
    `o3d.geometry.TriangleMesh` evaluate), calling it raises."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Missing(f"{self._name}.{attr}")

    def __call__(self, *a, **k):
        raise RuntimeError(f"{self._name}: its package is not installed in this environment (the feature is disabled)")

    def __or__(self, other):                   # `X | None` annotations
        return self

    __ror__ = __or__


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
        import importlib.machinery
        mod = types.ModuleType(n)
        mod.__spec__ = importlib.machinery.ModuleSpec(n, None)
        mod.__path__ = []

        def _getattr(attr, _n=n):
            if attr.startswith("__"):          # inspect / importlib probe dunders: must look like a plain module
                raise AttributeError(attr)
            return _Missing(f"{_n}.{attr}")

        mod.__getattr__ = _getattr
        sys.modules[n] = mod


def vendor(name: str) -> Path:
    studio = Path(os.environ.get("STUDIO_ROOT") or Path(__file__).resolve().parent.parent)
    return studio / "vendor" / name


# ------------------------------------------------------------------ GLB inspection / vertex-attribute carry-over (pure struct + json + numpy)
_GLB_CT = {5126: ("<f4", 4), 5125: ("<u4", 4), 5123: ("<u2", 2), 5121: ("u1", 1), 5122: ("<i2", 2), 5120: ("i1", 1)}
_GLB_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _glb_split(path):
    """-> (json dict, BIN bytes) of a binary glTF file."""
    data = Path(path).read_bytes()
    if len(data) < 20 or struct.unpack("<I", data[:4])[0] != 0x46546C67:
        raise ValueError("not a GLB file")
    jl = struct.unpack("<I", data[12:16])[0]
    j = json.loads(data[20:20 + jl].decode("utf-8"))
    off = 20 + jl
    blob = b""
    if off + 8 <= len(data):
        bl = struct.unpack("<I", data[off:off + 4])[0]
        blob = data[off + 8:off + 8 + bl]
    return j, blob


def glb_read_json(path) -> dict:
    """The JSON chunk of a GLB (no geometry is read)."""
    return _glb_split(path)[0]


def glb_flags(path) -> dict:
    """What a GLB contains, from its JSON chunk only: skins, animations, morph_targets, primitives, triangles."""
    j = glb_read_json(path)
    prims = [p for m in j.get("meshes", []) for p in m.get("primitives", [])]
    tris = 0
    for p in prims:
        if p.get("mode", 4) != 4:
            continue
        if "indices" in p:
            tris += j["accessors"][p["indices"]]["count"] // 3
        elif "POSITION" in p.get("attributes", {}):
            tris += j["accessors"][p["attributes"]["POSITION"]]["count"] // 3
    return {"skins": len(j.get("skins", [])), "animations": len(j.get("animations", [])),
            "morph_targets": any(p.get("targets") for p in prims), "primitives": len(prims), "triangles": int(tris)}


def _glb_accessor(j, blob, i):
    import numpy as np
    a = j["accessors"][i]
    bv = j["bufferViews"][a["bufferView"]]
    dt, size = _GLB_CT[a["componentType"]]
    n = _GLB_NC[a["type"]]
    stride = bv.get("byteStride") or size * n
    base = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    if stride == size * n:
        arr = np.frombuffer(blob, dt, a["count"] * n, base)
    else:
        arr = np.ndarray((a["count"], n), dt, blob, base, (stride, size)).copy()
    return arr.reshape(a["count"], n) if n > 1 else arr


def _glb_write(path, j, blob: bytes):
    jb = json.dumps(j, separators=(",", ":")).encode("utf-8")
    jb += b" " * (-len(jb) % 4)
    bb = bytes(blob) + b"\0" * (-len(blob) % 4)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(jb) + (8 + len(bb) if bb else 0)))
        fh.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        if bb:
            fh.write(struct.pack("<II", len(bb), 0x004E4942) + bb)


def carry_vertex_attrs(src_glb, dst_glb) -> bool:
    """Copy NORMAL / TANGENT and the primitive `extras` of `src_glb` onto the matching primitives of `dst_glb`.

    The rig worker round-trips the input through trimesh, which drops the authored NORMAL / TANGENT (and recomputes normals that
    differ at UV seams) and the primitive extras (e.g. `oas_quad_pairs` of the retopo task). For each primitive of dst (in order)
    the src primitive with the same vertex count and the same POSITIONs (1e-5 x bbox diagonal; an evenly spaced sample of 1000
    vertices first, then all) is looked up; its NORMAL / TANGENT are appended as new bufferViews and accessors and replace
    dst's (missing or re-computed) ones, and the extras are copied. JOINTS_0 / WEIGHTS_0 and everything else of dst stay untouched. Returns True when the file was
    rewritten; False (file untouched, byte for byte) when nothing matched."""
    import numpy as np
    sj, sb = _glb_split(src_glb)
    dj, db = _glb_split(dst_glb)
    sprims = [p for m in sj.get("meshes", []) for p in m.get("primitives", []) if "POSITION" in p.get("attributes", {})]
    if not sprims:
        return False
    cache = {}

    def pos_of(prim, j, blob, key):
        k = (key, prim["attributes"]["POSITION"])
        if k not in cache:
            cache[k] = _glb_accessor(j, blob, prim["attributes"]["POSITION"]).astype(np.float64)
        return cache[k]

    new_blob = bytearray(db)
    changed = False
    used = set()
    for m in dj.get("meshes", []):
        for dp in m.get("primitives", []):
            if "POSITION" not in dp.get("attributes", {}):
                continue
            dpos = pos_of(dp, dj, db, "d")
            n = len(dpos)
            diag = float(np.linalg.norm(dpos.max(0) - dpos.min(0))) or 1.0
            tol = 1e-5 * diag
            match = None
            for si, sp in enumerate(sprims):
                if si in used:
                    continue
                spos = pos_of(sp, sj, sb, "s")
                if len(spos) != n:
                    continue
                probe = np.linspace(0, n - 1, min(n, 1000)).astype(np.int64)
                if np.abs(spos[probe] - dpos[probe]).max() > tol or np.abs(spos - dpos).max() > tol:
                    continue
                match = si
                break
            if match is None:
                continue
            sp = sprims[match]
            used.add(match)
            for a in ("NORMAL", "TANGENT"):
                if a not in sp["attributes"]:
                    continue
                arr = np.ascontiguousarray(_glb_accessor(sj, sb, sp["attributes"][a]).astype("<f4"))
                if a in dp["attributes"]:                                 # trimesh may have re-computed it: only keep dst if identical
                    cur = _glb_accessor(dj, db, dp["attributes"][a])
                    if cur.shape == arr.shape and np.abs(cur - arr).max() < 1e-6:
                        continue
                while len(new_blob) % 4:
                    new_blob.append(0)
                dj.setdefault("bufferViews", []).append({"buffer": 0, "byteOffset": len(new_blob), "byteLength": arr.nbytes, "target": 34962})
                new_blob.extend(arr.tobytes())
                dj.setdefault("accessors", []).append({"bufferView": len(dj["bufferViews"]) - 1, "componentType": 5126, "count": n,
                                                       "type": "VEC3" if a == "NORMAL" else "VEC4"})
                dp["attributes"][a] = len(dj["accessors"]) - 1
                changed = True
            if "extras" in sp and "extras" not in dp:
                dp["extras"] = sp["extras"]
                changed = True
    if not changed:
        return False
    dj.setdefault("buffers", [{}])[0]["byteLength"] = len(new_blob)
    _glb_write(dst_glb, dj, bytes(new_blob))
    return True
