"""Tiny ctypes wrapper over upstream meshoptimizer (MIT, Arseny Kapoulkine) for the engine export.

Not the PyPI `meshoptimizer` package (unofficial, sdist-only, old core): `install/engine.yaml` builds `libmeshoptimizer.so` from the
pinned `vendor/meshoptimizer` checkout into the retopo venv (`$VIRTUAL_ENV/lib/`). Only the four functions the LOD chain needs are bound.

Library lookup (`lib()`): env `MESHOPT_LIB`, then `<sys.prefix>/lib/libmeshoptimizer.so`, then `<repo>/envs/retopo/lib/libmeshoptimizer.so`.
`available()` is False (instead of raising) when none loads, so tests in envs/core can skip.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

# meshopt_SimplifyOptions
LOCK_BORDER, SPARSE, ERROR_ABSOLUTE, PRUNE, REGULARIZE, PERMISSIVE = 1, 2, 4, 8, 16, 32

_REPO = Path(__file__).resolve().parent.parent
_LIB = None
_ERR = ""


def candidates() -> list[Path]:
    c = []
    if os.environ.get("MESHOPT_LIB"):
        c.append(Path(os.environ["MESHOPT_LIB"]))
    c.append(Path(sys.prefix) / "lib" / "libmeshoptimizer.so")
    c.append(_REPO / "envs" / "retopo" / "lib" / "libmeshoptimizer.so")
    return c


def lib():
    """The loaded CDLL with prototypes set; raises OSError("... run ./install.sh backend engine") when it cannot be found."""
    global _LIB, _ERR
    if _LIB is not None:
        return _LIB
    last = None
    for p in candidates():
        if not p.is_file():
            continue
        try:
            L = ctypes.CDLL(str(p))
        except OSError as e:
            last = e
            continue
        c = ctypes
        L.meshopt_simplifyWithAttributes.restype = c.c_size_t
        L.meshopt_simplifyWithAttributes.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_void_p, c.c_size_t, c.c_size_t,
                                                     c.c_void_p, c.c_size_t, c.c_void_p, c.c_size_t, c.c_void_p, c.c_size_t,
                                                     c.c_float, c.c_uint, c.POINTER(c.c_float)]
        L.meshopt_simplifyScale.restype = c.c_float
        L.meshopt_simplifyScale.argtypes = [c.c_void_p, c.c_size_t, c.c_size_t]
        L.meshopt_optimizeVertexCache.restype = None
        L.meshopt_optimizeVertexCache.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_size_t]
        L.meshopt_optimizeVertexFetchRemap.restype = c.c_size_t
        L.meshopt_optimizeVertexFetchRemap.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_size_t]
        _LIB = L
        return L
    _ERR = f"libmeshoptimizer.so not found or not loadable ({last or 'looked in ' + ', '.join(str(p) for p in candidates())}); run ./install.sh backend engine"
    raise OSError(_ERR)


def available() -> bool:
    try:
        lib()
        return True
    except OSError:
        return False


def simplify_scale(V) -> float:
    """meshopt's absolute scale of a position set (the extent that relative errors are multiplied by)."""
    pos = np.ascontiguousarray(V, np.float32)
    return float(lib().meshopt_simplifyScale(pos.ctypes.data, len(pos), 12))


def simplify(F, V, target_tris: int, target_error: float = 1.0, options: int = 0, lock=None):
    """Seam-preserving simplification. F (m,3) ints, V (n,3) floats -> (F2 (k,3) uint32 with the same vertex numbering, relative error).

    Vertices are never moved or merged here: the result only drops triangles, so every attribute array indexed by V stays valid
    (UVs, normals, tangents, joints, weights). `target_error` is relative to `simplify_scale(V)` unless ERROR_ABSOLUTE is set.
    `lock` (n,) uint8, optional: non-zero vertices are not moved. The index buffer is re-ordered for the vertex cache."""
    idx = np.ascontiguousarray(F, np.uint32).reshape(-1)
    pos = np.ascontiguousarray(V, np.float32)
    out = np.empty_like(idx)
    err = ctypes.c_float(0)
    lk = None if lock is None else np.ascontiguousarray(lock, np.uint8)
    L = lib()
    n = L.meshopt_simplifyWithAttributes(out.ctypes.data, idx.ctypes.data, idx.size, pos.ctypes.data, len(pos), 12,
                                         None, 0, None, 0, lk.ctypes.data if lk is not None else None,
                                         max(0, int(target_tris)) * 3, float(target_error), int(options), ctypes.byref(err))
    res = out[:n].copy()
    if res.size:
        L.meshopt_optimizeVertexCache(res.ctypes.data, res.ctypes.data, res.size, len(pos))
    return res.reshape(-1, 3), float(err.value)


def optimize_vertex_cache(F, n_vertices: int):
    idx = np.ascontiguousarray(F, np.uint32).reshape(-1).copy()
    if idx.size:
        lib().meshopt_optimizeVertexCache(idx.ctypes.data, idx.ctypes.data, idx.size, int(n_vertices))
    return idx.reshape(-1, 3)


def optimize_vertex_fetch_remap(F, n_vertices: int):
    """-> (remap (n_vertices,) uint32 with 0xffffffff for unused vertices, number of used vertices). The vertex order is the order of first use."""
    idx = np.ascontiguousarray(F, np.uint32).reshape(-1)
    remap = np.empty(int(n_vertices), np.uint32)
    used = lib().meshopt_optimizeVertexFetchRemap(remap.ctypes.data, idx.ctypes.data, idx.size, int(n_vertices))
    return remap, int(used)
