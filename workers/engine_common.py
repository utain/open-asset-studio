"""Pure helpers of the game-engine export (workers/engine_worker.py, app/plugins/engine.py, tests/test_engine_export.py).

numpy + PIL (+ scipy for convex hulls) only, so the unit tests import it in envs/core. No trimesh, no meshoptimizer here: the one native
call of the LOD chain (`simplify`) is passed in by the worker (workers/meshopt_ctypes.py), which keeps the chain logic testable.

Groups:
  names        safe_stem, engine_names                         engine naming (SM_ / SK_ / T_ / UCX_ / _LODn / -convcolonly)
  mesh         sanitize, compact, weld_positions, merge_prims   attribute clean-up before export
  lods         plan_lods, lod_stop, screen_sizes, build_lod_chain
  collision    simple_box / simple_obb / simple_capsule, hull_points_fps, validate_hulls, hull_of_points, box_mesh, capsule_mesh
  placement    placement, apply_placement                       size / pivot / yaw
  textures     pack_textures (PUBLIC, see its docstring), flip_green, downscale_pot, engine_material
  glb          GlbBuilder, read_prims, prune_unused, add_msft_lod, add_godot_hulls, image extraction / resizing
  reports      report, manifest, readme

Conventions (docs/CONTRACTS.md, "Engine export"): the canonical maps of the studio are an OpenGL (Y+) normal map and an ORM texture with
R = ambient occlusion, G = roughness, B = metallic (linear); base colour is sRGB. Only the engine package converts them.
"""
from __future__ import annotations

import copy
import io
import json
import math
import re
import struct
from pathlib import Path

import numpy as np

TOOL = "Open Asset Studio"
TARGETS = ("unreal", "unity", "godot", "gltf")
LOD_PRESETS = {"halves": [0.5, 0.25, 0.125], "aggressive": [0.4, 0.15, 0.06], "gentle": [0.7, 0.5, 0.35]}
# Unreal "Screen Size" of every level (the size at which the level starts); Unity LODGroup transition heights (where the level hands over)
UNREAL_SCREEN = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03]
UNITY_SCREEN = [0.6, 0.3, 0.15, 0.05, 0.02, 0.01]
MIN_GAIN = 0.85                   # a LOD must have at most 85 % of the triangles of the previous one, or it is not worth a level
COLLISION_PRESETS = {             # CoACD kwargs (threshold, hull cap, preprocess resolution, MCTS iterations, MCTS nodes or None = default)
    "fast": {"threshold": 0.10, "cap": 8, "preprocess_resolution": 30, "mcts_iterations": 60, "mcts_nodes": 15},
    "balanced": {"threshold": 0.05, "cap": 16, "preprocess_resolution": 50, "mcts_iterations": 100, "mcts_nodes": None},
    "detailed": {"threshold": 0.03, "cap": 32, "preprocess_resolution": 60, "mcts_iterations": 150, "mcts_nodes": None},
}
COLLISION_KINDS = ("auto", "none", "box", "obb", "capsule", "convex_hull", "decomposition")
PROXY_TRIS = 5000                 # CoACD runs on a proxy with at most this many triangles
FBX_UNIT_SCALE = 100.0            # metres -> centimetres in the FBX (UnitScaleFactor 1)


class EngineInputError(ValueError):
    """A user-facing problem (the worker turns it into InputError)."""


# ================================================================== names
def safe_stem(name, maxlen: int = 48, default: str = "Asset") -> str:
    """[A-Za-z0-9_] only (everything else becomes `_`, runs collapse, ends trimmed), at most `maxlen` characters. Never empty."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "")).strip("_")[:maxlen].strip("_")
    return s or default


def _prefixed(prefix: str, stem: str) -> str:
    return stem if (not prefix or stem.lower().startswith(prefix.lower())) else prefix + stem


def engine_names(target: str, name, rigged: bool = False, n_lods: int = 0, prefix_static: str | None = None,
                 prefix_texture: str | None = None) -> dict:
    """Names of everything an engine package contains.

    -> {stem, asset, group, lods: [..], file, hull(k), texture_prefix, material}
      stem    the safe user name without prefixes                  `Rock`
      asset   the base mesh name                                   unreal `SM_Rock` (static) / `SK_Rock` (rigged); others `Rock`
      group   the node that holds the LODs. Unreal needs it to differ from every mesh inside it (UE 5.5 drops UCX otherwise): `SM_Rock`
              versus the meshes `SM_Rock_LOD0`. Without LODs there is no group and the mesh is `SM_Rock`
      lods    mesh names `SM_Rock_LOD0 .. LODn` (n_lods + 1 of them), or [asset] when there are no LODs
      hull(k) collision hull `UCX_<LOD0 mesh name>_NN`; godot `<Rock>_hull_NN-convcolonly`; gltf `<Rock>_hull_NN`
      file    the main file name stem
    `prefix_static` / `prefix_texture` override SM_ / T_ (project setting engine.prefix_*); an existing prefix is never doubled."""
    stem = safe_stem(name)
    ps = ("SK_" if rigged else "SM_") if prefix_static is None else prefix_static
    pt = "T_" if prefix_texture is None else prefix_texture
    if target == "unreal":
        asset = _prefixed(ps, stem)
        mat = _prefixed("M_", stem)
    else:
        asset = stem
        mat = f"{stem}_Mat"
    lods = [f"{asset}_LOD{i}" for i in range(n_lods + 1)] if n_lods > 0 else [asset]
    hull_base = lods[0]

    def hull(k: int, kind: str = "convex") -> str:
        if target == "godot":
            return f"{stem}_hull_{k:02d}-convcolonly"
        if target == "gltf":
            return f"{stem}_hull_{k:02d}"
        return f"{'UBX' if kind == 'box' and target == 'unreal' else 'UCX'}_{hull_base}_{k:02d}"

    return {"stem": stem, "asset": asset, "group": asset if n_lods > 0 else None, "lods": lods, "file": asset,
            "hull": hull, "texture_prefix": pt if target == "unreal" else "", "material": mat}


# ================================================================== mesh
def _unit(v, fallback=(0.0, 1.0, 0.0)):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.where(n > 1e-12, v / np.maximum(n, 1e-30), np.asarray(fallback, v.dtype))
    return out


def face_normals(V, F):
    """Area-weighted (unnormalised) face normals."""
    V = np.asarray(V, np.float64)
    return np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])


def vertex_normals(V, F):
    n = np.zeros((len(V), 3), np.float64)
    fn = face_normals(V, F)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return _unit(n)


def compact(F, attrs: dict):
    """Drop the vertices no triangle uses and renumber. -> (F2 int64, attrs2). Every attribute array is indexed per vertex."""
    F = np.asarray(F, np.int64).reshape(-1, 3)
    used = np.unique(F)
    remap = np.full(len(attrs["POSITION"]), -1, np.int64)
    remap[used] = np.arange(len(used))
    return remap[F], {k: np.asarray(v)[used] for k, v in attrs.items()}


def sanitize(attrs: dict, F):
    """Make a primitive safe for the glTF validator and the engines. -> (attrs2, F2, report) with report = {counts..., "warnings": [..]}.

    NORMAL: zero normals become the area-weighted average of the faces around the vertex (fallback +Y), then everything is normalised;
    TANGENT: xyz normalised, w kept at +-1; WEIGHTS_0: rows sum to 1 and JOINTS_0 is 0 where the weight is 0 (u8 when it fits);
    degenerate triangles (a repeated index) and unused vertices are removed."""
    F = np.asarray(F, np.int64).reshape(-1, 3)
    a = {k: np.asarray(v) for k, v in attrs.items()}
    rep = {"zero_normals": 0, "non_unit_normals": 0, "tangents": 0, "weight_rows": 0, "joint_slots_cleared": 0,
           "degenerate_tris": 0, "unused_vertices": 0}
    good = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    rep["degenerate_tris"] = int((~good).sum())
    F = F[good]
    V = a["POSITION"].astype(np.float64)
    if "NORMAL" in a:
        n = a["NORMAL"].astype(np.float64)
        ln = np.linalg.norm(n, axis=1)
        zero = ln < 1e-6
        rep["zero_normals"] = int(zero.sum())
        rep["non_unit_normals"] = int((~zero & (np.abs(ln - 1) > 1e-3)).sum())
        if zero.any():
            n[zero] = vertex_normals(V, F)[zero]
        a["NORMAL"] = _unit(n).astype("<f4")
    if "TANGENT" in a:
        t = a["TANGENT"].astype(np.float64)
        w = np.where(t[:, 3] < 0, -1.0, 1.0)
        a["TANGENT"] = np.concatenate([_unit(t[:, :3], (1.0, 0.0, 0.0)), w[:, None]], 1).astype("<f4")
        rep["tangents"] = len(t)
    if "WEIGHTS_0" in a and "JOINTS_0" in a:
        w = a["WEIGHTS_0"].astype(np.float64)
        w = np.clip(w, 0.0, None)
        s = w.sum(1, keepdims=True)
        rep["weight_rows"] = int((np.abs(s[:, 0] - 1) > 1e-4).sum())
        w = np.where(s > 1e-12, w / np.maximum(s, 1e-30), np.array([1.0, 0.0, 0.0, 0.0])[: w.shape[1]])
        j = a["JOINTS_0"].copy()
        clear = w == 0
        rep["joint_slots_cleared"] = int((clear & (j != 0)).sum())
        j[clear] = 0
        a["WEIGHTS_0"] = w.astype("<f4")
        a["JOINTS_0"] = j.astype("u1" if int(j.max(initial=0)) <= 255 else "<u2")
    n0 = len(V)
    F, a = compact(F, a)
    rep["unused_vertices"] = int(n0 - len(a["POSITION"]))
    w_ = []
    if rep["zero_normals"]:
        w_.append(f"{rep['zero_normals']} zero normal(s) rebuilt from the faces")
    if rep["non_unit_normals"]:
        w_.append(f"{rep['non_unit_normals']} normal(s) renormalised")
    if rep["joint_slots_cleared"]:
        w_.append(f"{rep['joint_slots_cleared']} unused joint slot(s) cleared")
    if rep["degenerate_tris"]:
        w_.append(f"{rep['degenerate_tris']} degenerate triangle(s) removed")
    if rep["unused_vertices"]:
        w_.append(f"{rep['unused_vertices']} unused vertex(es) removed")
    rep["warnings"] = w_
    return a, F, rep


def weld_positions(V, F, rel_tol: float = 1e-6):
    """Merge vertices with the same position (UV / normal seams split them) -> (V2, F2). Degenerate triangles are dropped.
    CoACD and the volume ratio need a watertight surface, which only exists after welding."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64).reshape(-1, 3)
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    key = np.round(V / (diag * rel_tol)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    F2 = inv[F]
    good = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[good]
    used = np.unique(F2)
    remap = np.full(len(first), -1, np.int64)
    remap[used] = np.arange(len(used))
    return V[first][used], remap[F2]


def merge_prims(prims: list[dict]) -> dict:
    """Concatenate primitives of one LOD level into one triangle soup for FBX: -> {V, N, UV, F, mat (per face: index into `materials`), materials}.
    Missing normals are computed, a missing UV becomes zeros. `prims` items: {"attrs": {...}, "F": (m,3), "material": int|None}."""
    Vs, Ns, UVs, Fs, ms, off = [], [], [], [], [], 0
    mats: list = []
    for p in prims:
        a, F = p["attrs"], np.asarray(p["F"], np.int64)
        V = a["POSITION"].astype(np.float64)
        Vs.append(V)
        Ns.append(a["NORMAL"].astype(np.float64) if "NORMAL" in a else vertex_normals(V, F))
        UVs.append(a["TEXCOORD_0"].astype(np.float64) if "TEXCOORD_0" in a else np.zeros((len(V), 2)))
        Fs.append(F + off)
        m = p.get("material")
        if m not in mats:
            mats.append(m)
        ms.append(np.full(len(F), mats.index(m), np.int32))
        off += len(V)
    return {"V": np.concatenate(Vs), "N": np.concatenate(Ns), "UV": np.concatenate(UVs), "F": np.concatenate(Fs),
            "mat": np.concatenate(ms), "materials": mats}


# ================================================================== LOD chain
def parse_ratios(text) -> list[float]:
    """"0.5,0.25,0.1" or "6000,2500,800" (values above 1 are absolute triangle counts) -> floats; garbage is ignored."""
    out = []
    for t in re.split(r"[,\s;]+", str(text or "")):
        try:
            v = float(t)
        except ValueError:
            continue
        if v > 0 and math.isfinite(v):
            out.append(v)
    return out


def plan_lods(tris0: int, lods: int, lod_preset: str = "halves", lod_ratios=None, min_tris: int = 300) -> tuple[list[dict], list[str]]:
    """The LOD levels to try. -> (plan, skipped): plan items {level, ratio (of LOD0), target (triangles)}; skipped = human reasons.

    ratios: the preset table or `lod_ratios` (custom); a value above 1 is an absolute triangle target, otherwise a fraction of LOD0.
    target = max(min_tris, ...). A level whose target is not below 85 % of the previous kept target is skipped ("not worth a level"),
    and so is everything after a target that cannot go lower (the floor repeats)."""
    ratios = parse_ratios(lod_ratios) if lod_preset == "custom" else list(LOD_PRESETS.get(lod_preset) or LOD_PRESETS["halves"])
    plan, skipped = [], []
    prev = int(tris0)
    for r in ratios[: max(0, int(lods))]:
        target = max(int(min_tris), int(round(r * tris0)) if r <= 1 else int(r))
        if target >= MIN_GAIN * prev:
            skipped.append(f"LOD{len(plan) + 1} skipped: {target:,} triangles is not at least 15 % smaller than the previous level ({prev:,})")
            continue
        plan.append({"level": len(plan) + 1, "ratio": round(target / max(1, tris0), 4), "target": target})
        prev = target
    if lods > len(ratios):
        skipped.append(f"{lods - len(ratios)} LOD level(s) not made: the {lod_preset} steps have {len(ratios)}")
    return plan, skipped


def lod_stop(prev_tris: int, got_tris: int) -> bool:
    """True when the simplified level is not at least 15 % smaller than the previous one (the error budget stopped it)."""
    return got_tris >= MIN_GAIN * prev_tris


def screen_sizes(target: str, n_levels: int) -> list[float]:
    """Per-level screen size for the manifest: Unreal's 'Screen Size' (size where the level starts), Unity's LODGroup transition heights."""
    table = UNITY_SCREEN if target == "unity" else UNREAL_SCREEN
    return [table[min(i, len(table) - 1)] for i in range(max(1, n_levels))]


def switch_coverage(n_levels: int) -> list[float]:
    """MSFT_screencoverage: for each level the coverage below which the next level is used; the last one is 0."""
    ss = screen_sizes("unreal", n_levels)
    return [ss[i + 1] for i in range(n_levels - 1)] + [0.0]


def build_lod_chain(prims: list[dict], lods: int, lod_preset: str, lod_ratios, min_tris: int, max_error: float, skinned: bool,
                    simplify, scale_of=None, on_level=None) -> dict:
    """Simplify every primitive from LOD0 (not from the previous level: less drift) and apply the stop rules.

    prims: [{"attrs", "F", "material", "name"}] (sanitised, LOD0). `simplify(F, V, target_tris, target_error, options) -> (F2, rel_error)`
    is meshopt_ctypes.simplify (REGULARIZE = 16 for skinned meshes). Each primitive's target is the level target times its share of the
    LOD0 triangles (never below 4 triangles, and a primitive that small is kept as it is).
    -> {levels: [[prim..] per level incl. LOD0], info: [{level, tris, verts, ratio, target, error, error_abs}], skipped: [..], warnings}"""
    tris0 = sum(len(p["F"]) for p in prims)
    plan, skipped = plan_lods(tris0, lods, lod_preset, lod_ratios, min_tris)
    levels = [prims]
    info = [{"level": 0, "tris": tris0, "verts": sum(len(p["attrs"]["POSITION"]) for p in prims), "ratio": 1.0, "target": tris0,
             "error": 0.0, "error_abs": 0.0}]
    options = 16 if skinned else 0
    prev = tris0
    for step in plan:
        out, err_rel, err_abs = [], 0.0, 0.0
        for p in prims:
            share = len(p["F"]) / max(1, tris0)
            t = max(4, int(round(step["target"] * share)))
            if len(p["F"]) <= 4 or t >= len(p["F"]):
                out.append(p)
                continue
            F2, e = simplify(p["F"], p["attrs"]["POSITION"], t, max_error, options)
            if len(F2) == 0:
                out.append(p)
                continue
            F3, a3 = compact(F2.astype(np.int64), p["attrs"])
            out.append({**p, "attrs": a3, "F": F3})
            err_rel = max(err_rel, float(e))
            sc = scale_of(p["attrs"]["POSITION"]) if scale_of else 1.0
            err_abs = max(err_abs, float(e) * sc)
        tris = sum(len(p["F"]) for p in out)
        if lod_stop(prev, tris):
            skipped.append(f"LOD{step['level']} and below not made: the error limit ({max_error:g} of the model size) stops at {tris:,} triangles "
                           f"(previous level {prev:,})")
            break
        row = {"level": len(levels), "tris": tris, "verts": sum(len(p["attrs"]["POSITION"]) for p in out), "ratio": round(tris / max(1, tris0), 4),
               "target": step["target"], "error": round(err_rel, 6), "error_abs": round(err_abs, 6)}
        levels.append(out)
        info.append(row)
        prev = tris
        if on_level:
            on_level(row)
    return {"levels": levels, "info": info, "skipped": skipped}


# ================================================================== collision (pure numpy / scipy)
def _orient_outward(V, F):
    c = V.mean(0)
    n = face_normals(V, F)
    mid = V[F].mean(1)
    flip = (n * (mid - c)).sum(1) < 0
    F = F.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


def hull_of_points(P):
    """Convex hull of a point set -> (V, F) with outward winding, only hull vertices; None when it is degenerate (flat, < 4 points)."""
    from scipy.spatial import ConvexHull, QhullError

    P = np.asarray(P, np.float64)
    if len(P) < 4:
        return None
    try:
        h = ConvexHull(P)
    except QhullError:
        try:
            h = ConvexHull(P, qhull_options="QJ")
        except QhullError:
            return None
    used = np.unique(h.simplices)
    remap = np.full(len(P), -1, np.int64)
    remap[used] = np.arange(len(used))
    V = P[used]
    F = remap[h.simplices]
    return V, _orient_outward(V, F)


def hull_points_fps(P, n: int) -> np.ndarray:
    """Indices of at most `n` points chosen by farthest-point sampling, seeded with the six axis extremes (so the box stays covered)."""
    P = np.asarray(P, np.float64)
    if len(P) <= n:
        return np.arange(len(P))
    seed = []
    for k in range(3):
        seed += [int(P[:, k].argmin()), int(P[:, k].argmax())]
    sel = list(dict.fromkeys(seed))[:n]
    d = np.min(np.linalg.norm(P[:, None, :] - P[sel][None, :, :], axis=2), axis=1) if len(P) * len(sel) < 4_000_000 else \
        np.min(np.stack([np.linalg.norm(P - P[s], axis=1) for s in sel]), axis=0)
    while len(sel) < n:
        i = int(d.argmax())
        if d[i] <= 0:
            break
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.array(sel)


def validate_hulls(hulls, hull_max_verts: int = 64, bbox_volume: float | None = None, min_volume_frac: float = 1e-6):
    """Re-hull every convex piece so convexity and outward winding are guaranteed. -> (kept [(V, F)], dropped count).

    A piece is dropped when it has fewer than 4 vertices or its volume is below `min_volume_frac` of the bounding-box volume (a V-HACD
    flat piece crashed trimesh in the prototype). One with more than `hull_max_verts` vertices is reduced by farthest-point sampling and re-hulled."""
    kept, dropped = [], 0
    for V, _F in hulls:
        h = hull_of_points(V)
        if h is None:
            dropped += 1
            continue
        if len(h[0]) > hull_max_verts:
            idx = hull_points_fps(h[0], hull_max_verts)
            h = hull_of_points(h[0][idx])
            if h is None:
                dropped += 1
                continue
        if bbox_volume and hull_volume(*h) < min_volume_frac * bbox_volume:
            dropped += 1
            continue
        kept.append(h)
    return kept, dropped


def hull_volume(V, F) -> float:
    V = np.asarray(V, np.float64)
    c = V.mean(0)
    a, b, d = V[F[:, 0]] - c, V[F[:, 1]] - c, V[F[:, 2]] - c
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, d)).sum()) / 6.0)


def mesh_volume(V, F) -> float | None:
    """Volume of a closed triangle mesh (divergence theorem); None when the surface is open (an edge is not shared by exactly two faces)."""
    F = np.asarray(F, np.int64)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    if len(cnt) == 0 or not np.all(cnt == 2):
        return None
    V = np.asarray(V, np.float64)
    return float(abs(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum()) / 6.0)


def simple_box(P) -> dict:
    """Axis-aligned box: {center, axes (identity), half}."""
    P = np.asarray(P, np.float64)
    lo, hi = P.min(0), P.max(0)
    return {"center": (lo + hi) / 2, "axes": np.eye(3), "half": (hi - lo) / 2}


def simple_obb(P) -> dict:
    """Oriented box from the principal axes of the points: extents are exact for a rotated box."""
    P = np.asarray(P, np.float64)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    axes = vt
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    q = (P - c) @ axes.T
    lo, hi = q.min(0), q.max(0)
    mid = (lo + hi) / 2
    return {"center": c + mid @ axes, "axes": axes, "half": (hi - lo) / 2}


def box_mesh(box: dict):
    """8 vertices / 12 triangles of a box dict (simple_box / simple_obb), outward winding."""
    s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64)
    V = box["center"] + (s * box["half"]) @ np.asarray(box["axes"])
    return hull_of_points(V)


def simple_capsule(P, tight: bool = False) -> dict:
    """A capsule along the principal axis of largest variance: {center, axis (unit), radius, half_height (of the straight part)}.
    radius = largest distance to the axis (98th percentile with tight=True, for characters whose arms stick out); the total height is
    2 * (half_height + radius)."""
    P = np.asarray(P, np.float64)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    axis = vt[0]
    if axis[1] < 0:
        axis = -axis                      # point up when the axis is vertical
    along = (P - c) @ axis
    perp = (P - c) - along[:, None] * axis
    rad = np.linalg.norm(perp, axis=1)
    radius = float(np.percentile(rad, 98) if tight else rad.max())
    lo, hi = float(along.min()), float(along.max())
    half_total = (hi - lo) / 2
    radius = min(radius, half_total) if half_total > 0 else radius
    mid = (lo + hi) / 2
    return {"center": c + mid * axis, "axis": axis, "radius": radius, "half_height": max(0.0, half_total - radius)}


def capsule_mesh(cap: dict, segments: int = 12, rings: int = 3):
    """Convex mesh of a capsule dict: cylinder rings plus hemisphere rings -> (V, F)."""
    a = np.asarray(cap["axis"], np.float64)
    a = a / np.linalg.norm(a)
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    r, hh = float(cap["radius"]), float(cap["half_height"])
    pts = [np.asarray(cap["center"]) + a * (hh + r), np.asarray(cap["center"]) - a * (hh + r)]
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1)
    for sign in (1, -1):
        for k in range(rings):
            phi = (k / rings) * (np.pi / 2)
            rr, zz = r * math.cos(phi), r * math.sin(phi)
            for cs, sn in ring:
                pts.append(np.asarray(cap["center"]) + a * sign * (hh + zz) + u * rr * cs + v * rr * sn)
    return hull_of_points(np.array(pts))


# ================================================================== placement (size / pivot / yaw)
def placement(bmin, bmax, size_mode: str = "keep", height_m: float = 1.8, pivot: str = "keep", yaw: int = 0, up: int = 1) -> dict:
    """Uniform scale, pivot shift and a turn about the up axis (default +Y) as one transform p' = S * R * (p - c).
    -> {scale, yaw_deg, pivot_point, translation (for a node: T = -S R c), rotation (xyzw quaternion), identity}"""
    bmin, bmax = np.asarray(bmin, np.float64), np.asarray(bmax, np.float64)
    ext = bmax - bmin
    scale = 1.0
    if size_mode == "height" and ext[up] > 1e-12:
        scale = float(height_m) / float(ext[up])
    mid = (bmin + bmax) / 2
    c = {"keep": np.zeros(3), "center": mid, "bottom_center": np.where(np.arange(3) == up, bmin[up], mid)}.get(pivot, np.zeros(3))
    th = math.radians(int(yaw) % 360)
    R = _rot_up(th, up)
    t = -scale * (R @ c)
    h = th / 2
    q = [0.0, 0.0, 0.0, math.cos(h)]
    q[up] = math.sin(h)
    ident = scale == 1.0 and int(yaw) % 360 == 0 and not np.any(c)
    return {"scale": scale, "yaw_deg": int(yaw) % 360, "pivot_point": c.tolist(), "translation": t.tolist(), "rotation": q, "matrix_R": R, "identity": bool(ident)}


def _rot_up(th: float, up: int = 1) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    R = np.eye(3)
    i, j = [k for k in range(3) if k != up]
    # right-handed rotation about the `up` axis
    if up == 1:
        R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
    else:
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def apply_placement(V, pl: dict, normals: bool = False):
    """Apply `placement()` to vertices (or, with normals=True, only rotate direction vectors)."""
    V = np.asarray(V, np.float64)
    R = np.asarray(pl["matrix_R"])
    if normals:
        return V @ R.T
    return pl["scale"] * ((V - np.asarray(pl["pivot_point"])) @ R.T)


# ================================================================== textures
def flip_green(img):
    """OpenGL (Y+) <-> DirectX (Y-) normal map: invert the green channel only. Alpha is kept."""
    from PIL import Image

    a = np.asarray(img.convert("RGBA" if "A" in img.getbands() else "RGB")).copy()
    a[..., 1] = 255 - a[..., 1]
    return Image.fromarray(a)


def renormalize_normal(img):
    """Decode xyz = rgb*2-1, normalise, encode again (after a resize the vectors are shorter than 1)."""
    from PIL import Image

    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0 * 2 - 1
    n = np.linalg.norm(a, axis=2, keepdims=True)
    a = np.where(n > 1e-6, a / np.maximum(n, 1e-6), np.array([0.0, 0.0, 1.0], np.float32))
    return Image.fromarray(np.clip(np.round((a * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8))


def downscale_pot(img, max_size, normal: bool = False):
    """Lanczos downscale so the long side is at most `max_size` (never upscale). Power-of-two sizes stay power-of-two. A normal map is
    renormalised afterwards. `max_size` 0 / None / 'keep' = unchanged."""
    from PIL import Image

    try:
        m = int(max_size)
    except (TypeError, ValueError):
        return img
    w, h = img.size
    if m <= 0 or max(w, h) <= m:
        return img
    s = m / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    out = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return renormalize_normal(out) if normal else out


def _gray(img):
    from PIL import Image

    return img if img.mode == "L" else img.convert("L") if img.mode != "RGB" else Image.fromarray(np.asarray(img)[..., 0])


def compose_orm(maps: dict):
    """The canonical ORM image of a map set: maps['orm'] as it is, or built from separate 'ao' / 'roughness' / 'metallic' greys
    (a missing AO = 255, a missing metallic = 0, a missing roughness = 255). None when the set has neither."""
    from PIL import Image

    if maps.get("orm") is not None:
        return maps["orm"].convert("RGB")
    parts = {k: maps.get(k) for k in ("ao", "roughness", "metallic")}
    if not any(v is not None for v in (parts["roughness"], parts["metallic"])):
        return None
    size = next(v.size for v in parts.values() if v is not None)
    ch = []
    for k, dflt in (("ao", 255), ("roughness", 255), ("metallic", 0)):
        v = parts[k]
        ch.append(np.asarray(_gray(v).resize(size)) if v is not None else np.full((size[1], size[0]), dflt, np.uint8))
    return Image.fromarray(np.stack(ch, -1))


def engine_material(target: str, unity_pipeline: str = "both") -> dict:
    """The `material: {layout, normal}` block of an engine preset, for the seamless-material export (docs/CONTRACTS.md): which packing
    layout and normal convention `pack_textures` applies for a target."""
    if target == "unreal":
        return {"layout": "unreal", "normal": "dx"}
    if target == "unity":
        return {"layout": "unity_hdrp" if unity_pipeline == "hdrp" else "unity_urp", "normal": "gl"}
    if target == "godot":
        return {"layout": "godot", "normal": "gl"}
    return {"layout": "gltf", "normal": "gl"}


def pack_textures(target: str, maps: dict, stem: str, opts: dict | None = None) -> list[tuple[str, object, dict]]:
    """PUBLIC. Repack one PBR map set for an engine. Pure PIL / numpy, importable from envs/core (the seamless-material export uses it).

    target   'unreal' | 'unity' | 'godot' | 'gltf'
    maps     canonical maps, each a PIL image (missing ones are skipped):
               basecolor  RGB / RGBA, sRGB
               normal     RGB, OpenGL (Y+), linear
               orm        RGB: R = AO, G = roughness, B = metallic, linear      -- or the separate greys 'ao', 'roughness', 'metallic'
               emissive   RGB, sRGB
               height     L (only written for Unity HDRP, as `<stem>_Height.png`)
    stem     the asset name WITHOUT prefix (`Rock`); the target decides the file pattern
    opts     tex_max (int, 0 = keep), unity_pipeline ('urp' | 'hdrp' | 'both', default both), prefix (Unreal texture prefix, default 'T_';
             '' = none), normal ('gl' | 'dx': override the target's convention)
    -> [(filename, PIL.Image, meta)], meta = {slot, colorspace ('srgb'|'linear'), packing (what is in the channels), size [w, h], use (the import setting)}

    Files: unreal T_<stem>_BC / _N (DirectX, green flipped) / _ORM / _E; unity <stem>_BaseMap / _Normal (OpenGL) / URP _MetallicSmoothness (RGB metal,
    A = 1 - roughness) + _Occlusion (grey) / HDRP _MaskMap (R metal, G AO, B 255, A = 1 - roughness) / _Emission / _Height;
    godot <stem>_albedo / _normal / _orm / _emission; gltf <stem>_basecolor / _normal / _orm / _emissive. A missing ORM writes no packed map."""
    from PIL import Image

    o = opts or {}
    tex_max = o.get("tex_max") or 0
    out: list[tuple[str, object, dict]] = []

    def add(name, img, slot, cs, packing, use, normal=False):
        img = downscale_pot(img, tex_max, normal=normal)
        out.append((f"{name}.png", img, {"slot": slot, "colorspace": cs, "packing": packing, "size": list(img.size), "use": use}))

    bc, nm, em = maps.get("basecolor"), maps.get("normal"), maps.get("emissive")
    orm = compose_orm(maps)
    nconv = o.get("normal") or ("dx" if target == "unreal" else "gl")
    if nm is not None and nconv == "dx":
        nm = flip_green(nm)
    ntxt = "normal map, DirectX (green flipped)" if nconv == "dx" else "normal map, OpenGL"
    bcm = bc.convert("RGBA" if "A" in bc.getbands() else "RGB") if bc is not None else None
    if target == "unreal":
        p = o.get("prefix", "T_")
        if bcm is not None:
            add(f"{p}{stem}_BC", bcm, "basecolor", "srgb", "RGB base colour", "sRGB on, compression Default")
        if nm is not None:
            add(f"{p}{stem}_N", nm, "normal", "linear", ntxt, "compression Normalmap, sRGB off", normal=True)
        if orm is not None:
            add(f"{p}{stem}_ORM", orm, "orm", "linear", "R occlusion, G roughness, B metallic", "compression Masks, sRGB off")
        if em is not None:
            add(f"{p}{stem}_E", em, "emissive", "srgb", "RGB emissive", "sRGB on")
    elif target == "unity":
        pipe = o.get("unity_pipeline", "both")
        if bcm is not None:
            add(f"{stem}_BaseMap", bcm, "basecolor", "srgb", "RGB base colour", "sRGB on")
        if nm is not None:
            add(f"{stem}_Normal", nm, "normal", "linear", ntxt, "Texture Type Normal map", normal=True)
        if orm is not None:
            a = np.asarray(orm)
            smooth = 255 - a[..., 1]
            if pipe in ("hdrp", "both"):
                mask = np.stack([a[..., 2], a[..., 0], np.full(a.shape[:2], 255, np.uint8), smooth], -1)
                add(f"{stem}_MaskMap", Image.fromarray(mask, "RGBA"), "mask", "linear", "R metallic, G occlusion, B detail mask (255), A smoothness (1 - roughness)",
                    "HDRP Lit Mask Map, sRGB off")
            if pipe in ("urp", "both"):
                ms = np.stack([a[..., 2], a[..., 2], a[..., 2], smooth], -1)
                add(f"{stem}_MetallicSmoothness", Image.fromarray(ms, "RGBA"), "metallic_smoothness", "linear",
                    "RGB metallic, A smoothness (1 - roughness)", "URP Lit Metallic map, sRGB off, Smoothness source Metallic alpha")
                add(f"{stem}_Occlusion", Image.fromarray(a[..., 0]), "occlusion", "linear", "grey occlusion", "URP Lit Occlusion map, sRGB off")
        if em is not None:
            add(f"{stem}_Emission", em, "emissive", "srgb", "RGB emissive", "sRGB on")
        if maps.get("height") is not None and pipe in ("hdrp", "both"):
            add(f"{stem}_Height", _gray(maps["height"]), "height", "linear", "grey height", "HDRP height map, sRGB off")
    elif target == "godot":
        if bcm is not None:
            add(f"{stem}_albedo", bcm, "basecolor", "srgb", "RGB base colour", "Albedo texture")
        if nm is not None:
            add(f"{stem}_normal", nm, "normal", "linear", ntxt, "Normal map (OpenGL, Godot default)", normal=True)
        if orm is not None:
            add(f"{stem}_orm", orm, "orm", "linear", "R occlusion, G roughness, B metallic", "ORM texture (set Orm = on in the material)")
        if em is not None:
            add(f"{stem}_emission", em, "emissive", "srgb", "RGB emissive", "Emission texture")
    else:
        if bcm is not None:
            add(f"{stem}_basecolor", bcm, "basecolor", "srgb", "RGB base colour", "baseColorTexture")
        if nm is not None:
            add(f"{stem}_normal", nm, "normal", "linear", ntxt, "normalTexture", normal=True)
        if orm is not None:
            add(f"{stem}_orm", orm, "orm", "linear", "R occlusion, G roughness, B metallic", "metallicRoughnessTexture + occlusionTexture")
        if em is not None:
            add(f"{stem}_emissive", em, "emissive", "srgb", "RGB emissive", "emissiveTexture")
    return out


# ================================================================== GLB
_CT = {np.dtype("<f4"): 5126, np.dtype("<u4"): 5125, np.dtype("<u2"): 5123, np.dtype("u1"): 5121}
_TYPE = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}
_NORM_DIV = {5121: 255.0, 5123: 65535.0, 5120: 127.0, 5122: 32767.0}


def glb_split(data: bytes):
    """-> (json dict, BIN bytes) of binary glTF bytes."""
    if len(data) < 20 or struct.unpack("<I", data[:4])[0] != 0x46546C67:
        raise EngineInputError("This file is not a GLB (binary glTF). Export it as GLB first.")
    jl = struct.unpack("<I", data[12:16])[0]
    j = json.loads(data[20:20 + jl].decode("utf-8"))
    off = 20 + jl
    blob = b""
    if off + 8 <= len(data):
        bl = struct.unpack("<I", data[off:off + 4])[0]
        blob = data[off + 8:off + 8 + bl]
    return j, blob


def glb_bytes(j: dict, blob) -> bytes:
    j = {k: v for k, v in j.items() if not (isinstance(v, list) and not v)}          # an empty top-level array is a validator error
    jb = json.dumps(j, separators=(",", ":")).encode("utf-8")
    jb += b" " * (-len(jb) % 4)
    bb = bytes(blob) + b"\0" * (-len(blob) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(jb) + (8 + len(bb) if bb else 0))
    out += struct.pack("<II", len(jb), 0x4E4F534A) + jb
    if bb:
        out += struct.pack("<II", len(bb), 0x004E4942) + bb
    return out


def read_accessor(j, blob, i, as_float: bool = True):
    """Accessor i as an (n, k) array (k = 1 -> (n,)). Normalised integer accessors become floats in 0..1 (or -1..1) when as_float."""
    a = j["accessors"][i]
    if a.get("sparse"):
        raise EngineInputError("Sparse accessors (morph targets) are not supported. Export the model without shape keys.")
    bv = j["bufferViews"][a["bufferView"]]
    dt = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1", 5122: "<i2", 5120: "i1"}[a["componentType"]]
    size = np.dtype(dt).itemsize
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
    stride = bv.get("byteStride") or size * n
    base = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    if stride == size * n:
        arr = np.frombuffer(blob, dt, a["count"] * n, base).copy()
    else:
        arr = np.ndarray((a["count"], n), dt, blob, base, (stride, size)).copy().reshape(-1)
    arr = arr.reshape(a["count"], n) if n > 1 else arr
    if as_float and a.get("normalized") and a["componentType"] != 5126:
        arr = np.maximum(arr.astype(np.float32) / _NORM_DIV[a["componentType"]], -1.0)
    return arr


def _node_matrix(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.asarray(n["matrix"], np.float64).reshape(4, 4).T
    t = np.asarray(n.get("translation", [0, 0, 0]), np.float64)
    q = np.asarray(n.get("rotation", [0, 0, 0, 1]), np.float64)
    s = np.asarray(n.get("scale", [1, 1, 1]), np.float64)
    x, y, z, w = q / max(np.linalg.norm(q), 1e-12)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * s
    M[:3, 3] = t
    return M


def mesh_nodes(j: dict) -> list[tuple[int, np.ndarray]]:
    """[(node index, world matrix)] of every node with a mesh that the default scene reaches (depth first)."""
    out = []
    scenes = j.get("scenes") or []
    roots = scenes[j.get("scene", 0)]["nodes"] if scenes else list(range(len(j.get("nodes", []))))
    seen = set()

    def walk(i, parent):
        if i in seen:
            return
        seen.add(i)
        n = j["nodes"][i]
        M = parent @ _node_matrix(n)
        if n.get("mesh") is not None:
            out.append((i, M))
        for c in n.get("children", []):
            walk(c, M)

    for r in roots:
        walk(r, np.eye(4))
    return out


def read_prims(j: dict, blob, mesh_index: int) -> list[dict]:
    """Primitives of a mesh: [{"attrs": {name: array}, "F": (m,3) int64, "material": int|None, "name", "extras"}]. Triangles only."""
    out = []
    me = j["meshes"][mesh_index]
    for k, p in enumerate(me.get("primitives", [])):
        if p.get("mode", 4) != 4:
            raise EngineInputError("Only triangle meshes can be exported (this model has lines or points).")
        if p.get("targets"):
            raise EngineInputError("This model has morph targets (shape keys), which the engine export does not support yet.")
        attrs = {}
        for name, i in p["attributes"].items():
            if name in ("POSITION", "NORMAL", "TANGENT", "TEXCOORD_0", "TEXCOORD_1", "COLOR_0", "JOINTS_0", "WEIGHTS_0"):
                attrs[name] = read_accessor(j, blob, i, as_float=not name.startswith("JOINTS"))
        if "POSITION" not in attrs:
            raise EngineInputError("A mesh primitive has no positions.")
        n = len(attrs["POSITION"])
        F = read_accessor(j, blob, p["indices"], as_float=False).astype(np.int64).reshape(-1, 3) if "indices" in p else \
            np.arange(n, dtype=np.int64).reshape(-1, 3)
        for key in list(attrs):
            if key.startswith(("POSITION", "NORMAL", "TANGENT", "TEXCOORD", "COLOR", "WEIGHTS")):
                attrs[key] = attrs[key].astype("<f4")
        out.append({"attrs": attrs, "F": F, "material": p.get("material"), "name": me.get("name") or f"mesh{mesh_index}_{k}",
                    "extras": p.get("extras")})
    return out


def bake_node_transform(prim: dict, M: np.ndarray) -> dict:
    """Apply a world matrix to a primitive's positions / normals / tangents (a negative determinant flips the winding)."""
    if np.allclose(M, np.eye(4)):
        return prim
    a = dict(prim["attrs"])
    A = M[:3, :3]
    V = a["POSITION"].astype(np.float64) @ A.T + M[:3, 3]
    a["POSITION"] = V.astype("<f4")
    Ni = np.linalg.inv(A).T
    if "NORMAL" in a:
        a["NORMAL"] = _unit(a["NORMAL"].astype(np.float64) @ Ni.T).astype("<f4")
    if "TANGENT" in a:
        t = a["TANGENT"].astype(np.float64)
        a["TANGENT"] = np.concatenate([_unit(t[:, :3] @ A.T, (1.0, 0.0, 0.0)), t[:, 3:]], 1).astype("<f4")
    F = prim["F"]
    if np.linalg.det(A) < 0:
        F = F[:, [0, 2, 1]]
    return {**prim, "attrs": a, "F": F}


def merge_same_material(prims: list[dict]) -> list[dict]:
    """Combine primitives that share a material and an attribute set (nodes flattened into one mesh)."""
    groups: dict = {}
    for p in prims:
        groups.setdefault((p["material"], tuple(sorted(p["attrs"]))), []).append(p)
    out = []
    for (_m, keys), ps in groups.items():
        if len(ps) == 1:
            out.append(ps[0])
            continue
        off, Fs = 0, []
        for p in ps:
            Fs.append(p["F"] + off)
            off += len(p["attrs"]["POSITION"])
        out.append({"attrs": {k: np.concatenate([p["attrs"][k] for p in ps]) for k in keys}, "F": np.concatenate(Fs),
                    "material": ps[0]["material"], "name": ps[0]["name"], "extras": None})
    return out


class GlbBuilder:
    """Append accessors / meshes / nodes to a parsed GLB (json + bytearray) and write it back, pruning what nothing uses.

    Everything that is not touched stays: materials, textures, images, samplers, skins, joints and animations. New binary data is appended
    with 4-byte alignment, vertex attributes get target 34962, indices 34963, POSITION gets min / max."""

    TARGET = {"POSITION": 34962}

    def __init__(self, j: dict, blob):
        self.j = j
        self.b = bytearray(blob)
        for k in ("accessors", "bufferViews", "meshes", "nodes", "materials"):
            self.j.setdefault(k, [])
        if not self.j.get("buffers"):
            self.j["buffers"] = [{"byteLength": 0}]

    def view(self, data: bytes, target: int | None = None) -> int:
        while len(self.b) % 4:
            self.b.append(0)
        bv = {"buffer": 0, "byteOffset": len(self.b), "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.b.extend(data)
        self.j["bufferViews"].append(bv)
        return len(self.j["bufferViews"]) - 1

    def accessor(self, arr, target: int | None = None, minmax: bool = False) -> int:
        a = np.ascontiguousarray(arr)
        d = {"bufferView": self.view(a.tobytes(), target), "componentType": _CT[a.dtype], "count": int(len(a)),
             "type": _TYPE[a.shape[1] if a.ndim > 1 else 1]}
        if minmax:
            d["min"] = [float(x) for x in a.min(0)]
            d["max"] = [float(x) for x in a.max(0)]
        self.j["accessors"].append(d)
        return len(self.j["accessors"]) - 1

    def primitive(self, attrs: dict, F, material=None, extras=None) -> dict:
        at = {}
        for k, v in attrs.items():
            v = np.asarray(v)
            if k.startswith("JOINTS"):
                v = v.astype("u1" if int(v.max(initial=0)) <= 255 else "<u2")
            else:
                v = v.astype("<f4")
            at[k] = self.accessor(v, 34962, minmax=(k == "POSITION"))
        F = np.asarray(F).reshape(-1)
        idx = F.astype("<u2" if len(attrs["POSITION"]) < 65535 else "<u4")
        p = {"attributes": at, "indices": self.accessor(idx, 34963)}
        if material is not None:
            p["material"] = int(material)
        if extras:
            p["extras"] = extras
        return p

    def mesh(self, name: str, prims: list[dict]) -> int:
        self.j["meshes"].append({"name": name, "primitives": prims})
        return len(self.j["meshes"]) - 1

    def node(self, name: str, **kw) -> int:
        self.j["nodes"].append({"name": name, **kw})
        return len(self.j["nodes"]) - 1

    def finish(self, prune: bool = True, drop_images: bool = False) -> bytes:
        """The GLB bytes. prune: rebuild the buffer without the accessors nothing uses any more; drop_images: also leave out the images."""
        self.j["buffers"][0]["byteLength"] = len(self.b)
        self.j["asset"] = {**self.j.get("asset", {}), "version": "2.0", "generator": "Open Asset Studio engine export"}
        if prune or drop_images:
            j, blob = prune_unused(self.j, bytes(self.b), drop_images=drop_images)
            return glb_bytes(j, blob)
        return glb_bytes(self.j, bytes(self.b))


def prune_unused(j: dict, blob: bytes, drop_images: bool = False) -> tuple[dict, bytes]:
    """Rebuild the buffer with only the accessors / bufferViews that meshes, skins, animations and images still reference.
    `drop_images` removes images and textures too (and the material texture references) - used for the extra LOD files of a rigged
    model, which keep the material name only. Whole bufferViews are copied (interleaved views keep their stride)."""
    j = copy.deepcopy(j)
    if drop_images:
        for m in j.get("materials", []):
            pbr = m.get("pbrMetallicRoughness", {})
            for k in ("baseColorTexture", "metallicRoughnessTexture"):
                pbr.pop(k, None)
            for k in ("normalTexture", "occlusionTexture", "emissiveTexture"):
                m.pop(k, None)
        j.pop("textures", None)
        j.pop("images", None)
        j.pop("samplers", None)
    acc_used: set[int] = set()
    for me in j.get("meshes", []):
        for p in me.get("primitives", []):
            acc_used.update(p.get("attributes", {}).values())
            if "indices" in p:
                acc_used.add(p["indices"])
            for t in p.get("targets", []):
                acc_used.update(t.values())
    for s in j.get("skins", []):
        if "inverseBindMatrices" in s:
            acc_used.add(s["inverseBindMatrices"])
    for an in j.get("animations", []):
        for sm in an.get("samplers", []):
            acc_used.update((sm["input"], sm["output"]))
    bv_used: set[int] = set()
    for i in acc_used:
        a = j["accessors"][i]
        if "bufferView" in a:
            bv_used.add(a["bufferView"])
        for k in ("indices", "values"):
            if a.get("sparse", {}).get(k):
                bv_used.add(a["sparse"][k]["bufferView"])
    for im in j.get("images", []):
        if "bufferView" in im:
            bv_used.add(im["bufferView"])
    acc_map = {old: new for new, old in enumerate(sorted(acc_used))}
    bv_order = sorted(bv_used)
    bv_map = {old: new for new, old in enumerate(bv_order)}
    out = bytearray()
    new_views = []
    for old in bv_order:
        bv = dict(j["bufferViews"][old])
        while len(out) % 4:
            out.append(0)
        off = bv.get("byteOffset", 0)
        data = blob[off:off + bv["byteLength"]]
        bv["byteOffset"] = len(out)
        out.extend(data)
        new_views.append(bv)
    new_acc = []
    for old in sorted(acc_used):
        a = dict(j["accessors"][old])
        if "bufferView" in a:
            a["bufferView"] = bv_map[a["bufferView"]]
        if "sparse" in a:
            sp = copy.deepcopy(a["sparse"])
            sp["indices"]["bufferView"] = bv_map[sp["indices"]["bufferView"]]
            sp["values"]["bufferView"] = bv_map[sp["values"]["bufferView"]]
            a["sparse"] = sp
        new_acc.append(a)
    for me in j.get("meshes", []):
        for p in me.get("primitives", []):
            p["attributes"] = {k: acc_map[v] for k, v in p["attributes"].items()}
            if "indices" in p:
                p["indices"] = acc_map[p["indices"]]
            if p.get("targets"):
                p["targets"] = [{k: acc_map[v] for k, v in t.items()} for t in p["targets"]]
    for s in j.get("skins", []):
        if "inverseBindMatrices" in s:
            s["inverseBindMatrices"] = acc_map[s["inverseBindMatrices"]]
    for an in j.get("animations", []):
        for sm in an.get("samplers", []):
            sm["input"], sm["output"] = acc_map[sm["input"]], acc_map[sm["output"]]
    for im in j.get("images", []):
        if "bufferView" in im:
            im["bufferView"] = bv_map[im["bufferView"]]
    j["accessors"], j["bufferViews"] = new_acc, new_views
    j["buffers"] = [{"byteLength": len(out)}]
    return j, bytes(out)


# ---- images inside a GLB
def material_image_roles(j: dict) -> dict[int, str]:
    """image index -> 'basecolor' | 'normal' | 'orm' | 'emissive' | 'other' (the first role a material gives it)."""
    roles: dict[int, str] = {}

    def img_of(ref):
        if not ref or ref.get("index") is None:
            return None
        t = j.get("textures", [])[ref["index"]]
        return t.get("source")

    for m in j.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        for role, ref in (("basecolor", pbr.get("baseColorTexture")), ("normal", m.get("normalTexture")),
                          ("orm", pbr.get("metallicRoughnessTexture")), ("occlusion", m.get("occlusionTexture")),
                          ("emissive", m.get("emissiveTexture"))):
            i = img_of(ref)
            if i is not None:
                roles.setdefault(i, role)
    return roles


def load_image(j: dict, blob, i: int):
    from PIL import Image

    im = j["images"][i]
    if "bufferView" not in im:
        return None                     # external / data URI images are not read
    bv = j["bufferViews"][im["bufferView"]]
    off = bv.get("byteOffset", 0)
    img = Image.open(io.BytesIO(blob[off:off + bv["byteLength"]]))
    img.load()
    return img


def material_maps(j: dict, blob, mi: int) -> dict:
    """The canonical map set of material `mi` of a GLB: {basecolor, normal, orm, emissive} (PIL images; missing ones absent) and
    {"factors": {...}}. glTF's metallicRoughness texture already is (any, G rough, B metal); R is the occlusion only when an occlusion
    texture uses the same image, otherwise R = 255 (a different occlusion image is merged into R)."""
    from PIL import Image

    m = j["materials"][mi]
    pbr = m.get("pbrMetallicRoughness", {})
    cache: dict = {}

    def img(ref):
        if not ref or ref.get("index") is None:
            return None
        i = j["textures"][ref["index"]].get("source")
        if i is None:
            return None
        if i not in cache:
            cache[i] = load_image(j, blob, i)
        return cache[i]

    out: dict = {}
    bc, nm, mr, oc, em = (img(pbr.get("baseColorTexture")), img(m.get("normalTexture")), img(pbr.get("metallicRoughnessTexture")),
                          img(m.get("occlusionTexture")), img(m.get("emissiveTexture")))
    if bc is not None:
        out["basecolor"] = bc.convert("RGBA" if "A" in bc.getbands() else "RGB")
    if nm is not None:
        out["normal"] = nm.convert("RGB")
    if mr is not None:
        a = np.asarray(mr.convert("RGB")).copy()
        same = (pbr.get("metallicRoughnessTexture") or {}).get("index") is not None and \
            (m.get("occlusionTexture") or {}).get("index") is not None and \
            j["textures"][pbr["metallicRoughnessTexture"]["index"]].get("source") == j["textures"][m["occlusionTexture"]["index"]].get("source")
        if not same:
            if oc is not None:
                o = np.asarray(oc.convert("L").resize((a.shape[1], a.shape[0])))
                a[..., 0] = o
            else:
                a[..., 0] = 255
        rf, mf = float(pbr.get("roughnessFactor", 1.0)), float(pbr.get("metallicFactor", 1.0))
        if rf != 1.0:
            a[..., 1] = np.clip(a[..., 1] * rf, 0, 255).astype(np.uint8)
        if mf != 1.0:
            a[..., 2] = np.clip(a[..., 2] * mf, 0, 255).astype(np.uint8)
        out["orm"] = Image.fromarray(a)
    if em is not None:
        out["emissive"] = em.convert("RGB")
    out["factors"] = {"base_color": pbr.get("baseColorFactor", [1, 1, 1, 1]), "metallic": pbr.get("metallicFactor", 1.0),
                      "roughness": pbr.get("roughnessFactor", 1.0), "emissive": m.get("emissiveFactor", [0, 0, 0]),
                      "alpha_mode": m.get("alphaMode", "OPAQUE"), "double_sided": bool(m.get("doubleSided", False)),
                      "name": m.get("name") or f"material{mi}"}
    return out


def resize_glb_images(j: dict, blob: bytes, tex_max) -> tuple[dict, bytes, int]:
    """Downscale every embedded image to `tex_max` (long side; POT stays POT), normal maps renormalised. Re-encoded as PNG and appended as
    new bufferViews, the old ones are pruned by finish() / prune_unused. -> (j, blob, number of images changed)."""
    try:
        m = int(tex_max)
    except (TypeError, ValueError):
        return j, blob, 0
    if m <= 0:
        return j, blob, 0
    j = copy.deepcopy(j)
    b = GlbBuilder(j, blob)
    roles = material_image_roles(j)
    n = 0
    for i, im in enumerate(j.get("images", [])):
        src = load_image(j, blob, i)
        if src is None or max(src.size) <= m:
            continue
        is_n = roles.get(i) == "normal"
        out = downscale_pot(src.convert("RGBA" if "A" in src.getbands() else "RGB"), m, normal=is_n)
        buf = io.BytesIO()
        out.save(buf, "PNG", compress_level=6)
        im["bufferView"] = b.view(buf.getvalue())
        im["mimeType"] = "image/png"
        n += 1
    j["buffers"][0]["byteLength"] = len(b.b)
    return j, bytes(b.b), n


def add_msft_lod(j: dict, lod0_node: int, lod_nodes: list[int], coverage: list[float]) -> None:
    """MSFT_lod on the LOD0 node: the lower LODs are nodes outside the scene graph; `extras.MSFT_screencoverage` lists the coverage at which each
    level hands over (the last is 0). Listed in extensionsUsed only (a viewer without it shows LOD0)."""
    if not lod_nodes:
        return
    n = j["nodes"][lod0_node]
    n.setdefault("extensions", {})["MSFT_lod"] = {"ids": list(lod_nodes)}
    n.setdefault("extras", {})["MSFT_screencoverage"] = [round(float(c), 4) for c in coverage[: len(lod_nodes) + 1]]
    used = j.setdefault("extensionsUsed", [])
    if "MSFT_lod" not in used:
        used.append("MSFT_lod")


def add_hull_meshes(b: GlbBuilder, hulls, names: list[str], parent: int | None = None, in_scene: bool = False, scene_roots: list | None = None) -> list[int]:
    """Hull meshes (POSITION + indices, no material) as nodes named `names[k]`. parent: a node to add them to as children; in_scene: add them
    to the scene root list `scene_roots` instead. Returns the node indices."""
    ids = []
    for (V, F), nm in zip(hulls, names):
        mi = b.mesh(nm, [b.primitive({"POSITION": np.asarray(V, "<f4")}, np.asarray(F, np.int64))])
        ni = b.node(nm, mesh=mi)
        ids.append(ni)
        if parent is not None:
            b.j["nodes"][parent].setdefault("children", []).append(ni)
        elif in_scene and scene_roots is not None:
            scene_roots.append(ni)
    return ids


def add_godot_hulls(b: GlbBuilder, hulls, stem: str, parent: int | None = None, scene_roots: list | None = None) -> list[int]:
    """Godot collision: nodes `<stem>_hull_NN-convcolonly` (one StaticBody3D with a convex shape each on import)."""
    names = [f"{stem}_hull_{k:02d}-convcolonly" for k in range(len(hulls))]
    return add_hull_meshes(b, hulls, names, parent=parent, in_scene=parent is None, scene_roots=scene_roots)


def hull_colors(n: int) -> list[list[float]]:
    pal = [[0.95, 0.45, 0.20], [0.20, 0.65, 0.95], [0.40, 0.80, 0.35], [0.85, 0.30, 0.70], [0.95, 0.80, 0.20], [0.35, 0.40, 0.95]]
    return [pal[i % len(pal)] for i in range(n)]


# ================================================================== reports
def x_style_ref(meta: dict | None) -> dict | None:
    """{name, version, hash} of an asset's x_style record (docs/CONTRACTS.md section 11.2: export manifests copy it), or None."""
    x = (meta or {}).get("x_style")
    if not isinstance(x, dict) or not (x.get("name") or x.get("hash")):
        return None
    return {k: x.get(k) for k in ("name", "version", "hash")}


def report(target: str, source: dict, sanitize_rep: dict, lods: list[dict], collision: dict, textures: list[dict], license_: dict | None,
           warnings: list[str], timings: dict) -> dict:
    return {"target": target, "source": source, "sanitize": sanitize_rep, "lods": lods, "collision": collision, "textures": textures,
            "license": license_, "warnings": warnings, "timings": timings}


def manifest(target: str, name: str, source_asset: dict, license_: dict | None, units: dict, lods: list[dict], collision: dict,
             textures: list[dict], files: list[str], style: dict | None = None, rigged: bool = False, names: dict | None = None) -> dict:
    """manifest.json of an engine package (also read by the Unity postprocessor: `tool` and `lods[].screen_size`)."""
    return {"tool": TOOL, "version": 1, "target": target, "name": name, "rigged": rigged, "source_asset": source_asset, "license": license_,
            "units": units, "up_axis": "Y", "forward": "+Z (glTF convention)", "lods": lods, "collision": collision, "textures": textures,
            "files": files, "style": style, "names": names or {}}


_README = {
    "unreal": [
        "Unreal Engine 5.4 - 5.6. Drag the files into the Content Browser (keep the Textures folder next to the FBX).",
        "Import dialog: Import Mesh LODs ON, Generate Missing Collision OFF (the UCX_ hulls are the collision), Skeletal Mesh OFF for SM_ files.",
        "Textures: T_*_N = Compression Normalmap (it is already DirectX, green flipped); T_*_ORM = Masks, sRGB OFF; T_*_BC = Default, sRGB ON.",
        "Material: plug ORM as R = Ambient Occlusion, G = Roughness, B = Metallic. The FBX only links the base colour and the normal map.",
        "Screen sizes for the LODs are in manifest.json (UE computes its own on import; paste ours if you want them). Nanite meshes ignore LODs.",
        "BETA: the FBX is written by our own exporter and was checked with ufbx and assimp, not yet in the Unreal editor.",
    ],
    "unity": [
        "Unity 6 (6000.x), URP or HDRP. Put this folder under Assets/ - keep manifest.json next to the model.",
        "Copy Editor/OpenAssetStudioPostprocessor.cs to a folder named Editor (it is already in this package's Editor folder). It turns the UCX_ hulls into convex MeshColliders and sets the LOD Group heights.",
        "The model has Name_LOD0..n children, which Unity turns into a LOD Group on import. Convert Units stays ON (the file is in centimetres).",
        "URP: <Name>_BaseMap, <Name>_Normal (Texture Type Normal map), <Name>_MetallicSmoothness (sRGB OFF, smoothness from Metallic alpha), <Name>_Occlusion.",
        "HDRP: <Name>_MaskMap (R metallic, G occlusion, B detail, A smoothness; sRGB OFF).",
        "EXPERIMENTAL: the FBX and the Editor script were not tested in the Unity Editor (no Unity on the build machine).",
    ],
    "godot": [
        "Godot 4.3+. Copy this folder into your project (res://) and let it import.",
        "The .glb holds the mesh and the collision nodes named ...-convcolonly: Godot makes one convex StaticBody3D for each.",
        "Godot builds its own LODs when the model is imported, so there is only LOD0 here.",
        "The material is already linked (ORM texture). The Textures folder has loose copies if you want to edit them.",
    ],
    "gltf": [
        "Plain glTF 2.0 (binary). The .glb uses the MSFT_lod extension: viewers that do not know it show LOD0.",
        "<Name>_collision.glb holds the convex hulls, one node per hull, so they do not draw in ordinary viewers.",
        "Textures are embedded: base colour, normal (OpenGL), ORM (R occlusion, G roughness, B metallic).",
    ],
}


def readme(target: str, name: str, rigged: bool, lines_extra: list[str] | None = None) -> str:
    L = list(_README.get(target, []))
    if rigged and target in ("unreal", "unity"):
        L.append("Rigged / animated model: delivered as GLB with the skeleton and animation clips kept (skinned FBX is not written yet). "
                 "Unreal 5.3+ Interchange and Unity glTFast import GLB; the lower LODs are separate _LODn.glb files.")
    L += lines_extra or []
    return f"{name} - {TOOL} engine package ({target})\n\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(L)) + "\n"
