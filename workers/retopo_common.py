"""Prototype of workers/retopo_common.py: pure numpy / scipy helpers (no open3d, no native libs) so tests can import them.

Conventions
  * polygons from Instant Meshes: (F, 4) uint32; a triangle is a row with a repeated index (IM writes f[2] == f[3]).
  * glTF UV: origin top-left, v grows downwards. Image row r <-> v = (r + 0.5) / H. No flips in image space.
  * tangent frame (glTF / MikkTSpace / OpenGL "Y+" normal maps): T along +u, B = w * cross(N, T) along +v_up where
    v_up = 1 - v_gltf (image up). MikkTSpace is therefore fed (u, 1 - v).
"""
from __future__ import annotations

import json
import struct

import numpy as np


# Texts shared by the worker, the plugin handler and the UI tests (InputError / JobError messages, shown to the user as they are)
E1 = "Add a 3D model (GLB) to retopologise."
E3 = ("This model is rigged (it has a skeleton). Retopology rebuilds every vertex, so the skin and its animations would be lost. "
      "Run Retopo on the unrigged model (its source in the lineage), then Rig the result.")
E4 = "This model has animation or shape keys, which retopology cannot keep. Use the static model it was made from."


# ------------------------------------------------------------------ polygons
def split_polys(nf: np.ndarray):
    """(F,4) IM faces -> (quads (Q,4), tris (T,3)). Rows with a repeated index are triangles (any position)."""
    nf = np.asarray(nf, dtype=np.int64)
    if nf.ndim != 2 or nf.shape[1] not in (3, 4):
        raise ValueError("faces must be (F,3) or (F,4)")
    if nf.shape[1] == 3:
        return np.zeros((0, 4), np.int64), nf
    a, b, c, d = nf.T
    dup = (a == b) | (b == c) | (c == d) | (d == a) | (a == c) | (b == d)
    quads = nf[~dup]
    tri_rows = nf[dup]
    tris = np.array([list(dict.fromkeys(r.tolist()))[:3] for r in tri_rows], dtype=np.int64).reshape(-1, 3)
    tris = tris[[len(set(t)) == 3 for t in tris.tolist()]] if len(tris) else tris
    return quads, tris


def triangulate(quads: np.ndarray, tris: np.ndarray, verts: np.ndarray | None = None):
    """Quads -> 2 triangles each, emitted FIRST and in pairs (2k, 2k+1 share the diagonal), then the plain triangles.
    The diagonal is the shorter one when `verts` is given (better shading of non-planar quads).
    Returns (triangles (2Q+T,3), n_quads)."""
    q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    if verts is not None and len(q):
        v = np.asarray(verts)
        d02 = np.linalg.norm(v[q[:, 0]] - v[q[:, 2]], axis=1)
        d13 = np.linalg.norm(v[q[:, 1]] - v[q[:, 3]], axis=1)
        q = np.where((d13 < d02)[:, None], q[:, [1, 2, 3, 0]], q)
    pairs = np.stack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=1).reshape(-1, 3)
    return np.vstack([pairs, np.asarray(tris, dtype=np.int64).reshape(-1, 3)]), len(q)


def quad_wire_edges(tri: np.ndarray, n_quads: int) -> np.ndarray:
    """Unique edges to draw for a 'quad wireframe': every triangle edge except the shared diagonal of each quad pair."""
    tri = np.asarray(tri, dtype=np.int64)
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    keep = np.ones(len(e), bool)
    n = len(tri)
    # the diagonal of pair k is edge (0,2) of tri 2k (= index 2n+2k in the [20] block) and edge (0,1) of tri 2k+1
    k = np.arange(n_quads)
    keep[2 * n + 2 * k] = False           # tri 2k edge (2,0)
    keep[2 * k + 1] = False               # tri 2k+1 edge (0,1)
    e = np.sort(e[keep], axis=1)
    return np.unique(e, axis=0)


def poly_edges(polys: list[list[int]]):
    """{(a,b) sorted: [(face, a, b), ...]} for a list of polygons (any degree)."""
    E: dict = {}
    for fi, f in enumerate(polys):
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            E.setdefault((min(a, b), max(a, b)), []).append((fi, a, b))
    return E


def poly_stats(quads, tris) -> dict:
    """Topology report: faces, quad %, triangles in the engine, boundary / non-manifold edges, components, irregular verts."""
    polys = [list(q) for q in np.asarray(quads).tolist()] + [list(t) for t in np.asarray(tris).tolist()]
    E = poly_edges(polys)
    bnd = sum(1 for v in E.values() if len(v) == 1)
    nm = sum(1 for v in E.values() if len(v) > 2)
    nq, nt = len(quads), len(tris)
    val = np.bincount(np.asarray([i for f in polys for i in f], dtype=np.int64)) if polys else np.zeros(0)
    used = val > 0
    # interior-vertex valence in a quad mesh: 4 is regular (boundary vertices are not counted)
    irregular = float(np.mean(val[used] != 4)) if used.any() else 0.0
    return {"faces": nq + nt, "quads": nq, "tris": nt, "quad_pct": round(100.0 * nq / max(1, nq + nt), 1),
            "triangles": 2 * nq + nt, "boundary_edges": bnd, "nonmanifold_edges": nm,
            "components": components(polys)[0], "irregular_vert_pct": round(100 * irregular, 1)}


def components(polys):
    """Face components through shared manifold edges -> (count, labels)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    E = poly_edges(polys)
    r, c = [], []
    for v in E.values():
        if len(v) == 2:
            r.append(v[0][0]); c.append(v[1][0])
    n = len(polys)
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return connected_components(A, directed=False)


def boundary_loops_ex(polys):
    """Walk the boundary half-edges of a polygon soup -> (closed loops, unclosed chains).

    A hole runs opposite to its neighbour face, so each boundary edge (a, b) of a face becomes the directed edge b -> a. Every
    directed edge is used exactly once. At a pinched vertex (a figure-8: several outgoing edges) the walk prefers the edge that
    returns to a vertex already on the current path, and whenever the path revisits a vertex the sub-loop is cut off and emitted
    on its own, so two holes that touch in one vertex become two loops. A loop is oriented so that a fan (c, l[i], l[i+1]) faces
    the same way as the mesh. Chains that dead-end (a non-manifold border) are returned as the second value."""
    E = poly_edges(polys)
    out: dict[int, list[int]] = {}
    for v in E.values():
        if len(v) == 1:
            _, a, b = v[0]
            out.setdefault(b, []).append(a)
    loops, chains = [], []
    for s in list(out):
        while out.get(s):
            path, pos = [s], {s: 0}
            cur = out[s].pop()
            while True:
                if cur in pos:
                    i = pos[cur]
                    loop = path[i:]
                    if len(loop) >= 3:
                        loops.append(loop)
                    else:
                        chains.append(loop)
                    for q in path[i + 1:]:
                        del pos[q]
                    del path[i + 1:]
                else:
                    pos[cur] = len(path)
                    path.append(cur)
                nxt = out.get(cur)
                if not nxt:
                    break
                k = next((j for j, c in enumerate(nxt) if c in pos), len(nxt) - 1)
                cur = nxt.pop(k)
            if len(path) > 1:
                chains.append(path)
    return loops, chains


def boundary_loops(polys):
    """Closed boundary loops as vertex lists (see boundary_loops_ex)."""
    return boundary_loops_ex(polys)[0]


def clean_polys(verts, quads, tris, *, min_component_faces=24, keep_components_frac=0.01, max_hole_edges=64):
    """Post-remesh repair: drop faces on non-manifold edges, drop components smaller than
    max(min_component_faces, keep_components_frac * all faces) (the largest always stays), fan-fill holes of up to
    `max_hole_edges` edges. Returns (verts, quads, tris, report); report["open_holes"] counts loops left unfilled plus chains that
    could not be closed. New hole-fill vertices are appended (positions = loop centroid; the caller snaps them onto the surface)."""
    verts = np.asarray(verts, dtype=np.float64)
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    rep = {"dropped_nonmanifold_faces": 0, "dropped_component_faces": 0, "filled_holes": 0, "open_holes": 0}
    polys = [list(q) for q in quads.tolist()] + [list(t) for t in tris.tolist()]
    E = poly_edges(polys)
    bad = {fi for v in E.values() if len(v) > 2 for fi, _, _ in v}
    rep["dropped_nonmanifold_faces"] = len(bad)
    polys = [f for i, f in enumerate(polys) if i not in bad]
    n, lab = components(polys)
    if n > 1:
        size = np.bincount(lab, minlength=n)
        keep = size >= max(min_component_faces, keep_components_frac * len(polys))
        keep[int(np.argmax(size))] = True                       # the largest component always stays
        rep["dropped_component_faces"] = int(size[~keep].sum())
        polys = [f for f, l in zip(polys, lab) if keep[l]]
    new_v = []
    nv = len(verts)
    loops, chains = boundary_loops_ex(polys)
    rep["open_holes"] += len(chains)
    for L in loops:
        if len(L) > max_hole_edges:
            rep["open_holes"] += 1
            continue
        c = nv + len(new_v)
        new_v.append(verts[L].mean(0))
        for i in range(len(L)):
            polys.append([c, L[i], L[(i + 1) % len(L)]])
        rep["filled_holes"] += 1
    if new_v:
        verts = np.vstack([verts, np.asarray(new_v)])
    q = np.array([f for f in polys if len(f) == 4], dtype=np.int64).reshape(-1, 4)
    t = np.array([f for f in polys if len(f) == 3], dtype=np.int64).reshape(-1, 3)
    used = np.zeros(len(verts), bool)
    used[q.ravel()] = True; used[t.ravel()] = True
    remap = -np.ones(len(verts), np.int64); remap[used] = np.arange(used.sum())
    rep["hole_vertices"] = [int(remap[i]) for i in range(nv, len(verts)) if used[i]]
    return verts[used], remap[q], remap[t], rep


# ------------------------------------------------------------------ normals / tangents
def vertex_normals(v, f):
    v = np.asarray(v, dtype=np.float64); f = np.asarray(f, dtype=np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])          # area weighted
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def tangents_lengyel(v, n, uv, f):
    """Fallback per-vertex tangents (accumulate + Gram-Schmidt) in the same convention as MikkTSpace fed (u, 1-v)."""
    v = np.asarray(v, np.float64); n = np.asarray(n, np.float64); f = np.asarray(f, np.int64)
    st = np.c_[uv[:, 0], 1.0 - uv[:, 1]].astype(np.float64)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    w0, w1, w2 = st[f[:, 0]], st[f[:, 1]], st[f[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-20, 1e-20, r)
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]
    T = np.zeros_like(v); B = np.zeros_like(v)
    for k in range(3):
        np.add.at(T, f[:, k], sdir); np.add.at(B, f[:, k], tdir)
    T = T - n * np.sum(n * T, 1, keepdims=True)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-20)
    w = np.where(np.sum(np.cross(n, T) * B, 1) < 0, -1.0, 1.0)
    return np.c_[T, w]


# ------------------------------------------------------------------ tangent-space encode / decode
def encode_ts_normal(n_world, N, T, w):
    """World-space target normals -> tangent-space vectors for the shader basis (T, w*cross(N,T), N) with N, T the
    interpolated (unnormalised) vertex attributes. Uses the exact inverse of that basis, so normalize(TBN @ t) == n."""
    N = np.asarray(N, np.float64); T = np.asarray(T, np.float64)
    B = w[:, None] * np.cross(N, T)
    M = np.stack([T, B, N], axis=2)                       # columns T, B, N
    t = np.linalg.solve(M, np.asarray(n_world, np.float64)[..., None])[..., 0]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-20)
    t[:, 2] = np.maximum(t[:, 2], 0.0)                    # a normal map cannot point below the surface
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-20)


def decode_ts_normal(t, N, T, w):
    """What a glTF renderer does: n = normalize(t.x T + t.y (w cross(N,T)) + t.z N)."""
    B = w[:, None] * np.cross(N, T)
    n = t[:, 0:1] * T + t[:, 1:2] * B + t[:, 2:3] * N
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def to_rgb8(t):
    return np.clip(np.round((np.asarray(t) * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)


def from_rgb8(c):
    return np.asarray(c, np.float64) / 255.0 * 2.0 - 1.0


# ------------------------------------------------------------------ texture helpers
def texel_centers(res: int):
    """(res*res, 2) glTF UVs of texel centres, row-major (row 0 = v near 0 = image top)."""
    c = (np.arange(res) + 0.5) / res
    u, v = np.meshgrid(c, c)
    return np.c_[u.ravel(), v.ravel()]


def dilate(img: np.ndarray, mask: np.ndarray, pad: int | None = None, fill=None):
    """Edge padding: every texel outside `mask` copies its nearest texel inside (scipy EDT). pad=None: whole image;
    else only within `pad` texels, the rest gets `fill` (default: mean colour of the mask)."""
    from scipy.ndimage import distance_transform_edt
    if mask.all() or not mask.any():
        return img
    d, (ri, ci) = distance_transform_edt(~mask, return_indices=True)
    out = img[ri, ci]
    if pad is not None:
        far = d > pad
        out[far] = (np.asarray(fill) if fill is not None else img[mask].mean(0)).astype(img.dtype)
    out[mask] = img[mask]
    return out


def bilinear(img: np.ndarray, uv: np.ndarray, wrap=True):
    """Sample an (H,W,C) float image at glTF UVs (v down) with bilinear filtering (REPEAT wrap like glTF's default)."""
    H, W = img.shape[:2]
    x = uv[:, 0] * W - 0.5
    y = uv[:, 1] * H - 0.5
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    def at(yy, xx):
        if wrap:
            yy = np.mod(yy, H); xx = np.mod(xx, W)
        else:
            yy = np.clip(yy, 0, H - 1); xx = np.clip(xx, 0, W - 1)
        return img[yy, xx].astype(np.float64)
    return (at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy)
            + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy)


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def auto_cage(dist_hi_to_lo: np.ndarray, diag: float, lo=0.002, hi=0.05):
    """Ray distance from the sampled high->low distances: 1.5 x the 99.9th percentile, clipped to [lo, hi] x diagonal."""
    d = 1.5 * float(np.percentile(dist_hi_to_lo, 99.9)) if len(dist_hi_to_lo) else 0.0
    return float(np.clip(d, lo * diag, hi * diag))


def linear_to_srgb(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0.0), 1 / 2.4) - 0.055)


def next_request(req: int, target: int, got: int, tol: float = 0.08):
    """Face-count correction for Instant Meshes (it overshoots by ~10 %): None when `got` is within `tol` of `target`,
    otherwise the corrected request for a second run."""
    if got <= 0 or abs(got - target) / max(1, target) <= tol:
        return None
    return max(16, int(round(req * target / got)))


def choose_hit(t_front, t_back, cage):
    """Nearest of the +n and -n ray hits within the cage distance (ties -> front). Returns (use_front, hit); misses have hit False."""
    tf = np.where(np.asarray(t_front) <= cage, t_front, np.inf)
    tb = np.where(np.asarray(t_back) <= cage, t_back, np.inf)
    return tf <= tb, np.isfinite(np.minimum(tf, tb))


def bary_interp(attr, f, pid, puv):
    """Interpolate a per-vertex attribute at hits (Open3D convention: p = (1-a-b) v0 + a v1 + b v2, puv = (a, b))."""
    a, b = puv[:, 0:1], puv[:, 1:2]
    tri = f[pid]
    return attr[tri[:, 0]] * (1 - a - b) + attr[tri[:, 1]] * a + attr[tri[:, 2]] * b


def ao_directions(n, k, rng):
    """Cosine-weighted hemisphere directions around the unit normals n (M,3): returns (k, M, 3). Malley's method:
    r = sqrt(u1), phi = 2 pi u2, z = sqrt(1 - u1) in a frame built around n. Deterministic for a seeded rng."""
    n = np.asarray(n, np.float64)
    m = len(n)
    a = np.where(np.abs(n[:, 0:1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    t = np.cross(n, a)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    u1 = rng.random((k, m))
    u2 = rng.random((k, m))
    r, phi, z = np.sqrt(u1), 2 * np.pi * u2, np.sqrt(1 - u1)
    return (t[None] * (r * np.cos(phi))[..., None] + b[None] * (r * np.sin(phi))[..., None] + n[None] * z[..., None])


def bbox_match(a_min, a_max, b_min, b_max, tol: float = 0.01) -> bool:
    """True when both corners of two boxes agree within `tol` of the first box's diagonal."""
    a_min, a_max, b_min, b_max = (np.asarray(x, np.float64) for x in (a_min, a_max, b_min, b_max))
    diag = float(np.linalg.norm(a_max - a_min))
    err = max(float(np.abs(a_min - b_min).max()), float(np.abs(a_max - b_max).max()))
    return err <= tol * max(diag, 1e-12)


# ------------------------------------------------------------------ GLB / OBJ I/O (plain struct + json: importable in envs/core)
def _pad4(b: bytes, fill: bytes = b"\0") -> bytes:
    return b + fill * (-len(b) % 4)


def pack_glb(path, V, N, TAN, UV, F, images: dict, material: dict, extras: dict | None = None, generator: str = "Open Asset Studio retopo"):
    """Write a one-mesh, one-primitive GLB. V/N (n,3), TAN (n,4) or None, UV (n,2) glTF UVs, F (m,3) triangles,
    images {name: png bytes} embedded as bufferViews, material keys:
      name, base_color_image, base_color_factor, orm_image (metallicRoughness AND occlusion), normal_image,
      metallic_factor, roughness_factor, alpha_mode, double_sided.
    Samplers are linear / mipmapped / repeat. `extras` goes on the primitive."""
    blob = bytearray()
    views, accessors = [], []

    def view(data: bytes, target=None) -> int:
        while len(blob) % 4:
            blob.append(0)
        v = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            v["target"] = target
        blob.extend(data)
        views.append(v)
        return len(views) - 1

    def acc(arr, ctype, atype, target, minmax=False):
        a = np.ascontiguousarray(arr)
        d = {"bufferView": view(a.tobytes(), target), "componentType": ctype, "count": int(len(a)), "type": atype}
        if minmax:
            d["min"] = [float(x) for x in a.min(0)]
            d["max"] = [float(x) for x in a.max(0)]
        accessors.append(d)
        return len(accessors) - 1

    V = np.asarray(V, np.float64)
    N = np.asarray(N, np.float64)
    attrs = {"POSITION": acc(V.astype("<f4"), 5126, "VEC3", 34962, True),
             "NORMAL": acc((N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)).astype("<f4"), 5126, "VEC3", 34962)}
    if TAN is not None:
        attrs["TANGENT"] = acc(np.asarray(TAN).astype("<f4"), 5126, "VEC4", 34962)
    attrs["TEXCOORD_0"] = acc(np.asarray(UV).astype("<f4"), 5126, "VEC2", 34962)
    ind = acc(np.asarray(F).reshape(-1).astype("<u4"), 5125, "SCALAR", 34963)
    img_list, tex_of = [], {}
    for name, png in images.items():
        img_list.append({"bufferView": view(png), "mimeType": "image/png", "name": name})
        tex_of[name] = len(img_list) - 1
    m = material
    pbr = {"metallicFactor": float(m.get("metallic_factor", 1.0)), "roughnessFactor": float(m.get("roughness_factor", 1.0))}
    if m.get("base_color_image") in tex_of:
        pbr["baseColorTexture"] = {"index": tex_of[m["base_color_image"]]}
    pbr["baseColorFactor"] = [float(x) for x in m.get("base_color_factor", [1, 1, 1, 1])]
    mat = {"name": m.get("name", "retopo"), "pbrMetallicRoughness": pbr, "doubleSided": bool(m.get("double_sided", False))}
    if m.get("orm_image") in tex_of:
        pbr["metallicRoughnessTexture"] = {"index": tex_of[m["orm_image"]]}
        mat["occlusionTexture"] = {"index": tex_of[m["orm_image"]]}
    if m.get("normal_image") in tex_of:
        mat["normalTexture"] = {"index": tex_of[m["normal_image"]]}
    if m.get("alpha_mode"):
        mat["alphaMode"] = m["alpha_mode"]
    prim = {"attributes": attrs, "indices": ind, "material": 0}
    if extras:
        prim["extras"] = extras
    j = {"asset": {"version": "2.0", "generator": generator},
         "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": "retopo", "mesh": 0}],
         "meshes": [{"name": "retopo", "primitives": [prim]}],
         "materials": [mat], "accessors": accessors, "bufferViews": views,
         "buffers": [{"byteLength": len(blob)}]}
    if img_list:
        j["images"] = img_list
        j["textures"] = [{"source": i, "sampler": 0} for i in range(len(img_list))]
        j["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    jb = _pad4(json.dumps(j, separators=(",", ":")).encode(), b" ")
    bb = _pad4(bytes(blob))
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(jb) + 8 + len(bb)))
        fh.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        fh.write(struct.pack("<II", len(bb), 0x004E4942) + bb)


_CT = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1", 5122: "<i2", 5120: "i1"}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path) -> dict:
    """Read back a GLB written by pack_glb (first mesh, first primitive): {"json", "attributes": {name: float64 array},
    "indices" (m,3), "extras", "material" (the JSON material), "images" {texture index: png bytes}}. Tightly packed accessors only."""
    data = open(path, "rb").read()
    magic, ver, _total = struct.unpack("<III", data[:12])
    if magic != 0x46546C67:
        raise ValueError("not a GLB file")
    jl = struct.unpack("<I", data[12:16])[0]
    j = json.loads(data[20:20 + jl])
    off = 20 + jl
    blob = b""
    if off + 8 <= len(data):
        bl = struct.unpack("<I", data[off:off + 4])[0]
        blob = data[off + 8:off + 8 + bl]

    def acc(i):
        a = j["accessors"][i]
        bv = j["bufferViews"][a["bufferView"]]
        n = _NC[a["type"]]
        arr = np.frombuffer(blob, _CT[a["componentType"]], a["count"] * n, bv.get("byteOffset", 0) + a.get("byteOffset", 0))
        return arr.reshape(a["count"], n) if n > 1 else arr

    prim = j["meshes"][0]["primitives"][0]
    out = {"json": j, "attributes": {k: acc(i).astype(np.float64) for k, i in prim["attributes"].items()},
           "indices": acc(prim["indices"]).astype(np.int64).reshape(-1, 3), "extras": prim.get("extras"),
           "material": j["materials"][prim.get("material", 0)], "images": {}}
    for ti, t in enumerate(j.get("textures", [])):
        bv = j["bufferViews"][j["images"][t["source"]]["bufferView"]]
        out["images"][ti] = blob[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
    return out


def write_obj(path, verts, polys, uv, poly_uv, mtl_name: str | None = None, name: str = "retopo"):
    """Wavefront OBJ with real quads. verts (n,3); polys = list of vertex-index lists (3 or 4 each, 0-based); uv (m,2) glTF
    UVs (written as OBJ vt = (u, 1 - v)); poly_uv = per polygon the uv indices (same length as the polygon)."""
    uv = np.asarray(uv, np.float64)
    lines = [f"# Open Asset Studio retopo: {len(polys)} polygons"]
    if mtl_name:
        lines.append(f"mtllib {mtl_name}")
    lines.append(f"o {name}")
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in np.asarray(verts, np.float64)]
    lines += [f"vt {u:.6f} {1.0 - v:.6f}" for u, v in uv]
    if mtl_name:
        lines.append(f"usemtl {name}")
    lines.append("s 1")
    for pv, pt in zip(polys, poly_uv):
        lines.append("f " + " ".join(f"{a + 1}/{b + 1}" for a, b in zip(pv, pt)))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_mtl(path, name: str = "retopo", base_png: str | None = None, normal_png: str | None = None):
    lines = [f"newmtl {name}", "Ka 1 1 1", "Kd 1 1 1", "Ks 0 0 0", "d 1", "illum 2"]
    if base_png:
        lines.append(f"map_Kd {base_png}")
    if normal_png:
        lines.append(f"norm {normal_png}")
        lines.append(f"map_Bump {normal_png}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ warnings
def report_warnings(report: dict, params: dict) -> list[str]:
    """User-facing warnings W1-W8 from the measured report (so the thresholds are testable)."""
    w = []
    fh = int(params.get("fill_holes", 64))
    if report.get("open_holes", 0) > 0:
        k = int(report["open_holes"])
        w.append(f"{k} hole(s) larger than {fh} edges were left open (real openings such as sleeves are kept). The mesh is not watertight.")
    if report.get("ray_miss_pct", 0.0) > 2.0:
        w.append(f"{report['ray_miss_pct']:.1f}% of the texels found no surface within the ray distance; they use the closest point instead. "
                 "If the maps show smears, raise Ray distance.")
    if (report.get("colour_source") or {}).get("kind") == "none" and params.get("bake_color", True):
        w.append("The source has no texture or vertex colours, so only the normal and AO maps were baked.")
    t = int(report.get("target_faces", 0) or 0)
    got = int(report.get("faces", 0) or 0)
    if t and abs(got - t) / t > 0.15:
        w.append(f"The result has {got:,} faces for a target of {t:,} ({100.0 * (got - t) / t:+.0f}%).")
    p99 = report.get("dist_p99_frac")
    if p99 is not None and p99 > 0.005:
        w.append(f"The shape is coarse here: the low mesh is up to {p99:.2%} of the model size away from the original. "
                 "Raise the face target if fingers or thin parts look wrong.")
    if report.get("white_rejected"):
        w.append("The stored shape mesh (white_mesh.glb) does not line up with this model, so the model itself was used as the detail source.")
    if report.get("tangents") == "lengyel":
        w.append("Tangents were computed with the built-in fallback (MikkTSpace is not installed); the normal map is still correct in glTF viewers.")
    if report.get("predecimated_from"):
        w.append(f"The input was reduced to 2 million faces for remeshing; the bake still used all {int(report['predecimated_from']):,} faces.")
    return w


PRESETS = {    # target_faces is in polygons (quads + triangles); the engine sees about 1.9x triangles
    "mobile": {"target_faces": 2500, "tex_size": 1024, "ao_rays": 16},
    "pc": {"target_faces": 12000, "tex_size": 2048, "ao_rays": 16},
    "hero": {"target_faces": 30000, "tex_size": 4096, "ao_rays": 16},
}
