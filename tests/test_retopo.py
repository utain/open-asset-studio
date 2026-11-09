"""Retopo + bake (models.d/retopo.yaml, workers/retopo_common.py, app/plugins/retopo.py, m3d_common.carry_vertex_attrs).
No GPU, no weights, no envs/retopo: the pure helpers import in envs/core (numpy / scipy / PIL only)."""
import io
import json
import random
import struct
import sys

import numpy as np
import pytest
import yaml
from PIL import Image

from app import config
from app.registry import Registry

sys.path.insert(0, str(config.STUDIO / "workers"))
import m3d_common as M3  # noqa: E402
import retopo_common as C  # noqa: E402

MID = "retopo-instant-meshes"


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path_factory.mktemp("p") / "probe.json")


# ------------------------------------------------------------------ registry and wiring
def test_group_loads_without_errors(reg):
    assert not [e for e in reg.errors if "retopo" in e]
    assert MID in reg.models
    assert reg.models[MID]["_file"] == "retopo.yaml"


def test_task_and_backend(reg):
    assert reg.tasks["retopo"]["studio"] == "3d"
    b = reg.backends["retopo"]
    assert b["python"] == "envs/retopo/bin/python"
    assert (config.STUDIO / b["script"]).exists()
    assert b["device"] == "cpu"
    assert reg.is_gpu(MID) is False
    assert reg.models[MID]["default_for"] == ["retopo"]


def test_handler_registered():
    import app.plugins.retopo  # noqa: F401
    from app.jobs import HANDLERS
    assert HANDLERS["retopo.bake"]["gpu"] is False


def test_defaults_and_presets_validate(reg):
    assert isinstance(reg.validate_params(MID, {}), dict)
    names = {p["name"] for p in reg.models[MID]["params"]}
    presets = reg.models[MID]["presets"]
    assert set(presets) == {"mobile", "pc", "hero"}
    for k, preset in presets.items():
        assert set(preset) <= names, k
        v = reg.validate_params(MID, preset)
        assert isinstance(v["tex_size"], int)                      # the worker casts, but the choice values are ints already
        for key, val in C.PRESETS[k].items():
            if key in preset:
                assert preset[key] == val, (k, key)


def test_licence_commercial_and_install_recipe(reg):
    assert reg.models[MID]["license"]["commercial"] is True
    r = yaml.safe_load((config.STUDIO / "install" / "retopo.yaml").read_text())
    assert r["backend"] == "retopo" and r["env"] == "retopo" and r["lock"] == "retopo"
    assert (config.STUDIO / "install" / "files" / "pynim_pure_quad.patch").is_file()
    assert (config.STUDIO / "env-lock" / "retopo.txt").is_file()
    lock = (config.STUDIO / "vendor" / "vendor.lock").read_text().splitlines()
    assert any(ln.startswith("vendor/PyNanoInstantMeshes ") and "d02a725f93a1a66308ae0a63532779ae815ecc32" in ln for ln in lock)
    assert "PyNanoInstantMeshes" in (config.STUDIO / "THIRD_PARTY_NOTICES.md").read_text()


# ------------------------------------------------------------------ test meshes
def box_quads(n, drop=()):
    """Closed n x n x n box surface of outward quads on the integer lattice. drop = set of (face index 0..5, i, j) to leave out."""
    vid, V, Q = {}, [], []

    def vtx(p):
        if p not in vid:
            vid[p] = len(V)
            V.append(p)
        return vid[p]

    for f in range(6):
        a, s = f // 2, f % 2
        u, v = [i for i in range(3) if i != a]
        for i in range(n):
            for j in range(n):
                quad = []
                for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = [0, 0, 0]
                    p[a], p[u], p[v] = s * n, i + di, j + dj
                    quad.append(tuple(p))
                c = np.mean(quad, axis=0) - n / 2
                nrm = np.cross(np.subtract(quad[1], quad[0]), np.subtract(quad[3], quad[0]))
                if np.dot(nrm, c) < 0:
                    quad = quad[::-1]
                if (f, i, j) in drop:
                    continue
                Q.append([vtx(p) for p in quad])
    return np.array(V, float), np.array(Q, np.int64)


def test_split_polys():
    nf = np.array([[0, 1, 2, 3], [4, 5, 6, 6], [7, 7, 8, 9]])
    q, t = C.split_polys(nf)
    assert q.tolist() == [[0, 1, 2, 3]]
    assert sorted(map(sorted, t.tolist())) == [[4, 5, 6], [7, 8, 9]]


def test_triangulate_pairs_and_shorter_diagonal():
    V = np.array([[0, 0, 0], [4, 0, 0], [3, 1, 0], [0, 1, 0],            # quad A: diagonal 0-2 is shorter
                  [0, 0, 1], [4, 0, 1], [4, 1, 1], [1, 1, 1]], float)    # quad B: diagonal 1-3 (5-7) is shorter
    quads = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    tris = np.array([[0, 1, 4]])
    T, nq = C.triangulate(quads, tris, V)
    assert nq == 2 and T.shape == (5, 3)
    assert set(T[0]) & set(T[1]) == {0, 2}                                # pair 0 shares the 0-2 diagonal
    assert set(T[2]) & set(T[3]) == {5, 7}                                # pair 1 shares the shorter 5-7 diagonal
    assert T[4].tolist() == [0, 1, 4]                                     # plain triangles come after the pairs


def test_quad_wire_edges_grid_has_no_diagonals():
    V = [(x, y, 0) for y in range(3) for x in range(3)]
    q = np.array([[y * 3 + x, y * 3 + x + 1, (y + 1) * 3 + x + 1, (y + 1) * 3 + x] for y in range(2) for x in range(2)])
    T, nq = C.triangulate(q, np.zeros((0, 3), int), np.array(V, float))
    e = C.quad_wire_edges(T, nq)
    assert len(e) == 12
    for a, b in e:
        assert abs(V[a][0] - V[b][0]) + abs(V[a][1] - V[b][1]) == 1


def test_poly_stats_cube():
    V, Q = box_quads(1)
    s = C.poly_stats(Q, np.zeros((0, 3), int))
    assert (s["faces"], s["quads"], s["boundary_edges"], s["nonmanifold_edges"], s["components"], s["quad_pct"]) == (6, 6, 0, 0, 1, 100.0)


def test_clean_polys_fills_holes_and_drops_fragments():
    V, Q = box_quads(12, drop={(4, 3, 3), (5, 6, 6), (5, 7, 6), (5, 6, 7), (5, 7, 7)})     # a 4-edge hole and an 8-edge hole
    V = np.vstack([V, [[50, 0, 0], [51, 0, 0], [50, 1, 0], [51, 1, 0]]])
    n = len(V) - 4
    tris = np.array([[n, n + 1, n + 2], [n + 1, n + 3, n + 2]])                             # a 2-face fragment
    assert C.poly_stats(Q, tris)["boundary_edges"] == 12 + 4
    v2, q2, t2, rep = C.clean_polys(V, Q, tris)
    assert rep["filled_holes"] == 2 and rep["open_holes"] == 0 and rep["dropped_component_faces"] == 2
    assert len(rep["hole_vertices"]) == 2
    s = C.poly_stats(q2, t2)
    assert s["boundary_edges"] == 0 and s["components"] == 1
    assert len(t2) == 12 and len(v2) == len(V) - 4 + 2                                      # 4 + 8 fan triangles, 2 new centres


def test_clean_polys_big_hole_stays_open():
    V, Q = box_quads(25, drop={(0, i, j) for i in range(25) for j in range(25)})            # one whole face missing: 100 edges
    _, _, _, rep = C.clean_polys(V, Q, np.zeros((0, 3), int), max_hole_edges=64)
    assert rep["open_holes"] == 1 and rep["filled_holes"] == 0
    _, _, t, rep = C.clean_polys(V, Q, np.zeros((0, 3), int), max_hole_edges=128)
    assert rep["open_holes"] == 0 and rep["filled_holes"] == 1 and len(t) == 100


def test_boundary_loops_figure_8():
    """Two holes that touch in one vertex (a pinched / figure-8 boundary): the prototype walker lost them (16 edges stayed open)."""
    V, Q = box_quads(4, drop={(4, 0, 0), (4, 1, 1)})
    polys0 = [list(q) for q in Q.tolist()]
    pinch = [set(a) & set(b) for a, b in [(Q[i], Q[j]) for i in range(len(Q)) for j in range(i)]]
    assert pinch                                                                              # sanity: the fixture is a real box
    rnd = random.Random(1)
    for _ in range(8):                                                                        # the result must not depend on the face order
        polys = polys0[:]
        rnd.shuffle(polys)
        polys = [p[k:] + p[:k] for p in polys for k in [rnd.randrange(4)]]
        loops, chains = C.boundary_loops_ex(polys)
        assert not chains
        assert sorted(len(x) for x in loops) == [4, 4]
        E = C.poly_edges(polys)
        bnd = {k for k, v in E.items() if len(v) == 1}
        used = [frozenset((L[i], L[(i + 1) % len(L)])) for L in loops for i in range(len(L))]
        assert len(used) == len(set(used)) == len(bnd) == 8                                   # every boundary edge exactly once
    v2, q2, t2, rep = C.clean_polys(V, Q, np.zeros((0, 3), int))
    assert rep["filled_holes"] == 2 and rep["open_holes"] == 0
    assert C.poly_stats(q2, t2)["boundary_edges"] == 0


def test_boundary_loops_simple_and_unclosed():
    loops, chains = C.boundary_loops_ex([[0, 1, 2], [0, 2, 3], [0, 3, 4]])          # a fan: one loop of 5 edges
    assert [sorted(x) for x in loops] == [[0, 1, 2, 3, 4]] and not chains
    loops, chains = C.boundary_loops_ex([[0, 1, 2], [2, 1, 3]])                     # two triangles: one loop of 4 edges
    assert len(loops) == 1 and len(loops[0]) == 4 and not chains
    loops, chains = C.boundary_loops_ex([[0, 1, 2], [1, 2, 3]])                     # flipped neighbour: the border cannot close
    assert not loops and chains


def test_encode_decode_round_trip():
    rng = np.random.default_rng(3)
    n = 1000
    N = rng.normal(size=(n, 3))
    N *= rng.uniform(0.5, 2, (n, 1)) / np.linalg.norm(N, axis=1, keepdims=True)
    T = N[:, [1, 2, 0]] + rng.normal(size=(n, 3)) * 0.5                                       # non-orthogonal, unnormalised
    w = rng.choice([-1.0, 1.0], n)
    t = rng.normal(size=(n, 3)) * 0.4
    t[:, 2] = np.abs(t[:, 2]) + 0.6
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n_world = C.decode_ts_normal(t, N, T, w)
    t2 = C.encode_ts_normal(n_world, N, T, w)
    back = C.decode_ts_normal(t2, N, T, w)
    ang = np.arccos(np.clip(np.sum(back * n_world, 1), -1, 1))
    assert ang.max() < 1e-5
    flat = C.encode_ts_normal(N / np.linalg.norm(N, axis=1, keepdims=True), N, T, w)
    assert np.allclose(flat, [0, 0, 1], atol=1e-6)
    assert np.abs(C.to_rgb8(flat[:1]).astype(int) - [128, 128, 255]).max() <= 1
    assert np.allclose(C.from_rgb8(np.array([[0, 128, 255]])), [[-1, 0.0039, 1]], atol=1e-3)


def test_tangent_convention_matches_gltf():
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    N = np.tile([0.0, 0.0, 1.0], (4, 1))
    uv = np.c_[V[:, 0], 1.0 - V[:, 1]]                                                         # u = x, glTF v = 1 - y (v points down)
    tan = C.tangents_lengyel(V, N, uv, F)
    assert np.allclose(tan[:, :3], [1, 0, 0], atol=1e-6) and np.all(tan[:, 3] == 1)
    assert np.allclose(tan[:, 3:4] * np.cross(N, tan[:, :3]), [0, 1, 0], atol=1e-6)            # B points to image up = +Y
    uv2 = np.c_[1.0 - V[:, 0], 1.0 - V[:, 1]]
    tan2 = C.tangents_lengyel(V, N, uv2, F)
    assert np.all(tan2[:, 3] == -1)
    assert np.allclose(tan2[:, 3:4] * np.cross(N, tan2[:, :3]), [0, 1, 0], atol=1e-6)          # a mirrored u flips w, B stays image up


def test_texel_centers_and_bilinear_wrap():
    tc = C.texel_centers(4)
    assert tc.shape == (16, 2)
    assert np.allclose(tc[:4, 1], 0.125) and np.allclose(tc[:4, 0], [0.125, 0.375, 0.625, 0.875])
    assert tc[4, 1] == 0.375                                                                   # next rows go down the image
    img = np.array([[[0.0], [0.0], [0.0], [100.0]]])
    assert np.allclose(C.bilinear(img, np.array([[0.0, 0.5], [1.0, 0.5]])), 50.0)              # repeat wrap blends the edge columns
    assert np.allclose(C.bilinear(img, np.array([[0.875, 0.5]])), 100.0)


def test_dilate_fills_every_texel_with_the_nearest_valid_one():
    img = np.zeros((8, 8, 3), np.uint8)
    mask = np.zeros((8, 8), bool)
    img[2, 2], mask[2, 2] = (10, 20, 30), True
    img[6, 5], mask[6, 5] = (200, 100, 50), True
    out = C.dilate(img, mask)
    assert (out[2, 2] == (10, 20, 30)).all() and (out[6, 5] == (200, 100, 50)).all()
    assert (out[0, 0] == (10, 20, 30)).all() and (out[7, 7] == (200, 100, 50)).all() and (out[3, 2] == (10, 20, 30)).all()
    assert C.dilate(img, np.ones((8, 8), bool)) is img


def test_cage_request_and_hit_choice():
    d = np.full(1000, 1e-6)
    assert C.auto_cage(d, 10.0) == pytest.approx(0.02)                                         # clipped up to 0.002 x diag
    assert C.auto_cage(np.full(1000, 5.0), 10.0) == pytest.approx(0.5)                         # clipped down to 0.05 x diag
    assert C.auto_cage(np.full(1000, 0.1), 10.0) == pytest.approx(0.15)
    assert C.next_request(1000, 1000, 1100) == 909
    assert C.next_request(1000, 1000, 1079) is None and C.next_request(1000, 1000, 920) is None
    assert C.next_request(10, 1000, 100000) == 16
    use_f, hit = C.choose_hit(np.array([0.5, np.inf, 0.2, 0.3, 0.9]), np.array([0.4, np.inf, 0.2, np.inf, 0.8]), 0.6)
    assert use_f.tolist()[:4] == [False, True, True, True] and hit.tolist() == [True, False, True, True, False]


def test_srgb_round_trip():
    x = np.linspace(0, 1, 101)
    assert np.allclose(C.linear_to_srgb(C.srgb_to_linear(x)), x, atol=1e-9)
    assert C.linear_to_srgb(0.5) == pytest.approx(0.7354, abs=1e-3)


def test_ao_directions_cosine_weighted():
    rng = np.random.default_rng(0)
    n = np.tile([0.0, 0.0, 1.0], (20000, 1))
    d = C.ao_directions(n, 1, rng)[0]
    assert np.allclose(np.linalg.norm(d, axis=1), 1) and (d[:, 2] >= 0).all()
    assert abs(d[:, 2].mean() - 2 / 3) < 0.02
    nn = rng.normal(size=(500, 3))
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    assert (np.sum(C.ao_directions(nn, 4, rng) * nn[None], axis=2) >= -1e-9).all()
    a = C.ao_directions(nn, 2, np.random.default_rng(5))
    assert np.array_equal(a, C.ao_directions(nn, 2, np.random.default_rng(5)))


def _png(color):
    b = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(b, "PNG")
    return b.getvalue()


def _two_quads():
    V = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]], float)
    F = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]])
    N = np.tile([0.0, 0.0, 1.0], (6, 1))
    UV = np.c_[V[:, 0] / 2, 1 - V[:, 1]]
    TAN = np.tile([1.0, 0.0, 0.0, 1.0], (6, 1))
    return V, N, TAN, UV, F


def test_pack_glb_round_trip(tmp_path):
    V, N, TAN, UV, F = _two_quads()
    mat = {"base_color_image": "basecolor", "orm_image": "orm", "normal_image": "normal", "metallic_factor": 1.0, "roughness_factor": 1.0}
    p = tmp_path / "m.glb"
    C.pack_glb(p, V, N, TAN, UV, F, {"basecolor": _png("red"), "normal": _png((128, 128, 255)), "orm": _png("white")}, mat, {"oas_quad_pairs": 2})
    r = C.read_glb(p)
    A = r["attributes"]
    assert A["POSITION"].shape == (6, 3) and A["TANGENT"].shape == (6, 4) and A["TEXCOORD_0"].shape == (6, 2)
    assert np.allclose(A["POSITION"], V) and np.array_equal(r["indices"], F) and r["extras"] == {"oas_quad_pairs": 2}
    j = r["json"]
    acc = j["accessors"][j["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert acc["min"] == [0, 0, 0] and acc["max"] == [2, 1, 0]
    m = r["material"]
    assert "normalTexture" in m and "occlusionTexture" in m and "metallicRoughnessTexture" in m["pbrMetallicRoughness"]
    assert m["occlusionTexture"]["index"] == m["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]      # one ORM texture
    assert len(r["images"]) == 3 and Image.open(io.BytesIO(r["images"][0])).size == (4, 4)
    data = p.read_bytes()
    jl = struct.unpack("<I", data[12:16])[0]
    assert jl % 4 == 0 and len(data) % 4 == 0 and struct.unpack("<I", data[8:12])[0] == len(data)
    assert struct.unpack("<I", data[20 + jl:24 + jl])[0] % 4 == 0
    assert all(bv["byteOffset"] % 4 == 0 for bv in j["bufferViews"])
    assert M3.glb_flags(p)["triangles"] == 4
    C.pack_glb(tmp_path / "plain.glb", V, N, None, UV, F, {}, {})                                        # no maps, no tangents: still valid
    assert "TANGENT" not in C.read_glb(tmp_path / "plain.glb")["attributes"]


def test_write_obj(tmp_path):
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], float)
    UV = np.array([[0, 0.25], [1, 0.25], [1, 1], [0, 1], [0.5, 0.5]])
    C.write_obj(tmp_path / "a.obj", V, [[0, 1, 2, 3], [1, 4, 2]], UV, [[0, 1, 2, 3], [1, 4, 2]], mtl_name="a.mtl")
    lines = (tmp_path / "a.obj").read_text().splitlines()
    assert "mtllib a.mtl" in lines
    f = [ln for ln in lines if ln.startswith("f ")]
    assert f == ["f 1/1 2/2 3/3 4/4", "f 2/2 5/5 3/3"]
    vt = [ln.split() for ln in lines if ln.startswith("vt ")]
    assert float(vt[0][2]) == pytest.approx(0.75)                                                         # OBJ v = 1 - glTF v
    C.write_mtl(tmp_path / "a.mtl", "retopo", "b.png", "n.png")
    assert "map_Kd b.png" in (tmp_path / "a.mtl").read_text() and "norm n.png" in (tmp_path / "a.mtl").read_text()


def test_bbox_match():
    assert C.bbox_match([0, 0, 0], [10, 10, 10], [0.05, 0, 0], [10, 10, 10.1])
    assert not C.bbox_match([0, 0, 0], [10, 10, 10], [0.5, 0, 0], [10, 10, 10])


def test_report_warnings_thresholds():
    ok = {"open_holes": 0, "ray_miss_pct": 2.0, "target_faces": 10000, "faces": 11500, "dist_p99_frac": 0.005, "tangents": "mikktspace",
          "colour_source": {"kind": "texture"}}
    assert C.report_warnings(ok, {"bake_color": True}) == []
    w = C.report_warnings({**ok, "open_holes": 3}, {"fill_holes": 64})
    assert len(w) == 1 and w[0].startswith("3 hole(s) larger than 64 edges")
    assert len(C.report_warnings({**ok, "ray_miss_pct": 2.1}, {})) == 1
    assert len(C.report_warnings({**ok, "faces": 11501}, {})) == 1 and "+15%" in C.report_warnings({**ok, "faces": 11501}, {})[0]
    assert len(C.report_warnings({**ok, "faces": 8499}, {})) == 1
    assert len(C.report_warnings({**ok, "dist_p99_frac": 0.0051}, {})) == 1
    assert len(C.report_warnings({**ok, "colour_source": {"kind": "none"}}, {"bake_color": True})) == 1
    assert C.report_warnings({**ok, "colour_source": {"kind": "none"}}, {"bake_color": False}) == []
    assert len(C.report_warnings({**ok, "white_rejected": True}, {})) == 1
    assert len(C.report_warnings({**ok, "tangents": "lengyel"}, {})) == 1
    assert "2,500,000" not in C.report_warnings({**ok, "predecimated_from": 5_000_000}, {})[0] and "5,000,000" in C.report_warnings({**ok, "predecimated_from": 5_000_000}, {})[0]


# ------------------------------------------------------------------ rig chain: normals / tangents / extras survive the trimesh round trip
def _strip(path, drop_normal=True):
    j, blob = M3._glb_split(path)
    prim = j["meshes"][0]["primitives"][0]
    for a in ("NORMAL", "TANGENT"):
        if a in prim["attributes"] and (a == "TANGENT" or drop_normal):
            del prim["attributes"][a]
    prim.pop("extras", None)
    M3._glb_write(path, j, blob)


def test_carry_vertex_attrs(tmp_path):
    V, N, TAN, UV, F = _two_quads()
    N = N + np.array([0.0, 0.2, 0.0])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    src, dst = tmp_path / "src.glb", tmp_path / "dst.glb"
    for p in (src, dst):
        C.pack_glb(p, V, N, TAN, UV, F, {"normal": _png("blue")}, {"normal_image": "normal"}, {"oas_quad_pairs": 2})
    _strip(dst)
    before = C.read_glb(dst)
    assert "NORMAL" not in before["attributes"] and before["extras"] is None
    assert M3.carry_vertex_attrs(src, dst) is True
    a, b = C.read_glb(src), C.read_glb(dst)
    assert np.array_equal(a["attributes"]["NORMAL"], b["attributes"]["NORMAL"])
    assert np.array_equal(a["attributes"]["TANGENT"], b["attributes"]["TANGENT"])
    assert b["extras"] == {"oas_quad_pairs": 2}
    assert np.array_equal(a["indices"], b["indices"]) and np.array_equal(a["attributes"]["POSITION"], b["attributes"]["POSITION"])
    assert len(b["images"]) == 1
    # a second call changes nothing worth writing: the attributes are identical now, the extras are there
    assert M3.carry_vertex_attrs(src, dst) is False


def test_carry_vertex_attrs_replaces_recomputed_normals(tmp_path):
    V, N, TAN, UV, F = _two_quads()
    N2 = N + np.array([0.3, 0.0, 0.0])
    src, dst = tmp_path / "src.glb", tmp_path / "dst.glb"
    C.pack_glb(src, V, N2, TAN, UV, F, {}, {}, {"oas_quad_pairs": 2})
    C.pack_glb(dst, V, N, None, UV, F, {}, {})
    assert M3.carry_vertex_attrs(src, dst) is True
    assert np.allclose(C.read_glb(dst)["attributes"]["NORMAL"], C.read_glb(src)["attributes"]["NORMAL"])


def test_carry_vertex_attrs_no_match_leaves_file_untouched(tmp_path):
    V, N, TAN, UV, F = _two_quads()
    src, dst = tmp_path / "src.glb", tmp_path / "dst.glb"
    C.pack_glb(src, V, N, TAN, UV, F, {}, {}, {"oas_quad_pairs": 2})
    C.pack_glb(dst, V + [0.01, 0, 0], N, None, UV, F, {}, {})
    _strip(dst)
    raw = dst.read_bytes()
    assert M3.carry_vertex_attrs(src, dst) is False
    assert dst.read_bytes() == raw
    C.pack_glb(dst, V[:5], N[:5], None, UV[:5], F[:3], {}, {})                                          # different vertex count
    raw = dst.read_bytes()
    assert M3.carry_vertex_attrs(src, dst) is False and dst.read_bytes() == raw


def test_carry_vertex_attrs_keeps_skin_attributes(tmp_path):
    V, N, TAN, UV, F = _two_quads()
    src, dst = tmp_path / "src.glb", tmp_path / "dst.glb"
    C.pack_glb(src, V, N, TAN, UV, F, {}, {}, {"oas_quad_pairs": 2})
    C.pack_glb(dst, V, N, None, UV, F, {}, {})
    j, blob = M3._glb_split(dst)
    blob = bytearray(blob)
    j["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": 6 * 8, "target": 34962})
    blob.extend(bytes(range(48)))
    j["accessors"].append({"bufferView": len(j["bufferViews"]) - 1, "componentType": 5123, "count": 6, "type": "VEC4"})
    j["meshes"][0]["primitives"][0]["attributes"]["JOINTS_0"] = len(j["accessors"]) - 1
    j["buffers"][0]["byteLength"] = len(blob)
    M3._glb_write(dst, j, bytes(blob))
    assert M3.carry_vertex_attrs(src, dst) is True
    j2, blob2 = M3._glb_split(dst)
    ja = j2["meshes"][0]["primitives"][0]["attributes"]
    assert ja["JOINTS_0"] == len(j["accessors"]) - 1 and "TANGENT" in ja
    assert M3._glb_accessor(j2, blob2, ja["JOINTS_0"]).tobytes() == bytes(range(48))


def test_glb_flags(tmp_path):
    j = {"asset": {"version": "2.0"}, "skins": [{"joints": [0]}], "animations": [{"channels": []}, {"channels": []}],
         "accessors": [{"count": 300, "componentType": 5125, "type": "SCALAR"}, {"count": 100, "componentType": 5126, "type": "VEC3"}],
         "meshes": [{"primitives": [{"attributes": {"POSITION": 1}, "indices": 0, "targets": [{"POSITION": 1}]}]}]}
    p = tmp_path / "x.glb"
    M3._glb_write(p, j, b"")
    f = M3.glb_flags(p)
    assert f == {"skins": 1, "animations": 2, "morph_targets": True, "primitives": 1, "triangles": 100}
    assert M3.glb_read_json(p)["skins"]
    j.update(skins=[], animations=[])
    j["meshes"][0]["primitives"][0].pop("targets")
    M3._glb_write(p, j, b"")
    assert M3.glb_flags(p)["skins"] == 0 and not M3.glb_flags(p)["morph_targets"]
    with pytest.raises(ValueError):
        (tmp_path / "bad.glb").write_bytes(b"nope" * 10)
        M3.glb_read_json(tmp_path / "bad.glb")


# ------------------------------------------------------------------ plugin handler: refuses rigged input before the worker starts
def test_plugin_refuses_rigged_inputs(tmp_path):
    from app.jobs import JobError
    from app.plugins import retopo as plug
    ok = {"worker": {}, "info": {}}
    plug.check_input(ok, tmp_path / "missing.glb", C, M3)
    for meta, text in (({"worker": {"rigged": True}}, C.E3), ({"info": {"rigged": True}}, C.E3), ({"worker": {"animations": ["x"]}}, C.E4)):
        with pytest.raises(JobError) as e:
            plug.check_input(meta, tmp_path / "m.glb", C, M3)
        assert str(e.value) == text and e.value.kind == "input"
    j = {"asset": {"version": "2.0"}, "skins": [{"joints": [0]}], "meshes": [], "accessors": []}
    p = tmp_path / "rigged.glb"
    M3._glb_write(p, j, b"")
    with pytest.raises(JobError):
        plug.check_input(ok, p, C, M3)


def test_plugin_licence_rule():
    from app.plugins import retopo as plug
    nc, com = {"name": "X", "commercial": False}, {"name": "Y", "commercial": True}
    tool = {"name": "tool", "commercial": True}
    assert plug._lic([nc, tool]) is nc
    assert plug._lic([com, tool]) is com
    assert plug._lic([None, tool]) is tool
    assert plug._lic([None, None]) is None
