"""Game-engine export (models.d/engine.yaml, workers/engine_common.py, fbx_write.py, meshopt_ctypes.py, app/plugins/engine.py).

No GPU, no weights, no envs/retopo: the registry entry, the pure numpy / PIL helpers, the FBX writer (read back with a tiny reader
below), the GLB builder, the meshopt wrapper (skipped when libmeshoptimizer.so is not built), and the delivery security
(destinations, fd-based writer, REST refusals, MCP gating). The end-to-end worker run is covered live in docs/research/engine_export.md.
"""
import hashlib
import importlib.util
import json
import os
import stat
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from app import config
from app.registry import Registry

WORKERS = config.STUDIO / "workers"
if str(WORKERS) not in sys.path:
    sys.path.insert(0, str(WORKERS))

import engine_common as C  # noqa: E402
import fbx_write as FW  # noqa: E402
import meshopt_ctypes as MO  # noqa: E402
from app.plugins import engine as E  # noqa: E402


# ================================================================== fixtures and tiny builders
@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path_factory.mktemp("p") / "probe.json")


def cube(size=1.0):
    s = size / 2
    V = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)], np.float64)
    h = C.hull_of_points(V)
    return h


def uv_sphere(n_lat=24, n_lon=32, r=1.0):
    """UV sphere with a seam: the first and last longitude columns are separate vertices at the same position."""
    V, UV = [], []
    for i in range(n_lat + 1):
        th = np.pi * i / n_lat
        for k in range(n_lon + 1):
            ph = 2 * np.pi * k / n_lon
            V.append([r * np.sin(th) * np.cos(ph), r * np.cos(th), r * np.sin(th) * np.sin(ph)])
            UV.append([k / n_lon, i / n_lat])
    F = []
    w = n_lon + 1
    for i in range(n_lat):
        for k in range(n_lon):
            a, b, c, d = i * w + k, i * w + k + 1, (i + 1) * w + k, (i + 1) * w + k + 1
            if i > 0:
                F.append([a, c, b])
            if i < n_lat - 1:
                F.append([b, c, d])
    return np.array(V, np.float64), np.array(UV, np.float64), np.array(F, np.int64)


# ================================================================== 1. registry and wiring
def test_registry_entry_loads_and_is_wired(reg):
    assert not [e for e in reg.errors if "engine" in e], reg.errors
    assert reg.tasks["engine_export"]["studio"] == "3d"
    be = reg.backends["engine"]
    assert be["python"].endswith("envs/retopo/bin/python") and (config.STUDIO / be["script"]).is_file()
    m = reg.models["engine-export"]
    assert m["handler"] == "engine.export" and m["backend"] == "engine" and m["tasks"] == ["engine_export"]
    assert m["weights"] == [] and m["license"]["commercial"] is True
    assert reg.default_for("engine_export") == "engine-export"


# ================================================================== 2. params and presets
def test_every_param_validates_its_default_and_every_preset_applies(reg):
    m = reg.models["engine-export"]
    out = reg.validate_params("engine-export", {})
    for p in m["params"]:
        assert p["name"] in out, p["name"]
        assert out[p["name"]] == p["default"], p["name"]
    assert len(m["presets"]) >= 6
    for name, preset in m["presets"].items():
        got = reg.validate_params("engine-export", dict(preset))
        for k, v in preset.items():
            assert got[k] == v, (name, k)
    with pytest.raises(Exception):
        reg.validate_params("engine-export", {"target": "cryengine"})


# ================================================================== 3. install recipe
def test_install_recipe_is_valid_and_pins_the_vendor():
    import subprocess

    r = subprocess.run([str(config.STUDIO / "install.sh"), "validate"], capture_output=True, text=True, cwd=config.STUDIO, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    import yaml

    rec = yaml.safe_load((config.STUDIO / "install" / "engine.yaml").read_text())
    assert rec.get("shares_env_with") == "retopo" and "meshoptimizer" in (rec.get("vendor") or [])
    lock = (config.STUDIO / "vendor" / "vendor.lock").read_text()
    assert "vendor/meshoptimizer " in lock and "f0a684f11ead064e176775c9b796f82a55910e92" in lock
    assert "coacd==1.0.14" in (config.STUDIO / "env-lock" / "retopo.txt").read_text()


# ================================================================== 4. plugin registration
def test_plugin_registers_handler_and_router():
    from app.jobs import HANDLERS

    h = HANDLERS["engine.export"]
    assert (h.get("gpu") if isinstance(h, dict) else getattr(h, "gpu", False)) in (False, 0)
    paths = {(r.path, tuple(sorted(r.methods))) for r in E.router.routes}
    assert ("/destinations", ("GET",)) in paths and ("/deliver", ("POST",)) in paths


# ================================================================== 5. sanitize
def test_sanitize_rebuilds_normals_weights_and_compacts():
    V, F = cube()
    n = len(V)
    N = np.zeros((n, 3), np.float32)
    N[:4] = [0, 0, 5.0]                                   # non-unit
    attrs = {"POSITION": np.vstack([V, [[9, 9, 9]]]), "NORMAL": np.vstack([N, [[1, 0, 0]]]),   # last vertex unused
             "JOINTS_0": np.vstack([np.tile([1, 2, 3, 4], (n, 1)), [[0, 0, 0, 0]]]).astype(np.uint16),
             "WEIGHTS_0": np.vstack([np.tile([0.5, 0.5, 0, 0], (n, 1)), [[1, 0, 0, 0]]]).astype(np.float32),
             "TEXCOORD_0": np.zeros((n + 1, 2), np.float32)}
    F2 = np.vstack([F, [[0, 0, 1]]])                      # one degenerate triangle
    a, f, rep = C.sanitize(attrs, F2)
    assert len(a["POSITION"]) == n and len(f) == len(F)
    assert all(len(v) == n for v in a.values())           # still aligned after compaction
    ln = np.linalg.norm(a["NORMAL"], axis=1)
    assert np.allclose(ln, 1, atol=1e-5)                  # zeros rebuilt, 5.0 normalised
    assert rep["zero_normals"] == n - 4 and rep["non_unit_normals"] == 4 and rep["degenerate_tris"] == 1 and rep["unused_vertices"] == 1
    assert np.allclose(a["WEIGHTS_0"].sum(1), 1, atol=1e-5)
    assert (a["JOINTS_0"][:, 2:] == 0).all() and a["JOINTS_0"].dtype == np.uint8
    assert rep["warnings"]


# ================================================================== 6. LOD plan
def test_plan_lods_presets_floor_absolute_and_skip():
    plan, skipped = C.plan_lods(10000, 3, "halves", None, 300)
    assert [p["target"] for p in plan] == [5000, 2500, 1250] and not skipped
    plan, _ = C.plan_lods(10000, 3, "aggressive", None, 300)
    assert [p["target"] for p in plan] == [4000, 1500, 600]
    plan, _ = C.plan_lods(10000, 3, "gentle", None, 300)
    assert [p["target"] for p in plan] == [7000, 5000, 3500]
    plan, skipped = C.plan_lods(1000, 3, "halves", None, 300)          # the floor stops the chain: 500, then 300, then nothing smaller
    assert [p["target"] for p in plan] == [500, 300] and skipped
    plan, _ = C.plan_lods(10000, 3, "custom", "6000,2500,800", 300)    # absolute counts
    assert [p["target"] for p in plan] == [6000, 2500, 800]
    plan, skipped = C.plan_lods(10000, 3, "custom", "0.95,0.5", 300)  # 95 % is not worth a level
    assert [p["target"] for p in plan] == [5000] and skipped and "15 %" in skipped[0]
    assert C.plan_lods(10000, 0)[0] == []
    assert C.parse_ratios("0.5, abc;0.25 -1") == [0.5, 0.25]
    assert C.lod_stop(1000, 900) and not C.lod_stop(1000, 800)


# ================================================================== 7. pack_textures
def _img(rgb, size=(2, 2), mode="RGB"):
    from PIL import Image

    return Image.new(mode, size, rgb)


def test_pack_textures_channel_layouts():
    from PIL import Image

    orm = Image.new("RGB", (2, 2), (200, 100, 50))        # AO 200, roughness 100, metallic 50
    nm = Image.new("RGB", (2, 2), (128, 200, 255))
    maps = {"basecolor": _img((10, 20, 30)), "normal": nm, "orm": orm}
    ur = {n: (im, m) for n, im, m in C.pack_textures("unreal", maps, "Rock", {})}
    assert set(ur) == {"T_Rock_BC.png", "T_Rock_N.png", "T_Rock_ORM.png"}
    assert np.asarray(ur["T_Rock_N.png"][0])[0, 0].tolist() == [128, 255 - 200, 255]      # only the green channel is flipped
    assert np.asarray(ur["T_Rock_ORM.png"][0])[0, 0].tolist() == [200, 100, 50]
    assert ur["T_Rock_N.png"][1]["colorspace"] == "linear" and ur["T_Rock_BC.png"][1]["colorspace"] == "srgb"
    hd = {n: im for n, im, _ in C.pack_textures("unity", maps, "Rock", {"unity_pipeline": "hdrp"})}
    assert set(hd) == {"Rock_BaseMap.png", "Rock_Normal.png", "Rock_MaskMap.png"}
    assert np.asarray(hd["Rock_MaskMap.png"])[0, 0].tolist() == [50, 200, 255, 255 - 100]   # R metal, G AO, B 255, A smoothness
    assert np.asarray(hd["Rock_Normal.png"])[0, 0].tolist() == [128, 200, 255]               # Unity keeps OpenGL
    urp = {n: im for n, im, _ in C.pack_textures("unity", maps, "Rock", {"unity_pipeline": "urp"})}
    assert set(urp) == {"Rock_BaseMap.png", "Rock_Normal.png", "Rock_MetallicSmoothness.png", "Rock_Occlusion.png"}
    assert np.asarray(urp["Rock_MetallicSmoothness.png"])[0, 0].tolist() == [50, 50, 50, 155]
    assert np.asarray(urp["Rock_Occlusion.png"])[0, 0] == 200
    both = {n for n, _, _ in C.pack_textures("unity", maps, "Rock", {})}
    assert {"Rock_MaskMap.png", "Rock_MetallicSmoothness.png"} <= both
    gd = {n for n, _, _ in C.pack_textures("godot", maps, "Rock", {})}
    assert gd == {"Rock_albedo.png", "Rock_normal.png", "Rock_orm.png"}
    gl = {n for n, _, _ in C.pack_textures("gltf", maps, "Rock", {})}
    assert gl == {"Rock_basecolor.png", "Rock_normal.png", "Rock_orm.png"}
    # separate greys compose the ORM; a missing ORM writes no packed map at all
    sep = {n for n, _, _ in C.pack_textures("gltf", {"basecolor": maps["basecolor"], "roughness": _img(90, mode="L"), "metallic": _img(10, mode="L")}, "Rock")}
    assert "Rock_orm.png" in sep
    none = {n for n, _, _ in C.pack_textures("unity", {"basecolor": maps["basecolor"]}, "Rock", {})}
    assert none == {"Rock_BaseMap.png"}
    assert C.engine_material("unreal") == {"layout": "unreal", "normal": "dx"} and C.engine_material("unity", "hdrp")["layout"] == "unity_hdrp"


def test_pack_textures_downscale_keeps_pot_and_unit_normals():
    from PIL import Image

    rng = np.random.default_rng(1)
    v = rng.normal(size=(64, 64, 3))
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    nm = Image.fromarray(((v * 0.5 + 0.5) * 255).astype(np.uint8))
    out = {n: im for n, im, _ in C.pack_textures("gltf", {"normal": nm, "basecolor": _img((1, 2, 3), (64, 64))}, "X", {"tex_max": 16})}
    assert out["X_normal.png"].size == (16, 16) and out["X_basecolor.png"].size == (16, 16)
    a = np.asarray(out["X_normal.png"]).astype(float) / 255 * 2 - 1
    assert np.abs(np.linalg.norm(a, axis=2) - 1).max() < 0.06      # renormalised after the resize (8-bit rounding)
    odd = C.downscale_pot(_img((1, 1, 1), (100, 60)), 64)
    assert odd.size[0] in (64, 32, 128) and (odd.size[0] & (odd.size[0] - 1)) == 0


# ================================================================== 8. names
def test_names_and_prefixes():
    assert C.safe_stem("Mein Schöner Fels!") == "Mein_Sch_ner_Fels" and C.safe_stem("") == "Asset" and len(C.safe_stem("a" * 100)) <= 48
    assert C.safe_stem("../../etc/passwd") == "etc_passwd"
    n = C.engine_names("unreal", "Rock", n_lods=3)
    assert n["asset"] == "SM_Rock" and n["lods"] == ["SM_Rock_LOD0", "SM_Rock_LOD1", "SM_Rock_LOD2", "SM_Rock_LOD3"]
    assert n["group"] == "SM_Rock" and n["group"] not in n["lods"]                       # the LodGroup differs from every mesh in it
    assert n["hull"](0) == "UCX_SM_Rock_LOD0_00" and n["hull"](3, "box") == "UBX_SM_Rock_LOD0_03"
    assert n["texture_prefix"] == "T_" and n["material"] == "M_Rock"
    assert C.engine_names("unreal", "SM_Rock")["asset"] == "SM_Rock"                      # no doubled prefix
    assert C.engine_names("unreal", "Hero", rigged=True)["asset"] == "SK_Hero"
    assert C.engine_names("unreal", "Rock", prefix_static="Env_", prefix_texture="Tex_")["asset"] == "Env_Rock"
    assert C.engine_names("unreal", "Rock", prefix_texture="Tex_")["texture_prefix"] == "Tex_"
    assert C.engine_names("unreal", "Rock")["group"] is None                                 # no LODs, no group
    g = C.engine_names("godot", "Rock", n_lods=0)
    assert g["hull"](1) == "Rock_hull_01-convcolonly" and g["asset"] == "Rock"
    assert C.engine_names("gltf", "Rock")["hull"](0) == "Rock_hull_00"
    assert C.engine_names("unity", "Rock", n_lods=2)["hull"](2) == "UCX_Rock_LOD0_02"


# ================================================================== 9. simple shapes and hulls
def test_simple_shapes_hulls_and_fps():
    rng = np.random.default_rng(3)
    P = rng.uniform(-1, 1, (500, 3)) * [2.0, 0.5, 1.0]
    b = C.simple_box(P)
    assert np.allclose(b["half"], (P.max(0) - P.min(0)) / 2) and np.allclose(b["center"], (P.max(0) + P.min(0)) / 2)
    th = 0.6
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    corners = np.array([[x, y, z] for x in (-2, 2) for y in (-0.5, 0.5) for z in (-1, 1)], float)
    obb = C.simple_obb(corners @ R.T)
    assert np.allclose(sorted(obb["half"]), [0.5, 1.0, 2.0], atol=1e-6)                      # exact extents of a rotated box
    rad = 0.4
    ang, hh = np.meshgrid(np.linspace(0, 2 * np.pi, 48, endpoint=False), np.linspace(-1.0, 1.0, 40))
    cyl = np.stack([rad * np.cos(ang.ravel()), hh.ravel(), rad * np.sin(ang.ravel())], 1)       # a regular cylinder, 2 m tall
    cap = C.simple_capsule(cyl)
    assert abs(cap["radius"] - rad) / rad < 0.02
    total = 2 * (cap["half_height"] + cap["radius"])
    assert abs(total - 2.0) / 2.0 < 0.02 and abs(abs(cap["axis"][1]) - 1) < 1e-6
    V, F = C.capsule_mesh(cap)
    assert len(F) > 20
    ball = rng.normal(size=(3000, 3))
    ball /= np.linalg.norm(ball, axis=1, keepdims=True)
    idx = C.hull_points_fps(ball, 64)
    assert len(idx) <= 64
    hv, hf = C.hull_of_points(ball[idx])
    from scipy.spatial import ConvexHull

    eq = ConvexHull(hv).equations
    assert ((hv @ eq[:, :3].T + eq[:, 3]) <= 1e-9).all()                                  # every point is inside every face plane
    flat = (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.0]], float), None)
    big = C.hull_of_points(ball * 2.0)
    kept, dropped = C.validate_hulls([flat, big], hull_max_verts=32, bbox_volume=64.0)
    assert dropped == 1 and len(kept) == 1 and len(kept[0][0]) <= 32
    assert C.hull_volume(*cube(2.0)) == pytest.approx(8.0)
    assert C.mesh_volume(*cube(2.0)) == pytest.approx(8.0)
    assert C.mesh_volume(cube(2.0)[0], cube(2.0)[1][:-1]) is None                           # an open surface has no volume


def test_placement_height_pivot_yaw():
    pl = C.placement([-1, 0, -1], [1, 4, 1], "height", 2.0, "bottom_center", 90)
    assert pl["scale"] == pytest.approx(0.5)
    V = np.array([[-1.0, 0, -1], [1.0, 4, 1]])
    out = C.apply_placement(V, pl)
    assert out[:, 1].min() == pytest.approx(0.0) and out[:, 1].max() == pytest.approx(2.0)
    assert np.allclose((out[:, [0, 2]].min(0) + out[:, [0, 2]].max(0)) / 2, 0, atol=1e-9)


# ================================================================== 10. GlbBuilder round trip
def _skinned_glb() -> bytes:
    V, F = cube()
    j = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}],
         "nodes": [{"name": "Mesh", "mesh": 0, "skin": 0}, {"name": "Root", "children": [2]}, {"name": "Bone"}]}
    b = C.GlbBuilder(j, b"")
    n = len(V)
    prim = b.primitive({"POSITION": V, "JOINTS_0": np.zeros((n, 4), np.uint8), "WEIGHTS_0": np.tile([1.0, 0, 0, 0], (n, 1))}, F)
    b.mesh("body", [prim])
    j["skins"] = [{"joints": [2]}]
    t = b.accessor(np.array([0.0, 1.0], "<f4").reshape(-1, 1)[:, 0:1].reshape(-1), None)
    j["accessors"][t]["type"] = "SCALAR"
    j["accessors"][t]["min"], j["accessors"][t]["max"] = [0.0], [1.0]
    v = b.accessor(np.array([[0, 0, 0], [0, 1, 0]], "<f4"))
    j["animations"] = [{"name": "wave", "samplers": [{"input": t, "output": v, "interpolation": "LINEAR"}], "channels": [{"sampler": 0, "target": {"node": 2, "path": "translation"}}]}]
    return b.finish(prune=False)


def test_glb_builder_lods_msft_and_godot_hulls_round_trip(tmp_path):
    data = _skinned_glb()
    j, blob = C.glb_split(data)
    skins0, anims0 = len(j["skins"]), len(j["animations"])
    prims = C.read_prims(j, blob, 0)
    assert len(prims) == 1 and prims[0]["F"].shape == (12, 3)
    b = C.GlbBuilder(j, blob)
    lod_nodes = []
    for k in (1, 2):
        p = prims[0]
        keep = p["F"][: 12 - 3 * k]
        a, f = C.compact(keep, p["attrs"])[1], C.compact(keep, p["attrs"])[0]
        mi = b.mesh(f"body_LOD{k}", [b.primitive(a, f)])
        lod_nodes.append(b.node(f"body_LOD{k}", mesh=mi, skin=0))
    C.add_msft_lod(j, 0, lod_nodes, [1.0, 0.5, 0.25])
    roots = list(j["scenes"][0]["nodes"])
    hv = [cube(0.5), cube(0.25)]
    hull_nodes = C.add_godot_hulls(b, hv, "Rock", scene_roots=roots)
    j["scenes"][0]["nodes"] = roots
    out = b.finish(prune=True)
    j2, blob2 = C.glb_split(out)
    assert len(j2["skins"]) == skins0 and len(j2["animations"]) == anims0
    assert "MSFT_lod" in j2["extensionsUsed"]
    ids = j2["nodes"][0]["extensions"]["MSFT_lod"]["ids"]
    in_scene = set(j2["scenes"][0]["nodes"])
    assert len(ids) == 2 and all(0 <= i < len(j2["nodes"]) and i not in in_scene for i in ids)
    assert [j2["nodes"][i]["name"] for i in hull_nodes][0].endswith("-convcolonly") and set(hull_nodes) <= in_scene
    for bv in j2["bufferViews"]:
        assert bv["byteOffset"] % 4 == 0 and bv["byteOffset"] + bv["byteLength"] <= len(blob2)
    for a in j2["accessors"]:
        assert 0 <= a["bufferView"] < len(j2["bufferViews"])
    for m in j2["meshes"]:
        for p in m["primitives"]:
            n = j2["accessors"][p["attributes"]["POSITION"]]["count"]
            idx = C.read_accessor(j2, blob2, p["indices"], as_float=False)
            assert int(idx.max()) < n
    for an in j2["animations"]:
        for s in an["samplers"]:
            assert s["input"] < len(j2["accessors"]) and s["output"] < len(j2["accessors"])
    assert len(blob2) % 4 == 0 and out[:4] == b"glTF"
    # the generic reader of the studio reads it back too
    import m3d_common as M3

    f = tmp_path / "out.glb"
    f.write_bytes(out)
    jj, _bb = M3._glb_split(f)
    assert jj["meshes"]
    fl = M3.glb_flags(f)
    assert fl["skins"] == skins0 and fl["animations"] == anims0


# ================================================================== 11. FBX round trip
def read_fbx(path):
    """Tiny binary FBX 7.4 reader (the inverse of fbx_write._node / _prop): -> (version, roots [(name, props, kids)])."""
    d = Path(path).read_bytes()
    assert d[:20] == b"Kaydara FBX Binary  " and d[20:23] == b"\x00\x1a\x00"
    ver = struct.unpack_from("<I", d, 23)[0]

    def prop(o):
        t = chr(d[o])
        o += 1
        if t in "YCIFDL":
            fmt, n = {"Y": ("<h", 2), "C": ("<?", 1), "I": ("<i", 4), "F": ("<f", 4), "D": ("<d", 8), "L": ("<q", 8)}[t]
            return struct.unpack_from(fmt, d, o)[0], o + n
        if t in "SR":
            n = struct.unpack_from("<I", d, o)[0]
            raw = d[o + 4:o + 4 + n]
            return (raw.decode("utf-8") if t == "S" else raw), o + 4 + n
        n, enc, clen = struct.unpack_from("<III", d, o)
        o += 12
        raw = d[o:o + clen]
        if enc:
            raw = zlib.decompress(raw)
        dt = {"f": "<f4", "d": "<f8", "i": "<i4", "l": "<i8", "b": "u1"}[t]
        return np.frombuffer(raw, dt, n), o + clen

    def node(o):
        end, nprops, plen, nlen = struct.unpack_from("<IIIB", d, o)
        if end == 0:
            return None, o + 13
        name = d[o + 13:o + 13 + nlen].decode()
        p = o + 13 + nlen
        props = []
        for _ in range(nprops):
            v, p = prop(p)
            props.append(v)
        kids = []
        while p < end:
            k, p = node(p)
            if k is None:
                break
            kids.append(k)
        return (name, props, kids), end

    o, roots = 27, []
    while o < len(d):
        n, o = node(o)
        if n is None:
            break
        roots.append(n)
    assert d[-16:] == FW.FOOT_MAGIC
    return ver, roots


def _find(nodes, name):
    return [n for n in nodes if n[0] == name]


def test_fbx_round_trip_lod_group_ucx_textures(tmp_path):
    V, UV, F = uv_sphere(8, 12, 0.5)
    N = V / np.linalg.norm(V, axis=1, keepdims=True)
    sc = FW.Scene(up="Y", unit_cm_per_unit=1.0)
    mat = sc.material("M_Rock", {"DiffuseColor": "Textures/T_Rock_BC.png", "NormalMap": "Textures/T_Rock_N.png"})
    grp = sc.group("SM_Rock", 0, lod_levels=2)
    for i in range(2):
        sc.mesh(f"SM_Rock_LOD{i}", V, F, N, UV, materials=[mat], parent=grp, scale=100.0)
    hv, hf = cube(0.4)
    sc.mesh("UCX_SM_Rock_LOD0_00", hv, hf, None, None, None, parent=0, scale=100.0)
    p = tmp_path / "SM_Rock.fbx"
    sc.save(str(p))
    ver, roots = read_fbx(p)
    assert ver == 7400
    gs = _find(roots, "GlobalSettings")[0]
    p70 = {pp[0]: pp[4:] for pp in (k[1] for k in _find(gs[2], "Properties70")[0][2])}
    assert p70["UnitScaleFactor"][0] == 1.0 and p70["UpAxis"][0] == 1
    obj = _find(roots, "Objects")[0][2]
    attrs = [o for o in _find(obj, "NodeAttribute") if o[1][2] == "LodGroup"]
    assert len(attrs) == 1
    models = {o[1][0]: o[1][1].split("\x00")[0] for o in _find(obj, "Model")}
    by_name = {v: k for k, v in models.items()}
    conns = [(c[1][1], c[1][2]) for c in _find(_find(roots, "Connections")[0][2], "C")]
    grp_id = by_name["SM_Rock"]
    kids = {c for c, par in conns if par == grp_id}
    assert {by_name["SM_Rock_LOD0"], by_name["SM_Rock_LOD1"]} <= kids
    assert (by_name["UCX_SM_Rock_LOD0_00"], 0) in conns and by_name["UCX_SM_Rock_LOD0_00"] not in kids   # collision hangs off the root
    geo = [o for o in _find(obj, "Geometry") if o[1][1].startswith("SM_Rock_LOD0")][0]
    pvi = _find(geo[2], "PolygonVertexIndex")[0][1][0]
    assert len(pvi) == len(F) * 3 and (pvi[2::3] < 0).all() and (pvi[0::3] >= 0).all() and (pvi[1::3] >= 0).all()
    uvs = _find(_find(geo[2], "LayerElementUV")[0][2], "UV")[0][1][0].reshape(-1, 2)
    assert np.allclose(uvs[:, 1], 1.0 - UV[:, 1], atol=1e-9)                                       # v is flipped for FBX
    verts = _find(geo[2], "Vertices")[0][1][0].reshape(-1, 3)
    assert np.allclose(verts, V * 100.0)
    for t in _find(obj, "Texture"):
        rel = _find(t[2], "RelativeFilename")[0][1][0]
        assert not rel.startswith("/") and "home" not in rel and rel.startswith("Textures/")
    blob = p.read_bytes()
    assert b"/home" not in blob and str(tmp_path).encode() not in blob


# ================================================================== 12. meshoptimizer
@pytest.mark.skipif(not MO.available(), reason="libmeshoptimizer.so is not built (./install.sh backend engine)")
def test_meshopt_simplify_keeps_the_sphere_watertight_across_the_uv_seam():
    V, UV, F = uv_sphere(32, 48)
    F2, err = MO.simplify(F, V, len(F) // 4, 0.05, 0)
    assert len(F2) <= len(F) // 4 * 1.05 and err <= 0.05 + 1e-6
    assert np.array_equal(np.unique(F2).size > 0, True)
    # weld the seam copies by position: a crack across the seam would leave boundary edges
    key = {tuple(np.round(v, 9)): i for i, v in enumerate(V)}
    rep = np.array([key[tuple(np.round(v, 9))] for v in V])
    W = rep[F2]
    W = W[(W[:, 0] != W[:, 1]) & (W[:, 1] != W[:, 2]) & (W[:, 0] != W[:, 2])]
    e = np.sort(np.concatenate([W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    top = (np.abs(V[:, 1]) > 0.999)                                                                 # the poles are degenerate rows of the grid
    assert (cnt == 2).mean() > 0.97, f"{(cnt != 2).sum()} edges are not shared by two faces"
    assert not top.all()
    # positions are never moved: every index still points at the original vertex
    assert F2.max() < len(V)
    remap, used = MO.optimize_vertex_fetch_remap(F2, len(V))
    assert used == len(np.unique(F2)) and (remap != 0xFFFFFFFF).sum() == used


# ================================================================== 13. validate_destinations
def test_validate_destinations_accepts_and_refuses(tmp_path):
    game = tmp_path / "proj" / "Assets"
    game.mkdir(parents=True)
    ok, prob = E.validate_destinations([{"id": "g", "label": "My game", "path": str(game), "engine": "unity", "max_mb": 100, "allow_lan": True}])
    assert not prob and ok[0]["real"] == os.path.realpath(game) and ok[0]["subdir"] == "OpenAssetStudio" and ok[0]["allow_lan"] is True
    home = os.path.expanduser("~")
    base = {"id": "x", "path": str(game)}

    def bad(**kw):
        d = {**base, **kw}
        out, pr = E.validate_destinations([d], home=str(tmp_path / "homedir"))
        assert out == [] and len(pr) == 1, (kw, out, pr)

    bad(id="UPPER")
    bad(id="a b")
    bad(id="x" * 33)
    bad(id="")
    bad(path="relative/dir")
    bad(path=str(tmp_path / "missing"))
    bad(path="/")
    bad(path="/etc")
    bad(path="/usr/local")
    bad(path="/var/tmp")
    bad(path=str(config.STUDIO))                                       # the repo itself
    bad(path=str(config.STUDIO / "data"))
    bad(path=str(config.STUDIO / "envs"))
    bad(subdir="../x")
    bad(subdir="a/b")
    bad(subdir="")
    bad(engine="cryengine")
    bad(max_mb=5000)
    bad(max_mb=0)
    bad(allow_lan="yes")
    (tmp_path / "homedir").mkdir()
    out, pr = E.validate_destinations([{"id": "h", "path": str(tmp_path / "homedir")}], home=str(tmp_path / "homedir"))
    assert out == [] and pr                                            # the home directory itself
    out, pr = E.validate_destinations([{"id": "top", "path": "/tmp"}])
    assert out == [] and pr                                            # fewer than 3 path components
    two, pr = E.validate_destinations([{"id": "a", "path": str(game)}, {"id": "a", "path": str(game)}])
    assert len(two) == 1 and "duplicate" in pr[0]
    assert E.validate_destinations(None) == ([], []) and E.validate_destinations({"id": "x"})[1]
    link = tmp_path / "link"
    link.symlink_to(config.STUDIO)
    out, pr = E.validate_destinations([{"id": "l", "path": str(link)}])
    assert out == [] and pr                                            # a symlink into the repo is judged by its realpath
    extra, pr = E.validate_destinations([{"id": "d", "path": str(game)}], extra_forbidden=[str(tmp_path / "proj")])
    assert extra == [] and pr


# ================================================================== 14. deliver_files
def members(**kw):
    base = {"SM_Rock.fbx": b"FBX" * 100, "manifest.json": b"{}", "Textures/T_Rock_BC.png": b"\x89PNG" + b"0" * 50}
    base.update(kw)
    return base


@pytest.fixture
def game(tmp_path):
    d = tmp_path / "work" / "game" / "Assets"
    d.mkdir(parents=True)
    return str(d)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def test_deliver_writes_inside_subdir_name_and_keeps_a_ledger(game):
    res = E.deliver_files(game, "OpenAssetStudio", "Rock", members(), asset_id="a1")
    root = Path(game) / "OpenAssetStudio" / "Rock"
    assert sorted(res["written"]) == sorted(members()) and res["relpath"] == "OpenAssetStudio/Rock" and res["folder_created"]
    for rel, data in members().items():
        assert (root / rel).read_bytes() == data
    led = json.loads((root / E.LEDGER).read_text())
    assert led["files"]["SM_Rock.fbx"] == sha(b"FBX" * 100) and led["asset"] == "a1"
    assert not [p for p in root.rglob("*") if p.name.endswith(".tmp")]                 # no temp files left
    assert stat.S_IMODE((root / "SM_Rock.fbx").stat().st_mode) == 0o644
    everything = {p.relative_to(game).parts[0] for p in Path(game).rglob("*")}
    assert everything == {"OpenAssetStudio"}                                           # nothing outside


def test_deliver_redelivery_rules(game):
    E.deliver_files(game, "OpenAssetStudio", "Rock", members())
    root = Path(game) / "OpenAssetStudio" / "Rock"
    again = E.deliver_files(game, "OpenAssetStudio", "Rock", members())                # identical content: nothing to do
    assert again["written"] == [] and sorted(again["unchanged"]) == sorted(members())
    new = members(**{"SM_Rock.fbx": b"NEW" * 10})
    with pytest.raises(E.DeliveryError) as e:                                          # ours and unedited, but no overwrite flag
        E.deliver_files(game, "OpenAssetStudio", "Rock", new)
    assert e.value.status == 409 and e.value.code == "exists" and (root / "SM_Rock.fbx").read_bytes() == b"FBX" * 100
    res = E.deliver_files(game, "OpenAssetStudio", "Rock", new, overwrite=True)        # ours and unedited, overwrite on
    assert res["written"] == ["SM_Rock.fbx"] and (root / "SM_Rock.fbx").read_bytes() == b"NEW" * 10
    (root / "SM_Rock.fbx").write_bytes(b"user edit")                                  # edited by the user: never overwritten
    for ow in (False, True):
        with pytest.raises(E.DeliveryError) as e:
            E.deliver_files(game, "OpenAssetStudio", "Rock", members(), overwrite=ow)
        assert e.value.status == 409 and e.value.code == "edited"
    assert (root / "SM_Rock.fbx").read_bytes() == b"user edit"
    assert (root / "manifest.json").read_bytes() == b"{}"


def test_deliver_foreign_files_and_folders_are_never_touched(game):
    root = Path(game) / "OpenAssetStudio" / "Rock"
    root.mkdir(parents=True)
    (root / "mine.txt").write_text("hand made")
    with pytest.raises(E.DeliveryError) as e:                                          # a folder without our ledger
        E.deliver_files(game, "OpenAssetStudio", "Rock", members(), overwrite=True)
    assert e.value.status == 409 and e.value.code == "not_ours"
    assert sorted(p.name for p in root.iterdir()) == ["mine.txt"]
    # a foreign file inside a folder that has a ledger
    E.deliver_files(game, "OpenAssetStudio", "Other", {"a.json": b"1"})
    other = Path(game) / "OpenAssetStudio" / "Other"
    (other / "b.json").write_text("not ours")
    with pytest.raises(E.DeliveryError) as e:
        E.deliver_files(game, "OpenAssetStudio", "Other", {"a.json": b"1", "b.json": b"2"}, overwrite=True)
    assert e.value.code == "exists_foreign" and (other / "b.json").read_text() == "not ours"
    assert not (other / "Textures").exists()
    # a conflict writes nothing at all (plan first)
    with pytest.raises(E.DeliveryError):
        E.deliver_files(game, "OpenAssetStudio", "Other", {"c.json": b"3", "b.json": b"2"})
    assert not (other / "c.json").exists()


def test_deliver_refuses_symlinks_everywhere(game, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    gp = Path(game)
    # 1. the sub folder is a symlink
    (gp / "OpenAssetStudio").symlink_to(outside)
    with pytest.raises(E.DeliveryError) as e:
        E.deliver_files(game, "OpenAssetStudio", "Rock", members())
    assert e.value.code == "unsafe_path" and list(outside.iterdir()) == []
    (gp / "OpenAssetStudio").unlink()
    # 2. the asset folder is a symlink
    (gp / "OpenAssetStudio").mkdir()
    (gp / "OpenAssetStudio" / "Rock").symlink_to(outside)
    with pytest.raises(E.DeliveryError) as e:
        E.deliver_files(game, "OpenAssetStudio", "Rock", members())
    assert e.value.code == "unsafe_path" and list(outside.iterdir()) == []
    (gp / "OpenAssetStudio" / "Rock").unlink()
    # 3. Textures is a symlink inside a folder that is ours
    E.deliver_files(game, "OpenAssetStudio", "Rock", {"a.json": b"1"})
    (gp / "OpenAssetStudio" / "Rock" / "Textures").symlink_to(outside)
    with pytest.raises(E.DeliveryError) as e:
        E.deliver_files(game, "OpenAssetStudio", "Rock", members())
    assert e.value.code == "unsafe_path" and list(outside.iterdir()) == []
    (gp / "OpenAssetStudio" / "Rock" / "Textures").unlink()
    # 4. a target file is a symlink to a file outside: it is neither followed nor replaced
    victim = outside / "victim.txt"
    victim.write_text("precious")
    (gp / "OpenAssetStudio" / "Rock" / "b.json").symlink_to(victim)
    with pytest.raises(E.DeliveryError) as e:
        E.deliver_files(game, "OpenAssetStudio", "Rock", {"b.json": b"x"}, overwrite=True)
    assert e.value.code == "unsafe_path" and victim.read_text() == "precious"
    # 5. the ledger is a symlink
    (gp / "OpenAssetStudio" / "Rock" / "b.json").unlink()
    led = gp / "OpenAssetStudio" / "Rock" / E.LEDGER
    led.unlink()
    led.symlink_to(victim)
    with pytest.raises(E.DeliveryError) as e:
        E.deliver_files(game, "OpenAssetStudio", "Rock", {"a.json": b"1"}, overwrite=True)
    assert e.value.code == "bad_ledger" and victim.read_text() == "precious"
    # 6. the destination root itself is a symlink
    lnk = tmp_path / "rootlink"
    lnk.symlink_to(game)
    with pytest.raises(E.DeliveryError) as e:
        E.deliver_files(str(lnk), "OpenAssetStudio", "Rock2", {"a.json": b"1"})
    assert e.value.code == "unsafe_path"


def test_deliver_rejects_bad_member_and_folder_names(game):
    for bad in ("../x.json", "/abs.json", "a\\b.json", "Textures/../x.json", "sub/dir/x.json", ".hidden.json", "x" * 200 + ".json", "Other/x.json"):
        with pytest.raises(E.DeliveryError) as e:
            E.deliver_files(game, "OpenAssetStudio", "Rock", {bad: b"x"})
        assert e.value.status == 400, bad
    for name in ("", "../Rock", "a/b", "x" * 80, "Rock Rock"):
        with pytest.raises(E.DeliveryError):
            E.deliver_files(game, "OpenAssetStudio", name, {"a.json": b"1"})
    for sub in ("..", "a/b", ""):
        with pytest.raises(E.DeliveryError):
            E.deliver_files(game, sub, "Rock", {"a.json": b"1"})
    assert not any(Path(game).iterdir())


def _zip(path, files, mode="w", compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, mode, compression=compression) as z:
        for n, d in files.items():
            z.writestr(n, d)
    return path


def test_read_package_whitelist_limits_and_zip_bomb(tmp_path):
    ok = _zip(tmp_path / "ok.zip", {"SM_Rock.fbx": b"1", "Textures/a.png": b"2", "Editor/OpenAssetStudioPostprocessor.cs": b"3", "README.txt": b"4"})
    assert sorted(E.read_package(ok, max_bytes=1 << 20)) == ["Editor/OpenAssetStudioPostprocessor.cs", "README.txt", "SM_Rock.fbx", "Textures/a.png"]
    for bad in ("../evil.json", "/abs.fbx", "a\\b.fbx", "Textures/sub/a.png", "evil.exe", "evil.sh", "noext", ".hidden.json", "Other/a.png", "Textures/"):
        p = _zip(tmp_path / "bad.zip", {bad: b"x"})
        with pytest.raises(E.DeliveryError) as e:
            E.read_package(p, max_bytes=1 << 20)
        assert e.value.status == 400, bad
    many = _zip(tmp_path / "many.zip", {f"f{i}.json": b"1" for i in range(E.MAX_FILES + 1)})
    with pytest.raises(E.DeliveryError) as e:
        E.read_package(many, max_bytes=1 << 20)
    assert e.value.code == "too_many_files"
    big = _zip(tmp_path / "big.zip", {"a.json": b"0" * 3_000_000}, compression=zipfile.ZIP_DEFLATED)       # tiny on disk, big when unpacked
    assert big.stat().st_size < 10_000
    with pytest.raises(E.DeliveryError) as e:
        E.read_package(big, max_bytes=2 << 20)
    assert e.value.status == 413 and e.value.code == "too_big"
    assert E.read_package(big, max_bytes=4 << 20)["a.json"] == b"0" * 3_000_000
    dup = tmp_path / "dup.zip"
    with zipfile.ZipFile(dup, "w") as z:
        z.writestr("a.json", b"1")
        z.writestr("A.json", b"2")                       # same file on a case-insensitive disk (Windows, macOS)
    with pytest.raises(E.DeliveryError):
        E.read_package(dup, max_bytes=1 << 20)
    link = tmp_path / "link.zip"
    with zipfile.ZipFile(link, "w") as z:
        zi = zipfile.ZipInfo("a.json")
        zi.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(zi, "/etc/passwd")
    with pytest.raises(E.DeliveryError):
        E.read_package(link, max_bytes=1 << 20)
    with pytest.raises(E.DeliveryError):
        E.read_package(tmp_path / "missing.zip", max_bytes=1 << 20)
    (tmp_path / "junk.zip").write_bytes(b"not a zip")
    with pytest.raises(E.DeliveryError):
        E.read_package(tmp_path / "junk.zip", max_bytes=1 << 20)
    # a header that lies about the size is caught by counting the bytes really read
    liar = _zip(tmp_path / "liar.zip", {"a.json": b"0" * 5000}, compression=zipfile.ZIP_DEFLATED)
    raw = bytearray(liar.read_bytes())
    for sig in (b"PK\x03\x04", b"PK\x01\x02"):          # shrink the declared uncompressed size in the local and the central header
        i = raw.find(sig)
        off = 22 if sig == b"PK\x03\x04" else 24
        struct.pack_into("<I", raw, i + off, 10)
    liar.write_bytes(bytes(raw))
    with pytest.raises(Exception):
        E.read_package(liar, max_bytes=1 << 20)


# ================================================================== 15. REST API
def _glb_stub() -> bytes:
    j = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}]}
    return C.glb_bytes(j, b"")


@pytest.fixture
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main
    from app.core import Core

    main.core = Core()
    core = main.core
    monkeypatch.setattr(core.jobs, "_execute", lambda jid: None)
    root = tmp_path / "work" / "game" / "Assets"
    root.mkdir(parents=True)
    other = tmp_path / "work" / "unity" / "Assets"
    other.mkdir(parents=True)
    core.settings.update({"engine": {"destinations": [
        {"id": "ue", "label": "Unreal game", "path": str(root), "engine": "unreal", "max_mb": 50},
        {"id": "lan", "label": "Open to LAN", "path": str(other), "engine": "any", "allow_lan": True, "max_mb": 50},
        {"id": "unity", "label": "Unity game", "path": str(other), "engine": "unity", "subdir": "Gen"}]}})
    pkg = tmp_path / "engine_unreal.zip"
    _zip(pkg, {"SM_Rock.fbx": b"FBXDATA", "manifest.json": b"{}", "Textures/T_Rock_BC.png": b"PNG"})
    glb = tmp_path / "mesh.glb"
    glb.write_bytes(_glb_stub())
    with TestClient(main.app, client=("127.0.0.1", 50000)) as local, TestClient(main.app) as remote:
        if local.get("/api/projects/engine-test/settings").status_code == 404:
            assert local.post("/api/projects", json={"id": "engine-test", "title": "t"}).status_code == 200
        meta = {"name": "Rock", "worker": {"engine": {"target": "unreal", "name": "Rock", "package": "engine_unreal.zip"}}}
        a = core.assets.ingest("engine-test", glb, "mesh", meta, attach=[pkg])
        (tmp_path / "plain.glb").write_bytes(_glb_stub())
        plain = core.assets.ingest("engine-test", tmp_path / "plain.glb", "mesh", {"name": "Plain"})
        yield {"local": local, "remote": remote, "core": core, "asset": a["id"], "plain": plain["id"], "root": root, "other": other}
    core.settings.update({"engine": {"destinations": None}})
    main.core = None


def post(c, aid, dest, **kw):
    return c.post("/api/x/engine/deliver", json={"asset_id": aid, "destination": dest, **kw})


def test_api_settings_refuse_destinations_and_hide_paths(api):
    r = api["local"].patch("/api/settings", json={"engine": {"destinations": [{"id": "evil", "path": "/home"}]}})
    assert r.status_code == 400 and "file-only" in r.json()["error"]
    r = api["local"].patch("/api/projects/engine-test/settings", json={"engine": {"destinations": []}})
    assert r.status_code == 400
    ok = api["local"].patch("/api/projects/engine-test/settings", json={"engine": {"target": "unity", "height_m": 1.7, "prefix_static": "Env_"}})
    assert ok.status_code == 200 and ok.json()["effective"]["engine"] == {"target": "unity", "height_m": 1.7, "prefix_static": "Env_", "prefix_texture": None}
    assert "destinations" not in ok.json()["effective"]["engine"]
    assert api["local"].patch("/api/projects/engine-test/settings", json={"engine": {"prefix_static": "../x"}}).status_code == 400
    assert api["local"].patch("/api/projects/engine-test/settings", json={"engine": {"target": "cryengine"}}).status_code == 400
    boot = api["local"].get("/api/bootstrap?project=engine-test").json()["settings"]
    assert boot["engine"]["target"] == "unity" and "destinations" not in boot["engine"]
    api["local"].delete("/api/projects/engine-test/settings")
    loc = api["local"].get("/api/settings").json()["engine"]["destinations"]
    rem = api["remote"].get("/api/settings").json()["engine"]["destinations"]
    assert loc[0]["path"] and all("path" not in d for d in rem) and rem[0]["id"] == "ue"
    sch = api["local"].get("/api/settings/schema").json()
    f = next(x for x in sch["fields"] if x["key"] == "engine.destinations")
    assert f["editable"] is False and f["info"] and any(g["id"] == "engine" for g in sch["groups"])


def test_api_destinations_path_only_for_local_callers(api):
    loc = api["local"].get("/api/x/engine/destinations").json()
    rem = api["remote"].get("/api/x/engine/destinations").json()
    assert loc["local"] is True and {d["id"] for d in loc["destinations"]} == {"ue", "lan", "unity"} and all(d["path"] for d in loc["destinations"])
    assert rem["local"] is False and all("path" not in d for d in rem["destinations"])
    assert {d["id"]: d["can_deliver"] for d in rem["destinations"]} == {"ue": False, "lan": True, "unity": False}
    proxied = api["local"].get("/api/x/engine/destinations", headers={"X-Forwarded-For": "10.0.0.9"}).json()
    assert proxied["local"] is False and all("path" not in d for d in proxied["destinations"])      # a local reverse proxy is not local


def test_api_deliver_refusals(api):
    L, R, a = api["local"], api["remote"], api["asset"]
    r = post(R, a, "ue")                                                  # not the studio machine, destination is local-only
    assert r.status_code == 403 and r.json()["code"] == "not_local"
    r = L.post("/api/x/engine/deliver", json={"asset_id": a, "destination": "ue"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and r.json()["code"] == "origin"
    r = L.post("/api/x/engine/deliver", json={"asset_id": a, "destination": "ue"}, headers={"Origin": "http://evil.example", "Host": "127.0.0.1:7860"})
    assert r.status_code == 403
    r = post(L, a, "nope")
    assert r.status_code == 404 and r.json()["code"] == "no_destination"
    r = L.post("/api/x/engine/deliver", json={"asset_id": a, "destination": "ue", "path": "/etc"})     # a path is not accepted
    assert r.status_code == 422
    r = L.post("/api/x/engine/deliver", json={"asset_id": a, "destination": str(api["root"])})        # a path in place of the id is just an unknown id
    assert r.status_code == 404
    assert post(L, api["plain"], "ue").json()["code"] == "not_a_package"
    assert post(L, "20990101-000000-000000", "ue").status_code in (400, 404)
    r = post(L, a, "unity")                                               # a Unity folder for an Unreal package
    assert r.status_code == 400 and r.json()["code"] == "engine_mismatch"
    assert list(api["root"].iterdir()) == [] and list(api["other"].iterdir()) == []
    assert E._BUSY.acquire(blocking=False)
    try:
        r = post(L, a, "ue")
        assert r.status_code == 409 and r.json()["code"] == "busy"
    finally:
        E._BUSY.release()
    assert list(api["root"].iterdir()) == []


def test_api_deliver_success_overwrite_edit_and_audit(api):
    L, a, root = api["local"], api["asset"], api["root"]
    r = post(L, a, "ue")
    assert r.status_code == 200, r.text
    body = r.json()
    assert sorted(body["written"]) == ["SM_Rock.fbx", "Textures/T_Rock_BC.png", "manifest.json"] and body["relpath"] == "OpenAssetStudio/Rock"
    assert body["dir"] == str(root.resolve() / "OpenAssetStudio" / "Rock")
    assert (root / "OpenAssetStudio" / "Rock" / "SM_Rock.fbx").read_bytes() == b"FBXDATA"
    same = post(L, a, "ue")                                               # identical content: nothing to write, not an error
    assert same.status_code == 200 and same.json()["written"] == [] and len(same.json()["unchanged"]) == 3
    d = api["core"].assets.asset_dir("engine-test", a)
    _zip(d / "engine_unreal.zip", {"SM_Rock.fbx": b"FBXDATA-V2", "manifest.json": b"{}", "Textures/T_Rock_BC.png": b"PNG"})
    again = post(L, a, "ue")                                              # a newer package, files are ours and unedited, but no overwrite flag
    assert again.status_code == 409 and again.json()["code"] == "exists"
    assert (root / "OpenAssetStudio" / "Rock" / "SM_Rock.fbx").read_bytes() == b"FBXDATA"
    ow = post(L, a, "ue", overwrite=True)
    assert ow.status_code == 200 and ow.json()["written"] == ["SM_Rock.fbx"] and len(ow.json()["unchanged"]) == 2
    assert (root / "OpenAssetStudio" / "Rock" / "SM_Rock.fbx").read_bytes() == b"FBXDATA-V2"
    (root / "OpenAssetStudio" / "Rock" / "manifest.json").write_text("edited")
    assert post(L, a, "ue", overwrite=True).json()["code"] == "edited"
    dl = api["core"].assets.get(a)["worker"]["engine"]["deliveries"]
    assert dl and dl[0]["destination"] == "ue" and str(root) not in json.dumps(dl) and "/tmp" not in json.dumps(dl)
    r = post(api["remote"], a, "lan")                                     # allow_lan opens it to other machines, with no path in the answer
    assert r.status_code == 200 and "dir" not in r.json() and (api["other"] / "OpenAssetStudio" / "Rock" / "SM_Rock.fbx").is_file()


def test_api_deliver_size_limit(api):
    api["core"].settings.update({"engine": {"destinations": [{"id": "tiny", "path": str(api["root"]), "engine": "any", "max_mb": 1}]}})
    d = api["core"].assets.asset_dir("engine-test", api["asset"])
    _zip(d / "engine_unreal.zip", {"a.json": os.urandom(1_500_000)}, compression=zipfile.ZIP_STORED)
    r = post(api["local"], api["asset"], "tiny")
    assert r.status_code == 413 and r.json()["code"] == "too_big" and list(api["root"].iterdir()) == []


# ================================================================== 16. MCP
@pytest.fixture()
def mcp(monkeypatch):
    spec = importlib.util.spec_from_file_location("mcp_server_engine", config.STUDIO / "scripts" / "mcp_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []

    def fake(method, path, body=None, **kw):
        calls.append((method, path, body))
        if path == "/api/x/engine/destinations":
            return {"destinations": [{"id": "ue", "label": "G", "engine": "unreal", "path": "/home/me/Game/Assets", "can_deliver": True}], "local": True,
                    "problems": ["x"]}
        if path == "/api/x/engine/deliver":
            return {"ok": True, "dir": "/home/me/Game/Assets/OpenAssetStudio/Rock", "written": ["a"]}
        raise mod.StudioError("not faked: " + path, status=404)

    monkeypatch.setattr(mod, "http", fake)
    mod._calls = calls
    return mod


def _call(mod, name, args, allow_paths):
    return mod.call_tool(name, args, None, None, allow_paths)


def test_mcp_deliver_is_gated_on_the_callers_locality(mcp):
    res = _call(mcp, "deliver_engine_export", {"asset_id": "a", "destination": "ue"}, allow_paths=False)
    assert res.get("isError") and "studio's machine" in json.dumps(res)
    assert mcp._calls == []                                                # no HTTP call was made on behalf of a remote caller
    ok = _call(mcp, "deliver_engine_export", {"asset_id": "a", "destination": "ue", "overwrite": True}, allow_paths=True)
    assert not ok.get("isError") and mcp._calls == [("POST", "/api/x/engine/deliver", {"asset_id": "a", "destination": "ue", "overwrite": True})]
    schema = next(t for t in mcp.TOOLS if t["name"] == "deliver_engine_export")["inputSchema"]
    assert "path" not in schema["properties"] and set(schema["required"]) == {"asset_id", "destination"}
    assert "ask the user first" in next(t for t in mcp.TOOLS if t["name"] == "deliver_engine_export")["description"].lower()


def test_mcp_destinations_hide_paths_from_remote_callers(mcp):
    remote = json.dumps(_call(mcp, "engine_destinations", {}, allow_paths=False))
    assert "/home/me" not in remote and '"ue"' in remote.replace("\\", "")
    local = json.dumps(_call(mcp, "engine_destinations", {}, allow_paths=True))
    assert "/home/me/Game/Assets" in local


def test_mcp_documents_engine_export_and_summarises_the_package(mcp):
    desc = {t["name"]: t["description"] for t in mcp.TOOLS}
    assert "engine_export" in desc["list_models"] and "engine_export" in desc["submit_job"] and "deliver_engine_export" in desc["submit_job"]
    for n in ("engine_unreal.zip", "engine_unity.zip", "engine_godot.zip", "engine_gltf.zip", "engine_report.json", "engine_collision.glb"):
        assert n in mcp.MESH_ATTACHMENTS
    a = {"id": "m1", "project": "p", "type": "mesh", "name": "Rock", "main": "mesh.glb",
         "files": [{"name": "mesh.glb", "role": "main"}, {"name": "engine_unreal.zip", "role": "attachment"}],
         "worker": {"engine": {"target": "unreal", "name": "Rock", "package": "engine_unreal.zip", "hulls": 5, "collision": "decomposition",
                               "lods": [{"name": "SM_Rock_LOD0", "tris": 1000, "ratio": 1.0}], "deliveries": [{"destination": "ue", "at": "t", "relpath": "/secret"}]}}}
    v = mcp.asset_view(a, True)["mesh3d"]
    assert v["engine"]["target"] == "unreal" and v["engine"]["lods"] == [{"name": "SM_Rock_LOD0", "tris": 1000}] and v["engine"]["hulls"] == 5
    assert "engine_unreal.zip" in v["attachments"] and "/secret" not in json.dumps(v)
    fields = mcp._settings_fields({"fields": [{"key": "engine.destinations"}, {"key": "engine.target"}]})
    assert [f["key"] for f in fields] == ["engine.target"]                  # the folder paths are never listed by get_settings
