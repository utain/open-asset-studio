"""Seamless PBR materials (models.d/material.yaml, workers/material_common.py, app/plugins/material.py): registry entries, the seam metric,
tileability helpers, normal-map conventions, packing layouts, the PBRify weight re-parameterisation, wrap-safe resampling, sample meshes,
the `material` asset type and its zip export. No GPU, no weights."""
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import assets as assets_mod
from app import config
from app.assets import AssetStore
from app.registry import Registry

sys.path.insert(0, str(config.STUDIO / "workers"))
import material_common as mc  # noqa: E402

from app.plugins import material as plug  # noqa: E402


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path_factory.mktemp("p") / "probe.json")


def _noise(n=128, seed=0, sigma=2.0):
    rng = np.random.default_rng(seed)
    return mc.gauss_blur_wrap(rng.random((n, n), dtype=np.float32), sigma)


def _rgb(n=128, seed=0):
    return np.stack([_noise(n, seed + i) for i in range(3)], -1)


# ------------------------------------------------------------------ 1. registry
def test_registry_entries_load_cleanly(reg):
    assert not [e for e in reg.errors if "material" in e], reg.errors
    for t in ("material", "tileable"):
        assert reg.tasks[t]["studio"] == "material" and reg.tasks[t]["modality"] == "image"
    b = reg.backends["material"]
    assert b["python"].endswith("envs/qwen/bin/python") and b["script"] == "workers/material_worker.py" and b.get("max_loaded") == 4
    assert (config.STUDIO / b["script"]).is_file()
    m, t = reg.models["material-zimage"], reg.models["tileable-blend"]
    assert m["default_for"] == ["material"] and m["handler"] == "material.make" and m["license"]["commercial"] is True
    assert t["default_for"] == ["tileable"] and t["handler"] == "material.tileable" and t["weights"] == []
    assert reg.models["z-image-turbo"]["tasks"] == ["t2i", "tileable"]
    from app.jobs import HANDLERS
    for h in ("material.make", "material.tileable", "tool:material.sample_glb"):
        assert h in HANDLERS
    assert HANDLERS["tool:material.sample_glb"]["gpu"] is False


def test_params_validate_their_defaults_and_slots(reg):
    for mid in ("material-zimage", "tileable-blend", "z-image-turbo"):
        out = reg.validate_params(mid, {})
        for p in reg.models[mid]["params"]:
            if "default" in p:
                assert out[p["name"]] == p["default"], (mid, p["name"])
    m = reg.models["material-zimage"]
    assert [s["role"] for s in m["input_slots"]] == ["image", "base"]
    assert m["input_slots"][1]["types"] == ["material", "image"]
    # the weight entries: Z-Image files + the four PBRify files, all downloadable through one script
    paths = [w for w in m["weights"] if "path" in w]
    assert len(paths) == 4 and all(w["download_cmd"][-1] == "scripts/fetch_pbrify.py" for w in paths)
    assert (config.STUDIO / "scripts" / "fetch_pbrify.py").is_file()
    v = reg.validate_params("material-zimage", {"size": 4096, "metallic": 5})
    assert v["size"] == 4096 and v["metallic"] == 1


# ------------------------------------------------------------------ 2. seam metric
def test_seam_score_periodic_noise_is_clean():
    assert mc.seam_score(_noise())["score"] <= mc.SEAM_OK


def test_seam_score_detects_a_ramp_and_survives_constants():
    ramp = np.tile(np.linspace(0, 1, 128, dtype=np.float32), (128, 1))
    assert mc.seam_score(ramp)["x"] > mc.SEAM_WARN
    s = mc.seam_score(np.full((64, 64), 0.5, np.float32))
    assert np.isfinite(s["score"]) and s["score"] < 1.0
    assert mc.seam_class(1.0) == "ok" and mc.seam_class(1.1) == "warn" and mc.seam_class(1.5) == "bad"


# ------------------------------------------------------------------ 3. make tileable
def test_make_tileable_blend_removes_a_seam_and_keeps_contrast():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 128, dtype=np.float32)[None, :] * np.ones((128, 1), np.float32)
    nz = mc.gauss_blur_wrap(rng.random((128, 128), dtype=np.float32), 1.5)
    img = np.clip(0.5 + 0.25 * (x - 0.5) + 0.12 * (nz - nz.mean()) / nz.std(), 0, 1).astype(np.float32)
    assert mc.seam_score(img)["score"] > mc.SEAM_WARN
    out = mc.make_tileable_blend(img, 0.25, True)
    assert out.shape == img.shape and out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1
    assert mc.seam_score(out)["score"] <= 1.1
    assert abs(out.std() / img.std() - 1) < 0.15
    rgb = np.stack([img, img * 0.5, 1 - img], -1)
    assert mc.make_tileable_blend(rgb).shape == rgb.shape


# ------------------------------------------------------------------ 4. normal-map convention
def test_normal_convention_opengl_bump():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    h = np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 6.0 ** 2))).astype(np.float32)    # one bump, peak at row 32
    gl = mc.normal_from_height(h, 8.0)
    assert gl[40, 32, 1] < 0.5 < gl[24, 32, 1], "OpenGL: above the peak (smaller row) the surface faces up (+G); below it faces down"
    assert gl[32, 24, 0] < 0.5 < gl[32, 40, 0], "left of the peak the slope faces left (-R), right of it faces right (+R)"
    dx = mc.normal_from_height(h, 8.0, y_down=True)
    assert np.abs(dx - mc.flip_green(gl)).max() <= 1 / 255
    assert np.allclose(mc.flip_green(mc.flip_green(gl)), gl, atol=1e-6)
    assert abs(float(gl[..., 2].min()) - 0.5) > 0.0 and np.allclose(np.linalg.norm(gl * 2 - 1, axis=-1), 1, atol=1e-3)


def test_normal_scale_and_renormalize():
    gl = mc.normal_from_height(_noise(64, 5, 2.0), 6.0)
    s = mc.normal_scale(gl, 2.0)
    assert np.abs(s[..., :2] - 0.5).mean() > np.abs(gl[..., :2] - 0.5).mean()
    assert np.allclose(np.linalg.norm(s * 2 - 1, axis=-1), 1, atol=2e-3)
    assert np.allclose(mc.normal_scale(gl, 1.0), gl)
    assert np.allclose(np.linalg.norm(mc.renormalize_normal(gl * 0.9) * 2 - 1, axis=-1), 1, atol=2e-3)


# ------------------------------------------------------------------ 5. Frankot-Chellappa
def test_height_from_normal_round_trip():
    n = 128
    yy, xx = np.mgrid[0:n, 0:n] / n * 2 * np.pi
    h = (0.5 + 0.2 * np.sin(2 * xx) * np.cos(3 * yy) + 0.15 * np.sin(xx + 2 * yy)).astype(np.float32)
    nm = mc.normal_from_height(h, 6.0)
    r = mc.height_from_normal(nm)
    assert np.corrcoef(h.ravel(), r.ravel())[0, 1] > 0.99


# ------------------------------------------------------------------ 6. packing
def _maps(n=64):
    rng = np.random.default_rng(1)
    return {"basecolor": _rgb(n), "normal": mc.normal_from_height(_noise(n, 9, 2), 4.0), "height": _noise(n, 4),
            "roughness": _noise(n, 5) * 0.5 + 0.3, "metallic": np.full((n, n), 0.25, np.float32), "ao": _noise(n, 7) * 0.5 + 0.5,
            "orm": rng.random((n, n, 3), dtype=np.float32)}


def test_pack_orm_order():
    ao, r, m = (np.full((4, 4), v, np.float32) for v in (0.1, 0.5, 0.9))
    o = mc.pack_orm(ao, r, m)
    assert np.allclose(o[0, 0], [0.1, 0.5, 0.9])


@pytest.mark.parametrize("layout", list(mc.LAYOUTS))
def test_pack_layout_every_layout(layout):
    maps = _maps()
    out = mc.pack_layout(maps, layout, "Rock")
    names = [n for n, _, _ in out]
    assert len({n.lower() for n in names}) == len(names), names
    slots = {m["slot"] for _, _, m in out}
    assert "basecolor" in slots or any("BC" in n or "BaseMap" in n or "albedo" in n for n in names)
    assert any(n.endswith("_height.png") for n in names)
    assert all(img.size == (64, 64) for _, img, _ in out)
    assert next(i for n, i, _ in out if n.endswith("_height.png")).mode.startswith("I")        # 16-bit
    if layout == "separate":
        assert not any(n.lower().endswith(("_orm.png", "_maskmap.png")) or "metallicsmoothness" in n.lower() for n in names)
    else:
        assert any(m.get("slot") in ("orm", "packed", "maskmap", "metallicsmoothness") or "orm" in n.lower() or "mask" in n.lower() or "metallic" in n.lower()
                   for n, _, m in out)


def test_pack_layout_conventions():
    maps = _maps()
    gl = dict((n, np.asarray(i)) for n, i, _ in mc.pack_layout(maps, "gltf", "Rock"))
    dxo = dict((n, np.asarray(i)) for n, i, _ in mc.pack_layout(maps, "gltf", "Rock", normal="dx"))
    nk = next(k for k in gl if k.endswith("_normal.png"))
    assert np.abs(dxo[nk].astype(int)[..., 1] - (255 - gl[nk].astype(int)[..., 1])).max() <= 1
    un = {n: np.asarray(i) for n, i, _ in mc.pack_layout(maps, "unreal", "Rock")}
    nu = next(k for k in un if k.endswith("_N.png"))
    assert np.abs(un[nu].astype(int)[..., 1] - (255 - gl[nk].astype(int)[..., 1])).max() <= 1        # Unreal = DirectX
    urp = {n: np.asarray(i) for n, i, _ in mc.pack_layout(maps, "unity_urp", "Rock")}
    ms = next(k for k in urp if "MetallicSmoothness" in k)
    smooth = urp[ms][..., 3].astype(np.float32) / 255
    assert np.abs(smooth - (1 - maps["roughness"])).max() < 0.01                                      # smoothness = 1 - roughness
    orm = {n: np.asarray(i) for n, i, _ in mc.pack_layout(maps, "godot", "Rock")}
    o = next(v for k, v in orm.items() if k.endswith("_orm.png"))
    assert abs(o[..., 0].astype(float).mean() / 255 - maps["ao"].mean()) < 0.01 and abs(o[..., 2].mean() / 255 - 0.25) < 0.01


def test_pack_layout_size_never_upscales_and_uses_box_filter():
    maps = _maps(64)
    out = mc.pack_layout(maps, "separate", "R", size=32)
    assert all(img.size == (32, 32) for _, img, _ in out)
    assert mc.normalize_convert({"size": 4096}, 1024)["size"] == 0
    assert mc.normalize_convert({"size": 512}, 1024)["size"] == 512
    with pytest.raises(ValueError):
        mc.normalize_convert({"layout": "nope"})
    with pytest.raises(ValueError):
        mc.normalize_convert({"normal": "z"})
    assert mc.normalize_convert(None)["layout"] == mc.DEFAULT_LAYOUT and mc.normalize_convert({"glb": False})["glb"] is False


# ------------------------------------------------------------------ 7. PBRify weight re-parameterisation
def _conv1x1(x, w, b):
    return np.einsum("hwi,oi->hwo", x, w[:, :, 0, 0]) + b


def _conv3x3_valid(x, w, b):
    H, W = x.shape[0] - 2, x.shape[1] - 2
    out = np.zeros((H, W, w.shape[0]))
    for dy in range(3):
        for dx in range(3):
            out += np.einsum("hwi,oi->hwo", x[dy:dy + H, dx:dx + W], w[:, :, dy, dx])
    return out + b


def test_reparam_conv3xc_matches_the_training_branch():
    rng = np.random.default_rng(0)
    I, A, B, O = 3, 6, 6, 4
    w1, b1 = rng.standard_normal((A, I, 1, 1)), rng.standard_normal(A)
    w2, b2 = rng.standard_normal((B, A, 3, 3)) * 0.4, rng.standard_normal(B)
    w3, b3 = rng.standard_normal((O, B, 1, 1)), rng.standard_normal(O)
    sw, sb = rng.standard_normal((O, I, 1, 1)), rng.standard_normal(O)
    x = rng.standard_normal((10, 9, I))
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="wrap")
    ref = _conv1x1(_conv3x3_valid(_conv1x1(xp, w1, b1), w2, b2), w3, b3) + _conv1x1(x, sw, sb)
    W, Bi = mc.reparam_conv3xc(w1, b1, w2, b2, w3, b3, sw, sb)
    got = _conv3x3_valid(xp, W, Bi)
    assert W.shape == (O, I, 3, 3) and Bi.shape == (O,)
    assert np.abs(ref - got).max() < 1e-4, np.abs(ref - got).max()


# ------------------------------------------------------------------ 8. wrap-safe resampling
def test_box_and_bicubic_resampling_stay_seamless():
    a = _rgb(128, 11)
    assert mc.seam_score(a)["score"] <= mc.SEAM_OK
    assert mc.seam_score(mc.box_down(a, 2))["score"] <= mc.SEAM_OK
    assert mc.seam_score(mc.upscale_wrap(a, 2))["score"] <= mc.SEAM_OK
    im = Image.fromarray(mc.to_u8(a))
    assert mc.seam_score(np.asarray(mc.resize_wrap(im, (96, 96))))["score"] <= mc.SEAM_OK
    assert mc.box_down(a, 2).shape == (64, 64, 3) and mc.upscale_wrap(a[..., 0], 2).shape == (256, 256)


# ------------------------------------------------------------------ 9. meshes and glTF
@pytest.mark.parametrize("shape", ["sphere", "cube", "plane"])
def test_sample_mesh_is_well_formed(shape):
    m = mc.sample_mesh(shape, 1)
    P, N, T, uv, idx = m["positions"], m["normals"], m["tangents"], m["uvs"], m["indices"]
    assert P.shape[0] == N.shape[0] == T.shape[0] == uv.shape[0] and idx.dtype == np.uint32 and len(idx) % 3 == 0 and idx.max() < len(P)
    assert np.allclose(np.linalg.norm(N, axis=1), 1, atol=1e-4) and np.allclose(np.linalg.norm(T[:, :3], axis=1), 1, atol=1e-4)
    assert np.abs((T[:, :3] * N).sum(1)).max() < 1e-4 and set(np.unique(T[:, 3])) <= {-1.0, 1.0}
    assert np.allclose(mc.sample_mesh(shape, 4)["uvs"], uv * 4)
    tri = P[idx.reshape(-1, 3)]                           # counter-clockwise seen from outside: the face normal agrees with the vertex normals
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = N[idx.reshape(-1, 3)].mean(1)
    ok = (fn * vn).sum(1)
    assert (ok[np.linalg.norm(fn, axis=1) > 1e-9] > 0).all()
    with pytest.raises(ValueError):
        mc.sample_mesh("torus")


def test_gltf_and_glb_writers(tmp_path):
    mesh = mc.sample_mesh("sphere", 2)
    g = mc.write_gltf(tmp_path / "s.gltf", mesh, {"basecolor": "basecolor.png", "normal": "normal.png", "orm": "orm.png"})
    doc = json.loads(g.read_text())
    assert [i["uri"] for i in doc["images"]] == ["basecolor.png", "normal.png", "orm.png"]
    assert doc["buffers"][0]["uri"].startswith("data:application/octet-stream;base64,")
    acc = doc["accessors"]
    assert acc[0]["count"] == len(mesh["positions"]) and acc[4]["count"] == len(mesh["indices"])
    assert doc["samplers"][0]["wrapS"] == 10497
    png = plug.png_bytes(Image.new("RGB", (4, 4), "red"))
    b = mc.write_glb(tmp_path / "s.glb", mesh, {"basecolor": png, "normal": png, "orm": png})
    raw = b.read_bytes()
    magic, ver, total = struct.unpack("<4sII", raw[:12])
    assert magic == b"glTF" and ver == 2 and total == len(raw)
    jl, jt = struct.unpack("<I4s", raw[12:20])
    gd = json.loads(raw[20:20 + jl])
    assert jt == b"JSON" and all("bufferView" in i for i in gd["images"]) and gd["materials"][0]["normalTexture"]["index"] == 1


# ------------------------------------------------------------------ 10. height levels
def test_height_levels_do_not_stretch_a_flat_map_into_noise():
    rng = np.random.default_rng(2)
    flat = (0.5 + (rng.random((64, 64)) - 0.5) * 0.02).astype(np.float32)
    out = mc.height_levels(flat)
    assert abs(float(out.mean()) - 0.5) < 0.07 and float(out.max() - out.min()) < 0.2
    rough = _noise(64, 3, 2.0)
    full = mc.height_levels(rough)
    assert full.min() >= 0 and full.max() <= 1 and (full.max() - full.min()) > 0.9
    assert mc.height_levels(rough, 2.0).std() > mc.height_levels(rough, 0.5).std()


def test_build_maps_pure_numpy():
    n = 64
    alb = _rgb(n, 21)
    nrm_dx = mc.flip_green(mc.normal_from_height(_noise(n, 22, 2), 4.0))
    maps = plug.build_maps(mc, alb, nrm_dx, _noise(n, 23), _noise(n, 24), {"metallic": 0.4, "normal_strength": 1.5}, n)
    assert set(maps) == {"basecolor", "normal", "height", "roughness", "metallic", "ao", "orm"}
    assert np.allclose(maps["metallic"], 0.4) and maps["roughness"].min() >= 0.02 and maps["orm"].shape == (n, n, 3)
    assert np.allclose(maps["orm"][..., 2], 0.4)
    # the worker's DirectX map becomes OpenGL: a flipped green channel
    assert np.abs(mc.flip_green(mc.renormalize_normal(nrm_dx))[..., 0] - mc.renormalize_normal(nrm_dx)[..., 0]).max() < 1e-6
    big = plug.build_maps(mc, np.repeat(np.repeat(alb, 2, 0), 2, 1), nrm_dx, _noise(n, 23), _noise(n, 24), {}, n)
    assert big["normal"].shape == (2 * n, 2 * n, 3) and big["height"].shape == (2 * n, 2 * n)
    assert mc.seam_score(big["height"])["score"] <= mc.SEAM_OK


def test_with_suffix_is_idempotent():
    once = plug.with_suffix("mossy cobblestone")
    assert once.startswith("mossy cobblestone, seamless tileable texture")
    assert plug.with_suffix(once) == once


# ------------------------------------------------------------------ 11. the material asset type, thumbnail and export
def _make_material(tmp_path, store, project="p", size=64, name="Mossy rock"):
    maps = plug.build_maps(mc, _rgb(size, 31), mc.flip_green(mc.normal_from_height(_noise(size, 32, 2), 4.0)), _noise(size, 33), _noise(size, 34), {}, size)
    x = {"version": 1, "size": size, "map_res": size, "normal": "opengl", "tiling": {"method": "latent_roll+circular_vae"}, "seam": {"basecolor": 1.0}}
    d = tmp_path / "mat"
    files = plug.write_material_files(mc, maps, d, name, x)
    meta = {"name": name, "x_material": x, "license": {"name": "Apache-2.0", "commercial": True}}
    return store.ingest(project, d / files["basecolor"], "material", meta, attach=[d / v for k, v in files.items() if k != "basecolor"]), maps


@pytest.fixture()
def store(tmp_path):
    s = AssetStore(tmp_path / "projects")
    s.create_project("p")
    prev = dict(assets_mod.EXPORTERS)
    plug.setup(type("C", (), {})())
    yield s
    assets_mod.EXPORTERS.clear()
    assets_mod.EXPORTERS.update(prev)


def test_material_asset_thumb_comes_from_the_sphere_preview(tmp_path, store):
    a, _ = _make_material(tmp_path, store)
    assert a["type"] == "material" and a["main"] == "basecolor.png" and "material" in assets_mod.TYPES
    assert a["preview"]["thumb"] == "thumb.webp"
    d = store.asset_dir("p", a["id"])
    assert (d / "thumb.webp").is_file() and (d / "preview.png").is_file() and (d / "preview_sphere.gltf").is_file()
    names = {f["name"] for f in a["files"]}
    assert {"normal.png", "height.png", "roughness.png", "metallic.png", "ao.png", "orm.png", "material.json", "preview_cube.gltf"} <= names
    assert any(f["name"] == "preview_plane.gltf" and f["mime"] == "model/gltf+json" for f in a["files"])
    with Image.open(d / "height.png") as im:
        assert im.mode in ("I;16", "I")


def test_material_export_zip_default_layout(tmp_path, store):
    a, maps = _make_material(tmp_path, store)
    z = store.export([a["id"]], tmp_path / "out", convert={"material": {"layout": "gltf", "normal": "dx"}})
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        stem = "mossy_rock"
        assert any(n.startswith(f"materials/{stem}/") for n in names), names
        assert f"materials/{stem}/{stem}_basecolor.png" in names or any(n.endswith("_basecolor.png") for n in names)
        assert any(n.endswith("_sample.glb") for n in names)
        assert not any("preview" in n for n in names) and any(n.endswith(".material.json") for n in names)
        nm = next(n for n in names if n.endswith("_normal.png"))
        got = np.asarray(Image.open(zf.open(nm)).convert("RGB")).astype(int)
        want = mc.to_u8(mc.flip_green(maps["normal"])).astype(int)
        assert np.abs(got - want).max() <= 2, "dx export flips the green channel"
        glb = zf.read(next(n for n in names if n.endswith("_sample.glb")))
        assert glb[:4] == b"glTF"
        man = json.loads(zf.read("manifest.json"))
        assert man["assets"][0]["material"]["layout"] == "gltf" and man["assets"][0]["material"]["normal"] == "dx"


def test_material_export_options(tmp_path, store):
    a, _ = _make_material(tmp_path, store, size=512)
    z = store.export([a["id"]], tmp_path / "out", convert={"material": {"layout": "unreal", "size": 256, "glb": False}})
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert not any(n.endswith(".glb") for n in names)
        assert any("T_" in n and n.endswith("_ORM.png") for n in names)
        im = Image.open(zf.open(next(n for n in names if n.endswith("_BC.png"))))
        assert im.size == (256, 256)
    z2 = store.export([a["id"]], tmp_path / "out")                                  # no convert at all: defaults, nothing crashes
    assert "manifest.json" in zipfile.ZipFile(z2).namelist()
    with pytest.raises(assets_mod.AssetError):
        store.export([a["id"]], tmp_path / "out", convert={"material": {"layout": "bogus"}})


def test_other_asset_types_export_unchanged(tmp_path, store):
    src = tmp_path / "x.png"
    Image.new("RGB", (8, 8), "red").save(src)
    a = store.ingest("p", src, "image", {"name": "red"}, move=False)
    z = store.export([a["id"]], tmp_path / "out", convert={"material": {"layout": "unreal"}})
    names = zipfile.ZipFile(z).namelist()
    assert "images/red.png" in names and not any(n.startswith("materials/") for n in names)


def test_sample_glb_tool_writes_a_mesh_with_embedded_textures(tmp_path, store):
    a, _ = _make_material(tmp_path, store)
    d = store.asset_dir("p", a["id"])
    maps = plug.load_maps(mc, d, a, size=32)
    assert maps["basecolor"].shape == (32, 32, 3) and maps["height"].ndim == 2
    out = plug.write_sample_glb(mc, maps, tmp_path / "s.glb", "cube", 3, "x")
    gd = json.loads(out.read_bytes()[20:20 + struct.unpack("<I", out.read_bytes()[12:16])[0]])
    assert len(gd["images"]) == 3 and gd["meshes"][0]["name"] == "cube"


# ------------------------------------------------------------------ 12. MCP (scripts/mcp_server.py, the in-app assistant)
@pytest.fixture()
def mcp(monkeypatch, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("mcp_server_mat", config.STUDIO / "scripts" / "mcp_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "DATA", tmp_path)
    return mod


def _call(mod, name, args=None):
    res = mod.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args or {}}})["result"]
    return res, [json.loads(c["text"]) if c["type"] == "text" else c for c in res["content"]]


MAT = {"id": "mt1", "project": "default", "type": "material", "name": "Mossy rock", "main": "basecolor.png", "task": "material", "model": "material-zimage",
       "files": [{"name": n, "role": "main" if n == "basecolor.png" else "attachment", "mime": "image/png", "bytes": 1}
                 for n in ("basecolor.png", "normal.png", "height.png", "roughness.png", "metallic.png", "ao.png", "orm.png", "preview.png")],
       "x_material": {"size": 1024, "map_res": 1024, "normal": "opengl", "seam": {"basecolor": 0.93, "normal": 0.97, "height": 1.01},
                      "tiling": {"method": "latent_roll+circular_vae"}, "source": "text"}}


def test_mcp_material_descriptions_and_tool_table(mcp):
    desc = {t["name"]: t for t in mcp.TOOLS}
    assert "material" in desc["search_assets"]["inputSchema"]["properties"]["type"]["description"]
    assert "material" in desc["submit_job"]["description"] and "'base'" in desc["submit_job"]["description"] and "tileable" in desc["submit_job"]["description"]
    assert "material" in desc["export_assets"]["inputSchema"]["properties"] and "material" in desc["get_asset"]["description"]
    assert "material" in mcp.TASKS_HELP and "tileable" in mcp.TASKS_HELP
    t = mcp.TOOL_PARAMS["material.sample_glb"]
    assert set(t["params"]) == {"shape", "tiles", "size"} and "material" in t["in"]
    assert "materials" in mcp.INFO_TOPICS
    assert "material" in mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})["result"]["instructions"].lower()


def test_mcp_get_asset_material_block_and_previews(mcp, monkeypatch):
    monkeypatch.setattr(mcp, "http", lambda method, path, *a, **k: MAT if path == "/api/assets/mt1" else (_ for _ in ()).throw(mcp.StudioError("nf", status=404)))
    got = []
    monkeypatch.setattr(mcp, "_image_block", lambda asset, name: got.append(name) or [{"type": "image", "data": "x", "mimeType": "image/png"}])
    res, blocks = _call(mcp, "get_asset", {"id": "mt1", "preview": True})
    m = blocks[0]["material"]
    assert not res["isError"] and m["size"] == 1024 and m["normal"] == "opengl" and m["layout_hint"] == "gltf"
    assert set(m["maps"]) == {"basecolor", "normal", "height", "roughness", "metallic", "ao", "orm"} and m["seam"]["basecolor"] == 0.93
    assert "export_assets" in m["notes"]
    assert got == ["preview.png", "basecolor.png"] and [c["type"] for c in res["content"]] == ["text", "image", "image"]    # sphere first, then the base colour
    got.clear()
    mcp.preview_blocks(MAT, both=False)
    assert got == ["preview.png"]


def test_mcp_export_assets_sends_material_options_as_a_zip(mcp, monkeypatch):
    seen = []

    def fake_http(method, path, body=None, *, query=None, raw=None, ctype="", timeout=60):
        seen.append((method, path, body))
        if path == "/api/assets/mt1":
            return MAT
        if path == "/api/export":
            return {"url": "/api/export/export-20260101-000000-abcdef.zip", "name": "export-20260101-000000-abcdef.zip", "bytes": 10, "count": 1}
        raise mcp.StudioError("nf", status=404)

    monkeypatch.setattr(mcp, "http", fake_http)
    res, d = _call(mcp, "export_assets", {"ids": ["mt1"], "material": {"layout": "unreal", "size": 1024}})
    assert not res["isError"]
    post = next(s for s in seen if s[1] == "/api/export")
    assert post[2]["export"] == {"material": {"layout": "unreal", "size": 1024}} and post[2]["ids"] == ["mt1"]


def test_assistant_get_asset_passes_the_material_block(monkeypatch):
    from app.plugins import assistant

    class Store:
        def get(self, _id):
            return {**MAT, "project": "p"}

    monkeypatch.setattr(assistant, "_core", lambda: type("C", (), {"assets": Store()})())
    out = assistant.tool_get_asset("p", {"id": "mt1"})
    assert out["material"]["size"] == 1024 and out["material"]["seam"]["height"] == 1.01
    from pathlib import Path
    src = (Path(config.STUDIO) / "workers" / "studio_mcp.py").read_text()
    assert "material" in src and "tileable" in src


def test_qwen_worker_from_pipe_keeps_dtype():
    """diffusers' from_pipe defaults to float32 and casts the SHARED modules in place (regenerate took 25 s instead of 8 s, +20 GB, and every
    later t2i job in that worker was slow): the worker must pass the dtype the modules already have."""
    src = (Path(__file__).resolve().parent.parent / "workers" / "qwen_worker.py").read_text()
    assert "from_pipe(pipe, torch_dtype=torch.bfloat16)" in src
    assert src.count("from_pipe(") == 1
