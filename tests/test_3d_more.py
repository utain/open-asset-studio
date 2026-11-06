"""3D-more group (models.d/3d_more.yaml): registry entries are valid, the workers' helpers behave, the pure-torch spconv /
flash-attn stand-ins of the TRELLIS worker match dense references. No GPU, no weights; torch-dependent tests are skipped when
the running interpreter has no torch (envs/core has none; run them with envs/stab3d/bin/python -m pytest)."""
import importlib.util
import sys

import numpy as np
import pytest

from app import config
from app.registry import Registry

IDS = ["triposr", "sf3d", "spar3d", "trellis-image-large", "triposg", "hunyuan3d-2mini-turbo", "hunyuan3d-2mini",
       "zero123plus", "sv3d-u", "sv3d-p"]
GATED = {"sf3d", "spar3d", "sv3d-u", "sv3d-p"}
WORKERS = config.STUDIO / "workers"
sys.path.insert(0, str(WORKERS))


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path_factory.mktemp("p") / "probe.json")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, WORKERS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ registry
def test_group_loads_without_errors(reg):
    assert not [e for e in reg.errors if "3d_more" in e]
    for mid in IDS:
        assert mid in reg.models, mid
        assert reg.models[mid]["_file"] == "3d_more.yaml"


def test_tasks_backends_workers_exist(reg):
    assert reg.tasks["multiview"]["studio"] == "3d"
    for mid in IDS:
        m = reg.models[mid]
        assert set(m["tasks"]) <= set(reg.tasks), mid
        b = reg.backends[m["backend"]]
        assert (config.STUDIO / b["script"]).is_file(), b["script"]
        assert b["python"].startswith("envs/") and b["python"].split("/")[1] in ("stab3d", "shape3d", "trellis")


def test_licences_and_gating(reg):
    for mid in IDS:
        lic = reg.models[mid]["license"]
        assert lic["name"] and "commercial" in lic, mid
        if mid in GATED:
            assert lic.get("verify") is True and "Community" in lic["name"], mid
            assert "GATED" in reg.models[mid]["summary"], mid
    assert reg.models["triposr"]["license"]["name"] == "MIT"
    assert reg.models["triposg"]["license"]["name"] == "MIT"


def test_params_defaults_are_valid(reg):
    for mid in IDS:
        for p in reg.models[mid]["params"]:
            d = p.get("default")
            if p["type"] == "choice":
                assert d in p["choices"], (mid, p["name"])
            elif p["type"] in ("int", "float") and "min" in p:
                assert p["min"] <= d <= p["max"], (mid, p["name"])
        names = {p["name"] for p in reg.models[mid]["params"]}
        for preset, vals in reg.models[mid].get("presets", {}).items():
            assert set(vals) <= names, (mid, preset)


def test_input_slots_are_images(reg):
    for mid in IDS:
        slots = reg.models[mid]["input_slots"]
        assert len(slots) == 1 and slots[0]["role"] == "image" and slots[0]["types"] == ["image"] and slots[0]["required"]


def test_texture_feature_matches_description(reg):
    for mid in IDS:
        m = reg.models[mid]
        if m["tasks"] == ["shape"]:
            assert m["features"]["texture"] in ("vertex_colors", "pbr", "none")
    assert reg.models["triposg"]["features"]["texture"] == "none" and "Texture" in reg.models["triposg"]["summary"]
    assert reg.models["sf3d"]["features"]["pbr"] is True


def test_mesh_slot_type_is_known():
    from app.registry import Registry as R; SLOT_TYPES = R.SLOT_TYPES
    assert "image" in SLOT_TYPES and "mesh" in SLOT_TYPES


def test_vendor_lock_has_every_clone():
    lock = (config.STUDIO / "vendor" / "vendor.lock").read_text()
    for d in ("TripoSR", "stable-fast-3d", "stable-point-aware-3d", "generative-models", "TripoSG", "TRELLIS",
              "Hunyuan3D-2", "dinov2"):
        line = [l for l in lock.splitlines() if l.startswith(f"vendor/{d} ")]
        assert line, d
        assert len(line[0].split()[2]) == 40, d


# ------------------------------------------------------------------ m3d_common (numpy + PIL)
def test_seed_of():
    C = _load("m3d_common")
    assert C.seed_of({"seed": 5}) == 5
    s = C.seed_of({"seed": -1})
    assert 0 <= s < C.MAX_SEED
    assert 0 <= C.seed_of({}) < C.MAX_SEED


def test_crop_resize_foreground():
    from PIL import Image
    C = _load("m3d_common")
    im = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    im.paste((255, 0, 0, 255), (10, 20, 70, 60))          # 60 x 40 object
    out = C.crop_resize_foreground(im, 0.5, size=128)
    assert out.size == (128, 128)
    a = np.asarray(out.getchannel("A"))
    ys, xs = np.nonzero(a > 10)
    assert abs((xs.max() - xs.min() + 1) - 64) <= 3              # fills 50 % of the side
    assert abs((xs.min() + xs.max()) / 2 - 63.5) <= 2 and abs((ys.min() + ys.max()) / 2 - 63.5) <= 2
    with pytest.raises(ValueError):
        C.crop_resize_foreground(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))


def test_alpha_detection_and_composite():
    from PIL import Image
    C = _load("m3d_common")
    opaque = Image.new("RGBA", (16, 16), (10, 20, 30, 255))
    assert not C.has_useful_alpha(opaque)
    half = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    assert C.has_useful_alpha(half)
    rgb = C.composite_gray(half, 0.5)
    assert rgb.mode == "RGB" and tuple(np.asarray(rgb)[0, 0]) == (128, 128, 128)
    # an image that already has transparency is not sent through rembg
    assert C.remove_bg(half) is half


def test_prep_object_image_with_alpha(tmp_path):
    from PIL import Image
    C = _load("m3d_common")
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    im.paste((0, 255, 0, 255), (20, 20, 40, 50))
    p = tmp_path / "o.png"
    im.save(p)
    out = C.prep_object_image(str(p), do_remove_bg=True, fg_ratio=0.8)
    assert out.size[0] == out.size[1] and out.size[0] >= 30


def test_shim_missing_stub_raises():
    C = _load("m3d_common")
    C.shim_missing("definitely_not_a_module_3d")
    import definitely_not_a_module_3d as m
    with pytest.raises(RuntimeError):
        m.anything()
    sys.modules.pop("definitely_not_a_module_3d", None)


def test_export_glb_and_info(tmp_path):
    pytest.importorskip("trimesh")
    C = _load("m3d_common")
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    info = C.export_glb(tmp_path / "m.glb", v, f, vertex_colors=np.array([[1, 0, 0]] * 4, float))
    assert info == {"vertices": 4, "faces": 4}
    got = C.mesh_info(tmp_path / "m.glb")
    assert got["faces"] == 4


def test_simplify_noop_when_small():
    C = _load("m3d_common")
    v, f = np.zeros((3, 3)), np.array([[0, 1, 2]])
    assert C.simplify(v, f, None, 0)[1] is f
    assert C.simplify(v, f, None, 10)[1] is f


def test_multiview_sheet_layout(tmp_path):
    from PIL import Image
    sys.modules.setdefault("studio_worker", type(sys)("studio_worker"))
    sw = sys.modules["studio_worker"]
    for n in ("InputError", "Backend", "Cancelled", "serve"):
        if not hasattr(sw, n):
            setattr(sw, n, type(n, (Exception,), {}))
    mv = _load("stab3d_multiview")
    views = [Image.new("RGB", (32, 32), (i * 10, 0, 0)) for i in range(6)]
    sh = mv._sheet(views, tmp_path / "s.png")
    assert sh.size == (96, 64)                      # 3 columns x 2 rows
    sh21 = mv._sheet([views[0]] * 21, tmp_path / "s21.png")
    assert sh21.size == (7 * 32, 3 * 32)
    assert len(mv.ZERO123PLUS_VIEWS) == 6


# ------------------------------------------------------------------ torch stand-ins (skipped without torch)
@pytest.fixture(scope="module")
def shims():
    pytest.importorskip("torch")
    return _load("trellis_shims")


def test_subm_conv_matches_dense(shims):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    D = 8
    occ = torch.rand(1, D, D, D) < 0.25
    idx = torch.nonzero(occ).int()                               # [N, 4] (b, x, y, z)
    cin, cout = 3, 5
    feats = torch.randn(len(idx), cin)
    conv = shims.SubMConv3d(cin, cout, 3)
    x = shims.SparseConvTensor(feats, idx, [D, D, D], 1)
    out = conv(x).features
    dense = torch.zeros(1, cin, D, D, D)
    dense[0, :, idx[:, 1].long(), idx[:, 2].long(), idx[:, 3].long()] = feats.T
    # weight layout [out, kx, ky, kz, in] -> conv3d weight [out, in, kx, ky, kz]
    w = conv.weight.permute(0, 4, 1, 2, 3).contiguous()
    ref = F.conv3d(dense, w, conv.bias, padding=1)[0, :, idx[:, 1].long(), idx[:, 2].long(), idx[:, 3].long()].T
    assert torch.allclose(out, ref, atol=1e-5)


def test_varlen_attention_matches_per_sequence(shims):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(1)
    H, C = 2, 8
    lens = [5, 3, 7]
    cu = torch.tensor([0, 5, 8, 15], dtype=torch.int32)
    q, k, v = (torch.randn(15, H, C) for _ in range(3))
    out = shims.flash_attn_varlen_func(q, k, v, cu, cu, 7, 7)
    for i, n in enumerate(lens):
        s, e = int(cu[i]), int(cu[i + 1])
        ref = F.scaled_dot_product_attention(q[s:e].transpose(0, 1), k[s:e].transpose(0, 1), v[s:e].transpose(0, 1)).transpose(0, 1)
        assert torch.allclose(out[s:e], ref, atol=1e-5), i


def test_varlen_cross_attention_kv_lengths(shims):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(2)
    H, C = 2, 4
    cq = torch.tensor([0, 4, 6], dtype=torch.int32)
    ck = torch.tensor([0, 2, 7], dtype=torch.int32)
    q, k, v = torch.randn(6, H, C), torch.randn(7, H, C), torch.randn(7, H, C)
    out = shims.flash_attn_varlen_func(q, k, v, cq, ck, 4, 5)
    ref0 = F.scaled_dot_product_attention(q[:4].transpose(0, 1), k[:2].transpose(0, 1), v[:2].transpose(0, 1)).transpose(0, 1)
    ref1 = F.scaled_dot_product_attention(q[4:].transpose(0, 1), k[2:].transpose(0, 1), v[2:].transpose(0, 1)).transpose(0, 1)
    assert torch.allclose(out[:4], ref0, atol=1e-5) and torch.allclose(out[4:], ref1, atol=1e-5)


def test_install_registers_modules(shims):
    shims.install()
    import flash_attn
    import spconv.pytorch as sp
    assert sp.SubMConv3d is shims.SubMConv3d and hasattr(flash_attn, "flash_attn_varlen_func")
    for n in ("spconv", "spconv.pytorch", "flash_attn"):
        sys.modules.pop(n, None)


def test_torchmcubes_shim_sphere():
    torch = pytest.importorskip("torch")
    pytest.importorskip("skimage")
    C = _load("m3d_common")
    sys.modules.pop("torchmcubes", None)
    C.shim_torchmcubes()
    import torchmcubes
    g = torch.linspace(-1, 1, 24)
    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    vol = 0.6 - torch.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    v, f = torchmcubes.marching_cubes(vol, 0.0)
    assert len(v) > 100 and f.shape[1] == 3
    sys.modules.pop("torchmcubes", None)
