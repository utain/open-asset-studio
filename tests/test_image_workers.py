"""Worker-side logic that decides what reaches the models: prompt composition (RGBA trigger, extract, 3D-ready),
edit output size, Hunyuan3D parameter mapping. The workers are imported with a stub framework (no fd juggling)."""
import importlib.util
import sys
import types

import pytest

from app import config

WORKERS = config.STUDIO / "workers"
HY3D_APP = config.STUDIO / "engines" / "hy3d" / "app"
pytestmark = pytest.mark.skipif(not (HY3D_APP / "image_params.py").exists(), reason="engines/hy3d/app missing")


def _load(name):
    stub = types.ModuleType("studio_worker")

    class InputError(ValueError):
        pass

    stub.Backend, stub.InputError, stub.Cancelled = object, InputError, type("Cancelled", (Exception,), {})
    stub.log = stub.serve = lambda *a, **k: None
    stub.size_from_aspect = lambda aspect, side, multiple=16: (int(side), int(side))
    sys.modules["studio_worker"] = stub
    if str(HY3D_APP) not in sys.path:
        sys.path.append(str(HY3D_APP))
    spec = importlib.util.spec_from_file_location(f"t_{name}", WORKERS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def qw():
    return _load("qwen_worker")


TRIGGER = "This is an RGBA image with transparency."


def test_rgba_trigger_once(qw):
    text, transparent = qw.compose_prompt("rgba", "qwen21", {"prompt": "A potion bottle"})
    assert transparent and text.startswith(TRIGGER) and text.count(TRIGGER) == 1
    # the improver's qwen_rgba style already wrote the trigger: it must not be doubled
    text, _ = qw.compose_prompt("rgba", "qwen21", {"prompt": f"{TRIGGER} A potion bottle"})
    assert text.count(TRIGGER) == 1


def test_no_rgba_trigger_for_models_without_alpha(qw):
    text, transparent = qw.compose_prompt("t2i", "zimage", {"prompt": "A chest", "transparent": True})
    assert not transparent and TRIGGER not in text


def test_extract_and_edit(qw):
    text, transparent = qw.compose_prompt("extract", "qwen21", {"prompt": "the sword"})
    assert transparent and "the sword" in text and text.startswith(TRIGGER)
    text, _ = qw.compose_prompt("extract", "qwen21", {"prompt": ""})
    assert "main subject" in text
    with pytest.raises(ValueError):
        qw.compose_prompt("edit", "qwen21", {"prompt": " "})


def test_3d_ready_framing(qw):
    text, _ = qw.compose_prompt("t2i", "qwen21", {"prompt": "A chest", "threed_ready": True})
    assert "three-quarter" in text and "white background" in text
    text, _ = qw.compose_prompt("t2i", "zimage", {"prompt": "A chest", "threed_ready": True})
    assert "no cast shadow" not in text  # Z-Image turns negative phrasing into the thing itself


def test_edit_size_keeps_aspect_on_grid(qw):
    w, h = qw.edit_size(1600, 900, 1024, 32)
    assert w % 32 == 0 and h % 32 == 0 and abs(w / h - 16 / 9) < 0.05 and abs(w * h - 1024 ** 2) / 1024 ** 2 < 0.06


def test_hy3d_param_mapping():
    hy = _load("hy3d_worker")
    p = hy.engine_params({"seed": -1, "tex_seed": 7, "octree": "384", "steps": 500, "tex_size": 2048, "unknown": 1})
    assert p["randomize_seed"] is True and p["tex_randomize_seed"] is False and p["tex_seed"] == 7
    assert p["octree"] == 384 and p["steps"] == 100 and p["tex_size"] == 2048 and "unknown" not in p
    p = hy.engine_params({"seed": 99})
    assert p["randomize_seed"] is False and p["seed"] == 99
