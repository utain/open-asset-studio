"""Image + 3D registry entries: they load cleanly and every Hunyuan3D engine knob reaches the engine (no dead knobs)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from app import config
from app.registry import Registry

HY3D_APP = config.STUDIO / "engines" / "hy3d" / "app"


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path_factory.mktemp("p") / "probe.json")


@pytest.fixture(scope="module")
def specs():
    if not (HY3D_APP / "params.py").exists():
        pytest.skip("engines/hy3d/app missing")
    spec = importlib.util.spec_from_file_location("hy3d_params", HY3D_APP / "params.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hy3d_params"] = mod   # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod.SPECS


def test_drop_in_loads_without_errors(reg):
    assert not [e for e in reg.errors if "image.yaml" in e or "hunyuan" in e or "qwen" in e]
    for mid in ("qwen-image-2.1", "z-image-turbo", "hunyuan3d-2.1", "recenter-3d", "mesh-export", "rembg-u2net"):
        assert mid in reg.models
    assert reg.models["hunyuan3d-2.1"]["_file"] == "image.yaml"   # the override replaced the base entry
    assert {"recenter", "mesh_export"} <= set(reg.tasks)


def test_every_engine_knob_is_exposed(reg, specs):
    names = {p["name"] for m in ("hunyuan3d-2.1", "mesh-export") for p in reg.models[m]["params"]}
    internal = {"preset", "randomize_seed", "tex_randomize_seed"}   # derived: seed -1 = random
    missing = set(specs) - internal - names
    assert not missing, f"engine knobs without a form field: {sorted(missing)}"


def test_no_dead_knobs(reg, specs):
    studio_only = {"textured", "alpha_clean"}    # handled by the worker itself (job kind / input clean-up)
    for p in reg.models["hunyuan3d-2.1"]["params"] + reg.models["mesh-export"]["params"]:
        assert p["name"] in specs or p["name"] in studio_only, p["name"]


def test_choices_match_engine(reg, specs):
    for p in reg.models["hunyuan3d-2.1"]["params"]:
        s = specs.get(p["name"])
        if s is None or p["type"] != "choice" or s.choices is None:
            continue
        assert {str(c) for c in p["choices"]} <= {str(c) for c in s.choices}, p["name"]


def test_qwen_params_validate(reg):
    v = reg.validate_params("qwen-image-2.1", {"alpha_clean": 999, "steps": "12", "transparent": "true", "prompt": "x"})
    assert v["alpha_clean"] == 64 and v["steps"] == 12 and v["transparent"] is True and v["prompt"] == "x"
    v = reg.validate_params("recenter-3d", {"out_size": "0"})
    assert v["out_size"] == 0 and v["margin"] == 0.15
    with pytest.raises(ValueError):
        reg.validate_params("mesh-export", {"export_type": "fbx"})
