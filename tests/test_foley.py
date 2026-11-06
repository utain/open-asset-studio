"""Registry-level checks of models.d/foley.yaml (no GPU, no weights needed)."""
from pathlib import Path

import pytest

from app.config import Settings
from app.registry import Registry

STUDIO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("foley")
    return Registry(STUDIO / "models.yaml", Settings(tmp / "settings.json"), hf_cache=tmp / "hf", lmstudio=tmp / "lm",
                    probe_cache=tmp / "probe.json")


def foley_models(reg):
    return {k: m for k, m in reg.models.items() if m.get("backend") == "foley"}


def test_loads_without_errors(reg):
    assert not [e for e in reg.errors if "foley" in str(e)]
    assert foley_models(reg)


def test_v2a_task_declared(reg):
    t = reg.tasks["v2a"]
    assert t["modality"] == "audio" and t["studio"] == "sound" and "Foley" in t["label"]


def test_every_model_has_registry_basics(reg):
    for mid, m in foley_models(reg).items():
        assert "v2a" in m["tasks"], mid
        assert m["license"].get("commercial") in (True, False), mid
        assert m["memory_gb"] > 0 and m["est"]["run_s"] > 0, mid
        assert (STUDIO / "workers" / "foley_worker.py").exists()
        names = {p["name"] for p in m["params"]}
        assert {"seconds", "batch", "seed"} <= names, mid


def test_params_validate_and_clip(reg):
    out = reg.validate_params("mmaudio-large-44k-v2", {"seconds": "99", "batch": "0", "cfg_scale": "50"})
    assert out["seconds"] == 30 and out["batch"] == 1 and out["cfg_scale"] == 10
    assert out["seed"] == -1 and out["mux_video"] is True


def test_text_only_sfx_only_where_supported(reg):
    sfx = {k for k, m in foley_models(reg).items() if "sfx" in m["tasks"]}
    assert sfx == {k for k, m in foley_models(reg).items() if (m.get("features") or {}).get("text_only")}
