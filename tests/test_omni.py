"""Registry-level checks of models.d/omni.yaml (no GPU, no weights needed)."""
from pathlib import Path

import pytest

from app.config import Settings
from app.registry import Registry

STUDIO = Path(__file__).resolve().parent.parent
NC = {"qwen25-omni-3b", "llama-omni2-0_5b"}


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("omni")
    return Registry(STUDIO / "models.yaml", Settings(tmp / "settings.json"), hf_cache=tmp / "hf", lmstudio=tmp / "lm",
                    probe_cache=tmp / "probe.json")


def omni(reg):
    return {k: m for k, m in reg.models.items() if m.get("_file") == "omni.yaml"}


def test_loads_without_errors(reg):
    assert not [e for e in reg.errors if "omni" in str(e)]
    assert omni(reg)


def test_s2s_task_declared(reg):
    assert reg.tasks["s2s"]["studio"] == "sound"
    assert reg.tasks["s2s"]["label"].startswith("Voice reply")


def test_entries_complete(reg):
    for mid, m in omni(reg).items():
        assert m["backend"] in reg.backends and reg.backends[m["backend"]].get("script"), mid
        assert (STUDIO / reg.backends[m["backend"]]["script"]).exists(), mid
        assert m["tasks"] and set(m["tasks"]) <= set(reg.tasks), mid
        assert m["license"].get("name") and "commercial" in m["license"], mid
        assert m["memory_gb"] and m.get("summary"), mid
        names = {p["name"] for p in m["params"]}
        assert "seed" in names, mid
        if "s2s" in m["tasks"]:
            assert "seconds" in names, mid
            if m.get("features", {}).get("instruct"):
                assert "instruction" in names, mid


def test_nc_licences(reg):
    for mid in NC & set(omni(reg)):
        assert reg.models[mid]["license"]["commercial"] is False
