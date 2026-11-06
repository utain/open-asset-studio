"""Registry-level checks for models.d/vc_sing.yaml (no GPU, no weights needed)."""
import pytest

from app import config
from app.registry import Registry


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path_factory.mktemp("p") / "probe.json")


def test_loads_without_errors(reg):
    assert not [e for e in reg.errors if "vc_sing" in str(e)]


def test_tasks_declared(reg):
    assert reg.tasks["voice_convert"] == {"label": "Voice conversion", "modality": "audio", "studio": "sound"}


@pytest.mark.parametrize("mid", ["seed-vc", "rvc"])
def test_voice_convert_models(reg, mid):
    m = reg.models[mid]
    assert "voice_convert" in m["tasks"] and m["backend"] == "vc" and m["modality"] == "audio"
    assert m["license"]["name"] and "commercial" in m["license"]
    assert m["memory_gb"] > 0
    names = {p["name"] for p in m["params"]}
    assert "seed" in names
    # defaults are valid for their own declarations
    assert reg.validate_params(mid, {}) is not None
    assert reg.backends["vc"]["script"] == "workers/vc_worker.py"
