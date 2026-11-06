"""Voice-cloning group A (models.d/tts_clone_a.yaml): the entries load cleanly, use existing tasks, every backend has
its worker script, and every declared param validates its own default (no GPU, no weights needed)."""
import pytest

from app import config
from app.registry import Registry

IDS = ["xtts-v2", "chatterbox", "chatterbox-multilingual", "zonos-v0.1-transformer", "openvoice-v2", "spark-tts", "llasa-1b"]


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path_factory.mktemp("p") / "probe.json")


def test_group_loads_without_errors(reg):
    assert not [e for e in reg.errors if "tts_clone_a" in e]
    for mid in IDS:
        assert mid in reg.models, mid
        assert reg.models[mid]["_file"] == "tts_clone_a.yaml"


def test_tasks_exist_and_clone_models_take_a_reference(reg):
    for mid in IDS:
        m = reg.models[mid]
        assert set(m["tasks"]) <= set(reg.tasks), mid
        assert "voice_clone" in m["tasks"] and "tts" in m["tasks"], mid
        assert m["features"].get("needs_ref") == "clone", mid
    assert "voice_design" in reg.models["spark-tts"]["tasks"]


def test_backends_have_worker_scripts(reg):
    for mid in IDS:
        b = reg.backends[reg.models[mid]["backend"]]
        assert b["script_path"].exists() if "script_path" in b else (config.STUDIO / b["script"]).exists(), mid


def test_defaults_validate(reg):
    for mid in IDS:
        for task in reg.models[mid]["tasks"]:
            out = reg.validate_params(mid, {"text": "hello"})
            assert out["text"] == "hello", (mid, task)


def test_licences_flag_non_commercial(reg):
    for mid in ("xtts-v2", "spark-tts", "llasa-1b"):
        assert reg.models[mid]["license"]["commercial"] is False, mid
    for mid in ("chatterbox", "chatterbox-multilingual", "zonos-v0.1-transformer", "openvoice-v2"):
        assert reg.models[mid]["license"]["commercial"] is True, mid
