"""TTS / voice-clone group B (models.d/tts_clone_b.yaml): entries load cleanly, tasks exist, workers are present,
every param validates its own default, licences are flagged. No GPU and no weights needed."""
import pytest

from app import config
from app.registry import Registry

IDS = ["indextts-2", "indextts-2.5", "maskgct", "vevo", "vevo15", "dia-1.6b", "dia2-2b", "bark",
       "vibevoice-1.5b", "vibevoice-7b", "csm-1b"]
NON_COMMERCIAL = ["maskgct", "vevo", "vevo15"]


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    return Registry(config.STUDIO / "models.yaml", probe_cache=tmp_path_factory.mktemp("p") / "probe.json")


def test_group_loads_without_errors(reg):
    assert not [e for e in reg.errors if "tts_clone_b" in e]
    for mid in IDS:
        assert mid in reg.models, mid
        assert reg.models[mid]["_file"] == "tts_clone_b.yaml"


def test_tasks_exist(reg):
    for mid in IDS:
        assert set(reg.models[mid]["tasks"]) <= set(reg.tasks), mid
    assert reg.tasks["voice_convert"]["studio"] == "sound"
    assert "voice_convert" in reg.models["vevo"]["tasks"] and "voice_convert" in reg.models["vevo15"]["tasks"]


def test_backends_have_worker_scripts(reg):
    for mid in IDS:
        b = reg.backends[reg.models[mid]["backend"]]
        assert (config.STUDIO / b["script"]).exists(), mid


def test_defaults_validate(reg):
    for mid in IDS:
        out = reg.validate_params(mid, {})
        assert isinstance(out, dict), mid


def test_licences(reg):
    for mid in NON_COMMERCIAL:
        assert reg.models[mid]["license"]["commercial"] is False, mid
    for mid in ("dia-1.6b", "dia2-2b", "bark", "indextts-2"):
        assert reg.models[mid]["license"]["commercial"] is True, mid


def test_csm_is_wired_but_gated(reg):
    m = reg.models["csm-1b"]
    assert not m["features"].get("blocked")            # verified through the worker, see docs/research/audio_more/tts_clone_b.md
    assert m["license"]["name"] == "Apache-2.0" and m["license"]["commercial"] is True
    assert "hf auth login" in m["weights"][0]["fix"] and "huggingface.co/sesame/csm-1b" in m["weights"][0]["fix"]


def test_bark_voice_presets(reg):
    p = next(x for x in reg.models["bark"]["params"] if x["name"] == "voice")
    values = [c["value"] if isinstance(c, dict) else c for c in p["choices"]]
    assert p["default"] in values and "random" in values and len(values) > 100
