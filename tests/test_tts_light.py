"""Registry-level checks of models.d/tts_light.yaml (no GPU, no weights needed)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOC = yaml.safe_load((ROOT / "models.d" / "tts_light.yaml").read_text())
IDS = ["kokoro-82m", "piper-en", "piper-thai", "melo-english", "styletts2-libritts", "parler-mini", "parler-large"]


def test_models_present_and_backends_declared():
    models = {m["id"]: m for m in DOC["models"]}
    assert list(models) == IDS
    for m in models.values():
        assert m["backend"] in DOC["backends"]
        assert m["modality"] == "audio"
        assert "license" in m and "commercial" in m["license"]
        assert m["memory_gb"] > 0
        assert (ROOT / DOC["backends"][m["backend"]]["script"]).exists()


def test_no_new_task_ids():
    assert "tasks" not in DOC            # tts / voice_design / voice_clone already exist in models.yaml


def test_params_well_formed():
    for m in DOC["models"]:
        names = [p["name"] for p in m["params"]]
        assert len(names) == len(set(names)), m["id"]
        assert "text" in names
        for p in m["params"]:
            if p["type"] == "choice":
                assert p["default"] in [c if not isinstance(c, dict) else c["value"] for c in p["choices"]], (m["id"], p["name"])


def test_cpu_and_noncommercial_flags():
    models = {m["id"]: m for m in DOC["models"]}
    assert models["piper-en"]["device"] == "cpu" and models["piper-thai"]["device"] == "cpu"
    assert models["piper-thai"]["license"]["commercial"] is False
    assert models["parler-mini"]["tasks"] == ["tts", "voice_design"]
    assert "voice_clone" in models["styletts2-libritts"]["tasks"]
