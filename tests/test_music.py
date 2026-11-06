"""Registry-level checks of models.d/music.yaml (no GPU, no weights needed)."""
import ast
from pathlib import Path

import pytest

from app.config import Settings
from app.registry import Registry

STUDIO = Path(__file__).resolve().parent.parent
NC = {"musicgen-medium", "musicgen-melody", "musicgen-stereo-medium", "audiogen-medium", "magnet-medium-10secs",
      "diffrhythm-1.2", "diffrhythm-1.2-full", "songgeneration-levo"}


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("music")
    return Registry(STUDIO / "models.yaml", Settings(tmp / "settings.json"), hf_cache=tmp / "hf", lmstudio=tmp / "lm",
                    probe_cache=tmp / "probe.json")


def music_models(reg):
    return {k: m for k, m in reg.models.items() if str(m.get("backend", "")).startswith("music_")}


def test_loads_without_errors(reg):
    assert not [e for e in reg.errors if "music" in str(e)]
    assert len(music_models(reg)) >= 12


def test_tasks_and_styles_exist(reg):
    styles = set(Path(STUDIO / "prompts" / "core.yaml").read_text().split())
    for k, m in music_models(reg).items():
        assert set(m["tasks"]) <= {"music", "song", "sfx"}, k
        ps = m["prompt_style"]
        for s in (ps.values() if isinstance(ps, dict) else [ps]):
            assert f"{s}:" in styles, (k, s)


def test_noncommercial_licences(reg):
    m = music_models(reg)
    for k in NC & set(m):
        assert m[k]["license"]["commercial"] is False, k
    for k in ("acestep-1.5-turbo", "heartmula-3b"):
        assert m[k]["license"]["commercial"] is True, k


def test_song_models_take_lyrics(reg):
    for k, m in music_models(reg).items():
        if "song" in m["tasks"]:
            assert m["features"].get("lyrics"), k
            assert any(p["name"] == "lyrics" for p in m["params"]), k


def test_extend_models_have_extend_param(reg):
    for k, m in music_models(reg).items():
        if m["features"].get("extend"):
            assert any(p["name"] == "extend_seconds" for p in m["params"]), k


def test_blocked_entry_has_reason(reg):
    m = music_models(reg)["songgeneration-levo"]
    assert m["features"]["blocked"]
    assert reg.status["songgeneration-levo"]["status"] != "ready"


def test_workers_are_valid_python_and_use_the_protocol():
    for f in (STUDIO / "workers").glob("music_*_worker.py"):
        tree = ast.parse(f.read_text())
        src = f.read_text()
        assert "serve(" in src and "Backend" in src, f.name
        assert any(isinstance(n, ast.ClassDef) for n in tree.body), f.name
