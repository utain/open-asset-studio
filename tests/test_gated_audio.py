"""Gated audio models (Stable Audio Open 1.0 / Small via workers/sao_worker.py, Sesame CSM-1B): registry + pure helpers, no GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest

from app.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
GATED = {"stable-audio-open-1.0": "stabilityai/stable-audio-open-1.0",
         "stable-audio-open-small": "stabilityai/stable-audio-open-small",
         "csm-1b": "sesame/csm-1b"}


@pytest.fixture(scope="module")
def reg():
    r = Registry()
    assert not r.errors
    return r


def test_gated_entries_are_wired(reg):
    for mid, repo in GATED.items():
        m = reg.models[mid]
        assert not m["features"].get("blocked"), mid
        w = m["weights"][0]
        assert w["hf"] == repo and "hf auth login" in w["fix"] and f"huggingface.co/{repo}" in w["fix"], mid
        assert m["license"]["commercial"] is True and m["license"].get("name"), mid
    for mid in ("stable-audio-open-1.0", "stable-audio-open-small"):
        assert reg.models[mid]["backend"] == "sao" and reg.models[mid]["license"].get("verify") is True
    assert reg.models["csm-1b"]["backend"] == "cloneb_hf" and reg.models["csm-1b"]["license"]["name"] == "Apache-2.0"


def test_sao_backend_and_recipe():
    import yaml

    reg = yaml.safe_load((ROOT / "models.d" / "sfx_more.yaml").read_text())
    be = reg["backends"]["sao"]
    assert be["python"] == "envs/sao/bin/python" and (ROOT / be["script"]).exists()
    assert (ROOT / "install" / "sao.yaml").exists() and (ROOT / "env-lock" / "sao.txt").exists()
    assert "stable-audio-tools" in (ROOT / "env-lock" / "sao.txt").read_text()


def test_sao_worker_helpers(tmp_path, monkeypatch):
    pytest.importorskip("soundfile")
    monkeypatch.syspath_prepend(str(ROOT / "workers"))
    import sao_worker as w

    sr = 8000
    t = np.arange(sr) / sr
    out = tmp_path / "a.wav"
    meta = w.save_wav(out, np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2).astype(np.float32), sr)   # (C, T) in
    assert meta["duration"] == 1.0 and meta["sample_rate"] == sr and "peak_limited" not in meta
    hot = w.save_wav(tmp_path / "b.wav", 2.0 * np.sin(2 * np.pi * 440 * t).astype(np.float32), sr)
    assert hot["peak_limited"] > 0
    with pytest.raises(RuntimeError):
        w.save_wav(tmp_path / "c.wav", np.zeros(sr, dtype=np.float32), sr)
    assert w.take_name("a  door\nslam", 0, 1) == "a door slam" and w.take_name("x", 1, 3) == "x #2"
    assert w.seed0({"seed": 5}) == 5 and w.seed0({"seed": -1}) >= 0
