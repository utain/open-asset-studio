"""Registry compatibility status: ready / missing_weights / missing_env / broken."""
import os
import textwrap

import pytest

from app.config import Settings
from app.registry import Registry


@pytest.fixture
def env(tmp_path):
    hf = tmp_path / "hf"
    snap = hf / "models--org--good" / "snapshots" / "abc"
    (snap / "transformer").mkdir(parents=True)
    blob = hf / "models--org--good" / "blobs" / "b1"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x")
    os.symlink(blob, snap / "transformer" / "model.safetensors")
    # a repo whose only weight is a dangling blob link (= unfinished download)
    snap2 = hf / "models--org--partial" / "snapshots" / "def"
    snap2.mkdir(parents=True)
    os.symlink(hf / "models--org--partial" / "blobs" / "missing", snap2 / "model.safetensors")
    # a configs-only repo
    snap3 = hf / "models--org--configs" / "snapshots" / "ghi"
    snap3.mkdir(parents=True)
    (snap3 / "config.json").write_text("{}")
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("#!/bin/sh\n")
    (venv / "python").chmod(0o755)
    (tmp_path / "w_worker.py").write_text("# worker\n")
    yaml = textwrap.dedent(f"""
    backends:
      ok: {{python: {venv}/python, script: {tmp_path}/w_worker.py}}
      noenv: {{python: {tmp_path}/nope/bin/python, script: {tmp_path}/w_worker.py}}
      noworker: {{python: {venv}/python, script: {tmp_path}/missing_worker.py}}
    models:
      - {{id: ready, backend: ok, tasks: [t2i], weights: [{{hf: org/good}}],
          params: [{{name: steps, type: int, min: 1, max: 50, default: 20}}, {{name: mode, type: choice, choices: [a, b], default: a}},
                   {{name: seed, type: seed}}, {{name: fast, type: bool, default: false}}]}}
      - {{id: files_ok, backend: ok, tasks: [t2i], weights: [{{hf: org/good, files: ["transformer/*.safetensors"]}}]}}
      - {{id: files_missing, backend: ok, tasks: [t2i], weights: [{{hf: org/good, files: ["vae/*.safetensors"]}}]}}
      - {{id: no_repo, backend: ok, tasks: [t2i], weights: [{{hf: org/absent}}]}}
      - {{id: dangling, backend: ok, tasks: [t2i], weights: [{{hf: org/partial}}]}}
      - {{id: configs_only, backend: ok, tasks: [t2i], weights: [{{hf: org/configs}}]}}
      - {{id: path_missing, backend: ok, tasks: [upscale], weights: [{{path: {tmp_path}/nothing.pth}}]}}
      - {{id: env_missing, backend: noenv, tasks: [t2i], weights: [{{hf: org/good}}]}}
      - {{id: no_worker, backend: noworker, tasks: [t2i], weights: []}}
      - {{id: bad_backend, backend: ghost, tasks: [t2i]}}
    """)
    (tmp_path / "models.yaml").write_text(yaml)
    (tmp_path / "models.d").mkdir()
    (tmp_path / "models.d" / "extra.yaml").write_text("models:\n  - {id: dropin, backend: ok, tasks: [sfx], weights: []}\n")
    s = Settings(tmp_path / "settings.json")
    return Registry(tmp_path / "models.yaml", s, hf_cache=hf, lmstudio=tmp_path / "lm", probe_cache=tmp_path / "probe.json"), s


def test_status(env):
    r, _ = env
    st = {k: (v["status"], v["code"]) for k, v in r.status.items()}
    assert st["ready"] == ("ready", "ready")
    assert st["files_ok"] == ("ready", "ready")
    assert st["files_missing"][0] == "missing_weights"
    assert st["no_repo"][0] == "missing_weights"
    assert st["dangling"][0] == "missing_weights"
    assert st["configs_only"][0] == "missing_weights"
    assert st["path_missing"][0] == "missing_weights"
    assert st["env_missing"][0] == "missing_env"
    assert st["no_worker"] == ("broken", "no_worker")
    assert st["bad_backend"] == ("broken", "unknown_backend")
    assert st["dropin"][0] == "ready"  # models.d drop-ins are merged
    assert "hf download org/good --include 'vae/*.safetensors'" in r.status["files_missing"]["fix"]
    assert r.status["env_missing"]["fix"]


def test_probe_failure_marks_broken(env):
    r, _ = env
    r.probes["ready"] = {"ok": False, "reason": "ImportError: no torch", "stamp": r._env_stamp(r.backends["ok"])}
    r.reload()
    assert r.status["ready"]["status"] == "broken" and r.status["ready"]["code"] == "probe_failed"
    r.probes["ready"]["stamp"] = "stale"  # env changed since the probe: the old failure no longer counts
    r.reload()
    assert r.status["ready"]["status"] == "ready"


def test_validate_params(env):
    r, _ = env
    p = r.validate_params("ready", {"steps": "500", "mode": "b", "seed": "7", "fast": "true", "prompt": "hi"})
    assert p == {"steps": 50, "mode": "b", "seed": 7, "fast": True, "prompt": "hi"}
    assert r.validate_params("ready", {})["steps"] == 20
    with pytest.raises(ValueError):
        r.validate_params("ready", {"mode": "zzz"})
    with pytest.raises(ValueError):
        r.validate_params("ready", {"seed": 2**40})


def test_defaults(env):
    r, s = env
    assert r.default_for("t2i") in ("ready", "files_ok")  # a ready model wins over not-ready ones
    s.update({"defaults": {"t2i": "files_ok"}})
    assert r.default_for("t2i") == "files_ok"
    s.update({"defaults": {"t2i": "dropin"}})  # cannot do t2i -> ignored
    assert r.default_for("t2i") in ("ready", "files_ok")
    assert r.list(task="upscale")[0]["id"] == "path_missing"
