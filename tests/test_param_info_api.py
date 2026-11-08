"""The model API carries a beginner `info` text per param (the (i) balloons): the YAML `info:` wins, else app.param_info.
No GPU, no weights."""
import importlib.util
import textwrap

import pytest

from app import config, registry as regmod
from app.config import Settings
from app.registry import Registry


@pytest.fixture
def reg(tmp_path):
    (tmp_path / "w_worker.py").write_text("# worker\n")
    (tmp_path / "models.yaml").write_text(textwrap.dedent(f"""
    backends:
      b: {{python: {tmp_path}/py, script: {tmp_path}/w_worker.py}}
    models:
      - id: multi
        backend: b
        tasks: [t2i, edit]
        params:
          - {{name: steps, type: int, default: 20, help: "short help"}}
          - {{name: size, type: int, default: 512, info: "Own text of this model."}}
          - {{name: mystery, type: text}}
          - {{name: only_edit, type: text, tasks: [edit]}}
    """))
    r = Registry(tmp_path / "models.yaml", Settings(tmp_path / "settings.json"), hf_cache=tmp_path / "hf", lmstudio=tmp_path / "lm",
                 probe_cache=tmp_path / "probe.json")
    return r


def stub(monkeypatch, fn):
    monkeypatch.setattr(regmod, "_PARAM_INFO", fn)


def params(r):
    return {p["name"]: p for p in r.public(r.get("multi"))["params"]}


def test_yaml_info_wins_and_param_info_fills_the_rest(reg, monkeypatch):
    table = {("steps", None): "Generic steps.", ("steps", "edit"): "Steps while editing.", ("size", None): "Generic size.", ("only_edit", "edit"): "Edit only."}
    stub(monkeypatch, lambda key, task=None: table.get((key, task)) or table.get((key, None), ""))
    p = params(reg)
    assert p["size"]["info"] == "Own text of this model." and "info_by_task" not in p["size"]
    assert p["steps"]["info"] == "Generic steps." and p["steps"]["info_by_task"] == {"edit": "Steps while editing."}
    assert p["mystery"]["info"] == "" and "info_by_task" not in p["mystery"]
    assert p["only_edit"]["info"] == "Edit only."              # only task-specific text exists: it becomes the plain info
    assert p["steps"]["help"] == "short help"                   # help is untouched
    assert "info" not in reg.get("multi")["params"][0]          # the stored registry entry is not modified


def test_missing_or_broken_param_info_gives_empty_strings(reg, monkeypatch):
    stub(monkeypatch, False)                                    # app/param_info.py missing
    assert params(reg)["steps"]["info"] == "" and params(reg)["size"]["info"] == "Own text of this model."
    def boom(key, task=None):
        raise RuntimeError("x")
    stub(monkeypatch, boom)                                     # module there, function broken
    assert params(reg)["steps"]["info"] == ""


def test_real_registry_api_has_info_on_every_param():
    from fastapi.testclient import TestClient
    import app.main as main
    from app.core import Core

    main.core = Core()
    try:
        with TestClient(main.app) as c:
            models = c.get("/api/models", params={"include_test": "true"}).json()["models"]
            n = 0
            for m in models:
                for p in m.get("params") or []:
                    assert isinstance(p.get("info"), str), (m["id"], p.get("name"))
                    n += 1
            assert n
            if importlib.util.find_spec("app.param_info"):
                steps = [p for m in models for p in m.get("params", []) if p.get("name") == "steps"]
                assert steps and all(p["info"] for p in steps)
            one = next(m for m in models if m.get("params"))
            assert all("info" in p for p in c.get(f"/api/models/{one['id']}").json()["params"])
    finally:
        main.core = None


def test_mcp_get_model_passes_info_through():
    spec = importlib.util.spec_from_file_location("mcp_server_info", config.STUDIO / "scripts" / "mcp_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v = mod.model_view({"id": "m", "name": "M", "status": {"status": "ready"}, "params": [{"name": "steps", "type": "int", "info": "Passes."}, {"name": "x", "type": "int", "info": ""}]}, True)
    assert v["params"][0]["info"] == "Passes." and "info" not in v["params"][1]
