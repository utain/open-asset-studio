"""Settings page backend: the schema, validation, system reset, per-project overrides and their effects.
No GPU, no weights (jobs are queued but never executed)."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from app import config, profiles, settings_schema


@pytest.fixture
def client(monkeypatch):
    import app.main as main
    from app.core import Core

    main.core = Core()
    core = main.core
    monkeypatch.setattr(core.jobs, "_execute", lambda jid: None)      # jobs stay queued: nothing runs
    with TestClient(main.app) as c:
        for pid in ("p1", "p2"):
            if c.get(f"/api/projects/{pid}/settings").status_code == 404:
                assert c.post("/api/projects", json={"id": pid, "title": f"Project {pid}"}).status_code == 200
        yield c, core
    for pid in ("p1", "p2"):
        c.delete(f"/api/projects/{pid}/settings")
    core.settings.update({"memory": {"margin_gb": None}, "profile": None, "license": {"commercial_only": None, "strict": None},
                          "export": {"preset": None}, "defaults": {"test": None}, "textgen": {"improve": None}, "prompt": {"lang": None}})
    main.core = None


def _unflatten(flat: dict) -> dict:
    out: dict = {}
    for k, v in flat.items():
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def _flatten_defaults(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(_flatten_defaults(v, f"{prefix}{k}."))
        elif isinstance(v, dict):
            continue                    # empty maps (defaults, textgen, prompt.context, export) are expanded by the schema
        else:
            out[f"{prefix}{k}"] = v
    return out


def _nc_model(core):
    """A registry model that is not licensed for commercial use and does tasks 'test' (a copy of the swatch test model)."""
    m = copy.deepcopy(core.registry.models["test-swatch"])
    m.update(id="test-nc", name="Test NC", license={"name": "CC-BY-NC", "commercial": False})
    core.registry.models["test-nc"] = m
    core.registry.status["test-nc"] = dict(core.registry.status["test-swatch"])
    return m


# ------------------------------------------------------------------ schema
def test_schema_covers_every_default_leaf(client):
    c, core = client
    keys = {f["key"] for f in c.get("/api/settings/schema").json()["fields"]}
    for k in _flatten_defaults(config.DEFAULT_SETTINGS):
        assert k in keys, f"{k} has no schema field"
    for k in keys:
        if k.startswith(("defaults.", "textgen.", "prompt.context.")) or k == "export.preset":
            continue
        assert k in _flatten_defaults(config.DEFAULT_SETTINGS), f"{k} is not a default setting"
    assert {f"defaults.{t}" for t in core.registry.tasks} <= keys


def test_field_metadata(client):
    c, _ = client
    body = c.get("/api/settings/schema").json()
    groups = {g["id"] for g in body["groups"]}
    assert body["version"] == 1 and {"files", "env"} <= set(body) and "settings_json" in body["files"]
    for f in body["fields"]:
        k = f["key"]
        assert f["type"] in settings_schema.TYPES, k
        assert f["group"] in groups, k
        assert f["scope"] and set(f["scope"]) <= {"system", "project"}, k
        assert f["apply"] in settings_schema.APPLY, k
        assert f["label"], k
        if f["type"] in ("int", "float") and f.get("default") is not None:
            lo, hi = f.get("min"), f.get("max")
            assert (lo is None or f["default"] >= lo) and (hi is None or f["default"] <= hi), k
        if f["type"] == "choice":
            vals = [ch["value"] for ch in f["choices"]]
            assert vals and (f["default"] is None or f["default"] in vals), k
        if f["type"] == "model":
            assert f["task"], k


def test_defaults_validate(client):
    _, core = client
    for f in settings_schema.fields(core):
        if f.get("editable", True) and f.get("default") is not None:
            out = settings_schema.validate_patch(_unflatten({f["key"]: f["default"]}), "system", core)
            assert out, f["key"]


def test_validation_rejects(client):
    c, _ = client
    bad = [
        ({"license": {"commercial_only": "yes"}}, "system"),
        ({"workers": {"cancel_grace_s": 1.5}}, "system"),
        ({"workers": {"cancel_grace_s": 100000}}, "system"),
        ({"memory": {"nope": 1}}, "system"),
        ({"nope": 1}, "system"),
        ({"profile": "mystery"}, "system"),
        ({"export": {"preset": "mystery"}}, "system"),
        ({"defaults": {"t2i": "test-tone"}}, "system"),            # test-tone does not do t2i
        ({"defaults": {"no_such_task": "test-tone"}}, "system"),
        ({"llm": {"server": "/bin/sh"}}, "system"),                 # file-only
        ({"llm": {"args": ["-x"]}}, "system"),
        ({"textgen": {"cli": {"claude": {"binary": "/bin/sh"}}}}, "system"),
        ({"llm": {"primary": {"model": "relative/path.gguf"}}}, "system"),
        ({"textgen": {"improve": "bogus:thing"}}, "system"),
        ({"prompt": {"lang": "fr"}}, "system"),
        ({"prompt": {"context": {"image": "x" * 301}}}, "system"),
    ]
    for patch, _scope in bad:
        r = c.patch("/api/settings", json=patch)
        assert r.status_code == 400, patch
        assert r.json()["kind"] == "input"
    r = c.patch("/api/projects/p1/settings", json={"memory": {"margin_gb": 3}})        # system-only key
    assert r.status_code == 400 and "system-wide" in r.json()["error"]
    assert c.patch("/api/projects/p1/settings", json={"license": {"commercial_only": "yes"}}).status_code == 400
    assert c.patch("/api/projects/nope/settings", json={"profile": "film"}).status_code == 404


def test_legacy_patches_still_work(client):
    c, _ = client
    assert c.patch("/api/settings", json={"profile": "video"}).json()["profile"] == "video"
    assert c.patch("/api/settings", json={"license": {"commercial_only": True}}).json()["license"]["commercial_only"] is True
    assert c.patch("/api/settings", json={"export": {"preset": None}}).status_code == 200
    r = c.patch("/api/settings", json={"textgen": {"api_models": [{"provider": "openai", "model": "gpt-x"}]}})
    assert r.status_code == 200 and r.json()["textgen"]["api_models"][0]["model"] == "gpt-x"
    assert c.patch("/api/settings", json={"textgen": {"api_models": None}}).status_code == 200
    assert c.patch("/api/settings", json={"textgen": {"cli": {"claude": {"model": "opus"}}}}).status_code == 200
    assert c.patch("/api/settings", json={"textgen": {"cli": {"claude": {"model": None}}}}).status_code == 200
    assert c.patch("/api/settings", json={"ui": {"show_test_models": True}}).status_code == 200
    assert c.patch("/api/settings", json={"ui": {"show_test_models": False}}).status_code == 200
    c.patch("/api/settings", json={"profile": None, "license": {"commercial_only": None}})


def test_system_reset(client):
    c, core = client
    assert c.patch("/api/settings", json={"memory": {"margin_gb": 10}}).json()["memory"]["margin_gb"] == 10
    assert c.get("/api/settings").json()["memory"]["margin_gb"] == 10
    r = c.patch("/api/settings", json={"memory": {"margin_gb": None}}).json()
    assert r["memory"]["margin_gb"] == 6.0
    stored = json.loads(core.settings.path.read_text())
    assert "margin_gb" not in (stored.get("memory") or {}) and "memory" not in stored


# ------------------------------------------------------------------ projects
def test_project_license_override(client):
    c, core = client
    m = _nc_model(core)
    body = {"task": "test", "model": m["id"], "project": "p1", "params": {"steps": 1}}
    assert c.patch("/api/projects/p1/settings", json={"license": {"commercial_only": True}}).status_code == 200
    r = c.post("/api/jobs", json=body)
    assert r.status_code == 403 and r.json()["kind"] == "license"
    assert "p1" in r.json()["error"] and "Commercial-safe models only" in r.json()["error"]
    r = c.post("/api/jobs", json={**body, "project": "default"})
    assert r.status_code == 200 and r.json().get("kind") != "license"           # the system filter is off
    assert core.settings.get("license", "commercial_only") is False
    # a project may also be looser than the system filter (the UI warns)
    c.patch("/api/settings", json={"license": {"commercial_only": True}})
    assert c.post("/api/jobs", json={**body, "project": "p2"}).status_code == 403
    c.patch("/api/projects/p2/settings", json={"license": {"commercial_only": False}})
    assert c.post("/api/jobs", json={**body, "project": "p2"}).status_code == 200


def test_project_export_override(client):
    c, core = client
    c.patch("/api/projects/p1/settings", json={"export": {"preset": "podcast"}})
    assert profiles.resolve_export({}, core.settings_for("p1"))["preset"] == "podcast"
    assert profiles.resolve_export({}, core.settings)["preset"] != "podcast"
    assert core.settings.get("export", "preset") is None
    c.patch("/api/projects/p2/settings", json={"profile": "film"})
    assert profiles.resolve_export({}, core.settings_for("p2"))["preset"] == "film_master"
    body = c.get("/api/projects/p1/settings").json()
    assert body["resolved"]["export_preset"] == "podcast" and body["effective"]["export"]["preset"] == "podcast"


def test_project_default_model(client):
    c, core = client
    assert core.registry.default_for("test") is None
    c.patch("/api/projects/p1/settings", json={"defaults": {"test": "test-swatch"}})
    assert core.registry.default_for("test", settings=core.settings_for("p1")) == "test-swatch"
    assert core.registry.default_for("test") is None and core.settings.get("defaults") == {}
    assert c.get("/api/models?project=p1").json()["defaults"]["test"] == "test-swatch"
    assert c.get("/api/models").json()["defaults"]["test"] is None
    assert c.get("/api/bootstrap?project=p1").json()["settings"]["defaults"]["test"] == "test-swatch"
    j = core.jobs.submit({"task": "test", "project": "p1", "params": {"steps": 1}})
    assert j["model"] == "test-swatch"
    # PUT /api/settings/defaults with a project writes the override, not the system default
    r = c.put("/api/settings/defaults", json={"task": "test", "model": "test-tone", "project": "p2"})
    assert r.status_code == 200 and r.json()["model"] == "test-tone"
    assert core.settings.get("defaults") == {}
    assert c.get("/api/projects/p2/settings").json()["overrides"]["defaults"]["test"] == "test-tone"


def test_project_reset(client):
    c, core = client
    c.patch("/api/projects/p1/settings", json={"profile": "film", "license": {"commercial_only": True, "strict": True}})
    b = c.get("/api/projects/p1/settings").json()
    assert b["overridden"] == ["license.commercial_only", "license.strict", "profile"]
    assert [p for p in c.get("/api/projects").json()["projects"] if p["id"] == "p1"][0]["overrides"] == 3
    b = c.patch("/api/projects/p1/settings", json={"license": {"strict": None}}).json()
    assert b["overridden"] == ["license.commercial_only", "profile"] and b["effective"]["license"]["strict"] is False
    b = c.delete("/api/projects/p1/settings").json()
    assert b["overrides"] == {} and b["overridden"] == [] and b["effective"]["profile"] == "general"
    info = json.loads((core.assets.project_dir("p1") / "project.json").read_text())
    assert info["title"] == "Project p1" and "settings" not in info


def test_job_settings_snapshot(client):
    c, core = client
    c.patch("/api/projects/p1/settings", json={"export": {"preset": "podcast"}})
    j = core.jobs.submit({"task": "test", "model": "test-swatch", "project": "p1", "params": {"steps": 1}})
    c.patch("/api/projects/p1/settings", json={"export": {"preset": "broadcast"}})
    ctx = core.jobs.ctx[j["id"]]
    assert ctx.settings.get("export", "preset") == "podcast"
    assert core.settings_for("p1").get("export", "preset") == "broadcast"
    assert "_settings" not in core.jobs.public(core.jobs.get(j["id"]))
    j2 = core.jobs.retry(j["id"])                                                  # a retry re-resolves
    assert core.jobs.ctx[j2["id"]].settings.get("export", "preset") == "broadcast"


def test_update_project_keeps_settings(client):
    c, core = client
    c.patch("/api/projects/p1/settings", json={"profile": "video"})
    assert c.patch("/api/projects/p1", json={"title": "Renamed"}).json()["settings"] == {"profile": "video"}
    assert c.get("/api/projects/p1/settings").json()["effective"]["profile"] == "video"
    assert c.patch("/api/projects/p1", json={"title": "Project p1"}).status_code == 200


def test_improve_uses_project_backend(client, monkeypatch):
    c, core = client
    seen = {}

    def fake(text, **kw):
        seen.update(kw)
        return {"ok": True}

    monkeypatch.setattr(core.improver, "improve", fake)
    c.patch("/api/projects/p1/settings", json={"textgen": {"improve": "cli:claude@opus"}, "prompt": {"lang": "th"}})
    assert c.post("/api/improve", json={"text": "a cat", "project": "p1"}).status_code == 200
    assert seen["backend"] == "cli:claude@opus" and seen["lang"] == "th"
    c.post("/api/improve", json={"text": "a cat", "project": "p1", "backend": "local", "lang": "en"})
    assert seen["backend"] == "local" and seen["lang"] == "en"           # an explicit choice wins
    c.post("/api/improve", json={"text": "a cat"})
    assert seen["backend"] is None and seen["lang"] is None


def test_env_has_no_secrets(client):
    c, _ = client
    import re

    env = c.get("/api/settings/schema").json()["env"]
    assert env and all(not re.search(r"KEY|TOKEN|SECRET", e["key"]) for e in env)
    assert {"STUDIO_DATA", "HF_HUB_OFFLINE"} <= {e["key"] for e in env}


def test_settings_event_scope(client, monkeypatch):
    c, core = client
    events = []
    monkeypatch.setattr(core.bus, "publish", lambda t, d: events.append((t, d)))
    c.patch("/api/projects/p1/settings", json={"profile": "film"})
    t, d = [e for e in events if e[0] == "settings"][-1]
    assert d["scope"] == "project" and d["project"] == "p1" and d["profile"] == "film"
    events.clear()
    c.patch("/api/settings", json={"memory": {"margin_gb": 7}})
    assert not [e for e in events if e[0] == "settings"]                      # nothing the browser shows changed
    c.patch("/api/settings", json={"license": {"strict": True}})
    t, d = [e for e in events if e[0] == "settings"][-1]
    assert d["scope"] == "system" and d["project"] is None and d["license"]["strict"] is True and "ui" in d
    c.patch("/api/settings", json={"license": {"strict": None}})
