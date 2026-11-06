"""Commercial-use licence filter: states, the setting, and the job API refusing non-commercial models."""
import pytest
from fastapi.testclient import TestClient

from app import licensing


@pytest.fixture
def client():
    import app.main as main
    from app.core import Core

    main.core = Core()
    with TestClient(main.app) as c:
        yield c, main.core
    main.core.settings.update({"license": {"commercial_only": False, "strict": False}})
    main.core = None


def test_states():
    assert licensing.state({"commercial": True}) == licensing.COMMERCIAL
    assert licensing.state({"commercial": True, "verify": True}) == licensing.CHECK
    assert licensing.state({"commercial": False}) == licensing.NONCOMMERCIAL
    assert licensing.state(None) == licensing.NONCOMMERCIAL


def test_every_model_has_a_boolean_commercial_flag(client):
    _, core = client
    for mid, m in core.registry.models.items():
        assert isinstance(m["license"].get("commercial"), bool), mid


def test_blocked_follows_settings(client):
    _, core = client
    nc = {"name": "X", "license": {"name": "CC-BY-NC", "commercial": False}}
    chk = {"name": "Y", "license": {"name": "Custom", "commercial": True, "verify": True}}
    ok = {"name": "Z", "license": {"name": "Apache-2.0", "commercial": True}}
    assert not any(licensing.blocked(m, core.settings) for m in (nc, chk, ok))
    core.settings.update({"license": {"commercial_only": True}})
    assert licensing.blocked(nc, core.settings) and not licensing.blocked(chk, core.settings) and not licensing.blocked(ok, core.settings)
    core.settings.update({"license": {"strict": True}})
    assert licensing.blocked(chk, core.settings) and not licensing.blocked(ok, core.settings)
    assert "not licensed for commercial use" in licensing.refusal({**nc, "id": "x"})
    assert "licence verified" in licensing.refusal({**chk, "id": "y"})


def _pick(core, commercial: bool):
    for mid, m in core.registry.models.items():
        if m.get("tasks") and bool(m["license"].get("commercial")) == commercial and not m["license"].get("verify") and "test" not in m["tasks"]:
            return mid, m
    pytest.skip("no suitable model in the registry")


def test_job_api_refuses_noncommercial_models(client):
    c, core = client
    mid, m = _pick(core, False)
    task = m["tasks"][0]
    body = {"task": task, "model": mid, "params": {"prompt": "x"}}
    # off: the licence check does not run (the job may still fail for other reasons, never with kind=license)
    r = c.post("/api/jobs", json=body)
    assert r.json().get("kind") != "license"
    s = c.patch("/api/settings", json={"license": {"commercial_only": True}})
    assert s.status_code == 200 and s.json()["license"]["commercial_only"] is True
    r = c.post("/api/jobs", json=body)
    assert r.status_code == 403 and r.json()["kind"] == "license"
    assert "Commercial-safe models only" in r.json()["error"] and m["name"] in r.json()["error"]
    # retry/submit of a commercial model is not refused for licence reasons
    mid2, m2 = _pick(core, True)
    r = c.post("/api/jobs", json={"task": m2["tasks"][0], "model": mid2, "params": {"prompt": "x"}})
    assert r.json().get("kind") != "license"


def test_settings_roundtrip_and_event(client):
    c, core = client
    r = c.patch("/api/settings", json={"license": {"commercial_only": True, "strict": True}}).json()
    assert r["license"] == {"commercial_only": True, "strict": True}
    assert c.get("/api/profiles").json()["license"]["commercial_only"] is True
    assert c.get("/api/bootstrap").json()["settings"]["license"]["strict"] is True
    c.patch("/api/settings", json={"license": {"commercial_only": False, "strict": False}})
    assert c.get("/api/settings").json()["license"]["commercial_only"] is False
