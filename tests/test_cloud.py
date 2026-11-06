"""Cloud providers: secrets never leave the core, secrets file mode, budget / confirm, cloud jobs bypass the GPU
queue, the "Allow cloud" refusal. End-to-end adapter runs against tests/fixtures/cloud_mock.py are a separate
script (see docs/CONTRACTS.md section 8.6)."""
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.jobs import handler
from app.providers import PROVIDERS, ProviderError, Providers, estimate_cost

KEY = "sk-test-SUPERSECRET-0123456789abcdef"


@pytest.fixture(autouse=True)
def _no_env_keys(monkeypatch):
    for p in PROVIDERS.values():
        for e in p["env"]:
            monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("STUDIO_CLOUD_MOCK", "http://127.0.0.1:9")   # never reach a real provider from tests


@pytest.fixture
def client():
    import app.main as main
    from app.core import Core

    for n in ("secrets.json", "providers.json", "spend.jsonl"):
        (config.DATA / n).unlink(missing_ok=True)
    main.core = Core()
    with TestClient(main.app) as c:
        yield c, main.core
    main.core = None


def _wait(c, jid, t=20):
    t0 = time.time()
    while time.time() - t0 < t:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    return j


# ------------------------------------------------------------------ secrets
def test_secrets_file_mode_600_and_never_in_responses(client):
    c, core = client
    bodies = []

    def call(method, url, **kw):
        r = c.request(method, url, **kw)
        bodies.append(r.text)
        return r
    r = call("PUT", "/api/x/providers/keys/fal", json={"key": KEY})
    assert r.status_code == 200 and r.json()["key"] == {"set": True, "source": "file", "env": None, "last4": "cdef",
                                                        "env_names": ["FAL_KEY", "FAL_API_KEY"]}
    path = config.DATA / "secrets.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert KEY in path.read_text()                       # stored ...
    call("PUT", "/api/x/providers/keys/meshy", json={"key": KEY + "m"})
    assert path.stat().st_mode & 0o777 == 0o600          # ... still 600 after a rewrite
    os.chmod(path, 0o644)                                # loosened by hand: fixed on the next load
    Providers(config.DATA)
    assert path.stat().st_mode & 0o777 == 0o600
    call("PUT", "/api/x/providers/endpoints", json={"name": "lab", "base_url": "http://127.0.0.1:9/v1", "key": KEY + "e",
                                                     "models": ["m1"], "modalities": ["llm", "image"]})
    call("PUT", "/api/x/providers/allow", json={"on": True})
    for url in ("/api/x/providers", "/api/x/providers/spend", "/api/x/providers/catalog", "/api/models", "/api/settings",
                "/api/bootstrap", "/api/system", "/api/models/fal-flux2-pro", "/api/jobs"):
        assert call("GET", url).status_code == 200, url
    call("POST", "/api/x/providers/test/fal")                # network error (mock port 9): message must not echo it
    call("POST", "/api/x/providers/test/openai_compat:lab")
    call("POST", "/api/jobs", json={"task": "t2i", "model": "fal-flux2-pro", "params": {"prompt": "x"}})   # -> confirm/ok
    call("POST", "/api/x/providers/estimate", json={"model": "fal-flux2-pro", "params": {}})
    for b in bodies:
        assert KEY not in b and KEY[:-4] not in b
    # provider error texts are scrubbed before they reach a job / the UI
    assert KEY not in core.providers.secrets_in(f"bad key {KEY} given")


def test_env_key_is_used_but_not_shown(monkeypatch, tmp_path):
    monkeypatch.setenv("REPLICATE_API_TOKEN", KEY)
    p = Providers(tmp_path)
    assert p.get_key("replicate") == KEY
    info = p.key_info("replicate")
    assert info["source"] == "env" and info["last4"] == "cdef" and KEY not in str(info)
    p.set_key("replicate", KEY + "x")                    # the secrets file wins over the env var
    assert p.get_key("replicate") == KEY + "x" and p.key_info("replicate")["source"] == "file"
    with pytest.raises(ProviderError):
        p.set_key("replicate", "has space inside")


# ------------------------------------------------------------------ budget / confirm
def test_budget_and_confirm(tmp_path):
    p = Providers(tmp_path)
    with pytest.raises(ProviderError) as e:
        p.spend.check(0.01, "fal")
    assert e.value.kind == "cloud_disabled"
    p.set_allowed(True)
    p.set_limits(budget_usd=1.0, confirm_usd=0.5)
    p.spend.check(0.10, "fal")                                   # cheap: passes
    with pytest.raises(ProviderError) as e:
        p.spend.check(0.50, "fal")                               # at the threshold: confirm
    assert e.value.kind == "confirm" and e.value.data["estimate_usd"] == 0.5
    p.spend.check(0.50, "fal", confirmed=True)
    with pytest.raises(ProviderError) as e:
        p.spend.check(None, "fal")                               # unknown price: confirm
    assert e.value.kind == "confirm"
    p.spend.record(0.70, "fal", "fal-flux2-pro", "default", "j1")
    p.spend.reserve("tok", 0.20)                                 # a queued job counts against the budget
    with pytest.raises(ProviderError) as e:
        p.spend.check(0.20, "fal", confirmed=True)               # 0.70 + 0.20 + 0.20 > 1.00
    assert e.value.kind == "budget"
    p.spend.release("tok")
    p.spend.check(0.20, "fal", confirmed=True)
    s = p.spend.summary()
    assert s["total_usd"] == 0.7 and s["by_provider"][0]["key"] == "fal" and s["by_project"][0]["key"] == "default"
    assert Providers(tmp_path).spend.total() == 0.7                # persisted in spend.jsonl
    p.set_limits(budget_usd=0)                                   # 0 = no budget
    p.spend.check(50, "fal", confirmed=True)
    # a local endpoint is not "cloud": works with the switch off
    p.put_endpoint("lab", "http://127.0.0.1:7880/v1", ["m"], ["llm"])
    p.set_allowed(False)
    p.spend.check(0.0, "openai_compat:lab")


def test_estimates():
    assert estimate_cost({"per_image": 0.04}, {"batch": 3}) == 0.12
    assert estimate_cost({"first_megapixel": 0.03, "per_megapixel": 0.015}, {"size": 1024, "aspect": "1:1"}) == 0.03
    assert estimate_cost({"first_megapixel": 0.03, "per_megapixel": 0.015}, {"size": 2048, "aspect": "1:1"}) == 0.075
    assert estimate_cost({"per_request": 0.3, "by": {"resolution": {512: {"per_request": 0.25}}}}, {"resolution": 512}) == 0.25
    assert estimate_cost({"credits": 20, "per_credit": 0.02, "by": {"textured": {True: {"credits": 30}}}}, {"textured": True}) == 0.6
    assert estimate_cost({"per_second": 0.002}, {"seconds": 5, "batch": 2}) == 0.02
    assert estimate_cost({"per_1k_chars": 0.1}, {"text": "x" * 500}) == 0.05
    assert estimate_cost({"per_request": 0.375, "add": {"enable_pbr": {True: 0.15}}}, {"enable_pbr": True}) == 0.525
    assert estimate_cost(None, {}) is None


# ------------------------------------------------------------------ cloud-disabled refusal + status
def test_cloud_disabled_refusal(client):
    c, core = client
    before = set(core.jobs.jobs)
    m = c.get("/api/models/fal-flux2-pro").json()
    assert m["status"]["status"] == "cloud_disabled" and m["device"] == "cpu" and m["cloud"]["provider"] == "fal"
    r = c.post("/api/jobs", json={"task": "t2i", "model": "fal-flux2-pro", "params": {"prompt": "a chest"}})
    assert r.status_code == 409 and r.json()["kind"] == "cloud_disabled"
    c.put("/api/x/providers/allow", json={"on": True})
    assert c.get("/api/models/fal-flux2-pro").json()["status"]["status"] == "missing_key"
    r = c.post("/api/jobs", json={"task": "t2i", "model": "fal-flux2-pro", "params": {"prompt": "a chest"}})
    assert r.status_code == 409 and r.json()["kind"] == "missing_key"
    c.put("/api/x/providers/keys/fal", json={"key": KEY})
    assert c.get("/api/models/fal-flux2-pro").json()["status"]["status"] == "ready"
    c.put("/api/x/providers/allow", json={"on": False})
    r = c.post("/api/jobs", json={"task": "t2i", "model": "fal-flux2-pro", "params": {"prompt": "a chest"}, "confirm_cost": True})
    assert r.status_code == 409 and r.json()["kind"] == "cloud_disabled"
    assert set(core.jobs.jobs) == before                 # nothing was queued


# ------------------------------------------------------------------ GPU lock
def test_cloud_jobs_do_not_take_the_gpu_lock(client, monkeypatch):
    from app import cloud
    c, core = client
    release = threading.Event()
    started = threading.Event()

    @handler("test.block_gpu", gpu=True)
    def _block(ctx):                                     # holds the single GPU slot until released
        started.set()
        release.wait(20)

    def fake_fal(r):                                     # stands in for the HTTP adapter
        r.submitted = True
        r.request_ids.append("req-1")
        p = r.out / "x.png"
        Image.new("RGB", (8, 8), (1, 2, 3)).save(p)
        r.add_image(p, "fal_1", index=0)
    monkeypatch.setitem(cloud.ADAPTERS, "fal", fake_fal)
    c.put("/api/x/providers/allow", json={"on": True})
    c.put("/api/x/providers/keys/fal", json={"key": KEY})
    g = c.post("/api/jobs", json={"task": "test", "model": "test-swatch", "handler": "test.block_gpu", "params": {}}).json()
    assert g["gpu"] is True
    assert started.wait(10)
    # a second GPU job queues behind it ...
    g2 = c.post("/api/jobs", json={"task": "test", "model": "test-swatch", "handler": "test.block_gpu", "params": {}}).json()
    # ... but the cloud job runs right away, in the cloud pool
    r = c.post("/api/jobs", json={"task": "t2i", "model": "fal-flux2-pro", "params": {"prompt": "a chest"}, "confirm_cost": True})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["gpu"] is False and j["handler"] == "cloud" and j["id"] not in core.jobs.gpu_queue
    j = _wait(c, j["id"])
    try:
        assert j["status"] == "done", j.get("error")
        assert core.jobs.gpu_current == g["id"] and c.get(f"/api/jobs/{g2['id']}").json()["status"] == "queued"
        a = c.get(f"/api/assets/{j['outputs'][0]}").json()
        assert a["cloud"]["provider"] == "fal" and a["cost"]["usd"] == j["cost"]["usd"] == 0.03 and a["cost"]["estimated"]
        assert core.providers.spend.total() == 0.03
    finally:
        release.set()
        _wait(c, g["id"])
        _wait(c, g2["id"])
