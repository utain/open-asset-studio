"""scripts/mcp_server.py (the public MCP server): JSON-RPC handling, tool schemas, result shaping and error mapping.
The studio's REST API is faked, so no core and no network are needed."""
import importlib.util
import json

import pytest

from app import config


@pytest.fixture()
def mcp(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("mcp_server", config.STUDIO / "scripts" / "mcp_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "DATA", tmp_path)
    calls = []
    jobs = {"j1": [{"id": "j1", "status": "running", "progress": {"frac": 0.5, "msg": "half"}, "project": "default"},
                   {"id": "j1", "status": "done", "progress": {"frac": 1.0, "msg": "done"}, "project": "default", "outputs": ["a1"]}]}
    asset = {"id": "a1", "project": "default", "type": "audio", "name": "boom", "main": "audio.wav",
             "files": [{"name": "audio.wav", "role": "main", "mime": "audio/wav", "bytes": 10}], "seed": 7}

    def fake_http(method, path, body=None, *, query=None, raw=None, ctype="", timeout=60):
        calls.append((method, path, body, query))
        if path == "/api/models":
            return {"models": [{"id": "m1", "name": "M1", "tasks": ["sfx"], "status": {"status": "ready"}, "license": {"name": "MIT", "commercial": True},
                                "params": [{"name": "seconds", "type": "float", "default": 3}]},
                               {"id": "m2", "name": "M2", "tasks": ["sfx"], "status": {"status": "missing_weights", "fix": "./install.sh model m2"},
                                "license": {"name": "NC", "commercial": False}}],
                    "tasks": {"sfx": {"label": "SFX", "modality": "audio", "studio": "sound"}}, "defaults": {"sfx": "m1"}}
        if path == "/api/jobs" and method == "POST":
            if body.get("model") == "cloudy":
                raise mod.StudioError("This job costs about $0.40.", "confirm", status=409)
            return {"id": "j1", "status": "queued"}
        if path == "/api/jobs/j1":
            return jobs["j1"].pop(0) if len(jobs["j1"]) > 1 else jobs["j1"][0]
        if path == "/api/assets":
            return {"total": 1, "items": [asset]}
        raise mod.StudioError(f"Not found: {path}", status=404)

    monkeypatch.setattr(mod, "http", fake_http)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    mod._calls = calls
    return mod


def rpc(mod, method, params=None, mid=1):
    return mod.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def text(res):
    return json.loads(res["result"]["content"][0]["text"])


def test_initialize_and_tool_schemas(mcp):
    r = rpc(mcp, "initialize", {"protocolVersion": "2025-06-18"})["result"]
    assert r["protocolVersion"] == "2025-06-18" and r["capabilities"]["tools"]
    tools = rpc(mcp, "tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"list_models", "submit_job", "get_job", "cancel_job", "import_file", "search_assets"} <= names
    for t in tools:
        assert "fn" not in t and t["inputSchema"]["type"] == "object" and t["description"]
        json.dumps(t)
    assert mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert rpc(mcp, "nope")["error"]["code"] == -32601


def test_list_models_ready_only_and_status_object(mcp):
    d = text(rpc(mcp, "tools/call", {"name": "list_models", "arguments": {"task": "sfx"}}))
    assert [m["id"] for m in d["models"]] == ["m1"]
    assert d["models"][0]["status"] == "ready" and d["models"][0]["params"][0]["name"] == "seconds"
    assert d["tasks"]["sfx"]["default"] == "m1"
    d = text(rpc(mcp, "tools/call", {"name": "list_models", "arguments": {"task": "sfx", "ready_only": False}}))
    m2 = [m for m in d["models"] if m["id"] == "m2"][0]
    assert m2["fix"] == "./install.sh model m2" and "non-commercial" in m2["license"]


def test_submit_job_waits_and_returns_assets_with_paths(mcp):
    f = mcp.DATA / "projects" / "default" / "assets" / "a1" / "audio.wav"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    sent = []
    mcp.send = sent.append
    res = rpc(mcp, "tools/call", {"name": "submit_job", "arguments": {"task": "sfx", "params": {"prompt": "boom"}, "inputs": [{"asset": "a0", "role": "ref"}]},
                                  "_meta": {"progressToken": "t"}})
    d = text(res)
    assert not res["result"]["isError"] and d["status"] == "done"
    assert d["assets"][0]["files"][0]["path"] == str(f) and d["assets"][0]["files"][0]["url"].endswith("/files/default/a1/audio.wav")
    body = [c for c in mcp._calls if c[0] == "POST"][0][2]
    assert body["meta"]["studio"] == "sound" and body["meta"]["prompt"] == "boom" and body["inputs"] == [{"asset": "a0", "role": "ref"}]
    assert body["confirm_cost"] is False and "model" not in body
    assert any(m["method"] == "notifications/progress" and m["params"]["progressToken"] == "t" for m in sent)


def test_errors_are_tool_errors_with_hints(mcp):
    res = rpc(mcp, "tools/call", {"name": "submit_job", "arguments": {"task": "sfx", "model": "cloudy", "params": {}}})
    assert res["result"]["isError"] and text(res)["kind"] == "confirm" and "confirm_cost" in text(res)["hint"]
    res = rpc(mcp, "tools/call", {"name": "submit_job", "arguments": {}})
    assert res["result"]["isError"]
    res = rpc(mcp, "tools/call", {"name": "get_model", "arguments": {}})
    assert res["result"]["isError"] and "missing argument" in res["result"]["content"][0]["text"]
    assert rpc(mcp, "tools/call", {"name": "bogus"})["result"]["isError"]


def test_unreachable_studio_message(mcp, monkeypatch):
    spec = importlib.util.spec_from_file_location("mcp_server2", config.STUDIO / "scripts" / "mcp_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "API", "http://127.0.0.1:9")
    res = mod.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "studio_status", "arguments": {}}})
    assert res["result"]["isError"] and "run_studio.sh" in text(res)["error"]


def test_http_transport_plugin():
    """app/plugins/mcp.py: the same server over Streamable HTTP, mounted by the core at /api/x/mcp."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.plugins import mcp as plugin

    app = FastAPI()
    app.include_router(plugin.router, prefix="/api/x/mcp")
    c = TestClient(app)
    u = "/api/x/mcp"
    r = c.post(u, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
               headers={"Accept": "application/json, text/event-stream"})
    assert r.status_code == 200 and r.json()["result"]["serverInfo"]["name"] == "open-asset-studio"
    assert c.post(u, json={"jsonrpc": "2.0", "method": "notifications/initialized"}).status_code == 202
    names = {t["name"] for t in c.post(u, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).json()["result"]["tools"]}
    assert "submit_job" in names
    assert c.get(u).status_code == 405
    assert c.post(u, json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"}, headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.post(u, json={"jsonrpc": "2.0", "id": 4, "method": "tools/list"}, headers={"Origin": "http://localhost:5173"}).status_code == 200
    assert c.post(u, content=b"{nope", headers={"Content-Type": "application/json"}).status_code == 400


def test_import_file_refused_for_remote_http_callers(mcp, tmp_path):
    f = tmp_path / "x.png"
    f.write_bytes(b"png")
    res = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "import_file", "arguments": {"path": str(f)}}},
                     allow_paths=False)
    assert res["result"]["isError"] and "network" in text(res)["error"]


def test_error_extra_fields_reach_the_client(mcp, monkeypatch):
    def boom(*a, **k):
        raise mcp.StudioError("costs money", "confirm", status=409, extra={"estimate": {"usd": 0.4}})
    monkeypatch.setattr(mcp, "http", boom)
    res = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "submit_job", "arguments": {"tool": "x"}}})
    assert text(res)["estimate"] == {"usd": 0.4} and "confirm_cost" in text(res)["hint"]
