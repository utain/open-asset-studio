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


# ------------------------------------------------------------------ tools added for the full studio surface
def call(mod, name, args=None, **kw):
    res = mod.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args or {}}}, **kw)["result"]
    return res, json.loads(res["content"][0]["text"])


def fake(mod, monkeypatch, routes):
    """Replace mod.http with a router: routes maps (METHOD, path) -> value | callable(body, query); records every call."""
    seen = []

    def fake_http(method, path, body=None, *, query=None, raw=None, ctype="", timeout=60):
        seen.append((method, path, body, query))
        r = routes.get((method, path))
        if r is None:
            raise mod.StudioError(f"Not found: {method} {path}", status=404)
        return r(body, query) if callable(r) else r

    monkeypatch.setattr(mod, "http", fake_http)
    return seen


RIGGED = {"id": "r1", "project": "default", "type": "mesh", "name": "knight (rigged)", "main": "mesh.glb", "task": "rig",
          "files": [{"name": "mesh.glb", "role": "main"}, {"name": "skeleton.json", "role": "attachment"}, {"name": "rigged_plain.glb", "role": "attachment"}],
          "info": {"vertices": 2628, "faces": 3200}, "worker": {"rigged": True, "joints": 34, "animations": ["flex", "wiggle"]}}
PLAIN = {"id": "m1", "project": "default", "type": "mesh", "name": "knight", "main": "mesh.glb", "files": [{"name": "mesh.glb", "role": "main"}],
         "worker": {"textured": True}}


def test_tool_count_stays_reasonable_and_names_unique(mcp):
    names = [t["name"] for t in mcp.TOOLS]
    assert len(names) == len(set(names)) <= 24
    assert {"retry_job", "delete_asset", "improve_prompt", "save_preset", "export_assets", "studio_info"} <= set(names)
    desc = {t["name"]: t["description"] for t in mcp.TOOLS}
    assert "rig" in desc["list_models"] and "motion" in desc["list_models"] and "mesh.flex_preview" in desc["submit_job"]


def test_get_asset_surfaces_3d_info_and_lineage(mcp, monkeypatch):
    lin = {"ancestors": [{"id": "m1", "type": "mesh", "name": "knight", "task": "texture", "model": "x", "created": "t"}, {"id": "gone", "missing": True}],
           "children": [], "siblings": []}
    fake(mcp, monkeypatch, {("GET", "/api/assets/r1"): RIGGED, ("GET", "/api/assets/m1"): PLAIN, ("GET", "/api/assets/r1/lineage"): lin})
    res, d = call(mcp, "get_asset", {"id": "r1", "lineage": True})
    m = d["mesh3d"]
    assert not res["isError"] and m["rigged"] is True and m["joints"] == 34 and m["animations"] == ["flex", "wiggle"]
    assert set(m["attachments"]) == {"skeleton.json", "rigged_plain.glb"} and "hint" not in m
    assert d["lineage"]["ancestors"][0] == {"id": "m1", "type": "mesh", "name": "knight", "task": "texture", "model": "x"}
    assert d["lineage"]["ancestors"][1] == {"id": "gone", "missing": True} and d["lineage"]["children"] == []
    _, d = call(mcp, "get_asset", {"id": "m1"})
    assert d["mesh3d"]["rigged"] is False and d["mesh3d"]["textured"] is True and "rig" in d["mesh3d"]["hint"] and "lineage" not in d


def test_search_assets_mesh_has_compact_3d_info(mcp, monkeypatch):
    fake(mcp, monkeypatch, {("GET", "/api/assets"): {"total": 1, "items": [RIGGED]}})
    _, d = call(mcp, "search_assets", {"type": "mesh"})
    assert d["assets"][0]["mesh3d"]["animations"] == ["flex", "wiggle"] and "hint" not in d["assets"][0]["mesh3d"]


def test_list_tools_describes_params(mcp, monkeypatch):
    tools = {"mesh.flex_preview": "Animate preview", "audio.normalize": "Normalize", "future.tool": "New"}
    fake(mcp, monkeypatch, {("GET", "/api/bootstrap"): {"tools": tools}})
    _, d = call(mcp, "list_tools")
    by = {t["id"]: t for t in d["tools"]}
    assert set(by["mesh.flex_preview"]["params"]) == {"clips", "angle"} and "RIGGED" in by["mesh.flex_preview"]["in"]
    assert "lufs" in by["audio.normalize"]["params"] and "note" in by["future.tool"]
    for t in mcp.TOOL_PARAMS:                        # every documented tool must exist as a real handler id (checked below)
        assert "params" in mcp.TOOL_PARAMS[t] and "in" in mcp.TOOL_PARAMS[t]


def test_get_rig_reads_the_skeleton_and_documents_the_edit_flow(mcp, monkeypatch):
    sk = {"asset": "r1", "name": "Soldier", "needs_reskin": True, "count": 2, "height": 1.8, "bounds": [[0, 0, 0], [1, 1, 1]], "units": "m",
          "animations": [], "zero_weight": ["tail_01"], "humanoid": {"ok": False, "reason": "no arms"},
          "joints": [{"index": 0, "name": "hips", "parent": -1, "position": [0, 1, 0], "role": "hips", "children": 1, "weight_share": 0.9},
                     {"index": 1, "name": "tail_01", "parent": 0, "position": [0, 1, -0.1], "role": None, "children": 0, "weight_share": 0.0}]}
    seen = fake(mcp, monkeypatch, {("GET", "/api/x/rig/r1/skeleton"): sk})
    res, d = call(mcp, "get_rig", {"asset": "r1"})
    assert not res["isError"] and seen[0][:2] == ("GET", "/api/x/rig/r1/skeleton")
    assert d["count"] == 2 and d["joints"][1]["name"] == "tail_01" and d["zero_weight"] == ["tail_01"] and d["needs_reskin"] is True
    assert "reskin" in d["hint"] and "rig.edit" in d["hint"]
    ops = mcp.TOOL_PARAMS["rig.edit"]
    assert "add_chain" in ops["params"]["ops"] and "reskin" in ops["note"]
    assert "rig.edit" in {t["name"]: t for t in mcp.TOOLS}["submit_job"]["description"]


def test_tool_params_table_matches_registered_handlers():
    from app.jobs import HANDLERS
    from app.plugins import core_tools, mesh, rig, sound, video  # noqa: F401  (importing registers the handlers)

    import importlib.util
    spec = importlib.util.spec_from_file_location("mcp_server3", config.STUDIO / "scripts" / "mcp_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    registered = {k[5:] for k in HANDLERS if k.startswith("tool:")}
    assert set(mod.TOOL_PARAMS) == registered


def test_delete_asset_needs_explicit_confirm(mcp, monkeypatch):
    seen = fake(mcp, monkeypatch, {("GET", "/api/assets/m1"): PLAIN, ("GET", "/api/assets/m1/lineage"): {"children": [{"id": "r1"}]},
                                   ("DELETE", "/api/assets/m1"): {"id": "m1", "deleted": True}})
    for args in ({"id": "m1"}, {"id": "m1", "confirm": False}, {"id": "m1", "confirm": "yes"}):
        res, d = call(mcp, "delete_asset", args)
        assert res["isError"] and d["kind"] == "confirm_delete" and "confirm: true" in d["fix"] and "1 derived" in d["error"]
    assert not [c for c in seen if c[0] == "DELETE"]
    res, d = call(mcp, "delete_asset", {"id": "m1", "confirm": True})
    assert not res["isError"] and d["deleted"] is True and [c for c in seen if c[0] == "DELETE"]


def test_retry_job_passes_consent_and_waits(mcp, monkeypatch):
    done = {"id": "j2", "status": "done", "project": "default", "outputs": ["a1"]}
    seen = fake(mcp, monkeypatch, {("POST", "/api/jobs/j1/retry"): {"id": "j2", "status": "queued"}, ("GET", "/api/jobs/j2"): done,
                                   ("GET", "/api/assets"): {"items": [PLAIN]}})
    res, d = call(mcp, "retry_job", {"id": "j1", "confirm_cost": True})
    assert not res["isError"] and d["status"] == "done" and d["assets"][0]["id"] == "m1"
    assert seen[0][2] == {"confirm_cost": True, "confirm_upload": False}


def test_improve_prompt(mcp, monkeypatch):
    seen = fake(mcp, monkeypatch, {("POST", "/api/improve"): {"prompt": "a knight", "negative": "", "variations": []},
                                   ("GET", "/api/improve/styles"): {"styles": [{"id": "qwen_image", "label": "Q", "modality": "image", "chips": [{"id": "lowpoly", "label": "Low-poly", "hint": "x"}]}]}})
    _, d = call(mcp, "improve_prompt", {"list_styles": True})
    assert d["styles"][0]["chips"] == [{"id": "lowpoly", "label": "Low-poly"}]
    _, d = call(mcp, "improve_prompt", {"text": "knight", "model_id": "qwen-image-2.1", "task": "t2i", "chips": ["lowpoly"], "n": 2})
    body = seen[-1][2]
    assert d["prompt"] == "a knight" and body == {"text": "knight", "model_id": "qwen-image-2.1", "task": "t2i", "chips": ["lowpoly"], "n": 2}
    res, d = call(mcp, "improve_prompt", {})
    assert res["isError"] and d["kind"] == "input"
    res, d = call(mcp, "improve_prompt", {"text": "knight", "backend": "api:openrouter/x"})      # off-machine backend: needs consent
    assert res["isError"] and d["kind"] == "consent" and len(seen) == 2
    res, _ = call(mcp, "improve_prompt", {"text": "knight", "backend": "api:openrouter/x", "confirm_upload": True})
    assert not res["isError"] and seen[-1][2]["backend"] == "api:openrouter/x"


def test_save_preset_and_confirmed_delete(mcp, monkeypatch):
    seen = fake(mcp, monkeypatch, {("PUT", "/api/presets/m1"): {"builtin": {}, "user": {"mine": {"steps": 9}}},
                                   ("DELETE", "/api/presets/m1/mine"): {"builtin": {}, "user": {}}})
    _, d = call(mcp, "save_preset", {"model": "m1", "name": "mine", "params": {"steps": 9}})
    assert d["user"]["mine"] == {"steps": 9} and seen[-1][2] == {"name": "mine", "params": {"steps": 9}}
    assert call(mcp, "save_preset", {"model": "m1", "name": "mine"})[1]["kind"] == "input"
    res, d = call(mcp, "save_preset", {"model": "m1", "name": "mine", "delete": True})
    assert res["isError"] and d["kind"] == "confirm_delete" and seen[-1][0] == "PUT"
    res, d = call(mcp, "save_preset", {"model": "m1", "name": "mine", "delete": True, "confirm": True})
    assert not res["isError"] and seen[-1][0] == "DELETE"


def test_get_model_shows_user_presets(mcp, monkeypatch):
    fake(mcp, monkeypatch, {("GET", "/api/models/m1"): {"id": "m1", "name": "M1", "tasks": ["t2i"], "status": {"status": "ready"}, "license": {"name": "MIT", "commercial": True},
                                                       "presets": {"fast": {"steps": 8}}, "user_presets": {"mine": {"steps": 9}}, "params": []}})
    _, d = call(mcp, "get_model", {"id": "m1"})
    assert d["presets"] == {"fast": {"steps": 8}} and d["user_presets"] == {"mine": {"steps": 9}}


def test_export_assets_modes(mcp, monkeypatch, tmp_path):
    (tmp_path / "exports").mkdir()
    (tmp_path / "exports" / "export-20260101-000000-abcdef.zip").write_bytes(b"z")
    (tmp_path / "exports" / "render-0123456789ab").mkdir()
    (tmp_path / "exports" / "render-0123456789ab" / "boom.mp3").write_bytes(b"m")
    seen = fake(mcp, monkeypatch, {
        ("POST", "/api/export"): {"url": "/api/export/export-20260101-000000-abcdef.zip", "name": "export-20260101-000000-abcdef.zip", "bytes": 1, "count": 2},
        ("POST", "/api/assets/a1/render"): {"url": "/api/export/render/0123456789ab/boom.mp3", "name": "boom.mp3", "bytes": 1, "report": {}, "asset": "a1"},
        ("POST", "/api/x/sound/export"): {"url": "/api/x/sound/export/sound-export-20260101-000000-abcdef.zip", "name": "sound-export-20260101-000000-abcdef.zip", "count": 2},
        ("GET", "/api/assets/m1"): PLAIN})
    _, d = call(mcp, "export_assets", {"ids": ["a1", "a2"], "preset": "podcast"})
    assert d["url"] == mcp.API + "/api/export/export-20260101-000000-abcdef.zip" and d["path"].endswith("export-20260101-000000-abcdef.zip")
    assert seen[-1][2]["export"] == {"preset": "podcast"} and seen[-1][2]["include_attachments"] is True
    _, d = call(mcp, "export_assets", {"ids": ["a1"], "preset": "podcast"})
    assert seen[-1][1] == "/api/assets/a1/render" and seen[-1][2] == {"export": {"preset": "podcast"}}
    assert d["path"] == str(tmp_path / "exports" / "render-0123456789ab" / "boom.mp3") and d["report"] == {}
    _, d = call(mcp, "export_assets", {"ids": ["a1", "a2"], "game_audio": {"format": "ogg", "folder": "sfx"}})
    assert seen[-1][1] == "/api/x/sound/export" and seen[-1][2] == {"ids": ["a1", "a2"], "format": "ogg", "folder": "sfx"} and "path" not in d
    _, d = call(mcp, "export_assets", {"ids": ["m1"]})                       # one asset, no preset: just its files
    assert d["id"] == "m1" and d["files"][0]["url"].endswith("/files/default/m1/mesh.glb")
    assert call(mcp, "export_assets", {"ids": []})[0]["isError"]


def test_studio_info_topics(mcp, monkeypatch):
    seen = fake(mcp, monkeypatch, {
        ("GET", "/api/profiles"): {"current": "game", "default_export": "original", "license": {"commercial_only": False},
                                   "profiles": [{"id": "game", "label": "Game", "summary": "s", "export": "original", "export_presets": ["original"], "hints": {"x": 1}}],
                                   "export_presets": [{"id": "original", "label": "Original files"}]},
        ("GET", "/api/x/video/formats"): {"aspects": {"16:9": [1920, 1080]}},
        ("GET", "/api/x/sound/voices"): {"total": 0, "items": []},
        ("GET", "/api/x/media_audio/dub/options"): {"tts_models": []},
        ("GET", "/api/x/pipelines/capabilities"): {"tasks": {"t2i": {"ready": ["m1"]}}},
        ("GET", "/api/x/pipelines/flows"): {"flows": [{"id": "f1"}]},
        ("GET", "/api/x/pipelines/flows/f1"): {"id": "f1", "state": {}},
        ("GET", "/api/x/providers"): {"allow_cloud": False, "budget_usd": 20, "confirm_usd": 0.5, "spend": {"total_usd": 0},
                                      "providers": [{"id": "fal", "label": "fal.ai", "key": {"set": False, "last4": None}, "modalities": ["image"], "models": 2}],
                                      "models": [{"id": "fal-x", "name": "X", "tasks": ["t2i"], "status": {"status": "cloud_disabled"}, "example_usd": 0.04}]},
        ("GET", "/api/improve/status"): {"state": "stopped"}, ("GET", "/api/x/image/comfyui"): {"reachable": False}, ("GET", "/api/downloads"): {"downloads": []},
        ("GET", "/api/system/logs"): {"name": "studio", "lines": ["a"], "available": ["studio"], "bytes": 1}})
    _, d = call(mcp, "studio_info")
    assert {"profiles", "voices", "video_formats", "dub_options", "cloud", "flows", "logs"} <= set(d["topics"])
    _, d = call(mcp, "studio_info", {"topic": "profiles"})
    assert d["current"] == "game" and d["export_presets"][0]["id"] == "original" and "hints" not in d["profiles"][0]
    assert call(mcp, "studio_info", {"topic": "video_formats"})[1]["aspects"]["16:9"] == [1920, 1080]
    assert call(mcp, "studio_info", {"topic": "voices", "project": "p"})[1]["total"] == 0 and seen[-1][3] == {"project": "p"}
    assert call(mcp, "studio_info", {"topic": "dub_options"})[1] == {"tts_models": []}
    assert call(mcp, "studio_info", {"topic": "ready_tasks"})[1]["tasks"]["t2i"]["ready"] == ["m1"]
    d = call(mcp, "studio_info", {"topic": "flows"})[1]
    assert d["flows"] == [{"id": "f1"}] and "t2i" in d["ready_tasks"]
    assert call(mcp, "studio_info", {"topic": "flows", "id": "f1"})[1]["id"] == "f1"
    d = call(mcp, "studio_info", {"topic": "cloud"})[1]
    assert d["providers"][0]["key_set"] is False and d["models"][0]["status"] == "cloud_disabled" and "key" not in d["providers"][0]
    assert call(mcp, "studio_info", {"topic": "llm"})[1]["state"] == "stopped"
    assert call(mcp, "studio_info", {"topic": "comfyui"})[1]["reachable"] is False
    assert call(mcp, "studio_info", {"topic": "downloads"})[1] == {"downloads": []}
    call(mcp, "studio_info", {"topic": "logs", "lines": 5000})
    assert seen[-1][3]["lines"] == 300


def test_create_project_update_mode(mcp, monkeypatch):
    seen = fake(mcp, monkeypatch, {("PATCH", "/api/projects/p1"): {"id": "p1", "title": "New"}, ("POST", "/api/projects"): {"id": "p1"}})
    _, d = call(mcp, "create_project", {"id": "p1", "title": "New", "update": True})
    assert d["title"] == "New" and seen[-1][:3] == ("PATCH", "/api/projects/p1", {"title": "New"})
    call(mcp, "create_project", {"id": "p1"})
    assert seen[-1][0] == "POST"


def test_assistant_mcp_stays_read_propose_only(monkeypatch):
    """workers/studio_mcp.py: the Assistant's server keeps its 5 tools (no submit / delete), knows rig + motion, and its get_asset shows 3D info."""
    import sys
    sys.path.insert(0, str(config.STUDIO / "workers"))
    try:
        import studio_mcp
    finally:
        sys.path.pop(0)
    from app.plugins import assistant
    assert {t["name"] for t in studio_mcp.TOOLS} == {"list_models", "search_assets", "get_asset", "get_rig", "propose_plan"}
    step = [t for t in studio_mcp.TOOLS if t["name"] == "propose_plan"][0]["inputSchema"]["properties"]["steps"]["items"]["properties"]
    assert "rig" in step["task"]["description"] and "motion" in step["task"]["description"] and "reskin" in step["task"]["description"]

    class Store:
        def get(self, aid):
            return RIGGED

    class Core:
        assets = Store()

    monkeypatch.setattr(assistant, "CORE", Core())
    d = assistant.tool_get_asset("default", {"id": "r1"})
    assert d["mesh3d"] == {"rigged": True, "joints": 34, "animations": ["flex", "wiggle"], "textured": None}


def test_proxied_callers_are_not_local():
    """`tailscale serve` / nginx connect from 127.0.0.1 but forward other machines: they must not get import_file."""
    from starlette.requests import Request

    from app.plugins import mcp as plugin

    def req(headers):
        return Request({"type": "http", "client": ("127.0.0.1", 5), "headers": [(k.encode(), v.encode()) for k, v in headers.items()]})
    assert plugin._is_local(req({}))
    assert not plugin._is_local(req({"x-forwarded-for": "203.0.113.7"}))
    assert not plugin._is_local(req({"tailscale-user-login": "someone@example.com"}))
