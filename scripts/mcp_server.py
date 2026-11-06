#!/usr/bin/env python3
"""Open Asset Studio as an MCP server (stdio) for Claude Code, Claude Desktop or any MCP client.

  claude mcp add studio -- python3 /path/to/open-asset-studio/scripts/mcp_server.py     # or the repo's .mcp.json
  STUDIO_URL=http://127.0.0.1:7860   (default; the core must be running: ./run_studio.sh)

The same tools are served over HTTP by the core itself (app/plugins/mcp.py, http://127.0.0.1:7860/api/x/mcp), so they
are up whenever the studio is; this file is the stdio transport for clients that cannot use HTTP.

Unlike workers/studio_mcp.py (the in-app Assistant's restricted, token-gated server), this one can SUBMIT jobs. It is a
thin client of the public REST API (docs/CONTRACTS.md §3-4), so every job goes through the same queue, GPU lock, memory
checks and "Commercial-safe models only" filter as the web UI. Cloud jobs that cost money or upload files are refused
until the caller passes confirm_cost / confirm_upload, which should only happen after the user agreed.

Stdlib only (runs with any python3 >= 3.9). Newline-delimited JSON-RPC 2.0 on stdin/stdout; logs go to stderr.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

PROTOCOL = "2025-06-18"
API = os.environ.get("STUDIO_URL", "http://127.0.0.1:7860").rstrip("/")
REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("STUDIO_DATA", REPO / "data"))   # for local file paths of assets (same machine only)
TERMINAL = {"done", "failed", "cancelled"}
MAX_WAIT_S = 600


class StudioError(Exception):
    def __init__(self, message: str, kind: str = "", fix: str = "", status: int = 0, extra: dict | None = None):
        super().__init__(message)
        self.kind, self.fix, self.status, self.extra = kind, fix, status, extra or {}


# ------------------------------------------------------------------ HTTP
def http(method: str, path: str, body=None, *, query: dict | None = None, raw: bytes | None = None,
         ctype: str = "application/json", timeout: float = 60):
    url = API + path
    if query:
        q = {k: (",".join(v) if isinstance(v, list) else str(v).lower() if isinstance(v, bool) else v)
             for k, v in query.items() if v not in (None, "", [])}
        if q:
            url += "?" + urllib.parse.urlencode(q)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": ctype} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (STUDIO_URL is set by the user)
            return json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        try:
            d = json.loads(e.read().decode())
        except ValueError:
            d = {}
        err = d.get("error") or d.get("detail") or str(e)
        if isinstance(err, dict):
            d, err = {**d, **err}, err.get("message") or str(err)
        extra = {k: v for k, v in d.items() if k not in ("error", "detail", "kind", "fix", "message")}
        raise StudioError(str(err), d.get("kind", ""), d.get("fix", ""), e.code, extra) from None
    except (urllib.error.URLError, OSError) as e:
        raise StudioError(f"Open Asset Studio is not reachable at {API} ({e}). Start it with ./run_studio.sh "
                          "(or set STUDIO_URL).", "unreachable") from None


def multipart(fields: dict, file_field: str, path: Path) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{path.name}"\r\n'
                 f"Content-Type: {mime}\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={b}"


# ------------------------------------------------------------------ shaping results for the model
def asset_view(a: dict, full: bool = False) -> dict:
    """An asset with its file URLs and, when the studio runs on this machine, absolute local paths."""
    files = []
    for f in a.get("files") or []:
        name = f.get("name")
        url = f"{API}/files/{a.get('project')}/{a.get('id')}/{urllib.parse.quote(name)}"
        p = DATA / "projects" / str(a.get("project")) / "assets" / str(a.get("id")) / name
        files.append({"name": name, "role": f.get("role"), "mime": f.get("mime"), "bytes": f.get("bytes"), "url": url,
                      **({"path": str(p)} if p.is_file() else {})})
    keys = ["id", "project", "type", "name", "main", "model", "task", "prompt", "seed", "tags", "rating", "created", "license", "info"]
    if full:
        keys += ["params", "improved_prompt", "final_prompt", "negative", "parents", "job", "studio", "notes", "source", "worker"]
    out = {k: a[k] for k in keys if a.get(k) not in (None, "", [], {})}
    out["files"] = files
    return out


def job_view(j: dict) -> dict:
    keys = ["id", "status", "task", "model", "tool", "title", "project", "queue_position", "progress", "error", "outputs", "timings", "created"]
    out = {k: j[k] for k in keys if j.get(k) not in (None, "", [])}
    if j.get("status") != "done":
        out["log_tail"] = (j.get("log") or [])[-5:]
    return out


def status_of(m: dict) -> str:
    st = m.get("status")
    return st.get("status", "") if isinstance(st, dict) else str(st or "")


def model_view(m: dict, detail: bool) -> dict:
    out = {k: m.get(k) for k in ("id", "name", "modality", "tasks", "summary", "speed", "memory_gb") if m.get(k) not in (None, "", [])}
    st = m.get("status") if isinstance(m.get("status"), dict) else {"status": m.get("status")}
    out["status"] = st.get("status")
    lic = m.get("license") or {}
    out["license"] = f"{lic.get('name', '?')} ({'commercial OK' if lic.get('commercial') else 'non-commercial'}{', verify' if lic.get('verify') else ''})"
    if out["status"] != "ready":
        out.update({k: st[k] for k in ("reason", "fix") if st.get(k)})
    if m.get("backend") == "cloud":
        out["cloud"] = True
    if detail:
        out["params"] = [{k: p[k] for k in ("name", "type", "default", "min", "max", "choices", "unit", "label") if k in p}
                         for p in m.get("params") or []]
        if m.get("presets"):
            out["presets"] = m["presets"]
        if m.get("input_slots"):
            out["input_slots"] = m["input_slots"]
        if m.get("features"):
            out["features"] = m["features"]
    return out


# ------------------------------------------------------------------ tools
def t_status(_a, _ctx):
    sysinfo = http("GET", "/api/system")
    mem = sysinfo.get("memory") or {}
    jobs = http("GET", "/api/jobs", query={"status": "running,queued", "limit": 20})
    settings = http("GET", "/api/settings")
    return {"studio": API, "health": http("GET", "/api/health"), "queue": jobs.get("counts"),
            "active_jobs": [job_view(j) for j in jobs.get("jobs", [])],
            "memory": {k: mem.get(k) for k in ("available_gb", "total_gb", "ours_gb")},
            "loaded_models": [x.get("label") for x in mem.get("loaded", [])],
            "profile": settings.get("profile"), "license_filter": settings.get("license")}


def t_list_models(a, _ctx):
    d = http("GET", "/api/models", query={"task": a.get("task"), "modality": a.get("modality")})
    models = d.get("models", [])
    if a.get("ready_only", True):
        models = [m for m in models if status_of(m) == "ready"]
    if a.get("commercial_only"):
        models = [m for m in models if (m.get("license") or {}).get("commercial")]
    detail = bool(a.get("task")) and a.get("detail", len(models) <= 6)
    tasks = {k: {"label": v.get("label"), "modality": v.get("modality"), "studio": v.get("studio"), "default": d["defaults"].get(k)}
             for k, v in (d.get("tasks") or {}).items() if not a.get("task") or k == a["task"]}
    return {"tasks": tasks, "models": [model_view(m, detail) for m in models],
            "hint": "get_model(id) shows a model's params, presets and input slots; the task default is used when model is omitted."}


def t_get_model(a, _ctx):
    return model_view(http("GET", f"/api/models/{urllib.parse.quote(a['id'])}"), True)


def t_list_tools(_a, _ctx):
    tools = http("GET", "/api/bootstrap").get("tools") or {}
    return {"tools": tools, "hint": "Run one with submit_job(tool=<id>, params=..., inputs=[{asset: <id>}]). "
                                    "Params per tool: docs/CONTRACTS.md and app/plugins/*.py."}


def t_list_projects(_a, _ctx):
    return http("GET", "/api/projects")


def t_create_project(a, _ctx):
    return http("POST", "/api/projects", {"id": a["id"], "title": a.get("title") or a["id"], "description": a.get("description", "")})


def t_search_assets(a, _ctx):
    d = http("GET", "/api/assets", query={"project": a.get("project", "default"), "q": a.get("q"), "type": a.get("type"),
                                          "tag": a.get("tag"), "model": a.get("model"), "task": a.get("task"),
                                          "sort": a.get("sort", "new"), "limit": min(int(a.get("limit", 20)), 100)})
    return {"total": d.get("total"), "assets": [asset_view(x) for x in d.get("items", [])]}


def t_get_asset(a, _ctx):
    asset = http("GET", f"/api/assets/{urllib.parse.quote(a['id'])}")
    out = [{"type": "text", "text": json.dumps(asset_view(asset, True), ensure_ascii=False)}]
    if a.get("preview"):
        out += preview_blocks(asset)
    return out


def preview_blocks(asset: dict) -> list:
    """The image (or its thumbnail) as MCP image content, so the client can look at it. Images only, max 4 MB."""
    names = [(asset.get("preview") or {}).get("thumb")] if asset.get("type") != "image" else [asset.get("main"), (asset.get("preview") or {}).get("thumb")]
    for name in filter(None, names):
        url = f"/files/{asset['project']}/{asset['id']}/{urllib.parse.quote(name)}"
        try:
            with urllib.request.urlopen(API + url, timeout=30) as r:  # noqa: S310
                blob = r.read(4_000_001)
                mime = r.headers.get_content_type()
        except (urllib.error.URLError, OSError):
            continue
        if len(blob) <= 4_000_000 and mime in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            return [{"type": "image", "data": base64.b64encode(blob).decode(), "mimeType": mime}]
    return []


def wait_job(jid: str, wait_s: float, ctx) -> dict:
    deadline = time.monotonic() + min(max(wait_s, 0), MAX_WAIT_S)
    last = None
    while True:
        j = http("GET", f"/api/jobs/{urllib.parse.quote(jid)}")
        pr = j.get("progress") or {}
        key = (j.get("status"), round(float(pr.get("frac") or 0), 2), pr.get("msg"))
        if key != last:
            last = key
            ctx.progress(float(pr.get("frac") or 0), f"{j.get('status')}: {pr.get('msg') or pr.get('stage') or ''}".strip())
        if j.get("status") in TERMINAL or time.monotonic() >= deadline:
            return j
        time.sleep(1.0)


def finished(j: dict, preview: bool) -> list:
    out = job_view(j)
    blocks = []
    if j.get("status") == "done" and j.get("outputs"):
        items = http("GET", "/api/assets", query={"ids": j["outputs"], "project": j.get("project"), "limit": 100}).get("items", [])
        out["assets"] = [asset_view(x) for x in items]
        out.pop("outputs", None)
        if preview:
            for x in items[:4]:
                blocks += preview_blocks(x)
    elif j.get("status") not in TERMINAL:
        out["hint"] = "Still running: call get_job(id, wait_s=...) to keep waiting, or cancel_job(id)."
    return [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}] + blocks


def t_submit_job(a, ctx):
    if not a.get("task") and not a.get("tool"):
        raise StudioError("Give a task (with an optional model) or a tool.", "input")
    meta = dict(a.get("meta") or {})
    meta.setdefault("tags", ["mcp"])
    if a.get("task") and "studio" not in meta:
        task = (http("GET", "/api/models", query={"task": a["task"]}).get("tasks") or {}).get(a["task"]) or {}
        if task.get("studio"):
            meta["studio"] = task["studio"]              # lets the UI's Regenerate open the right studio
    params = dict(a.get("params") or {})
    if params.get("prompt") and "prompt" not in meta:
        meta["prompt"] = params["prompt"]
    spec = {"task": a.get("task"), "model": a.get("model"), "tool": a.get("tool"), "params": params,
            "inputs": [{"asset": i["asset"], **({"role": i["role"]} if i.get("role") else {})} for i in a.get("inputs") or []],
            "project": a.get("project", "default"), "title": a.get("title", ""), "meta": meta,
            "confirm_cost": bool(a.get("confirm_cost")), "confirm_upload": bool(a.get("confirm_upload"))}
    j = http("POST", "/api/jobs", {k: v for k, v in spec.items() if v is not None})
    if a.get("wait_s", 300) > 0:
        j = wait_job(j["id"], float(a.get("wait_s", 300)), ctx)
    return finished(j, bool(a.get("preview", True)))


def t_get_job(a, ctx):
    j = wait_job(a["id"], float(a.get("wait_s", 0)), ctx) if a.get("wait_s") else http("GET", f"/api/jobs/{urllib.parse.quote(a['id'])}")
    return finished(j, bool(a.get("preview", False)))


def t_list_jobs(a, _ctx):
    d = http("GET", "/api/jobs", query={"status": a.get("status"), "project": a.get("project"), "limit": min(int(a.get("limit", 20)), 100)})
    return {"counts": d.get("counts"), "jobs": [job_view(j) for j in d.get("jobs", [])]}


def t_cancel_job(a, _ctx):
    return http("POST", f"/api/jobs/{urllib.parse.quote(a['id'])}/cancel")


def t_import_file(a, ctx):
    if not ctx.allow_paths:
        raise StudioError("import_file reads a file on the studio's machine, so it only works for MCP clients on that machine "
                          "(this request came over the network). Upload through the web UI or POST /api/upload instead.", "input")
    p = Path(a["path"]).expanduser()
    if not p.is_file():
        raise StudioError(f"No such file: {p}", "input")
    fields = {"project": a.get("project", "default"), "name": a.get("name", ""), "tags": ",".join(a.get("tags") or ["mcp"]),
              "type": "voice" if a.get("as_voice") else ""}
    body, ctype = multipart(fields, "file", p)
    return asset_view(http("POST", "/api/upload", raw=body, ctype=ctype, timeout=300), True)


def t_update_asset(a, _ctx):
    patch = {k: a[k] for k in ("name", "tags", "rating", "notes", "favorite") if k in a}
    return asset_view(http("PATCH", f"/api/assets/{urllib.parse.quote(a['id'])}", patch), True)


S, I, B, O = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}, {"type": "object"}
PROJECT = {"type": "string", "description": "project id (default: default)"}
INPUTS = {"type": "array", "description": "input assets, e.g. [{asset: '<id>', role: 'ref'}] (roles per model: get_model -> input_slots)",
          "items": {"type": "object", "properties": {"asset": S, "role": S}, "required": ["asset"]}}
RO = {"readOnlyHint": True, "openWorldHint": False}
TOOLS = [
    {"name": "studio_status", "description": "Studio health, queue (running / queued jobs), free memory, loaded models, profile and licence filter.",
     "inputSchema": {"type": "object", "properties": {}}, "annotations": RO, "fn": t_status},
    {"name": "list_models", "description": "Generation tasks (t2i, edit, rgba, shape, sfx, music, tts, voice_clone, ...) and the models for them, with status, licence and speed. Filter by task for params.",
     "inputSchema": {"type": "object", "properties": {"task": S, "modality": {"type": "string", "enum": ["image", "mesh", "audio", "video", "text"]},
                                                      "ready_only": {**B, "description": "default true"}, "commercial_only": B, "detail": B}},
     "annotations": RO, "fn": t_list_models},
    {"name": "get_model", "description": "One model: params (name, type, default, range, choices), presets, input slots, features, licence.",
     "inputSchema": {"type": "object", "properties": {"id": S}, "required": ["id"]}, "annotations": RO, "fn": t_get_model},
    {"name": "list_tools", "description": "CPU tools (audio.normalize, audio.loop, image.convert, video.trim, ...) that post-process existing assets.",
     "inputSchema": {"type": "object", "properties": {}}, "annotations": RO, "fn": t_list_tools},
    {"name": "list_projects", "description": "Projects (asset libraries).", "inputSchema": {"type": "object", "properties": {}}, "annotations": RO, "fn": t_list_projects},
    {"name": "create_project", "description": "Create a project. id: lowercase letters, digits, - and _.",
     "inputSchema": {"type": "object", "properties": {"id": S, "title": S, "description": S}, "required": ["id"]}, "fn": t_create_project},
    {"name": "search_assets", "description": "Search a project's assets (newest first). Results include file URLs and local paths.",
     "inputSchema": {"type": "object", "properties": {"project": PROJECT, "q": S, "type": {"type": "string", "description": "image | audio | voice | mesh | video | text (comma list ok)"},
                                                      "tag": S, "model": S, "task": S, "sort": {"type": "string", "enum": ["new", "old", "rating", "name"]}, "limit": I}},
     "annotations": RO, "fn": t_search_assets},
    {"name": "get_asset", "description": "One asset: prompt, model, params, seed, lineage, info (size / duration / LUFS ...), files with local paths. preview=true also returns the image.",
     "inputSchema": {"type": "object", "properties": {"id": S, "preview": B}, "required": ["id"]}, "annotations": RO, "fn": t_get_asset},
    {"name": "submit_job", "description": (
        "Generate assets: run a model on a task (or a CPU tool on existing assets) through the studio's queue. One GPU job runs at a time. "
        "Waits up to wait_s (default 300, max 600) and returns the new assets with local file paths (and image previews). "
        "Use list_models / get_model first for valid params. seed=-1 is random; the real seed is in the asset. "
        "Cloud models: only set confirm_cost / confirm_upload after the user agreed to the cost / to uploading the inputs."),
     "inputSchema": {"type": "object", "properties": {
         "task": S, "model": {**S, "description": "model id (omit = the task's default)"}, "tool": {**S, "description": "CPU tool id instead of task/model"},
         "params": {**O, "description": "model params by name, e.g. {prompt, seconds, batch, seed, aspect, steps}"}, "inputs": INPUTS,
         "project": PROJECT, "title": S, "meta": {**O, "description": "optional asset meta (tags, asset_extra: {x_*})"},
         "wait_s": {**I, "description": "seconds to wait for the result; 0 = return the queued job at once"},
         "preview": {**B, "description": "return image previews of the outputs (default true)"}, "confirm_cost": B, "confirm_upload": B},
     }, "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}, "fn": t_submit_job},
    {"name": "get_job", "description": "A job's status / progress / error, and its output assets when done. wait_s > 0 waits for it to finish.",
     "inputSchema": {"type": "object", "properties": {"id": S, "wait_s": I, "preview": B}, "required": ["id"]}, "annotations": RO, "fn": t_get_job},
    {"name": "list_jobs", "description": "Recent jobs. status: comma list of queued, running, done, failed, cancelled.",
     "inputSchema": {"type": "object", "properties": {"status": S, "project": PROJECT, "limit": I}}, "annotations": RO, "fn": t_list_jobs},
    {"name": "cancel_job", "description": "Cancel a queued or running job (cooperative: a running stage may finish first).",
     "inputSchema": {"type": "object", "properties": {"id": S}, "required": ["id"]}, "fn": t_cancel_job},
    {"name": "import_file", "description": "Add a local file (image, audio, GLB, video) to a project as an asset, e.g. to use it as a job input. as_voice=true marks audio as a voice reference.",
     "inputSchema": {"type": "object", "properties": {"path": S, "project": PROJECT, "name": S, "tags": {"type": "array", "items": S}, "as_voice": B}, "required": ["path"]},
     "fn": t_import_file},
    {"name": "update_asset", "description": "Rename, tag, rate (0-5), favourite or annotate an asset.",
     "inputSchema": {"type": "object", "properties": {"id": S, "name": S, "tags": {"type": "array", "items": S}, "rating": I, "notes": S, "favorite": B}, "required": ["id"]},
     "fn": t_update_asset},
]
BY_NAME = {t["name"]: t for t in TOOLS}


# ------------------------------------------------------------------ JSON-RPC
def send(msg: dict):
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class Ctx:
    def __init__(self, token, emit=None, allow_paths=True):
        self.token, self.emit, self.allow_paths = token, emit or send, allow_paths

    def progress(self, frac: float, msg: str):
        if self.token is not None:
            self.emit({"jsonrpc": "2.0", "method": "notifications/progress",
                       "params": {"progressToken": self.token, "progress": round(frac * 100, 1), "total": 100, "message": msg[:200]}})


def call_tool(name: str, args: dict, token=None, emit=None, allow_paths=True) -> dict:
    """Run one tool. emit(msg) receives progress notifications (default: stdout, the stdio transport).
    allow_paths=False (HTTP from another host) refuses tools that read a path on this machine."""
    t = BY_NAME.get(name)
    if not t:
        return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}
    try:
        res = t["fn"](args, Ctx(token, emit, allow_paths))
    except StudioError as e:
        err = {"error": str(e), **({"kind": e.kind} if e.kind else {}), **({"fix": e.fix} if e.fix else {}), **e.extra}
        if e.kind in ("confirm", "consent"):
            err["hint"] = "Ask the user, then call again with confirm_cost / confirm_upload = true."
        return {"content": [{"type": "text", "text": json.dumps(err, ensure_ascii=False)}], "isError": True}
    except KeyError as e:
        return {"content": [{"type": "text", "text": f"missing argument {e}"}], "isError": True}
    except Exception as e:  # noqa: BLE001  (never kill the server on one bad call)
        print(f"studio-mcp: {name} failed: {e!r}", file=sys.stderr)
        return {"content": [{"type": "text", "text": f"internal error: {e}"}], "isError": True}
    blocks = res if isinstance(res, list) else [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}]
    return {"content": blocks, "isError": False}


def handle(msg: dict, emit=None, allow_paths=True) -> dict | None:
    mid, method = msg.get("id"), msg.get("method")
    if mid is None:
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": (msg.get("params") or {}).get("protocolVersion") or PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "open-asset-studio", "version": "1.0"},
            "instructions": ("Open Asset Studio makes images, 3D models, sound and video with local open-weight models. "
                             "Typical flow: list_models(task=...) -> get_model(id) -> submit_job(...) -> use the returned file paths. "
                             "Check each model's licence before commercial use; many are non-commercial.")}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [{k: v for k, v in t.items() if k != "fn"} for t in TOOLS]}}
    if method == "tools/call":
        p = msg.get("params") or {}
        token = (p.get("_meta") or {}).get("progressToken")
        return {"jsonrpc": "2.0", "id": mid, "result": call_tool(p.get("name"), p.get("arguments") or {}, token, emit, allow_paths)}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        out = handle(msg)
        if out is not None:
            send(out)


if __name__ == "__main__":
    main()
