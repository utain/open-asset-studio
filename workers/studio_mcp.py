#!/usr/bin/env python3
"""Tiny stdio MCP server that gives the Assistant (claude -p) the studio's restricted tool set -- and nothing else.

Tools (all read-only, except propose_plan, which only *records* a proposal; nothing here can submit a job):
  list_models    models with status, cost, licence, tasks and their params
  search_assets  list / search assets of the current project
  get_asset      one asset's meta
  propose_plan   a structured plan (steps of task / model / params / inputs), validated by the studio

Every call is forwarded to the studio core: POST $STUDIO_API/api/x/assistant/tools/<name> with the run token in
X-Studio-Token (the core only accepts tokens of a live assistant run). Stdlib only, newline-delimited JSON-RPC 2.0
(MCP stdio transport). Launched by the core through `claude --mcp-config ... --strict-mcp-config`.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

PROTOCOL = "2025-06-18"
API = os.environ.get("STUDIO_API", "http://127.0.0.1:7860").rstrip("/")
TOKEN = os.environ.get("STUDIO_MCP_TOKEN", "")

STEP = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "short step id, e.g. s1, props_img, amb"},
        "title": {"type": "string"},
        "task": {"type": "string", "description": "registry task id (t2i, rgba, shape, texture, rig, motion, sfx, ambience, music, tts, ...; list_models shows them all)"},
        "tool": {"type": "string", "description": "instead of task+model: a CPU tool id such as audio.loop or audio.normalize"},
        "model": {"type": "string", "description": "model id from list_models (omit = the task's default)"},
        "params": {"type": "object", "description": "model params by name (prompt, negative, seconds, batch, seed, ...)"},
        "inputs": {"type": "array", "items": {"type": "object", "properties": {
            "step": {"type": "string", "description": "id of an earlier step whose outputs feed this one"},
            "asset": {"type": "string", "description": "existing asset id"},
            "use": {"type": "string", "enum": ["first", "all"], "description": "which outputs of the step (default first)"},
            "role": {"type": "string"}}}},
        "notes": {"type": "string"},
    },
    "required": ["id", "title", "params"],
}
TOOLS = [
    {"name": "list_models", "description": "List the studio's generation models with status (ready or not), est. time, cloud cost, licence (commercial or not), tasks and parameters. Filter by task or modality.",
     "inputSchema": {"type": "object", "properties": {"task": {"type": "string"}, "modality": {"type": "string", "enum": ["image", "mesh", "audio", "video", "text"]},
                                                      "ready_only": {"type": "boolean"}}}},
    {"name": "search_assets", "description": "List or search existing assets of the current project (newest first).",
     "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}, "type": {"type": "string", "enum": ["image", "mesh", "audio", "voice", "video"]},
                                                      "tag": {"type": "string"}, "limit": {"type": "integer"}}}},
    {"name": "get_asset", "description": "Read one asset's metadata (prompt, model, params, parents, tags, rating; meshes also mesh3d: rigged, joints, animation clip names).",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "propose_plan", "description": "Propose a plan to the user. It is shown as an editable card; the USER approves and runs it. Returns validation errors to fix (call again with a corrected plan).",
     "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}, "summary": {"type": "string"},
                                                      "steps": {"type": "array", "items": STEP, "minItems": 1}},
                     "required": ["title", "steps"]}},
]
NAMES = {t["name"] for t in TOOLS}


def call_core(name: str, args: dict) -> dict:
    req = urllib.request.Request(f"{API}/api/x/assistant/tools/{name}", data=json.dumps(args).encode(),
                                 headers={"Content-Type": "application/json", "X-Studio-Token": TOKEN}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (loopback URL from the core)
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return {"error": json.loads(e.read().decode()).get("error", str(e))}
        except ValueError:
            return {"error": str(e)}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"error": f"studio not reachable: {e}"}


def handle(msg: dict) -> dict | None:
    mid, method = msg.get("id"), msg.get("method")
    if mid is None:          # notifications (initialized, cancelled): no answer
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": (msg.get("params") or {}).get("protocolVersion") or PROTOCOL,
                                                       "capabilities": {"tools": {"listChanged": False}},
                                                       "serverInfo": {"name": "studio", "version": "1.0"}}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        p = msg.get("params") or {}
        name = p.get("name")
        if name not in NAMES:
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}}
        res = call_core(name, p.get("arguments") or {})
        return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}],
                                                       "isError": bool(res.get("error")) if isinstance(res, dict) else False}}
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
            sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
