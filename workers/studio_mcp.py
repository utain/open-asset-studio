#!/usr/bin/env python3
"""Tiny stdio MCP server that gives the Assistant (claude -p) the studio's restricted tool set -- and nothing else.

Tools (all read-only, except propose_plan, which only *records* a proposal; nothing here can submit a job):
  list_models    models with status, cost, licence, tasks and their params
  search_assets  list / search assets of the current project
  get_asset      one asset's meta
  get_rig        the bones of a rigged mesh (name, index, parent, position, role, weight share), to plan a rig.edit step
  preview_rig_edit  dry run of a rig.edit step: what the ops would do (nothing is written)
  get_settings   the studio settings (system or this project's effective values), read-only
  get_style_guide  the project style guide (look, avoid, palette, variants), read-only
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
        "task": {"type": "string", "description": "registry task id (t2i, rgba, shape, texture, retopo, rig, reskin, motion, engine_export, sfx, ambience, music, tts, ...; list_models shows them all)"},
        "tool": {"type": "string", "description": "instead of task+model: a CPU tool id such as audio.loop, audio.normalize or rig.edit (params.ops = list of rig edit operations; see the system prompt)"},
        "model": {"type": "string", "description": "model id from list_models (omit = the task's default)"},
        "params": {"type": "object", "description": "model params by name (prompt, negative, seconds, batch, seed, ...)"},
        "inputs": {"type": "array", "items": {"type": "object", "properties": {
            "step": {"type": "string", "description": "id of an earlier step whose outputs feed this one"},
            "asset": {"type": "string", "description": "existing asset id"},
            "use": {"type": "string", "enum": ["first", "all"], "description": "which outputs of the step (default first)"},
            "role": {"type": "string"}}}},
        "style": {"description": "project style guide for this step: omit = the studio applies it by itself; 'off' = skip it; or {variant, strength}",
                  "anyOf": [{"type": "string", "enum": ["auto", "off"]}, {"type": "object"}]},
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
    {"name": "get_rig", "description": "The skeleton of a rigged mesh asset: bones with index, name, parent, world position (metres, +Y up, +Z front, +X the character's left), humanoid role (hips, left_forearm, head ...), child count and skin-weight share; bones without weight. Call it before proposing a tool step rig.edit (add a tail, move / delete / rename bones) and copy bone names from it.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string", "description": "mesh asset id"}}, "required": ["id"]}},
    {"name": "preview_rig_edit", "description": "Dry run of a rig.edit step on a rigged mesh asset: checks the ops (same list as the step's params.ops) and reports what they would do (bones added / moved / deleted, bones left without skin weight, animations dropped, notes). Writes nothing. Use it before proposing the rig.edit step; errors name the candidate bones.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string", "description": "rigged mesh asset id"}, "ops": {"type": "array", "items": {"type": "object"}, "description": "rig edit operations"}}, "required": ["id", "ops"]}},
    {"name": "get_settings", "description": "Read the studio settings (read-only; you cannot change them). No arguments: setting groups and the values that differ from their defaults, for the current project (its overrides applied). group / q / keys: those settings with help, default and range (profile, commercial-only filter, default model per task, export preset, prompt language ...).",
     "inputSchema": {"type": "object", "properties": {"group": {"type": "string", "description": "general | models | prompts | memory | workers | llm"}, "q": {"type": "string"}, "keys": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "get_style_guide", "description": "Read the project's style guide (read-only): the look, avoid list, palette and variants the studio appends to the prompt of every image / texture / video / sound step by itself. Do not copy it into prompts; a step can opt out with style 'off'.",
     "inputSchema": {"type": "object", "properties": {}}},
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
