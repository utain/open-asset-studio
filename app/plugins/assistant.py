"""Assistant plugin: a planning chat over a restricted studio tool set, writing tools, text-backend settings.

The agent (any text backend from app/textgen.py) may ONLY: list models, list / search assets, read an asset's meta and
propose a plan. It can never submit jobs. A plan is a list of steps {id, title, task | tool, model, params, inputs:
[{step, use} | {asset}]} validated here against the registry; the user edits / approves it in the UI and the studio
then executes it through the normal job API (`core.jobs.submit`, i.e. POST /api/jobs semantics, so the cloud cost guard
of the job manager applies to every step).

How each backend gets the tools:
  claude CLI   workers/studio_mcp.py (stdio MCP) via --mcp-config + --strict-mcp-config, --tools "" and
               --allowedTools limited to mcp__studio__*; the MCP server calls POST /tools/<name> with a run token
  local / API  OpenAI function calling, executed in-process (same functions); falls back to one JSON answer
  other CLIs   one JSON answer {message, plan} with a compact model catalogue in the prompt

Routes (/api/x/assistant):
  GET  /backends                      text backend options + defaults + ToS note      PUT /settings {improve, write, assistant, cli, api_models, endpoints}
  POST /backends/test {backend}       one tiny call (spends quota)
  GET  /chats?project= · POST /chats {project, backend, title} · GET|PATCH|DELETE /chats/{id}?project=
  POST /chats/{id}/message {project, text, backend}   (async: progress via SSE "assistant" events)
  POST /chats/{id}/cancel · POST /chats/{id}/validate {project, plan} · POST /chats/{id}/run {project, message, plan, steps}
  POST /runs/{run}/cancel?project=&chat=
  POST /tools/{name}  (MCP proxy, X-Studio-Token of a live run only)
  POST /write {kind, project, backend, input}  kinds: barks | sfx_events | asset_list | asset_meta | character_audio (Pipelines: character → voice / barks / SFX defaults)
Files: data/projects/<project>/assistant/<chat id>.json
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from .. import config, licensing, profiles
from .. import style_common as sc
from ..assets import AssetError, check_project
from ..jobs import HANDLERS, TERMINAL, JobError
from ..textgen import CLI_INFO, TOS_NOTE, TextGenError, parse_backend

router = APIRouter()
CORE = None
CHAT_RE = re.compile(r"^c[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
STEP_RE = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
MAX_STEPS = 40
_LOCK = threading.RLock()
RUNS_BY_TOKEN: dict[str, dict] = {}       # MCP token -> {chat, project, proposals: [], tools: []}
TURNS: dict[str, threading.Event] = {}    # chat id -> cancel event of the turn in flight
PLAN_RUNS: dict[str, dict] = {}           # run id -> {cancel, thread}
MCP_TOOLS = ["list_models", "search_assets", "get_asset", "get_rig", "preview_rig_edit", "get_settings", "get_style_guide", "propose_plan"]
ALLOWED_MCP = [f"mcp__studio__{n}" for n in MCP_TOOLS]

SHARED_PARAMS = {"prompt", "negative", "batch", "seed"}       # accepted by every model job (not always in params:)
TOOL_PARAMS = {
    "audio.trim": {"threshold_db", "pad_ms"},
    "audio.fade": {"in_ms", "out_ms"},
    "audio.normalize": {"lufs", "ceiling_db"},
    "audio.loop": {"xfade_ms"},
    "audio.convert": {"format", "bit_depth", "quality"},
    "audio.channels": {"channels"},
    "audio.resample": {"sample_rate"},
    "audio.process": {"trim", "fade", "loop", "normalize", "sample_rate", "channels", "format", "threshold_db", "pad_ms",
                      "fade_in_ms", "fade_out_ms", "xfade_ms", "lufs", "ceiling_db", "bit_depth", "quality"},
    "image.convert": {"format", "max_side", "square"},
    "asset.export": {"preset", "audio", "image"},       # export preset: loudness / sample rate / format / crop; derived copies
    "rig.edit": {"ops", "clips"},                       # edit the skeleton of a rigged mesh (workers/rig_edit.py); ops is a list of objects
    "style.palette": {"mode", "strength", "colors"},    # recolour images with the project style palette (app/plugins/style.py)
}
JSON_TOOL_PARAMS = {"rig.edit": {"ops"}, "style.palette": {"colors"}}                 # tool params that may be lists / objects (all others must be scalars)


def setup(core):
    global CORE
    CORE = core
    core.assistant = sys.modules[__name__]


def _core():
    if CORE is None:
        raise HTTPException(503, "studio core not ready")
    return CORE


def _show_test() -> bool:
    return bool(_core().settings.get("ui", "show_test_models", default=False))


# ===================================================================== tools (shared by MCP / function calling / JSON)
def _est(m: dict, batch: int = 1) -> tuple[float | None, float | None]:
    e = m.get("est") or {}
    est_s = None
    if e.get("run_s") is not None or e.get("load_s") is not None:
        est_s = float(e.get("load_s") or 0) + float(e.get("run_s") or 0) * max(1, batch)
    usd = None
    for key in ("est_usd", "cost_usd"):
        if m.get(key) is not None:
            usd = float(m[key]) * max(1, batch)
    pr = m.get("pricing") or (m.get("cloud") or {}).get("pricing")
    if usd is None and isinstance(pr, dict):
        per = pr.get("per_output", pr.get("per_image", pr.get("per_call", pr.get("usd"))))
        if per is not None:
            usd = float(per) * max(1, batch)
    if usd is None and not _is_cloud(m):
        usd = 0.0
    return est_s, usd


def _is_cloud(m: dict) -> bool:
    return bool((m.get("features") or {}).get("remote") and m.get("backend") != "comfyui") or bool(m.get("provider")) or m.get("backend") in ("cloud",)


def tool_list_models(args: dict, project: str | None = None) -> dict:
    c = _core()
    reg = c.registry
    eff = c.settings_for(project)          # system settings + the project's overrides (licence filter, default models)
    task, modality = args.get("task"), args.get("modality")
    ready_only = args.get("ready_only", False)
    out = []
    for m in reg.list(task or None, modality or None, include_test=_show_test()):
        if m.get("modality") == "text" or "improve" in m["tasks"]:
            continue
        if licensing.blocked(m, eff):     # "Commercial-safe models only": the planner must not propose them
            continue
        st = m["status"]
        if ready_only and st["status"] != "ready":
            continue
        if st["status"] != "ready":     # not usable in a plan: id + reason only (keeps small contexts small)
            out.append({"id": m["id"], "name": m["name"], "tasks": m["tasks"], "status": st["status"], "reason": (st.get("reason") or "")[:120]})
            continue
        est_s, usd = _est(m)
        out.append({"id": m["id"], "name": m["name"], "modality": m.get("modality"), "tasks": m["tasks"],
                    "status": "ready", "summary": (m.get("summary") or "")[:160],
                    "license": {"name": (m.get("license") or {}).get("name"), "commercial": bool((m.get("license") or {}).get("commercial"))},
                    "cloud": _is_cloud(m), "est_s_per_take": est_s, "est_usd_per_take": usd,
                    "default_for": [t for t in m["tasks"] if reg.default_for(t, settings=eff) == m["id"]],
                    "params": [_param_public(p) for p in m.get("params") or [] if p.get("type") not in ("asset", "assets")]})
    return {"models": out, "tasks": {k: v.get("label") for k, v in reg.tasks.items() if k not in ("improve", "test")},
            "tools": {k[5:]: v["label"] for k, v in HANDLERS.items() if k.startswith("tool:") and k[5:] in TOOL_PARAMS},
            "shared_params": sorted(SHARED_PARAMS)}


def _param_public(p: dict) -> str:
    """Compact param description: 'steps:int[4..60]=28', 'aspect:choice{1:1|16:9}=1:1', 'prompt:text'."""
    d = f"{p['name']}:{p['type']}"
    if "min" in p or "max" in p:
        d += f"[{p.get('min', '')}..{p.get('max', '')}]"
    if p.get("choices"):
        d += "{" + "|".join(str(c["value"] if isinstance(c, dict) else c) for c in p["choices"][:16]) + "}"
    if "default" in p and p["default"] not in (None, ""):
        d += f"={p['default']}"
    if p.get("tasks"):
        d += f" (tasks: {','.join(p['tasks'])})"
    return d


def _rig_edit():
    if str(config.WORKERS) not in sys.path:
        sys.path.append(str(config.WORKERS))
    import rig_edit
    return rig_edit


def tool_get_rig(project: str, args: dict) -> dict:
    """The bones of a rigged mesh, compact: [index, name, parent index, [x, y, z], role, child count, weight share]."""
    c = _core()
    try:
        a = c.assets.get(str(args.get("id") or args.get("asset") or ""))
    except (AssetError, KeyError):
        return {"error": "asset not found"}
    if a.get("project") != project:
        return {"error": "asset not in this project"}
    if a.get("type") != "mesh" or not str(a.get("main") or "").lower().endswith(".glb"):
        return {"error": "get_rig needs a rigged mesh asset (GLB)"}
    re_ = _rig_edit()
    try:
        d = re_.describe(c.assets.file_path(a["project"], a["id"], a["main"]))
    except re_.NoSkinError:
        return {"error": "this mesh has no skeleton; propose the task rig first"}
    except re_.RigAnimError as e:
        return {"error": str(e)}
    hum = d.get("humanoid") or {}
    return {"id": a["id"], "name": a.get("name"), "count": d["count"], "height_m": d["height"], "units": d["units"],
            "columns": ["index", "name", "parent", "position", "role", "children", "weight_share"],
            "joints": [[r["index"], r["name"], r["parent"], [round(x, 3) for x in r["position"]], r["role"], r["children"], round(r["weight_share"], 3)]
                       for r in d["joints"]],
            "humanoid": ({"roles": hum["roles"]} if hum.get("ok") else {"roles": None, "reason": hum.get("reason")}),
            "animations": d["animations"], "bones_without_weight": d["zero_weight"], "needs_reskin": bool((a.get("info") or {}).get("needs_reskin")),
            "hint": "Edit with a step {tool: 'rig.edit', params: {ops: [...]}}; name bones exactly as listed here (or role words like 'left elbow')."}


def tool_preview_rig_edit(project: str, args: dict) -> dict:
    """Dry run of a rig.edit step: what the ops would do. Reads the GLB, writes nothing."""
    c = _core()
    try:
        a = c.assets.get(str(args.get("id") or args.get("asset") or ""))
    except (AssetError, KeyError):
        return {"error": "asset not found"}
    if a.get("project") != project:
        return {"error": "asset not in this project"}
    if a.get("type") != "mesh" or not str(a.get("main") or "").lower().endswith(".glb"):
        return {"error": "preview_rig_edit needs a rigged mesh asset (GLB)"}
    ops = args.get("ops")
    if not isinstance(ops, list) or not ops:
        return {"error": "give ops: the list of rig edit operations of the rig.edit step"}
    re_ = _rig_edit()
    try:
        _, rep = re_.edit(c.assets.file_path(a["project"], a["id"], a["main"]), ops)
    except re_.NoSkinError:
        return {"error": "this mesh has no skeleton; propose the task rig first"}
    except (re_.RigEditError, re_.RigAnimError) as e:
        return {"error": str(e)}
    rep.pop("bytes_before", None)
    rep.pop("bytes_after", None)
    rep["written"] = False
    rep["hint"] = ("Nothing was changed. If it is what the user asked for, propose the rig.edit step with these ops"
                   + (" and then a reskin step (new bones have no skin weight)." if rep.get("needs_reskin") or rep.get("zero_weight_bones") else "."))
    return rep


def tool_get_style_guide(project: str, args: dict) -> dict:
    """Read-only: the project's style guide (look, avoid list, palette, variants). The studio adds it to the prompt of image / texture / video /
    sound jobs on its own, so a plan must NOT repeat it in prompts; a step may carry style 'off' or {variant, strength} to override it."""
    c = _core()
    svc = getattr(c, "style", None)
    g = svc.get(project) if svc else None
    if not g:
        return {"guide": None, "hint": "This project has no style guide; write prompts as usual."}
    keep = ("name", "version", "enabled", "strength", "positive", "background", "avoid", "palette", "variants")
    out = {k: g[k] for k in keep if k in g}
    out["refs"] = [r.get("name") or r.get("asset") for r in g.get("refs") or []]
    out["hint"] = ("The studio appends this to the prompt of every image / texture / video / sound step by itself: do not copy it into prompts. "
                   "A step may add style: 'off' (skip it) or {variant: image|texture|sprite|ui_icon|video|audio, strength: light|normal|strong}.")
    return out


def tool_get_settings(project: str, args: dict) -> dict:
    """Read-only settings for the plan: no arguments = groups + values that differ from the defaults; group / q / keys = fields
    with help, default and range. Values are the project's effective ones. The Assistant cannot change settings."""
    from .. import settings_schema as ss

    c = _core()
    view = c.settings_for(project)
    ov = config.flatten(getattr(view, "overrides", {}) or {})
    fields = [f for f in ss.fields(c) if f.get("gui") is not False]

    def val(f):
        v = view.get(*f["key"].split("."))
        return v

    keys, group, q = args.get("keys") or [], args.get("group"), str(args.get("q") or "").strip().lower()
    if not (keys or group or q):
        groups = [{"id": g["id"], "label": g["label"], "fields": sum(1 for f in fields if f["group"] == g["id"])} for g in ss.GROUPS if g["id"] != "env"]
        return {"project": project, "groups": groups,
                "changed_from_default": {f["key"]: val(f) for f in fields if f.get("editable", True) and val(f) != f.get("default")},
                "overridden_in_project": {k: v for k, v in sorted(ov.items())},
                "hint": "get_settings(group=<id>) or (q=<text>) or (keys=[...]) lists fields with help, default and range. You cannot change settings: tell the user which one to change on the Settings page."}
    out = []
    for f in fields:
        if (keys and f["key"] not in keys) or (group and f["group"] != group):
            continue
        if q and q not in f"{f['key']} {f.get('label', '')} {f.get('help', '')}".lower():
            continue
        row = {"key": f["key"], "label": f.get("label"), "value": val(f), "default": f.get("default"), "help": f.get("help"),
               "overridden_in_project": f["key"] in ov}
        if f.get("min") is not None or f.get("max") is not None:
            row["range"] = f"{f.get('min')}..{f.get('max')}" + (f" {f['unit']}" if f.get("unit") else "")
        ch = ss._choices(f)
        if ch:
            row["choices"] = [x["value"] for x in ch]
        if f.get("editable") is False:
            row["file_only"] = True
        out.append({k: v for k, v in row.items() if v not in (None, "")} | {"value": val(f)})
    return {"project": project, "total": len(out), "settings": out[:40], **({"truncated": True} if len(out) > 40 else {})}


def tool_search_assets(project: str, args: dict) -> dict:
    lim = max(1, min(int(args.get("limit") or 30), 100))
    r = _core().assets.list(project, type=args.get("type") or None, tag=args.get("tag") or None, q=args.get("q") or None, limit=lim)
    return {"total": r["total"], "assets": [_asset_brief(a) for a in r["items"]]}


def _asset_brief(a: dict) -> dict:
    return {"id": a["id"], "type": a["type"], "name": a.get("name"), "task": a.get("task"), "model": a.get("model"),
            "prompt": (a.get("prompt") or "")[:200], "tags": a.get("tags") or [], "rating": a.get("rating", 0), "created": a.get("created")}


def tool_get_asset(project: str, args: dict) -> dict:
    try:
        a = _core().assets.get(str(args.get("id") or ""))
    except (AssetError, KeyError):
        return {"error": "asset not found"}
    if a.get("project") != project:
        return {"error": "asset not in this project"}
    keep = ("id", "type", "name", "task", "model", "model_name", "prompt", "improved_prompt", "final_prompt", "negative", "params",
            "seed", "parents", "tags", "rating", "favorite", "notes", "info", "license", "created")
    out = {k: a.get(k) for k in keep}
    if a.get("type") == "mesh":   # rig / motion results: is it rigged, how many joints, which animation clips
        w, info = a.get("worker") or {}, a.get("info") or {}
        out["mesh3d"] = {"rigged": bool(w.get("rigged") or info.get("rigged")), "joints": w.get("joints") or info.get("joints"),
                         "animations": w.get("animations") or info.get("animations") or [], "textured": w.get("textured")}
        if w.get("retopo"):
            out["mesh3d"]["retopo"] = {k: w["retopo"][k] for k in ("faces", "quad_pct", "triangles", "tex_size") if k in w["retopo"]}
    return out


# ===================================================================== plan validation
def validate_plan(plan: dict, project: str, *, registry=None, assets=None, include_test: bool | None = None) -> dict:
    """Normalize + check a plan. Returns {plan, ok, errors: {step_id|'plan': [..]}, warnings, totals}.
    Rejects unknown tasks / models / tools / params, steps a model cannot do, not-ready models, bad input refs."""
    reg = registry or _core().registry
    store = assets if assets is not None else (_core().assets if CORE else None)
    include_test = _show_test() if include_test is None else include_test
    eff_settings = _core().settings_for(project) if CORE else None     # project default models
    errors: dict[str, list[str]] = {}
    warnings: list[str] = []

    def err(k, msg):
        errors.setdefault(k, []).append(msg)
    if not isinstance(plan, dict):
        return {"plan": {}, "ok": False, "errors": {"plan": ["A plan must be an object."]}, "warnings": [], "totals": {}}
    steps_in = plan.get("steps")
    if not isinstance(steps_in, list) or not steps_in:
        err("plan", "A plan needs at least one step.")
        steps_in = []
    if len(steps_in) > MAX_STEPS:
        err("plan", f"At most {MAX_STEPS} steps.")
        steps_in = steps_in[:MAX_STEPS]
    seen: dict[str, dict] = {}
    out_steps = []
    tot_s, tot_usd, usd_unknown = 0.0, 0.0, False
    for i, s in enumerate(steps_in):
        if not isinstance(s, dict):
            err("plan", f"Step {i + 1} is not an object.")
            continue
        sid = str(s.get("id") or f"s{i + 1}")
        key = sid if STEP_RE.match(sid) else f"s{i + 1}"
        if not STEP_RE.match(sid):
            err(key, f"Invalid step id '{sid[:30]}'.")
            sid = key
        if sid in seen:
            err(sid, f"Duplicate step id '{sid}'.")
        st = {"id": sid, "title": str(s.get("title") or "")[:120], "notes": str(s.get("notes") or "")[:400],
              "params": {}, "inputs": [], "enabled": s.get("enabled", True) is not False}
        params = s.get("params") if isinstance(s.get("params"), dict) else {}
        if s.get("params") is not None and not isinstance(s.get("params"), dict):
            err(sid, "params must be an object.")
        tool = s.get("tool")
        if not tool and str(s.get("task") or "").startswith("tool:"):
            tool = str(s["task"])[5:]
        model = None
        if tool:
            tool = str(tool).removeprefix("tool:")
            if f"tool:{tool}" not in HANDLERS or tool not in TOOL_PARAMS:
                err(sid, f"Unknown tool '{tool}'.")
            else:
                bad = sorted(set(params) - TOOL_PARAMS[tool])
                if bad:
                    err(sid, f"Unknown params for {tool}: {', '.join(bad)}.")
            st["tool"] = tool
            st["task"] = None
            json_ok = JSON_TOOL_PARAMS.get(tool, set())
            st["params"] = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool)) or v is None or (k in json_ok and isinstance(v, (list, dict)))}
            if tool == "rig.edit":
                if not params.get("ops"):
                    err(sid, "rig.edit needs params.ops (a list of operations).")
                else:
                    for msg in _rig_edit().validate_ops(params.get("ops")):
                        err(sid, msg)
            if not s.get("inputs"):
                err(sid, f"The tool {tool} needs an input (an earlier step or an asset).")
        else:
            task = s.get("task") or None
            mid = s.get("model") or None
            if not task and not mid:
                err(sid, "A step needs a task (or a model, or a CPU tool).")
            if task and task not in reg.tasks:
                err(sid, f"Unknown task '{task}'.")
            if task in ("improve",):
                err(sid, "Prompt improving is not a plan step.")
            if mid and mid not in reg.models:
                err(sid, f"Unknown model '{mid}'.")
                mid = None
            elif not mid and task in reg.tasks:
                mid = reg.default_for(task, settings=eff_settings)
                if not mid:
                    err(sid, f"No model can do '{task}'.")
            if mid:
                model = reg.models[mid]
                if not task:
                    task = model["tasks"][0]
                if task not in model["tasks"]:
                    err(sid, f"{model['name']} cannot do '{task}' (it does {', '.join(model['tasks'])}).")
                if model.get("test") and not include_test:
                    err(sid, f"{model['name']} is a self-test model.")
                stt = reg.status.get(mid, {})
                if stt.get("status") != "ready":
                    err(sid, f"{model['name']} is not ready: {stt.get('reason') or stt.get('status')}.")
                declared = {p["name"]: p for p in model.get("params") or []}
                unknown = sorted(k for k in params if k not in declared and k not in SHARED_PARAMS)
                if unknown:
                    err(sid, f"Unknown params for {model['name']}: {', '.join(unknown)}.")
                asset_p = [k for k in params if declared.get(k, {}).get("type") in ("asset", "assets")]
                if asset_p:
                    err(sid, f"Pass assets as inputs, not params ({', '.join(asset_p)}).")
                clean = {k: v for k, v in params.items() if k not in unknown and k not in asset_p}
                try:
                    clean = reg.validate_params(mid, clean)
                except ValueError as e:
                    err(sid, str(e))
                for k in ("batch",):
                    if k in clean:
                        try:
                            clean[k] = max(1, min(int(clean[k]), 16))
                        except (TypeError, ValueError):
                            err(sid, "batch must be a number.")
                st["params"] = clean
            st["task"] = task
            st["model"] = mid
        for inp in s.get("inputs") or []:
            if not isinstance(inp, dict):
                err(sid, "Each input is an object {step} or {asset}.")
                continue
            role = str(inp.get("role") or "input")[:24]
            if inp.get("step"):
                ref = str(inp["step"])
                if ref not in seen:
                    err(sid, f"Input step '{ref}' must be an earlier step.")
                use = inp.get("use") if inp.get("use") in ("first", "all", "each") else "first"
                st["inputs"].append({"step": ref, "use": use, "role": role})
            elif inp.get("asset"):
                aid = str(inp["asset"])
                ok = False
                if store is not None:
                    try:
                        a = store.get(aid)
                        ok = a.get("project") == project
                    except (AssetError, KeyError):
                        ok = False
                if not ok:
                    err(sid, f"Asset {aid[:40]} not found in this project.")
                st["inputs"].append({"asset": aid, "role": role})
            else:
                err(sid, "An input needs 'step' or 'asset'.")
        if model is not None:
            if st.get("task") in ("t2i", "rgba", "sfx", "ambience", "music", "song", "voice_design") and not model.get("test") \
                    and not (st.get("params") or {}).get("prompt") and not st["inputs"]:
                warnings.append(f"{sid}: no prompt.")
            fanout = next((x for x in st["inputs"] if x.get("use") == "each"), None)
            mult = 1
            if fanout and fanout["step"] in seen:
                mult = int((seen[fanout["step"]].get("params") or {}).get("batch") or 1)
            est_s, usd = _est(model, int(st["params"].get("batch") or 1))
            if s.get("est_s") is not None and est_s is None:
                try:
                    est_s = float(s["est_s"])
                except (TypeError, ValueError):
                    pass
            st["est_s"] = round(est_s * mult, 1) if est_s is not None else None
            st["est_usd"] = round(usd * mult, 4) if usd is not None else None
            st["cloud"] = _is_cloud(model)
            st["license"] = model.get("license")
            st["model_name"] = model["name"]
            if not (model.get("license") or {}).get("commercial"):
                warnings.append(f"{sid}: {model['name']} is non-commercial.")
            tot_s += st["est_s"] or 0
            if st["est_usd"] is None:
                usd_unknown = True
            else:
                tot_usd += st["est_usd"]
        elif tool:
            st["est_s"], st["est_usd"], st["cloud"] = 2.0, 0.0, False
            tot_s += 2
        sty = s.get("style")
        if sty not in (None, "", "auto"):
            if sty == "off" or (isinstance(sty, dict) and not set(sty) - {"apply", "variant", "strength"}
                                and sty.get("variant", "image") in sc.VARIANTS and sty.get("strength", "normal") in sc.STRENGTHS
                                and isinstance(sty.get("apply", True), bool)):
                st["style"] = sty
            else:
                err(sid, "style must be 'auto', 'off' or {apply, variant, strength} (variant: " + ", ".join(sc.VARIANTS) + "; strength: " + ", ".join(sc.STRENGTHS) + ").")
        seen[sid] = st
        out_steps.append(st)
    norm = {"title": str(plan.get("title") or "Plan")[:120], "summary": str(plan.get("summary") or "")[:1200], "steps": out_steps}
    totals = {"steps": len(out_steps), "est_s": round(tot_s, 1), "est_usd": round(tot_usd, 4), "usd_unknown": usd_unknown,
              "cloud_steps": sum(1 for s in out_steps if s.get("cloud"))}
    return {"plan": norm, "ok": not errors, "errors": errors, "warnings": warnings, "totals": totals}


def tool_propose_plan(run: dict, args: dict) -> dict:
    v = validate_plan(args, run["project"])
    run.setdefault("proposals", []).append({"plan": args, "validation": v})
    if v["ok"]:
        return {"ok": True, "message": "Plan recorded and shown to the user for approval. Do not call propose_plan again unless the user asks for changes.",
                "totals": v["totals"], "warnings": v["warnings"]}
    return {"ok": False, "errors": v["errors"], "hint": "Fix these and call propose_plan again with the full corrected plan."}


def execute_tool(run: dict, name: str, args: dict) -> dict:
    args = args if isinstance(args, dict) else {}
    run.setdefault("tools", []).append(name)
    if run.get("on_tool"):
        run["on_tool"](name)
    try:
        if name == "list_models":
            return tool_list_models(args, run.get("project"))
        if name == "search_assets":
            return tool_search_assets(run["project"], args)
        if name == "get_asset":
            return tool_get_asset(run["project"], args)
        if name == "get_rig":
            return tool_get_rig(run["project"], args)
        if name == "preview_rig_edit":
            return tool_preview_rig_edit(run["project"], args)
        if name == "get_settings":
            return tool_get_settings(run["project"], args)
        if name == "get_style_guide":
            return tool_get_style_guide(run["project"], args)
        if name == "propose_plan":
            return tool_propose_plan(run, args)
    except (ValueError, TypeError, AssetError) as e:
        return {"error": str(e)}
    return {"error": f"unknown tool {name}"}


@router.post("/tools/{name}")
def tool_endpoint(name: str, args: dict, x_studio_token: str = Header(default="")):
    run = RUNS_BY_TOKEN.get(x_studio_token or "")
    if not run:
        raise HTTPException(403, "No live assistant run for this token.")
    if name not in MCP_TOOLS:
        raise HTTPException(404, "Unknown tool.")
    return execute_tool(run, name, args)


# ===================================================================== chats (files)
def chats_dir(project: str) -> Path:
    return config.DATA / "projects" / check_project(project) / "assistant"


def chat_path(project: str, cid: str) -> Path:
    if not isinstance(cid, str) or not CHAT_RE.match(cid):
        raise HTTPException(400, "Invalid chat id.")
    return chats_dir(project) / f"{cid}.json"


def load_chat(project: str, cid: str) -> dict:
    p = chat_path(project, cid)
    if not p.is_file():
        raise HTTPException(404, "Chat not found.")
    return json.loads(p.read_text())


def save_chat(chat: dict):
    with _LOCK:
        p = chat_path(chat["project"], chat["id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        chat["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(chat, ensure_ascii=False, indent=1, default=str))
        os.replace(tmp, p)


def publish(chat: dict, kind: str, **data):
    if CORE is not None:
        CORE.bus.publish("assistant", {"chat": chat["id"], "project": chat["project"], "kind": kind, **data})


def _mid(chat) -> str:
    return f"m{len(chat.get('messages') or []) + 1}-{secrets.token_hex(2)}"


# ===================================================================== the agent turn
SYSTEM = """You are the planning assistant of a local media-asset studio running on one machine. It makes images, 3D models, sound
effects, ambience, music, voices and video for games, video, podcasts, social content and film.
You help the user turn a request into a PLAN of generation steps. You can ONLY look things up and propose; you can never
run anything yourself. The user reviews, edits and approves the plan, then the studio runs it.

Rules:
- Use only model ids that exist and whose status is "ready". Prefer the task's default model unless the request needs
  something else. Prefer commercial-licence models; if you pick a non-commercial one, say so.
- Use only params that the chosen model lists, plus the shared ones: prompt, negative, batch, seed. Put input assets in
  "inputs", never in params. batch = takes / variations per step (keep 1-4).
- Chain steps with inputs: {"step": "<earlier step id>", "use": "first"|"all"|"each"} ("each" = one job per output of that
  step, e.g. turn every concept image into a 3D shape). Existing assets: {"asset": "<id>"}.
- Typical chains: concept image (t2i or rgba) -> shape -> texture for 3D props (-> retopo for a game-ready low poly with baked maps, before rig; -> rig for a skeleton; for a humanoid, -> motion
  with an English prompt such as "walks forward", whose input is the rigged mesh); sfx / ambience with 3 takes;
  ambience then tool audio.loop; voice_design then tts; thumbnail or b-roll stills (t2i at 16:9 or 9:16); podcast or video
  intro stings and music beds (music, short); asset.export to deliver loudness-normalised audio or platform-sized images. Write prompts in English, in the target model's style
  (Stable Audio: comma-separated sound tags; images: concrete visual description, plain background for 3D).
- Never retopo a rigged mesh (the skeleton and skin would be lost): retopo the unrigged one, then rig.
- Honour the USE CASE line when present (aspect ratios, lengths, loudness targets) and keep game support intact when the
  request is about a game.
- Keep plans short (usually <= 12 steps). Step ids: short snake_case. Give every step a clear title.
- Editing an existing rig ("add a 5-bone tail", "move the left elbow down a bit", "delete the gun bones", "rename to Mixamo names",
  "re-skin"): first call get_rig(id) on the rigged mesh (find it with search_assets; mesh3d.rigged must be true) and copy bone names
  from it; never invent bones. Check the ops with preview_rig_edit(id, ops) (a dry run: reports what changes and which bones will
  lack weights; fix the ops if it errors). Then propose ONE step {"tool": "rig.edit", "params": {"ops": [...]}, "inputs": [{"asset": id}]}.
  Ops (applied in order; units metres, +Y up, the character faces +Z, its left is +X; a bone is a name, an index or a role word like
  "left elbow", "head", "hips"):
  {"op":"rename","bone":B,"to":NAME} | {"op":"rename_standard","scheme":"mixamo"|"vrm"|"unreal"} (humanoids) |
  {"op":"move_joint","bone":B,"offset":[x,y,z]} (or "delta":[x,y,z] as fractions of the bone length, or "to":[x,y,z]; "children":"follow"
  moves the whole subtree, "mirror":true moves the opposite side too) | {"op":"delete_bone","bone":B} (or "match":"gun*",
  "include_children":true; skin weights go to the parent) | {"op":"add_bone","parent":B,"direction":"up"|"down"|"left"|"right"|"forward"|"back"|[x,y,z],
  "length":m,"name":NAME} | {"op":"add_chain","parent":B,"count":5,"direction":"back","length":0.08,"name_prefix":"tail"} (tails, ears, hair) |
  {"op":"reparent","bone":B,"parent":B2}. Example 5-bone tail: [{"op":"add_chain","parent":"hips","count":5,"direction":"back","length":0.1,"name_prefix":"tail"}].
  New bones start with no skin weight: when the edit adds bones (or the user says re-skin) add a second step {"task": "reskin",
  "inputs": [{"step": "<edit step id>"}]} (keeps the skeleton, recomputes the weights, GPU ~35 s). A rename or a move does not need it.
- The project may have a style guide (get_style_guide). The studio appends it to the prompt of every image, texture, video and sound step by
  itself, so describe only the subject in prompts and never copy the style in. Add "style": "off" to a step that must not be styled.
- Reply briefly in the user's language. If something cannot be done with the ready models, say what is missing.
"""


def _history_text(chat: dict, limit: int = 12) -> list[dict]:
    turns = []
    for m in (chat.get("messages") or [])[-limit:]:
        if m.get("role") not in ("user", "assistant") or m.get("pending"):
            continue
        t = m.get("text") or ""
        if m.get("plan"):
            pl = m["plan"].get("plan") or {}
            t += "\n[plan proposed: " + json.dumps({"title": pl.get("title"), "steps": [
                {k: s.get(k) for k in ("id", "task", "tool", "model", "params", "inputs") if s.get(k) is not None} for s in pl.get("steps") or []]},
                ensure_ascii=False)[:3000] + "]"
        if m.get("run"):
            r = m["run"]
            t += f"\n[plan run {r.get('status')}: " + ", ".join(f"{sid}: {st.get('status')} ({len(st.get('outputs') or [])} outputs: {', '.join((st.get('outputs') or [])[:4])})"
                                                           for sid, st in (r.get("steps") or {}).items()) + "]"
        if t.strip():
            turns.append({"role": m["role"], "content": t})
    return turns


def _catalog_text(project: str) -> str:
    d = tool_list_models({"ready_only": True}, project)
    lines = []
    for m in d["models"]:
        ps = ", ".join(m["params"][:14])
        lines.append(f"- {m['id']} ({m['name']}) tasks={','.join(m['tasks'])} licence={'commercial' if m['license']['commercial'] else 'NON-commercial'}"
                     f"{' cloud' if m['cloud'] else ''} est={m['est_s_per_take']}s params: {ps}")
    assets = tool_search_assets(project, {"limit": 15})["assets"]
    al = [f"- {a['id']} {a['type']} '{a['name']}'" for a in assets]
    tools = ", ".join(d["tools"])
    c = _core()
    eff = c.settings_for(project)
    pid = profiles.profile_id(eff)
    prof = profiles.PROFILES[pid]
    use = (f"USE CASE: {prof['label']}. {prof['summary']} Default aspect ratio: {prof['aspect'] or 'model default'}.\n"
           if pid != "general" else "")
    if licensing.settings_of(eff)[0]:
        use += "COMMERCIAL-SAFE MODE IS ON: only commercial-licence models are listed; never propose a non-commercial model.\n"
    return (use + "READY MODELS:\n" + "\n".join(lines) + f"\nCPU TOOLS (step 'tool'): {tools}\nRECENT ASSETS:\n" + ("\n".join(al) or "- none"))


PLAN_JSON_SCHEMA = {
    "type": "object", "required": ["message", "plan"],
    "properties": {
        "message": {"type": "string"},
        "plan": {"type": ["object", "null"], "properties": {
            "title": {"type": "string"}, "summary": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "object", "required": ["id", "title", "task", "model", "params", "inputs"], "properties": {
                "id": {"type": "string"}, "title": {"type": "string"},
                "task": {"type": "string", "description": "registry task id, or tool:<id> for a CPU tool (then model is \"\")"},
                "model": {"type": "string"}, "params": {"type": "object"},
                "inputs": {"type": "array", "items": {"type": "object", "properties": {
                    "step": {"type": "string"}, "asset": {"type": "string"}, "use": {"type": "string", "enum": ["first", "all", "each"]}}}}}}}}},
    },
}


def _plan_check(project: str):
    """generate(validate=...): plan errors become the one retry's feedback."""
    def check(data):
        if not isinstance(data, dict) or not data.get("plan"):
            return []
        v = validate_plan(data["plan"], project)
        return [f"step {k}: {' '.join(e)}" for k, e in v["errors"].items()][:8]
    return check


def _tool_specs() -> list[dict]:
    sys.path.insert(0, str(config.WORKERS))
    try:
        import studio_mcp  # the MCP server's tool definitions are the single source of truth
    finally:
        sys.path.pop(0)
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}}
            for t in studio_mcp.TOOLS]


def _mcp_tool_defs() -> list[dict]:
    sys.path.insert(0, str(config.WORKERS))
    try:
        import studio_mcp
    finally:
        sys.path.pop(0)
    return studio_mcp.TOOLS


def run_turn(project: str, cid: str, text: str, backend: str | None, cancel: threading.Event):
    c = _core()
    tg = c.textgen
    chat = load_chat(project, cid)
    t0 = time.time()
    msg = {"id": _mid(chat), "role": "assistant", "text": "", "pending": True, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tools": []}
    try:
        spec = tg.resolve(backend or chat.get("backend"), "assistant", c.settings_for(project))
    except TextGenError as e:
        spec = None
        msg.update(text="", error={"message": str(e), "kind": e.kind, "fix": e.fix})
    if spec:
        msg["backend"] = tg.public_spec(spec)
    chat["messages"].append(msg)
    save_chat(chat)
    publish(chat, "message", message=msg)
    if not spec:
        msg.pop("pending", None)
        save_chat(chat)
        publish(chat, "message", message=msg)
        return
    token = secrets.token_urlsafe(24)
    run = {"chat": cid, "project": project, "proposals": [], "tools": []}

    def on_tool(name):
        msg["tools"].append(name)
        publish(chat, "status", message_id=msg["id"], activity=name)
    run["on_tool"] = on_tool
    RUNS_BY_TOKEN[token] = run
    history = _history_text(chat)[:-1]       # the pending message is last; the user message just before it
    try:
        mode = "mcp" if spec["kind"] == "cli" and spec["name"] == "claude" else "tools" if spec["kind"] in ("local", "api") else "json"
        answer = ""
        if mode == "mcp":
            port = int(os.environ.get("STUDIO_PORT", "7860"))
            mcp = {"mcpServers": {"studio": {"type": "stdio", "command": sys.executable, "args": [str(config.WORKERS / "studio_mcp.py")],
                                             "env": {"STUDIO_API": f"http://127.0.0.1:{port}", "STUDIO_MCP_TOKEN": token}}}}
            prompt = _transcript(history) + f"\n\n[project: {project}]\nUser request:\n{text}\n\nUse the studio tools to check models (and assets if relevant), then call propose_plan when a plan fits. Finish with a short message to the user."
            p = tg.claude_agent(system=SYSTEM, prompt=prompt, mcp_config=mcp, allowed_tools=ALLOWED_MCP, model=spec.get("model"),
                                timeout=float(tg.cfg.get("assistant_timeout_s") or 420), cancel=cancel, max_turns=14)
            answer = p["text"] or (p["texts"][-1] if p["texts"] else "")
            msg["verified_tools"] = (p.get("init") or {}).get("tools")
            bad = [u["name"] for u in p["tool_uses"] if not str(u.get("name", "")).startswith("mcp__studio__")]
            if bad:
                msg.setdefault("warnings", []).append(f"blocked non-studio tool attempts: {', '.join(bad)}")
        elif mode == "tools":
            messages = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": f"[project: {project}]\n{text}"}]
            try:
                r = tg.api_tool_chat(spec, messages, _tool_specs(), lambda n, a: execute_tool(run, n, a), cancel=cancel, project=project)
                answer = r["text"]
                if r.get("truncated") and not answer:
                    answer = "I ran out of tool rounds; here is what I have so far."
                if not run["proposals"]:
                    # many local models describe the plan in prose instead of calling propose_plan: ask once for JSON
                    on_tool("propose_plan")
                    fm = [{"role": "system", "content": SYSTEM + "\n\n" + _catalog_text(project)}] + history + \
                         [{"role": "user", "content": text}, {"role": "assistant", "content": answer or "(no answer)"},
                          {"role": "user", "content": "Return that answer as JSON {message, plan}: message = your short reply to me, "
                                                      "plan = the plan as structured steps (null if no plan is needed)."}]
                    fr = tg.generate(messages=fm, schema=PLAN_JSON_SCHEMA, backend=spec, purpose="assistant", temperature=0.3,
                                     max_tokens=3500, cancel=cancel, project=project, validate=_plan_check(project))
                    fd = fr.get("data") or {}
                    if fd.get("plan"):
                        tool_propose_plan(run, fd["plan"])
                        answer = fd.get("message") or answer
            except TextGenError as e:
                if e.kind in ("cancelled", "budget", "confirm", "not_logged_in", "quota", "rate_limited", "unavailable", "timeout"):
                    raise
                mode = "json"
                msg.setdefault("warnings", []).append(f"tool calling failed ({e}); used a single JSON answer")
        if mode == "json":
            messages = [{"role": "system", "content": SYSTEM + "\n\n" + _catalog_text(project) +
                         "\n\nAnswer with JSON {message, plan}; plan is null when you only answer a question."}] + history + \
                       [{"role": "user", "content": text}]
            r = tg.generate(messages=messages, schema=PLAN_JSON_SCHEMA, backend=spec, purpose="assistant", temperature=0.4,
                            max_tokens=3500, cancel=cancel, project=project, validate=_plan_check(project))
            data = r.get("data") or {}
            answer = data.get("message") or ""
            if data.get("plan"):
                tool_propose_plan(run, data["plan"])
        prop = next((p for p in reversed(run["proposals"]) if p["validation"]["ok"]), run["proposals"][-1] if run["proposals"] else None)
        if prop:
            msg["plan"] = prop["validation"]
        msg["text"] = (answer or "").strip() or ("Here is a plan." if prop else "(no answer)")
    except TextGenError as e:
        msg["error"] = {"message": str(e), "kind": e.kind, "fix": e.fix}
    except Exception as e:  # noqa: BLE001
        msg["error"] = {"message": f"{type(e).__name__}: {e}", "kind": "internal", "fix": "See logs/studio.log."}
    finally:
        RUNS_BY_TOKEN.pop(token, None)
        TURNS.pop(cid, None)
    msg.pop("pending", None)
    msg["seconds"] = round(time.time() - t0, 1)
    with _LOCK:
        chat = load_chat(project, cid)
        chat["messages"] = [msg if m.get("id") == msg["id"] else m for m in chat["messages"]]
        if chat.get("title") in (None, "", "New chat"):
            chat["title"] = text[:60]
        save_chat(chat)
    publish(chat, "message", message=msg)


def _transcript(turns: list[dict]) -> str:
    if not turns:
        return "(new conversation)"
    return "Conversation so far:\n" + "\n\n".join(f"[{t['role'].upper()}]\n{t['content']}" for t in turns)


# ===================================================================== plan execution (through the job API)
def _step_spec(step: dict, plan: dict, project: str, chat_id: str, run_id: str, inputs: list[dict], reg=None) -> dict:
    reg = reg or _core().registry
    studio = (reg.tasks.get(step.get("task") or "") or {}).get("studio") or "assistant"
    tags = ["assistant"]
    meta = {"studio": studio, "tags": tags, "asset_extra": {"x_assistant": {"chat": chat_id, "run": run_id, "step": step["id"], "plan": plan.get("title")}}}
    if (step.get("params") or {}).get("prompt"):
        meta["prompt"] = step["params"]["prompt"]
    title = step.get("title") or step["id"]
    if step.get("tool"):
        return {"tool": step["tool"], "params": dict(step.get("params") or {}), "inputs": inputs, "project": project, "title": title, "meta": meta}
    if step.get("style"):
        meta["style"] = step["style"]                 # per-step override of the project style guide (applied by the core at submit time)
    return {"task": step["task"], "model": step["model"], "params": dict(step.get("params") or {}), "inputs": inputs,
            "project": project, "title": title, "meta": meta}


def execute_plan(project: str, chat_id: str, message_id: str, plan: dict, *, submit=None, get_job=None,
                 cancel: threading.Event | None = None, on_change=None, poll_s: float = 0.4, run_id: str | None = None,
                 registry=None) -> dict:
    """Run a validated plan: submit each step through the job API as soon as its input steps are done.
    submit(spec) -> job (default core.jobs.submit), get_job(id) -> job. Returns the run record."""
    c = CORE
    registry = registry or (c.registry if c else None)
    submit = submit or c.jobs.submit
    get_job = get_job or (lambda jid: c.jobs.public(c.jobs.get(jid)))
    cancel = cancel or threading.Event()
    run_id = run_id or f"r{time.strftime('%Y%m%d%H%M%S')}{secrets.token_hex(2)}"
    steps = [s for s in plan["steps"] if s.get("enabled", True)]
    run = {"id": run_id, "status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "steps": {s["id"]: {"status": "waiting", "jobs": [], "outputs": [], "error": None} for s in steps}}

    def changed():
        if on_change:
            on_change(run)
    changed()
    pending = list(steps)
    while pending:
        if cancel.is_set():
            for s in pending:
                run["steps"][s["id"]]["status"] = "cancelled"
            for st in run["steps"].values():
                for jid in st["jobs"]:
                    try:
                        if c is not None:
                            c.jobs.cancel(jid)
                    except Exception:  # noqa: BLE001
                        pass
            break
        progressed = False
        for s in list(pending):
            st = run["steps"][s["id"]]
            if st["status"] == "waiting":
                deps = [i["step"] for i in s.get("inputs") or [] if i.get("step")]
                dst = [run["steps"].get(d, {"status": "skipped"})["status"] for d in deps]
                if any(x in ("failed", "cancelled", "skipped", "blocked") for x in dst):
                    st.update(status="skipped", error="an input step did not finish")
                    pending.remove(s)
                    progressed = True
                    continue
                if not all(x == "done" for x in dst):
                    continue
                groups: list[list[dict]] = [[]]
                for i in s.get("inputs") or []:
                    if i.get("asset"):
                        for g in groups:
                            g.append({"asset": i["asset"], "role": i.get("role") or "input"})
                        continue
                    outs = run["steps"][i["step"]]["outputs"]
                    if not outs:
                        continue
                    if i.get("use") == "each":
                        groups = [g + [{"asset": o, "role": i.get("role") or "input"}] for g in groups for o in outs]
                    elif i.get("use") == "all":
                        for g in groups:
                            g.extend({"asset": o, "role": i.get("role") or "input"} for o in outs)
                    else:
                        for g in groups:
                            g.append({"asset": outs[0], "role": i.get("role") or "input"})
                if s.get("inputs") and not any(groups):
                    st.update(status="skipped", error="the input step produced no outputs")
                    pending.remove(s)
                    progressed = True
                    continue
                try:
                    for g in groups[:24]:
                        j = submit(_step_spec(s, plan, project, chat_id, run_id, g, registry))
                        st["jobs"].append(j["id"])
                    st["status"] = "running"
                except JobError as e:
                    st.update(status="blocked" if e.kind in ("budget", "confirm") else "failed", error=str(e), kind=e.kind, fix=e.fix)
                    pending.remove(s)
                except Exception as e:  # noqa: BLE001
                    st.update(status="failed", error=f"{type(e).__name__}: {e}")
                    pending.remove(s)
                progressed = True
            elif st["status"] == "running":
                jobs = [get_job(j) for j in st["jobs"]]
                if all(j["status"] in TERMINAL for j in jobs):
                    st["outputs"] = [o for j in jobs for o in j.get("outputs") or []]
                    bad = [j for j in jobs if j["status"] != "done"]
                    if bad and not st["outputs"]:
                        st.update(status="cancelled" if all(j["status"] == "cancelled" for j in bad) else "failed",
                                  error=(bad[0].get("error") or {}).get("message") or bad[0]["status"])
                    else:
                        st["status"] = "done"
                        if bad:
                            st["error"] = f"{len(bad)} of {len(jobs)} jobs failed"
                    pending.remove(s)
                    progressed = True
        if progressed:
            changed()
        else:
            time.sleep(poll_s)
    sts = [v["status"] for v in run["steps"].values()]
    run["status"] = "cancelled" if cancel.is_set() else "done" if all(x == "done" for x in sts) else "partial" if "done" in sts else "failed"
    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    changed()
    return run


def _start_run(project: str, cid: str, message_id: str, plan: dict) -> dict:
    cancel = threading.Event()
    run_id = f"r{time.strftime('%Y%m%d%H%M%S')}{secrets.token_hex(2)}"
    holder = {}

    def on_change(run):
        with _LOCK:
            chat = load_chat(project, cid)
            for m in chat["messages"]:
                if m.get("id") == message_id:
                    m["run"] = run
                    m.setdefault("runs", [])
                    if run["id"] not in m["runs"]:
                        m["runs"].append(run["id"])
            save_chat(chat)
        publish(chat, "run", message_id=message_id, run=run)

    def go():
        try:
            holder["run"] = execute_plan(project, cid, message_id, plan, cancel=cancel, on_change=on_change, run_id=run_id)
        finally:
            PLAN_RUNS.pop(run_id, None)
    th = threading.Thread(target=go, daemon=True, name=f"plan-{run_id}")
    PLAN_RUNS[run_id] = {"cancel": cancel, "thread": th, "chat": cid}
    th.start()
    return {"run": run_id}


# ===================================================================== writing tools
WRITE_KINDS = {
    "barks": {
        "system": ("You write short voice lines for characters (in-game barks, short video / podcast / ad lines, animation one-liners). Each line is "
                   "something the character says aloud: short (2-12 words usually), in character, varied, no stage directions inside the line, no quotes. "
                   "emotion = a short acting direction for a TTS model (e.g. 'gruff, amused', 'panicked, out of breath'). "
                   "situation = when the line plays (e.g. combat_start, hit, idle, low_health, victory, greeting, intro, outro, cta). Write the "
                   "lines in the requested language; directions in English."),
        "schema": {"type": "object", "additionalProperties": False, "required": ["lines", "notes"], "properties": {
            "lines": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": ["line", "emotion", "situation"],
                                                                "properties": {"line": {"type": "string", "minLength": 1}, "emotion": {"type": "string"}, "situation": {"type": "string"}}}},
            "notes": {"type": "string"}}},
    },
    "sfx_events": {
        "system": ("You are a sound designer. From the design text (game design, video script, storyboard, podcast outline), list the sound effect events the project needs. "
                   "event = snake_case file-friendly id (e.g. door_wood_open, ui_click_soft, footstep_stone, transition_whoosh_short). prompt = an English "
                   "text-to-audio prompt in Stable Audio style: comma-separated concrete sound tags (source, material, action, "
                   "space, character), no music unless asked. seconds = sensible clip length (0.3-10). category = ui | foley | "
                   "combat | ambience | creature | magic | vehicle | transition | misc. Avoid duplicates; cover the important interactions."),
        "schema": {"type": "object", "additionalProperties": False, "required": ["events", "notes"], "properties": {
            "events": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": ["event", "prompt", "seconds", "category"],
                                                                 "properties": {"event": {"type": "string", "minLength": 1}, "prompt": {"type": "string", "minLength": 3},
                                                                                "seconds": {"type": "number"}, "category": {"type": "string"}}}},
            "notes": {"type": "string"}}},
    },
    "asset_list": {
        "system": ("You are a production planner. From the design text (game, video, podcast, social or film project), list the assets to make. kind = image (2D art, "
                   "icons, concepts, thumbnails, b-roll stills, storyboard frames), mesh (3D prop / character), sfx, ambience, music or voice. name = short snake_case id. "
                   "prompt = an English generation prompt suited to the kind (images/meshes: concrete visual description, for 3D "
                   "a single object on a plain background; audio: comma-separated sound tags). notes = one short line on use."),
        "schema": {"type": "object", "additionalProperties": False, "required": ["assets", "notes"], "properties": {
            "assets": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": ["name", "kind", "prompt", "notes"],
                                                                 "properties": {"name": {"type": "string", "minLength": 1},
                                                                                "kind": {"type": "string", "enum": ["image", "mesh", "sfx", "ambience", "music", "voice"]},
                                                                                "prompt": {"type": "string", "minLength": 3}, "notes": {"type": "string"}}}},
            "notes": {"type": "string"}}},
    },
    "character_audio": {
        "system": ("You are an audio director. From a character's description (game, animation, film or video character), infer how the character sounds and "
                   "write the defaults for its audio kit. summary = one line: apparent gender, age, build, personality, role. "
                   "voice.description = an English voice-design prompt for a TTS voice-design model: gender, age, pitch, timbre, "
                   "pace, accent if any, emotion / attitude (e.g. 'A gruff, warm male voice in his fifties, low pitch, slow and "
                   "confident, slight rasp'); describe only the voice, never the looks. voice.sample_line = one natural in-character "
                   "sentence (8-20 words) to audition the voice, in the requested language. barks = short in-character lines (2-12 words, "
                   "no quotes, no stage directions) covering greeting, attack, hit, low_health, victory and idle, in the requested "
                   "language; emotion = a short English acting direction. sfx = the character's own sound effects (footsteps on a "
                   "fitting surface with the right weight / footwear, its weapon or ability swings and hits, armour / clothing / "
                   "body movement, creature noises if not human); event = snake_case id, prompt = an English text-to-audio prompt "
                   "of comma-separated concrete sound tags (source, material, action, size, space), no music, no speech."),
        "schema": {"type": "object", "additionalProperties": False, "required": ["summary", "voice", "barks", "sfx", "notes"], "properties": {
            "summary": {"type": "string"},
            "voice": {"type": "object", "additionalProperties": False, "required": ["description", "sample_line"],
                      "properties": {"description": {"type": "string", "minLength": 10}, "sample_line": {"type": "string", "minLength": 3}}},
            "barks": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": ["line", "emotion", "situation"],
                                                                "properties": {"line": {"type": "string", "minLength": 1}, "emotion": {"type": "string"}, "situation": {"type": "string"}}}},
            "sfx": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": ["event", "prompt"],
                                                              "properties": {"event": {"type": "string", "minLength": 1}, "prompt": {"type": "string", "minLength": 3}}}},
            "notes": {"type": "string"}}},
    },
    "asset_meta": {
        "system": ("You name and tag media assets (game, video, podcast, social, film) for a library. For each asset (given with its id, type, prompt and model) write: "
                   "name = short human-readable title (2-5 words, Title Case), tags = 3-6 lowercase single-word or kebab-case tags "
                   "(subject, material, style, use), description = one sentence for the team. Keep each id exactly as given."),
        "schema": {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {
            "items": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": ["id", "name", "tags", "description"],
                                                                "properties": {"id": {"type": "string"}, "name": {"type": "string"},
                                                                               "tags": {"type": "array", "items": {"type": "string"}}, "description": {"type": "string"}}}}}},
    },
}


def write_prompt(kind: str, inp: dict, project: str) -> str:
    if kind == "barks":
        sit = ", ".join(inp.get("situations") or []) or "a mix of combat, idle, hurt, victory and greeting"
        return (f"CHARACTER: {inp.get('character') or 'unnamed'}\nDESCRIPTION: {inp.get('description') or '-'}\n"
                f"PERSONALITY: {inp.get('personality') or '-'}\nSITUATIONS: {sit}\nCOUNT: {int(inp.get('count') or 8)} lines\n"
                f"LANGUAGE: {inp.get('language') or 'en'}\nPROJECT CONTEXT: {inp.get('context') or '-'}")
    if kind == "character_audio":
        return (f"CHARACTER: {str(inp.get('character') or '-')[:3000]}\n"
                f"DETAILED VISUAL PROMPT: {str(inp.get('description') or '-')[:3000]}\n"
                f"ART DIRECTION / PROJECT CONTEXT: {str(inp.get('context') or '-')[:1000]}\n"
                f"BARKS: {max(1, min(int(inp.get('count') or 6), 20))} lines\nSFX: {max(1, min(int(inp.get('sfx_count') or 4), 12))} events\n"
                f"LANGUAGE (sample line + barks): {inp.get('language') or 'en'}")
    if kind in ("sfx_events", "asset_list"):
        return f"DESIGN TEXT (game design, script, outline):\n{str(inp.get('text') or '')[:12000]}\n\nAbout {int(inp.get('count') or 12)} items. EXTRA: {inp.get('context') or '-'}"
    if kind == "asset_meta":
        rows = []
        for aid in (inp.get("asset_ids") or [])[:40]:
            try:
                a = _core().assets.get(aid)
            except (AssetError, KeyError):
                continue
            if a.get("project") != project:
                continue
            rows.append(json.dumps({"id": a["id"], "type": a["type"], "name": a.get("name"), "prompt": (a.get("prompt") or a.get("final_prompt") or "")[:300],
                                    "model": a.get("model_name"), "task": a.get("task"), "tags": a.get("tags")}, ensure_ascii=False))
        if not rows:
            raise HTTPException(400, "Pick assets of this project first.")
        return "ASSETS:\n" + "\n".join(rows) + f"\nCONTEXT: {inp.get('context') or '-'}"
    raise HTTPException(400, "Unknown writing tool.")


class WriteReq(BaseModel):
    kind: str
    project: str = "default"
    backend: str | None = None
    input: dict = Field(default_factory=dict)
    confirm_cost: bool = False


@router.post("/write")
def write(req: WriteReq):
    if req.kind not in WRITE_KINDS:
        raise HTTPException(400, "Unknown writing tool.")
    check_project(req.project)
    w = WRITE_KINDS[req.kind]
    prompt = write_prompt(req.kind, req.input, req.project)
    r = _core().textgen.generate(messages=[{"role": "system", "content": w["system"]}, {"role": "user", "content": prompt}],
                                 schema=w["schema"], backend=req.backend, purpose="write", temperature=0.8, max_tokens=3000,
                                 project=req.project, confirm_cost=req.confirm_cost)
    return {"kind": req.kind, "data": r["data"], "backend": r["backend"], "seconds": r["seconds"], "usd": r["usd"],
            "warnings": r.get("warnings") or []}


# ===================================================================== routes: backends / settings
@router.get("/backends")
def backends(check_auth: bool = True, project: str | None = None):
    c = _core()
    tg = c.textgen
    eff = c.settings_for(project) if project else None      # a project's text-backend defaults
    return {"backends": tg.backends(check_auth=check_auth),
            "defaults": {p: tg.default_for(p, eff) for p in ("improve", "write", "assistant")},
            "settings": tg.cfg, "tos": TOS_NOTE, "cli": {k: {"tested": v["tested"], "models": v["models"]} for k, v in CLI_INFO.items()}}


class TGSettings(BaseModel):
    improve: str | None = None
    write: str | None = None
    assistant: str | None = None
    cli: dict | None = None
    api_models: list | None = None
    endpoints: dict | None = None


@router.put("/settings")
def put_settings(req: TGSettings):
    patch = {}
    for k in ("improve", "write", "assistant"):
        v = getattr(req, k)
        if v is not None:
            if v != "":
                parse_backend(v)
            patch[k] = v or None
    if req.cli is not None:
        cli = {}
        for name, v in req.cli.items():
            if name in CLI_INFO and isinstance(v, dict):
                m = str(v.get("model") or "")
                if m and not re.fullmatch(r"[A-Za-z0-9._:\-\[\]]{1,80}", m):
                    raise HTTPException(400, "Invalid model name.")
                cli[name] = {"model": m or None}
        patch["cli"] = cli
    if req.api_models is not None:
        rows = []
        for e in req.api_models[:50]:
            if isinstance(e, dict) and e.get("provider") and e.get("model"):
                parse_backend(f"api:{e['provider']}/{e['model']}")
                rows.append({"provider": str(e["provider"]), "model": str(e["model"])[:120], "label": str(e.get("label") or "")[:60] or None,
                             "vision": bool(e.get("vision"))})
        patch["api_models"] = rows
    if req.endpoints is not None:
        eps = {}
        for name, url in req.endpoints.items():
            if re.fullmatch(r"[A-Za-z0-9_.\-]{1,40}", name) and isinstance(url, str) and re.match(r"^https?://", url):
                eps[name] = url.rstrip("/")[:300]
        patch["endpoints"] = eps
    c = _core()
    if patch:
        cur = dict(c.settings.get("textgen") or {})
        cur.update(patch)
        c.settings.update({"textgen": None})
        c.settings.update({"textgen": {k: v for k, v in cur.items() if v is not None}})
    return backends(check_auth=False)


class TestReq(BaseModel):
    backend: str


@router.post("/backends/test")
def test_backend(req: TestReq):
    r = _core().textgen.generate(messages=[{"role": "system", "content": "You are a connectivity test. Answer in JSON."},
                                           {"role": "user", "content": "Reply with ok=true and a three-word greeting for a media studio."}],
                                 schema={"type": "object", "additionalProperties": False, "required": ["ok", "greeting"],
                                         "properties": {"ok": {"type": "boolean"}, "greeting": {"type": "string"}}},
                                 backend=req.backend, purpose="write", temperature=0.2, max_tokens=60)
    return {"ok": True, "data": r["data"], "backend": r["backend"], "seconds": r["seconds"], "usd": r["usd"]}


# ===================================================================== routes: chats
class NewChat(BaseModel):
    project: str = "default"
    backend: str | None = None
    title: str = ""


@router.get("/chats")
def list_chats(project: str = "default", limit: int = 100):
    d = chats_dir(project)
    out = []
    if d.is_dir():
        for p in d.glob("c*.json"):
            if not CHAT_RE.match(p.stem):
                continue
            try:
                ch = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            last = (ch.get("messages") or [{}])[-1]
            out.append({"id": ch["id"], "title": ch.get("title") or "New chat", "updated": ch.get("updated"), "created": ch.get("created"),
                        "backend": ch.get("backend"), "messages": len(ch.get("messages") or []), "last": (last.get("text") or "")[:120],
                        "busy": ch["id"] in TURNS})
    out.sort(key=lambda x: x.get("updated") or "", reverse=True)
    return {"chats": out[:max(1, min(limit, 500))]}


@router.post("/chats")
def new_chat(req: NewChat):
    check_project(req.project)
    if req.backend:
        parse_backend(req.backend)
    cid = f"c{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    chat = {"id": cid, "project": req.project, "title": req.title[:80] or "New chat", "backend": req.backend or None,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "messages": []}
    save_chat(chat)
    return chat


@router.get("/chats/{cid}")
def get_chat(cid: str, project: str = "default"):
    ch = load_chat(project, cid)
    ch["busy"] = cid in TURNS
    return ch


@router.patch("/chats/{cid}")
def patch_chat(cid: str, patch: dict, project: str = "default"):
    with _LOCK:
        ch = load_chat(project, cid)
        if "title" in patch:
            ch["title"] = str(patch["title"] or "New chat")[:80]
        if "backend" in patch:
            if patch["backend"]:
                parse_backend(patch["backend"])
            ch["backend"] = patch["backend"] or None
        save_chat(ch)
    return ch


@router.delete("/chats/{cid}")
def delete_chat(cid: str, project: str = "default"):
    chat_path(project, cid).unlink(missing_ok=True)
    return {"id": cid, "deleted": True}


class MessageReq(BaseModel):
    project: str = "default"
    text: str
    backend: str | None = None


@router.post("/chats/{cid}/message")
def post_message(cid: str, req: MessageReq):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "Write a message first.")
    if len(text) > 8000:
        raise HTTPException(400, "Message too long (8000 characters max).")
    if cid in TURNS:
        raise HTTPException(409, "The assistant is still answering the previous message.")
    if req.backend:
        parse_backend(req.backend)
    with _LOCK:
        ch = load_chat(req.project, cid)
        um = {"id": _mid(ch), "role": "user", "text": text, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        ch["messages"].append(um)
        if req.backend:
            ch["backend"] = req.backend
        save_chat(ch)
    publish(ch, "message", message=um)
    cancel = threading.Event()
    TURNS[cid] = cancel
    threading.Thread(target=run_turn, args=(req.project, cid, text, req.backend, cancel), daemon=True, name=f"assistant-{cid}").start()
    return {"ok": True, "message": um}


@router.post("/chats/{cid}/cancel")
def cancel_turn(cid: str):
    ev = TURNS.get(cid)
    if ev:
        ev.set()
    return {"ok": True, "cancelled": bool(ev)}


class PlanReq(BaseModel):
    project: str = "default"
    plan: dict
    message: str | None = None
    steps: list[str] | None = None


@router.post("/chats/{cid}/validate")
def validate_endpoint(cid: str, req: PlanReq):
    load_chat(req.project, cid)
    return validate_plan(req.plan, req.project)


@router.post("/chats/{cid}/run")
def run_plan(cid: str, req: PlanReq):
    if not req.message:
        raise HTTPException(400, "message (the plan's message id) is required.")
    v = validate_plan(req.plan, req.project)
    plan = v["plan"]
    if req.steps is not None:
        keep = set(req.steps)
        for s in plan["steps"]:
            s["enabled"] = s["id"] in keep
        # a selected step whose input step is not selected: its inputs come from an earlier run of that step, if any
    sel = [s for s in plan["steps"] if s.get("enabled", True)]
    if not sel:
        raise HTTPException(400, "Select at least one step.")
    sel_ids = {s["id"] for s in sel}
    errs = {k: e for k, e in v["errors"].items() if k in sel_ids or k == "plan"}
    if errs:
        return {"ok": False, **v, "errors": errs}
    with _LOCK:
        ch = load_chat(req.project, cid)
        msg = next((m for m in ch["messages"] if m.get("id") == req.message), None)
        if msg is None:
            raise HTTPException(404, "Plan message not found.")
        prev = (msg.get("run") or {}).get("steps") or {}
        if msg.get("run", {}).get("status") == "running":
            raise HTTPException(409, "This plan is already running.")
        # unselected input steps that already ran: re-use their outputs (lets the user re-run one step)
        for s in sel:
            for i in s.get("inputs") or []:
                ref = i.get("step")
                if ref and ref not in sel_ids:
                    outs = (prev.get(ref) or {}).get("outputs") or []
                    if not outs:
                        return {"ok": False, **v, "errors": {s["id"]: [f"Input step '{ref}' is not selected and has no earlier outputs."]}}
                    i.pop("step")
                    if i.get("use") in ("all", "each"):
                        i["assets"] = outs
                    i["asset"] = outs[0]
        msg["plan"] = {**v, "plan": {**plan, "steps": [{k: x for k, x in s.items() if k != "enabled"} | {"enabled": s.get("enabled", True)} for s in plan["steps"]]}}
        msg["approved"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_chat(ch)
    run_plan_obj = {"title": plan["title"], "steps": _expand_asset_lists(sel)}
    return {"ok": True, **_start_run(req.project, cid, req.message, run_plan_obj)}


def _expand_asset_lists(steps: list[dict]) -> list[dict]:
    out = []
    for s in steps:
        s = dict(s)
        ins = []
        for i in s.get("inputs") or []:
            if i.get("assets") and i.get("use") == "all":
                ins += [{"asset": a, "role": i.get("role")} for a in i["assets"]]
            else:
                ins.append({k: v for k, v in i.items() if k != "assets"})
        s["inputs"] = ins
        out.append(s)
    return out


@router.post("/runs/{rid}/cancel")
def cancel_run(rid: str):
    r = PLAN_RUNS.get(rid)
    if r:
        r["cancel"].set()
    return {"ok": True, "cancelled": bool(r)}
