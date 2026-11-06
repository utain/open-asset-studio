"""Pipelines plugin: server-side state of the guided flows (Concept -> 3D prop, Character kit).

The flows themselves are normal jobs submitted by the Pipelines page (every result is an ordinary asset with
lineage). What lives here is the *flow record*: which step the user is on, what was picked / approved, the job and
asset ids of each step. It is stored per project, so a flow can be resumed from any browser on the LAN.

  GET    /api/x/pipelines/flows?project=      -> {flows: [record, ...]}   (newest first, without big fields)
  GET    /api/x/pipelines/flows/{id}?project= -> record
  PUT    /api/x/pipelines/flows/{id}          body {project, kind, title, step, state, ...} -> record
  DELETE /api/x/pipelines/flows/{id}?project=
  GET    /api/x/pipelines/capabilities        -> which tasks have a ready model (Character kit sound steps)
Files: data/pipelines/<project>/<flow id>.json (ids are validated; nothing else is ever opened).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .. import config
from ..assets import check_project

router = APIRouter()
CORE = None
FLOW_RE = re.compile(r"^f[0-9]{8}-[0-9]{6}-[0-9a-z]{4,12}$")
KINDS = ("concept3d", "character")
MAX_BYTES = 256 * 1024
SOUND_TASKS = ("voice_design", "tts", "voice_clone", "sfx")


def flows_dir(project: str) -> Path:
    return config.DATA / "pipelines" / check_project(project)


def flow_path(project: str, fid: str) -> Path:
    if not isinstance(fid, str) or not FLOW_RE.match(fid):
        raise HTTPException(400, "Invalid flow id.")
    d = flows_dir(project)
    p = (d / f"{fid}.json").resolve()
    if p.parent != d.resolve():
        raise HTTPException(400, "Invalid flow id.")
    return p


def validate(rec: dict, fid: str) -> dict:
    if not isinstance(rec, dict):
        raise HTTPException(400, "A flow record must be an object.")
    kind = rec.get("kind")
    if kind not in KINDS:
        raise HTTPException(400, f"kind must be one of {', '.join(KINDS)}")
    out = {k: v for k, v in rec.items() if k in ("kind", "title", "step", "state", "status", "project", "created")}
    out["id"] = fid
    out["title"] = str(out.get("title") or "")[:120]
    out["status"] = out.get("status") if out.get("status") in ("active", "done", "archived") else "active"
    if not isinstance(out.get("state"), dict):
        out["state"] = {}
    out["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out.setdefault("created", out["updated"])
    if len(json.dumps(out, default=str)) > MAX_BYTES:
        raise HTTPException(413, "Flow record too large.")
    return out


@router.get("/flows")
def list_flows(project: str = "default", limit: int = 50):
    d = flows_dir(project)
    items = []
    if d.is_dir():
        for p in d.glob("f*.json"):
            if not FLOW_RE.match(p.stem):
                continue
            try:
                r = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            items.append({k: r.get(k) for k in ("id", "kind", "title", "step", "status", "created", "updated")} |
                         {"thumb": (r.get("state") or {}).get("thumb")})
    items.sort(key=lambda r: r.get("updated") or "", reverse=True)
    return {"flows": items[:max(1, min(limit, 500))]}


@router.get("/flows/{fid}")
def get_flow(fid: str, project: str = "default"):
    p = flow_path(project, fid)
    if not p.is_file():
        raise HTTPException(404, "Flow not found.")
    return json.loads(p.read_text())


@router.put("/flows/{fid}")
def put_flow(fid: str, rec: dict):
    project = rec.get("project") or "default"
    p = flow_path(project, fid)
    out = validate(rec, fid)
    out["project"] = project
    if p.is_file():
        try:
            out["created"] = json.loads(p.read_text()).get("created") or out["created"]
        except (OSError, ValueError):
            pass
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, p)
    return out


@router.delete("/flows/{fid}")
def delete_flow(fid: str, project: str = "default"):
    p = flow_path(project, fid)
    p.unlink(missing_ok=True)
    return {"id": fid, "deleted": True}


@router.get("/capabilities")
def capabilities():
    reg = CORE.registry if CORE else None
    out = {}
    for t in ("t2i", "edit", "extract", "rmbg", "recenter", "shape", "texture", "mesh_export", *SOUND_TASKS):
        ready = [m["id"] for m in (reg.models.values() if reg else []) if t in m.get("tasks", []) and not m.get("test")
                 and reg.status.get(m["id"], {}).get("status") == "ready"]
        out[t] = {"ready": ready, "default": reg.default_for(t) if reg else None}
    return {"tasks": out}


def setup(core):
    global CORE
    CORE = core
