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
    if a.get("type") == "mesh":
        out["mesh3d"] = mesh_view(a, files, full)
    return out


# attachments the 3D tasks write next to mesh.glb, and what they are for
MESH_ATTACHMENTS = {
    "skeleton.json": "joint names / hierarchy of the rig (rig task, tool rig.edit)",
    "rigged_plain.glb": "the same rigged mesh without the flex / wiggle test clips (rig task)",
    "white_mesh.glb": "untextured mesh (shape / texture tasks)",
    "motion.npz": "raw motion data (motion task)",
    "retarget.json": "how the motion was mapped onto this skeleton (motion task)",
}


def mesh_view(a: dict, files: list, full: bool = False) -> dict:
    """3D facts of a mesh asset: rigged, joints, animation clip names, textured, size, motion, attachment files."""
    info, w = a.get("info") or {}, a.get("worker") or {}
    anims = w.get("animations") or info.get("animations") or []
    rigged = bool(w.get("rigged") or info.get("rigged"))
    out = {"rigged": rigged, "joints": w.get("joints") or info.get("joints"), "animations": list(anims),
           "needs_reskin": True if info.get("needs_reskin") else None,
           "textured": w.get("textured") if "textured" in w else None,
           "vertices": info.get("vertices"), "faces": info.get("faces"), "motion": w.get("motion")}
    out = {k: v for k, v in out.items() if v not in (None, [], {})}
    out["rigged"], out["animations"] = rigged, list(anims)
    att = [f["name"] for f in files if f.get("role") == "attachment"]
    if att:
        out["attachments"] = {n: MESH_ATTACHMENTS.get(n, "attachment") for n in att}
    if not full:
        return out
    if info.get("needs_reskin"):
        out["hint"] = ("Rig edited: new bones have no skin weight yet. Run submit_job(task='reskin', inputs=[{asset: <this id>, role: 'mesh'}]) "
                       "(GPU, ~35 s) so the mesh follows them; get_rig shows the bones.")
    elif rigged and not anims:
        out["hint"] = "Rigged but no clips: run submit_job(tool='mesh.flex_preview') or task 'motion' on it."
    elif not rigged:
        out["hint"] = "Not rigged: run task 'rig' on it before 'motion' or tool 'mesh.flex_preview'."
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
        if m.get("user_presets"):
            out["user_presets"] = m["user_presets"]
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


# Params of the CPU tools (handlers in app/plugins/*.py). `in` = the input assets the tool needs.
AUDIO_OUT = {"format": "wav | ogg | mp3 | flac (default wav)", "bit_depth": "16 | 24 (default 16)", "quality": "0..1 for ogg / mp3 (default 0.6)"}
TOOL_PARAMS = {
    "image.convert": {"in": "1+ image assets", "params": {"format": "png | webp | jpg (default png)", "max_side": "16..8192, 0 = keep size", "square": "bool, pad to a transparent square"}},
    "asset.export": {"in": "1+ audio / voice / image assets", "params": {"preset": "export preset id (studio_info topic=profiles)", "audio": "{format, sample_rate, bit_depth, lufs, channels, ...}", "image": "{format, width, height, aspect, fit, max_side, quality}"},
                     "note": "Writes NEW derived assets (originals untouched). To download a converted copy instead use export_assets."},
    "mesh.flex_preview": {"in": "1+ mesh assets (GLB) that are RIGGED (have a skin; make one with task 'rig')",
                          "params": {"clips": "list of 'flex' (each joint swings in turn, a skinning test) and 'wiggle' (gentle loop); default both", "angle": "degrees, 5..80 (default 30)"},
                          "note": "Writes a NEW mesh asset with those animation clips (get_asset -> mesh3d.animations). Fails with kind 'input' when the mesh has no skeleton."},
    "rig.edit": {"in": "ONE rigged mesh asset (GLB with a skeleton; look at it with get_rig first)",
                 "params": {"ops": ("list of operations, applied in order (later ones see earlier results). Bones are named by node name, by the index get_rig shows, or by role word on a humanoid "
                                    "('left elbow', 'left forearm', 'head', 'hips', 'spine 2'). Units: metres, +Y up, the character faces +Z, its left is +X. Ops: "
                                    "{op:'rename', bone, to} | {op:'rename_standard', scheme:'mixamo'|'vrm'|'unreal'} (humanoids only) | "
                                    "{op:'move_joint', bone, offset:[x,y,z] | delta:[x,y,z] (fractions of the bone length) | to:[x,y,z], children:'stay'(default)|'follow', mirror:bool} | "
                                    "{op:'delete_bone', bone | match:'gun*', merge_weights_to?, include_children?:bool} (weights go to the parent, the mesh stays bound) | "
                                    "{op:'add_bone', parent, to:[x,y,z] | direction:'up'|'down'|'left'|'right'|'forward'|'back'|[x,y,z] + length, name?} | "
                                    "{op:'add_chain', parent, count, direction, length (per bone) | total_length, name_prefix?} (tails, ears, hair) | {op:'reparent', bone, parent}"),
                            "clips": "bool, rebuild the flex / wiggle preview clips when the input had them (default true)"},
                 "note": ("Writes a NEW mesh asset (rest pose, mesh and textures unchanged; other animation clips are dropped). New bones have ZERO skin weight: when get_asset -> mesh3d.needs_reskin is true, "
                          "run submit_job(task='reskin') on the result. Examples: ops=[{op:'add_chain', parent:'hips', count:5, direction:'back', length:0.08, name_prefix:'tail'}] (a 5-bone tail); "
                          "[{op:'move_joint', bone:'left elbow', offset:[0,-0.03,0]}]; [{op:'delete_bone', match:'gun*', include_children:true}]; [{op:'rename_standard', scheme:'mixamo'}]. "
                          "Errors (kind 'input') name the candidates when a bone is unknown or ambiguous.")},
    "audio.process": {"in": "1+ audio / voice assets", "params": {"ops": "{trim:{threshold_db,pad_ms}, fade:{in_ms,out_ms}, loop:{xfade_ms}, normalize:{lufs,ceiling_db}, sample_rate, channels:'mono'|'stereo'} (or the flat keys trim, fade, loop, normalize, lufs, sample_rate, channels)", **AUDIO_OUT}, "note": "A chain of the single audio tools in one pass."},
    "audio.trim": {"in": "audio", "params": {"threshold_db": "default -50", "pad_ms": "default 30", **AUDIO_OUT}},
    "audio.fade": {"in": "audio", "params": {"in_ms": "default 10", "out_ms": "default 200", **AUDIO_OUT}},
    "audio.normalize": {"in": "audio", "params": {"lufs": "target loudness (default -16)", "ceiling_db": "true-peak ceiling (default -1)", **AUDIO_OUT}},
    "audio.loop": {"in": "audio", "params": {"xfade_ms": "cross-fade length (default 500)", **AUDIO_OUT}},
    "audio.convert": {"in": "audio", "params": dict(AUDIO_OUT)},
    "audio.channels": {"in": "audio", "params": {"channels": "mono | stereo (required)", **AUDIO_OUT}},
    "audio.resample": {"in": "audio", "params": {"sample_rate": "Hz (default 44100)", **AUDIO_OUT}},
    "audio.make_voice": {"in": "1 audio asset", "params": {"name": "voice name", "ref_text": "transcript of the clip", "language": "language code", "design_prompt": "text", "character": "text"},
                         "note": "Saves a trimmed, -20 LUFS mono copy (max 30 s) as a voice asset, the reference for task voice_clone."},
    "video.frame": {"in": "1+ video assets", "params": {"at": "'last' (default) | 'first' | seconds"}, "note": "Writes an image asset (e.g. the start frame for i2v)."},
    "video.trim": {"in": "video", "params": {"start": "seconds (default 0)", "end": "seconds, 0 = to the end"}},
    "video.loop": {"in": "video", "params": {"mode": "pingpong (default, clips < 60 s) | repeat", "times": "2..8 (default 2)"}},
    "video.speed": {"in": "video", "params": {"speed": "0.25..4 (default 1)"}},
    "video.reframe": {"in": "video", "params": {"aspect": "16:9 | 9:16 | 1:1 | 4:5 | 4:3 | 3:4 (default 9:16)", "fit": "crop (default) | blur | pad", "short_side": "360..2160 (default 1080)"}},
    "video.convert": {"in": "video", "params": {"format": "mp4 (default) | webm", "max_side": "0 = keep, up to 7680", "fps": "0 = keep", "crf": "10..40 (default 20, lower = better)"}},
    "video.concat": {"in": "2+ video assets, joined in order", "params": {}, "note": "Output size / fps follow the first clip; clips without sound get silence."},
    "video.mux": {"in": "one video + one audio / voice asset", "params": {"volume": "0..4 (default 1)", "keep_original": "bool, mix with the video's own sound", "fade_out": "seconds 0..10 (default 0.5)"}},
}


def t_list_tools(_a, _ctx):
    tools = http("GET", "/api/bootstrap").get("tools") or {}
    out = [{"id": k, "label": v, **TOOL_PARAMS.get(k, {"note": "params: see docs/CONTRACTS.md and app/plugins/*.py"})} for k, v in tools.items()]
    return {"tools": out, "hint": "Run one with submit_job(tool=<id>, params=..., inputs=[{asset: <id>}]); the result is a new asset whose parents are the inputs."}


def t_get_rig(a, _ctx):
    d = http("GET", f"/api/x/rig/{urllib.parse.quote(a['asset'])}/skeleton")
    hum = d.get("humanoid") or {}
    rows = [{"index": r["index"], "name": r["name"], "parent": r["parent"], "position": [round(x, 3) for x in r["position"]],
             **({"role": r["role"]} if r.get("role") else {}), "children": r["children"], "weight": round(r["weight_share"], 3)} for r in d.get("joints", [])]
    out = {"asset": d.get("asset"), "name": d.get("name"), "joints": rows, "count": d.get("count"), "height": d.get("height"), "bounds": d.get("bounds"),
           "units": d.get("units"), "humanoid": hum if hum.get("ok") else {"ok": False, "reason": hum.get("reason")},
           "animations": d.get("animations"), "zero_weight": d.get("zero_weight"), "needs_reskin": d.get("needs_reskin")}
    out["hint"] = ("Edit with submit_job(tool='rig.edit', params={ops: [...]}, inputs=[{asset: <id>, role: 'mesh'}]) - list_tools shows the ops; "
                   "bone = name, index or role word (a humanoid's roles are listed above). " +
                   ("This rig has bones without weights: run submit_job(task='reskin') on it." if d.get("zero_weight") else "After adding bones, run task 'reskin'."))
    return out


def t_list_projects(_a, _ctx):
    return http("GET", "/api/projects")


def t_create_project(a, _ctx):
    if a.get("update"):
        patch = {k: a[k] for k in ("title", "description") if k in a}
        return http("PATCH", f"/api/projects/{urllib.parse.quote(a['id'])}", patch)
    return http("POST", "/api/projects", {"id": a["id"], "title": a.get("title") or a["id"], "description": a.get("description", "")})


def t_search_assets(a, _ctx):
    d = http("GET", "/api/assets", query={"project": a.get("project", "default"), "q": a.get("q"), "type": a.get("type"),
                                          "tag": a.get("tag"), "model": a.get("model"), "task": a.get("task"),
                                          "sort": a.get("sort", "new"), "limit": min(int(a.get("limit", 20)), 100)})
    return {"total": d.get("total"), "assets": [asset_view(x) for x in d.get("items", [])]}


def t_get_asset(a, _ctx):
    asset = http("GET", f"/api/assets/{urllib.parse.quote(a['id'])}")
    view = asset_view(asset, True)
    if a.get("lineage"):
        lin = http("GET", f"/api/assets/{urllib.parse.quote(a['id'])}/lineage")
        brief = lambda m: {k: m[k] for k in ("id", "type", "name", "task", "model", "missing") if m.get(k)}  # noqa: E731
        view["lineage"] = {k: [brief(m) for m in lin.get(k) or []] for k in ("ancestors", "children", "siblings")}
    out = [{"type": "text", "text": json.dumps(view, ensure_ascii=False)}]
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


def t_retry_job(a, ctx):
    j = http("POST", f"/api/jobs/{urllib.parse.quote(a['id'])}/retry",
             {"confirm_cost": bool(a.get("confirm_cost")), "confirm_upload": bool(a.get("confirm_upload"))})
    if a.get("wait_s", 300) > 0:
        j = wait_job(j["id"], float(a.get("wait_s", 300)), ctx)
    return finished(j, bool(a.get("preview", True)))


def t_delete_asset(a, _ctx):
    """Soft delete: the studio moves the folder to the project's .trash (recoverable by hand). Needs confirm=true."""
    aid = urllib.parse.quote(a["id"])
    meta = http("GET", f"/api/assets/{aid}")
    if a.get("confirm") is not True:
        kids = http("GET", f"/api/assets/{aid}/lineage").get("children") or []
        raise StudioError(f"Deleting '{meta.get('name')}' ({meta.get('type')}, {a['id']}) removes it from the library"
                          + (f"; {len(kids)} derived asset(s) keep their lineage but lose this parent" if kids else "") + ".",
                          "confirm_delete", "Ask the user, then call delete_asset again with confirm: true.")
    return http("DELETE", f"/api/assets/{aid}")


def t_improve_prompt(a, _ctx):
    if a.get("list_styles"):
        d = http("GET", "/api/improve/styles")
        return {"styles": [{"id": x.get("id"), "label": x.get("label"), "modality": x.get("modality"),
                            "chips": [{"id": c.get("id"), "label": c.get("label")} for c in x.get("chips") or []]} for x in d.get("styles", [])],
                "hint": "improve_prompt(text, model_id=<model>) picks the style of that model; pass target_style to force one."}
    if not (a.get("text") or "").strip():
        raise StudioError("Give the idea in `text` (Thai or English), or list_styles=true.", "input")
    backend = a.get("backend")
    if backend and backend != "local" and not a.get("confirm_upload"):
        raise StudioError(f"Backend '{backend}' sends the text to a third party (CLI / API).", "consent",
                          "Leave backend empty to use the local LLM, or ask the user and pass confirm_upload: true.")
    body = {k: a[k] for k in ("text", "target_style", "model_id", "task", "lang", "n", "image_asset", "context", "chips", "backend") if a.get(k) not in (None, "")}
    return http("POST", "/api/improve", body, timeout=300)


def t_save_preset(a, _ctx):
    mid = urllib.parse.quote(a["model"])
    if a.get("delete"):
        if a.get("confirm") is not True:
            raise StudioError(f"Deleting the preset '{a['name']}' of {a['model']} cannot be undone.", "confirm_delete",
                              "Ask the user, then call save_preset again with delete: true and confirm: true.")
        return http("DELETE", f"/api/presets/{mid}/{urllib.parse.quote(a['name'], safe='')}")
    if not isinstance(a.get("params"), dict) or not a["params"]:
        raise StudioError("Give the param values to store in `params` (names as in get_model).", "input")
    return http("PUT", f"/api/presets/{mid}", {"name": a["name"], "params": a["params"]})


def _local_export_path(url: str) -> str | None:
    """/api/export/<zip> | /api/x/sound/export/<zip> | /api/export/render/<token>/<name> -> the file under data/exports."""
    parts = url.split("/")
    p = DATA / "exports" / (f"render-{parts[-2]}/{parts[-1]}" if "/render/" in url else parts[-1])
    return str(p) if p.is_file() else None


def t_export_assets(a, _ctx):
    ids = a.get("ids") or []
    if not ids:
        raise StudioError("Give the asset ids to export.", "input")
    spec = {k: a[k] for k in ("preset", "audio", "image") if a.get(k)}
    if a.get("game_audio") is not None:
        g = dict(a["game_audio"] or {})
        d = http("POST", "/api/x/sound/export", {"ids": ids, **g}, timeout=600)
    elif len(ids) == 1 and spec and not a.get("zip"):
        d = http("POST", f"/api/assets/{urllib.parse.quote(ids[0])}/render", {"export": spec}, timeout=600)
    elif len(ids) == 1 and not spec and not a.get("zip"):
        return asset_view(http("GET", f"/api/assets/{urllib.parse.quote(ids[0])}"), True)   # nothing to convert: the files as stored
    else:
        d = http("POST", "/api/export", {"ids": ids, "naming": a.get("naming", "name"), "include_meta": a.get("include_meta", True),
                                         "include_attachments": a.get("include_attachments", True), **({"export": spec} if spec else {})}, timeout=600)
    out = {k: v for k, v in d.items() if k != "url"}
    out["url"] = API + d["url"]
    path = _local_export_path(d["url"])
    if path:
        out["path"] = path
    return out


def _info_profiles(_a):
    d = http("GET", "/api/profiles")
    return {"current": d.get("current"), "default_export": d.get("default_export"), "license_filter": d.get("license"),
            "profiles": [{k: p.get(k) for k in ("id", "label", "summary", "aspect", "aspects", "export", "export_presets")} for p in d.get("profiles", [])],
            "export_presets": d.get("export_presets"),
            "hint": "Use a preset id with export_assets(preset=...) or tool asset.export."}


def _info_cloud(_a):
    d = http("GET", "/api/x/providers")
    return {"allow_cloud": d.get("allow_cloud"), "budget_usd": d.get("budget_usd"), "confirm_usd": d.get("confirm_usd"), "spend": d.get("spend"),
            "providers": [{"id": p.get("id"), "label": p.get("label"), "key_set": bool((p.get("key") or {}).get("set")), "modalities": p.get("modalities"),
                           "models": p.get("models")} for p in d.get("providers", [])],
            "models": [{"id": m.get("id"), "name": m.get("name"), "tasks": m.get("tasks"), "status": status_of(m), "example_usd": m.get("example_usd")}
                       for m in d.get("models", [])],
            "hint": "Cloud models cost money and upload inputs: submit_job needs confirm_cost / confirm_upload after the user agreed."}


def _info_flows(a):
    if a.get("id"):
        return http("GET", f"/api/x/pipelines/flows/{urllib.parse.quote(a['id'])}", query={"project": a.get("project", "default")})
    return {"flows": http("GET", "/api/x/pipelines/flows", query={"project": a.get("project", "default"), "limit": 50}).get("flows"),
            "ready_tasks": http("GET", "/api/x/pipelines/capabilities").get("tasks"),
            "hint": "Flows are the Pipelines page's saved guided runs (Concept to 3D prop, Character kit); their steps are ordinary jobs and assets. "
                    "To do the same from here, chain submit_job calls (see the 3D recipe in the server instructions)."}


def _info_logs(a):
    d = http("GET", "/api/system/logs", query={"name": a.get("name", "studio"), "lines": min(int(a.get("lines", 60)), 300)})
    return {k: d.get(k) for k in ("name", "available", "lines")}


# topic -> (what it is, function(args))
INFO_TOPICS = {
    "profiles": ("use-case profiles, export presets (loudness / format / size) and the licence filter", _info_profiles),
    "voices": ("the project's voice library (voice assets usable as clone references)", lambda a: http("GET", "/api/x/sound/voices", query={"project": a.get("project", "default")})),
    "video_formats": ("aspect ratios (px) and per-platform defaults for video reframing", lambda a: http("GET", "/api/x/video/formats")),
    "dub_options": ("dubbing: voice-clone models with their languages, translation backends, ready tasks", lambda a: http("GET", "/api/x/media_audio/dub/options")),
    "ready_tasks": ("which models are ready for each pipeline task (t2i, edit, rmbg, shape, texture, tts, ...)", lambda a: http("GET", "/api/x/pipelines/capabilities")),
    "flows": ("saved Pipelines flows of a project (pass id for one flow)", _info_flows),
    "cloud": ("cloud providers (keys set?, budget, spend) and cloud models with example prices", _info_cloud),
    "llm": ("the local prompt-improver LLM: running or stopped, model", lambda a: http("GET", "/api/improve/status")),
    "comfyui": ("whether the optional ComfyUI server is reachable", lambda a: http("GET", "/api/x/image/comfyui")),
    "downloads": ("model weight downloads in progress", lambda a: http("GET", "/api/downloads")),
    "logs": ("tail of a studio log (name: studio, worker-<backend>, ...; lines up to 300)", _info_logs),
}


def t_studio_info(a, _ctx):
    topic = a.get("topic")
    if topic not in INFO_TOPICS:
        return {"topics": {k: v[0] for k, v in INFO_TOPICS.items()}}
    return INFO_TOPICS[topic][1](a)


S, I, B, O = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}, {"type": "object"}
PROJECT = {"type": "string", "description": "project id (default: default)"}
INPUTS = {"type": "array", "description": "input assets, e.g. [{asset: '<id>', role: 'ref'}] (roles per model: get_model -> input_slots)",
          "items": {"type": "object", "properties": {"asset": S, "role": S}, "required": ["asset"]}}
RO = {"readOnlyHint": True, "openWorldHint": False}
TASKS_HELP = 'Tasks: image (t2i, edit, multi_ref, inpaint, rgba, extract, rmbg, upscale, recenter), 3D (multiview, text3d, shape, texture, rig = auto-rig a mesh, reskin = recompute the skin weights of a rigged mesh after editing its skeleton with tool rig.edit, motion = text to motion for a rigged humanoid, mesh_export), audio (sfx, ambience, music, song, tts, voice_design, voice_clone, voice_convert, sing, s2s, v2a, separate, dub, qa_clap, qa_asr), text (transcribe, subtitles, audio_ask), video (t2v, i2v, flf2v, interpolate).'
TOOLS = [
    {"name": "studio_status", "description": "Studio health, queue (running / queued jobs), free memory, loaded models, profile and licence filter.",
     "inputSchema": {"type": "object", "properties": {}}, "annotations": RO, "fn": t_status},
    {"name": "list_models", "description": "Generation tasks and the models for them, with status, licence and speed. Filter by task for params and input slots. " + TASKS_HELP,
     "inputSchema": {"type": "object", "properties": {"task": S, "modality": {"type": "string", "enum": ["image", "mesh", "audio", "video", "text"]},
                                                      "ready_only": {**B, "description": "default true"}, "commercial_only": B, "detail": B}},
     "annotations": RO, "fn": t_list_models},
    {"name": "get_model", "description": "One model: params (name, type, default, range, choices), presets, input slots, features, licence.",
     "inputSchema": {"type": "object", "properties": {"id": S}, "required": ["id"]}, "annotations": RO, "fn": t_get_model},
    {"name": "list_tools", "description": "CPU tools that post-process existing assets (audio.*, image.convert, video.*, asset.export, mesh.flex_preview, rig.edit), each with its input assets and params. Run with submit_job(tool=...).",
     "inputSchema": {"type": "object", "properties": {}}, "annotations": RO, "fn": t_list_tools},
    {"name": "get_rig", "description": ("The skeleton of a rigged mesh: every bone with index, name, parent, world position, role on a humanoid (hips, left_forearm, head ...), "
                                       "child count and share of the skin weight; humanoid detection (role -> bone name), animation clips, bones without weights. Look at it before "
                                       "editing a rig with submit_job(tool='rig.edit') (add a tail, move / delete / rename bones) and re-skinning with task 'reskin'."),
     "inputSchema": {"type": "object", "properties": {"asset": {**S, "description": "mesh asset id (a rigged GLB)"}}, "required": ["asset"]}, "annotations": RO, "fn": t_get_rig},
    {"name": "list_projects", "description": "Projects (asset libraries).", "inputSchema": {"type": "object", "properties": {}}, "annotations": RO, "fn": t_list_projects},
    {"name": "create_project", "description": "Create a project (id: lowercase letters, digits, - and _), or with update=true change the title / description of an existing one.",
     "inputSchema": {"type": "object", "properties": {"id": S, "title": S, "description": S, "update": B}, "required": ["id"]}, "fn": t_create_project},
    {"name": "search_assets", "description": "Search a project's assets (newest first). Results include file URLs and local paths.",
     "inputSchema": {"type": "object", "properties": {"project": PROJECT, "q": S, "type": {"type": "string", "description": "image | audio | voice | mesh | video | text (comma list ok)"},
                                                      "tag": S, "model": S, "task": S, "sort": {"type": "string", "enum": ["new", "old", "rating", "name"]}, "limit": I}},
     "annotations": RO, "fn": t_search_assets},
    {"name": "get_asset", "description": ("One asset: prompt, model, params, seed, info (size / duration / LUFS ...), files with local paths. Meshes add mesh3d: "
                                                     "rigged, joints, animation clip names, textured, attachments (skeleton.json, rigged_plain.glb, motion.npz ...). "
                                                     "lineage=true adds ancestors / derived assets / batch siblings. preview=true also returns the image."),
     "inputSchema": {"type": "object", "properties": {"id": S, "preview": B, "lineage": B}, "required": ["id"]}, "annotations": RO, "fn": t_get_asset},
    {"name": "submit_job", "description": (
        "Generate assets: run a model on a task (or a CPU tool on existing assets) through the studio's queue. One GPU job runs at a time. "
        "Waits up to wait_s (default 300, max 600) and returns the new assets with local file paths (and image previews). "
        "Use list_models / get_model first for valid params and input slots (roles); list_tools for CPU tools. seed=-1 is random; the real seed is in the asset. "
        "3D chain: t2i / rgba -> recenter -> shape -> texture -> rig (inputs=[{asset: mesh}]) -> motion (inputs=[{asset: rigged mesh, role: 'mesh'}], params.prompt); "
        "or tool 'mesh.flex_preview' on a rigged mesh for test clips. "
        "Edit a rig: get_rig(asset) to see the bones, then tool 'rig.edit' with params.ops (add_chain / add_bone / move_joint / delete_bone / rename / rename_standard / reparent; list_tools has the full syntax), "
        "e.g. submit_job(tool='rig.edit', params={ops:[{op:'add_chain', parent:'hips', count:5, direction:'back', length:0.08, name_prefix:'tail'}]}, inputs=[{asset:'<rigged mesh>', role:'mesh'}]); "
        "the result is a NEW asset; if it reports needs_reskin (new bones have no weights) run submit_job(task='reskin', inputs=[{asset:'<edited mesh>', role:'mesh'}]) (GPU, ~35 s, keeps the skeleton, recomputes the weights). "
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
    {"name": "retry_job", "description": "Run a failed or cancelled job again with the same settings (new job, same queue). Cloud jobs need confirm_cost / confirm_upload again.",
     "inputSchema": {"type": "object", "properties": {"id": S, "wait_s": I, "preview": B, "confirm_cost": B, "confirm_upload": B}, "required": ["id"]},
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}, "fn": t_retry_job},
    {"name": "delete_asset", "description": "Delete an asset (moved to the project's .trash). Without confirm=true it only explains what would be deleted: ask the user first.",
     "inputSchema": {"type": "object", "properties": {"id": S, "confirm": {**B, "description": "must be true, after the user agreed"}}, "required": ["id"]},
     "annotations": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False}, "fn": t_delete_asset},
    {"name": "improve_prompt", "description": (
        "Turn a rough idea (Thai or English) into a model-ready prompt with variations, using the local prompt LLM (it may load for a few seconds). "
        "Pass model_id (and task) so the style matches the model you will use, then give the result's `prompt` / `negative` to submit_job. "
        "list_styles=true lists the prompt styles and their style chips instead."),
     "inputSchema": {"type": "object", "properties": {"text": S, "model_id": S, "task": S, "target_style": S, "lang": {"type": "string", "enum": ["th", "en"]},
                                                      "n": {**I, "description": "variations, 0..6 (default 3)"}, "chips": {"type": "array", "items": S},
                                                      "context": S, "image_asset": {**S, "description": "image asset id to describe / base the prompt on"},
                                                      "backend": {**S, "description": "default: the local LLM. cli:* / api:* backends send the text off-machine and need confirm_upload"},
                                                      "confirm_upload": B, "list_styles": B}},
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}, "fn": t_improve_prompt},
    {"name": "save_preset", "description": "Save (or overwrite) a user preset of param values for a model, shown next to the built-in ones in get_model. delete=true + confirm=true removes one.",
     "inputSchema": {"type": "object", "properties": {"model": S, "name": S, "params": O, "delete": B, "confirm": B}, "required": ["model", "name"]},
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}, "fn": t_save_preset},
    {"name": "export_assets", "description": (
        "Package assets for delivery as DERIVED files (originals are never modified). One asset with an export preset gives one converted file; several give a zip "
        "(files + meta.json + attachments such as skeleton.json). preset ids: studio_info(topic=profiles). game_audio={format, folder, sample_rate, channels, normalize_lufs, by_character} "
        "instead builds an engine-ready sound folder zip with a manifest. Returns url (+ local path)."),
     "inputSchema": {"type": "object", "properties": {"ids": {"type": "array", "items": S}, "preset": S, "audio": O, "image": O, "zip": {**B, "description": "force a zip for one asset"},
                                                      "naming": {"type": "string", "enum": ["name", "id"]}, "include_meta": B, "include_attachments": B, "game_audio": O},
                     "required": ["ids"]},
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}, "fn": t_export_assets},
    {"name": "studio_info", "description": "Studio reference data. topic: " + "; ".join(f"{k} ({v[0]})" for k, v in INFO_TOPICS.items()) + ". Omit topic to list them.",
     "inputSchema": {"type": "object", "properties": {"topic": {"type": "string", "enum": list(INFO_TOPICS)}, "project": PROJECT, "id": {**S, "description": "flows: a flow id"},
                                                      "name": {**S, "description": "logs: log name"}, "lines": I}},
     "annotations": RO, "fn": t_studio_info},
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
                             "Typical flow: list_models(task=...) -> get_model(id) -> [improve_prompt] -> submit_job(...) -> use the returned file paths; "
                             "export_assets packages results. 3D: shape -> texture -> rig -> motion (get_asset shows mesh3d: rigged, joints, animation clips); edit a rig with get_rig + submit_job(tool='rig.edit') then task 'reskin'. "
                             "Check each model's licence before commercial use; many are non-commercial. "
                             "Never delete, spend cloud money or upload files without the user's explicit OK.")}}
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
