"""FastAPI app: REST + SSE API and the single-page web UI. Run: envs/core/bin/python -m app.main (see run_studio.sh).

Every route that touches files takes ids (project, asset id, file name) that are validated by regex; no route
accepts a server-side path. API reference: docs/CONTRACTS.md.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .assets import AssetError
from .core import Core
from .downloads import DownloadError, Downloads
from .improver import ImproveError
from .jobs import HANDLERS, JobError
from .textgen import TextGenError

log = logging.getLogger("studio")
core: Core | None = None


def C() -> Core:
    assert core is not None
    return core


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global core
    if core is None:
        core = Core()
    core.load_plugins(app)
    core.downloads = Downloads(core.registry, core.bus)
    # plugin routers are included after the catch-all static mount ("/"): keep that mount last so /api/x/* is reachable
    web = [r for r in app.router.routes if getattr(r, "name", None) == "web"]
    for r in web:
        app.router.routes.remove(r)
        app.router.routes.append(r)
    task = asyncio.create_task(_ticker())
    log.info("studio core ready: %d models, %d plugins", len(core.registry.models), len(core.plugins))
    try:
        yield
    finally:
        task.cancel()
        core.downloads.shutdown()
        core.shutdown()


async def _ticker():
    """Memory meter + queue counts for every client, every 3 s."""
    while True:
        await asyncio.sleep(3)
        try:
            c = C()
            c.bus.publish("system", {"memory": c.memory.snapshot(), "jobs": c.jobs.counts(), "llm": c.llm.status()})
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(title="Open Asset Studio", version="0.1.0", lifespan=lifespan)


# ------------------------------------------------------------------ errors
@app.exception_handler(AssetError)
async def _asset_err(_r, e: AssetError):
    return JSONResponse({"error": str(e), "kind": "input"}, status_code=e.status)


@app.exception_handler(JobError)
async def _job_err(_r, e: JobError):
    body = {**(getattr(e, "data", None) or {}), "error": str(e), "kind": e.kind, "fix": e.fix}
    return JSONResponse(body, status_code=400 if e.kind == "input" else 403 if e.kind == "license" else 409)


@app.exception_handler(ImproveError)
async def _improve_err(_r, e: ImproveError):
    return JSONResponse({"error": str(e), "kind": "improve", "fix": e.fix}, status_code=e.status)


@app.exception_handler(TextGenError)
async def _textgen_err(_r, e: TextGenError):
    return JSONResponse({"error": str(e), "kind": e.kind, "fix": e.fix}, status_code=e.status if 400 <= e.status < 600 else 502)


@app.exception_handler(DownloadError)
async def _dl_err(_r, e: DownloadError):
    return JSONResponse({"error": str(e), "kind": e.kind, "fix": e.fix}, status_code=400 if e.kind == "input" else 502)


@app.exception_handler(KeyError)
async def _key_err(_r, e: KeyError):
    return JSONResponse({"error": f"Not found: {e}"}, status_code=404)


@app.exception_handler(ValueError)
async def _val_err(_r, e: ValueError):
    return JSONResponse({"error": str(e), "kind": "input"}, status_code=400)


@app.middleware("http")
async def _no_cache_ui(request: Request, call_next):
    resp = await call_next(request)
    p = request.url.path
    if not p.startswith(("/api", "/files", "/vendor")) and (p.endswith((".js", ".css", ".html")) or p == "/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


# ------------------------------------------------------------------ bootstrap / system
@app.get("/api/health")
def health():
    return {"ok": True, "version": C().version, "uptime_s": round(time.time() - C().started)}


def _studio_modules() -> list[str]:
    d = config.WEB / "studios"
    return sorted(p.name for p in d.glob("*.js")) if d.is_dir() else []


@app.get("/api/bootstrap")
def bootstrap():
    c = C()
    return {"version": c.version, "projects": c.assets.projects(), "tasks": c.registry.tasks,
            "settings": {"ui": c.settings.get("ui"), "profile": c.settings.get("profile"), "license": c.settings.get("license"),
                         "export": c.settings.get("export"), "defaults": {t: c.registry.default_for(t) for t in c.registry.tasks}},
            "modules": _studio_modules(), "llm": c.llm.status(), "jobs": c.jobs.counts(),
            "memory": c.memory.snapshot(), "plugins": sorted(c.plugins), "plugin_errors": c.plugin_errors,
            "tools": {k[5:]: v["label"] for k, v in HANDLERS.items() if k.startswith("tool:")},
            "registry_errors": c.registry.errors}


@app.get("/api/ui/modules")
def ui_modules():
    return {"modules": _studio_modules()}


@app.get("/api/system")
def system():
    c = C()
    return {"memory": c.memory.snapshot(), "workers": [w.status() for w in c.pool.workers.values()],
            "llm": c.llm.status(), "jobs": c.jobs.counts(), "uptime_s": round(time.time() - c.started),
            "version": c.version, "plugins": sorted(c.plugins), "plugin_errors": c.plugin_errors}


class UnloadReq(BaseModel):
    backend: str | None = None
    model: str | None = None
    llm: bool = False
    all: bool = False


@app.post("/api/system/unload")
def system_unload(req: UnloadReq):
    c = C()
    done = []
    if req.llm or req.all:
        c.llm.release()
        done.append("llm")
    if req.all:
        for name, w in list(c.pool.workers.items()):
            if not w.busy:
                c.pool.unload(name)
                done.append(name)
    elif req.backend:
        w = c.pool.workers.get(req.backend)
        if w and w.busy:
            raise HTTPException(409, f"{req.backend} is running a job; cancel it first.")
        c.pool.unload(req.backend, req.model)
        done.append(f"{req.backend}:{req.model or '*'}")
    return {"unloaded": done, "memory": c.memory.snapshot()}


def _log_files() -> dict[str, Path]:
    return {p.stem: p for p in sorted(config.LOGS.glob("*.log"))}


@app.get("/api/system/logs")
def system_logs(name: str = "studio", lines: int = 200):
    files = _log_files()
    if name not in files:  # only known log files: the name is a key, never a path
        return {"name": name, "lines": [], "available": sorted(files)}
    p = files[name]
    lines = max(10, min(lines, 2000))
    with p.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - lines * 300))
        text = f.read().decode("utf-8", "replace")
    return {"name": name, "lines": text.splitlines()[-lines:], "available": sorted(files), "bytes": size}


class SelfTestReq(BaseModel):
    kind: str = "audio"
    project: str = "default"
    steps: int = 20


@app.post("/api/system/selftest")
def selftest(req: SelfTestReq):
    model = "test-swatch" if req.kind == "image" else "test-tone"
    return C().jobs.submit({"task": "test", "model": model, "project": req.project, "title": f"Self-test ({req.kind})",
                            "params": {"steps": req.steps, "batch": 2 if req.kind == "audio" else 4,
                                       "prompt": "self-test"}, "meta": {"studio": "system", "tags": ["self-test"]}})


# ------------------------------------------------------------------ models / settings / presets
@app.get("/api/models")
def models(task: str | None = None, modality: str | None = None, include_test: bool | None = None):
    c = C()
    if include_test is None:
        include_test = bool(c.settings.get("ui", "show_test_models", default=False))
    return {"models": c.registry.list(task, modality, include_test), "tasks": c.registry.tasks,
            "defaults": {t: c.registry.default_for(t) for t in c.registry.tasks}, "errors": c.registry.errors,
            "backends": [c.registry.backend_public(b) for b in c.registry.backends.values()]}


@app.get("/api/models/{model_id}")
def model(model_id: str):
    c = C()
    m = c.registry.public(c.registry.get(model_id))
    m["user_presets"] = c.presets.get(model_id)
    return m


@app.post("/api/models/refresh")
def models_refresh():
    c = C()
    c.registry.refresh()
    c.templates.reload()
    c.bus.publish("models", {"refreshed": True})
    return models()


class DownloadReq(BaseModel):
    accept_license: bool = False


@app.get("/api/downloads")
def downloads_list():
    return {"downloads": C().downloads.list()}


@app.get("/api/models/{model_id}/download")
def download_plan(model_id: str):
    return C().downloads.plan(model_id)


@app.post("/api/models/{model_id}/download")
def download_start(model_id: str, req: DownloadReq):
    return C().downloads.start(model_id, req.accept_license)


@app.post("/api/models/{model_id}/download/cancel")
def download_cancel(model_id: str):
    return C().downloads.cancel(model_id)


class ProbeReq(BaseModel):
    backend: str


@app.post("/api/models/probe")
def models_probe(req: ProbeReq):
    c = C()
    res = c.registry.probe_backend(req.backend)
    c.bus.publish("models", {"probed": req.backend})
    return {"backend": req.backend, "results": res}


@app.get("/api/settings")
def settings_get():
    return C().settings.data


@app.patch("/api/settings")
def settings_patch(patch: dict):
    allowed = {"defaults", "ui", "memory", "workers", "jobs", "llm", "textgen", "profile", "license", "export"}
    bad = set(patch) - allowed
    if bad:
        raise HTTPException(400, f"Unknown settings: {', '.join(sorted(bad))}")
    from . import profiles

    if "profile" in patch and patch["profile"] not in profiles.PROFILES:
        raise HTTPException(400, f"Unknown profile '{patch['profile']}' (one of {', '.join(profiles.PROFILE_IDS)}).")
    if isinstance(patch.get("export"), dict) and patch["export"].get("preset") not in (None, *profiles.EXPORT_PRESETS):
        raise HTTPException(400, f"Unknown export preset '{patch['export']['preset']}'.")
    if "license" in patch and not isinstance(patch["license"], dict):
        raise HTTPException(400, "license must be an object: {commercial_only, strict}.")
    c = C()
    out = c.settings.update(patch)
    if {"profile", "license", "export"} & set(patch):
        c.bus.publish("settings", {"profile": out.get("profile"), "license": out.get("license"), "export": out.get("export")})
    return out


@app.get("/api/profiles")
def profiles_get():
    """Use-case profiles + export presets + the current selection (docs/CONTRACTS.md "Profiles and export presets")."""
    from . import profiles

    return {**profiles.public(C().settings), "license": C().settings.get("license")}


class DefaultReq(BaseModel):
    task: str
    model: str | None = None


@app.put("/api/settings/defaults")
def set_default(req: DefaultReq):
    c = C()
    if req.model is not None:
        m = c.registry.get(req.model)
        if req.task not in m["tasks"]:
            raise HTTPException(400, f"{m['name']} does not support {req.task}")
    c.settings.update({"defaults": {req.task: req.model}})
    c.bus.publish("models", {"defaults": True})
    return {"task": req.task, "model": c.registry.default_for(req.task)}


class PresetReq(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


@app.get("/api/presets/{model_id}")
def presets_get(model_id: str):
    c = C()
    m = c.registry.get(model_id)
    return {"builtin": m.get("presets") or {}, "user": c.presets.get(model_id)}


@app.put("/api/presets/{model_id}")
def presets_put(model_id: str, req: PresetReq):
    c = C()
    c.registry.get(model_id)
    c.presets.put(model_id, req.name, req.params)
    return presets_get(model_id)


@app.delete("/api/presets/{model_id}/{name}")
def presets_delete(model_id: str, name: str):
    C().presets.delete(model_id, name)
    return presets_get(model_id)


# ------------------------------------------------------------------ projects / assets
class ProjectReq(BaseModel):
    id: str
    title: str = ""
    description: str = ""


@app.get("/api/projects")
def projects():
    return {"projects": C().assets.projects()}


@app.post("/api/projects")
def project_create(req: ProjectReq):
    return C().assets.create_project(req.id, req.title, req.description)


@app.patch("/api/projects/{project}")
def project_update(project: str, patch: dict):
    return C().assets.update_project(project, patch)


@app.get("/api/assets")
def assets_list(project: str | None = None, type: str | None = None, model: str | None = None,  # noqa: A002
                tag: str | None = None, q: str | None = None, rating_min: int | None = None, favorite: bool | None = None,
                job: str | None = None, parent: str | None = None, ids: str | None = None, sort: str = "new",
                limit: int = 200, offset: int = 0, task: str | None = None, studio: str | None = None):
    return C().assets.list(project, type=type, model=model, tag=tag, q=q, rating_min=rating_min, favorite=favorite,
                           job=job, parent=parent, ids=ids.split(",") if ids else None, sort=sort, limit=limit,
                           offset=offset, task=task, studio=studio)


@app.get("/api/assets/{asset_id}")
def asset_get(asset_id: str):
    return C().assets.get(asset_id)


@app.patch("/api/assets/{asset_id}")
def asset_patch(asset_id: str, patch: dict):
    return C().assets.update(asset_id, patch)


@app.delete("/api/assets/{asset_id}")
def asset_delete(asset_id: str):
    return C().assets.delete(asset_id)


@app.get("/api/assets/{asset_id}/lineage")
def asset_lineage(asset_id: str):
    return C().assets.lineage(asset_id)


async def _save_upload(f: UploadFile, limit_mb: int = 500) -> Path:
    ext = Path(f.filename or "").suffix.lower()[:10]
    if not ext or any(ch in ext for ch in "/\\"):
        raise AssetError("The file needs an extension (.png, .wav, .glb ...).")
    d = config.DATA / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"up-{secrets.token_hex(8)}{ext}"
    n = 0
    with tmp.open("wb") as out:
        while chunk := await f.read(1 << 20):
            n += len(chunk)
            if n > limit_mb * 2**20:
                out.close()
                tmp.unlink(missing_ok=True)
                raise AssetError(f"Upload larger than {limit_mb} MB.", 413)
            out.write(chunk)
    return tmp


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), project: str = Form("default"), name: str = Form(""),
                 tags: str = Form(""), type: str = Form("")):  # noqa: A002
    c = C()
    c.assets.project_dir(project)
    tmp = await _save_upload(file)
    try:
        meta = {"tags": [t for t in tags.split(",") if t.strip()]}
        if name:
            meta["name"] = name
        if type == "voice":
            meta["type"] = "voice"
        return c.assets.upload(project, file.filename or tmp.name, tmp, meta)
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/api/assets/{asset_id}/thumb")
async def asset_thumb(asset_id: str, file: UploadFile = File(...)):
    c = C()
    c.assets.get(asset_id)
    tmp = await _save_upload(file, limit_mb=20)
    try:
        return c.assets.set_thumb(asset_id, tmp)
    finally:
        tmp.unlink(missing_ok=True)


class ExportReq(BaseModel):
    ids: list[str]
    naming: str = "name"
    include_meta: bool = True
    include_attachments: bool = True
    # Export preset: {"preset": "<id>"} and/or {"audio": {...}, "image": {...}} (app/profiles.py). Omitted = the stored
    # files as they are (the web UI sends the profile's default preset). Originals are never modified.
    export: dict | None = None


class RenderReq(BaseModel):
    export: dict | None = None


def _export_spec(spec: dict | None) -> dict | None:
    """Request body -> {audio, image} option dicts (None for 'original')."""
    from . import profiles

    if spec is None:
        return None
    r = profiles.resolve_export(spec, C().settings)
    return {"audio": r["audio"], "image": r["image"], "preset": r["preset"]} if (r["audio"] or r["image"]) else None


@app.post("/api/export")
def export(req: ExportReq):
    z = C().assets.export(req.ids, config.DATA / "exports", naming=req.naming, include_meta=req.include_meta,
                          include_attachments=req.include_attachments, convert=_export_spec(req.export))
    return {"url": f"/api/export/{z.name}", "name": z.name, "bytes": z.stat().st_size, "count": len(req.ids)}


@app.post("/api/assets/{asset_id}/render")
def asset_render(asset_id: str, req: RenderReq):
    """Convert one asset for download (loudness / sample rate / format / crop / size). Returns a URL to a derived file."""
    r = C().assets.render(asset_id, _export_spec(req.export), config.DATA / "exports")
    return {"url": f"/api/export/render/{r['token']}/{r['name']}", "name": r["name"], "bytes": r["bytes"], "report": r["report"],
            "asset": asset_id}


@app.get("/api/export/render/{token}/{name}")
def render_get(token: str, name: str):
    import re

    from .assets import FILE_RE
    from . import media

    if not re.fullmatch(r"[0-9a-f]{12}", token) or not FILE_RE.match(name) or ".." in name:
        raise HTTPException(404)
    p = config.DATA / "exports" / f"render-{token}" / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type=media.MIME.get(p.suffix.lower(), "application/octet-stream"), filename=name)


@app.get("/api/export/{name}")
def export_get(name: str):
    import re

    if not re.fullmatch(r"export-\d{8}-\d{6}-[0-9a-f]{6}\.zip", name):
        raise HTTPException(404)
    p = config.DATA / "exports" / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/zip", filename=name)


@app.get("/files/{project}/{asset_id}/{name}")
def files(project: str, asset_id: str, name: str, download: bool = False):
    p = C().assets.file_path(project, asset_id, name)
    from . import media

    return FileResponse(p, media_type=media.MIME.get(p.suffix.lower(), "application/octet-stream"),
                        filename=name if download else None, headers={"Cache-Control": "private, max-age=3600"})


# ------------------------------------------------------------------ jobs + events
class JobReq(BaseModel):
    task: str | None = None
    model: str | None = None
    tool: str | None = None
    params: dict = Field(default_factory=dict)
    inputs: list = Field(default_factory=list)
    project: str = "default"
    title: str = ""
    meta: dict = Field(default_factory=dict)
    handler: str | None = None
    confirm_cost: bool = False      # cloud: the user accepted the cost estimate (409 kind "confirm" otherwise)
    confirm_upload: bool = False    # cloud: the user accepted sending input files to the provider (409 kind "consent")


@app.post("/api/jobs")
def job_create(req: JobReq):
    return C().jobs.submit(req.model_dump())


@app.get("/api/jobs")
def jobs_list(status: str | None = None, limit: int = 100, project: str | None = None):
    c = C()
    return {"jobs": c.jobs.list(status, limit, project), "counts": c.jobs.counts()}


@app.get("/api/jobs/{job_id}")
def job_get(job_id: str):
    c = C()
    return c.jobs.public(c.jobs.get(job_id))


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    return C().jobs.cancel(job_id)


class RetryReq(BaseModel):
    confirm_cost: bool = False
    confirm_upload: bool = False


@app.post("/api/jobs/{job_id}/retry")
def job_retry(job_id: str, req: RetryReq | None = None):
    req = req or RetryReq()
    return C().jobs.retry(job_id, confirm_cost=req.confirm_cost, confirm_upload=req.confirm_upload)


@app.delete("/api/jobs")
def jobs_clear():
    C().jobs.clear_finished()
    return {"ok": True}


@app.get("/api/events")
async def events(request: Request, types: str | None = None, job: str | None = None):
    c = C()
    q = c.bus.subscribe({"types": set(types.split(",")) if types else None, "job": job})

    async def gen():
        try:
            yield "retry: 2000\n\n"
            yield c.bus.format({"type": "hello", "data": {"version": c.version, "jobs": c.jobs.counts()}, "seq": 0})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15)
                    yield c.bus.format(item)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            c.bus.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ improver
class ImproveReq(BaseModel):
    text: str = ""
    target_style: str | None = None
    model_id: str | None = None
    task: str | None = None
    lang: str | None = None
    n: int = 3
    image_asset: str | None = None
    context: str = ""
    chips: list[str] = Field(default_factory=list)
    backend: str | None = None      # textgen backend id ("local", "cli:claude", "api:openrouter/<model>"); None = settings


@app.post("/api/improve")
def improve(req: ImproveReq):
    c = C()
    image_path = None
    if req.image_asset:
        m = c.assets.get(req.image_asset)
        if m["type"] != "image":
            raise HTTPException(400, "image_asset must be an image")
        image_path = c.assets.file_path(m["project"], m["id"], m["main"])
    style = req.target_style
    if not style and req.model_id and req.model_id in c.registry.models:
        style = c.registry.prompt_style(req.model_id, req.task)
    return c.improver.improve(req.text, target_style=style, model_id=req.model_id, lang=req.lang, n=req.n,
                              image_path=image_path, context=req.context, chips=req.chips, backend=req.backend)


@app.get("/api/improve/styles")
def improve_styles():
    return {"styles": C().templates.public(), "chipsets": C().templates.chipsets}


@app.get("/api/improve/status")
def improve_status():
    return C().llm.status()


@app.post("/api/improve/warmup")
def improve_warmup():
    C().llm.ensure()
    return C().llm.status()


@app.post("/api/improve/release")
def improve_release():
    C().llm.release()
    return C().llm.status()


# ------------------------------------------------------------------ web UI (last: catch-all static)
app.mount("/", StaticFiles(directory=str(config.WEB), html=True), name="web")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("STUDIO_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("STUDIO_PORT", "7860")))
    a = ap.parse_args()
    os.environ["STUDIO_PORT"] = str(a.port)   # the Assistant's MCP server calls back into this port
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, log_level="info", timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
