"""Project style guide (docs/CONTRACTS.md section 11): one versioned guide per project that the studio applies to every styled job.

  data/projects/<p>/style/guide.json             the current guide (includes version + hash)
  data/projects/<p>/style/versions/v0001.json    an immutable snapshot per saved version (written BEFORE guide.json)

How it is applied (nothing is monkey-patched; these are the core hooks of docs/CONTRACTS.md 3.3):
  jobs.SUBMIT_HOOKS          submit_hook()  adds the style as a SUFFIX of params.prompt, the avoid list to a real negative prompt, reference
                             images as inputs with role `style_ref`, and records `x_style` in meta.asset_extra. Runs for every path that ends in
                             JobManager.submit (UI, Pipelines, Assistant plans, MCP, retry). Retry / regenerate never double-append: the user's
                             own text is restored from x_style first (app/style_common.py compose, step 1).
  jobs.OUTPUT_HOOKS          output_hook()  the optional palette post-process (tint / lock) of image outputs; keeps the raw image as an attachment.
  core.improve_context_hooks improve_context()  one line about the style for the prompt improver.

Other plugins: `core.style.register_task("<task>", "texture")`, or a module-level `STYLE_TASKS = {"<task>": "<variant>"}` in the plugin file
(read in setup and again lazily, so load order does not matter). The tool `tool:style.palette` applies the palette to existing images (derived assets).
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import config, jobs, licensing
from .. import style_common as sc
from ..assets import AssetError, check_project
from ..jobs import JobContext, JobError, handler

log = logging.getLogger("studio")
router = APIRouter()
CORE = None
SVC: "StyleService | None" = None
BY = ("ui", "mcp", "assistant", "restore", "api")
IMG_SUFFIX = (".png", ".jpg", ".jpeg", ".webp")

# (i) balloon texts of the Style page and the run-form chip (40-600 characters, plain words, never just the label)
INFO = {
    "enabled": ("Style guide on", "Turns the whole guide on or off for this project. When it is off, no job is changed and the Style chip disappears from the run forms. Your settings stay saved, so you can switch it back on later."),
    "strength": ("Strength", "How much of the guide reaches the model. Light adds the written look only. Normal also attaches up to 2 of your reference images on models that can use them. Strong attaches up to 4 and names the palette colours in the prompt. More references make jobs slower."),
    "positive": ("The look", "Describe how things should be drawn, never what is drawn: the medium, the line work, the shading, the lighting and the colour mood. For example “hand-painted fantasy game art, thick dark outlines, flat two-tone shading”. It is added after the prompt of every job. Do not name a subject, and avoid words like “no” or “without”."),
    "background": ("Background", "What is behind the subject, for example “plain cream parchment background”. It is dropped automatically for transparent images and for 3D-ready images, because those must have no background."),
    "avoid": ("Avoid list", "Looks that would break your style, for example “photorealistic”, “3d render”, “text” or “watermark”. They go into the model's negative prompt, but only models that use one at their current settings (for example Wan, Lumina, or Qwen-Image with True CFG above 1). Other models ignore the list, and the preview tells you which."),
    "refs": ("Reference images", "Up to 4 finished images from this project that show the style. On models that can take style references (Qwen-Image 2.1 for now) they are attached to the job so colours, background and framing follow them, at the cost of about 20 s more per image. Use only images you have the rights to."),
    "palette": ("Palette", "The colours of the style, up to 12. They are not written into the prompt unless you ask (colour words tend to leak into the subject, for example a teal sword). They can be applied to the finished picture with the post-process below, and the 3D and sprite tools can read them."),
    "palette_extract": ("Extract from a reference", "Picks the most distinct colours from one of your reference images, with the background colour first. It takes a fraction of a second and replaces the swatches you see now, which you can still edit before saving."),
    "palette_in_prompt": ("Name colours in the prompt", "Adds a short “colour palette: crimson, teal, cream” phrase to the prompt. It helps some models keep to the colours, but others start painting the subject in those colours. It is always on at Strong strength."),
    "palette_post": ("Palette post-process", "Changes the finished picture after it is made. Tint nudges the overall colours toward the palette and keeps the details. Lock snaps every pixel to the nearest palette colour, which suits pixel art and flat icons but ruins photos and soft shading. The untouched original is kept as an attachment."),
    "tint_strength": ("Tint amount", "How far the colours move toward the palette. 0.1 is barely visible, 0.35 is a gentle pull toward the palette, 0.8 is strong and can give a colour cast. Start at 0.35 and look at the result."),
    "variants": ("Variants", "Different kinds of work can need a different version of the style: a sprite wants a crisp silhouette, a texture wants even light and no perspective, a sound wants a mood instead of a look. Each tab changes only what differs; everything else is inherited from the main guide."),
    "variant_mode": ("Extend or replace", "Extend adds this variant's text after the main look, so both apply. Replace uses only this variant's text and ignores the main look. Replace is the right choice for Audio mood, because a visual look makes no sense in a sound prompt."),
    "variant_refs": ("Use references", "Whether the reference images are attached to jobs of this kind. Turn it off for textures, because a picture of a whole scene used as a reference breaks the tiling, and for video, which takes its look from the first frame."),
    "variant_enabled": ("Use this variant", "When it is off, jobs of this kind are not styled at all, even if the main guide is on. Audio mood is off by default, because most sound prompts should stay purely about sound."),
    "variant_positive": ("Extra look", "Text added for this kind of job only, for example “crisp silhouette, side view” for sprites. In Extend mode it comes after the main look; in Replace mode it is all that is used."),
    "variant_background": ("Background for this variant", "Inherit uses the main background. None adds no background clause at all, which is right for textures and sprites. Custom lets you write a different one."),
    "variant_avoid": ("Extra avoid terms", "More terms for the negative prompt, used only for this kind of job, for example “perspective, shadows” for textures. They are combined with the main avoid list."),
    "variant_post": ("Palette post-process here", "Inherit follows the palette setting above. Off never touches this kind of picture. Tint or Lock applies only to this kind, for example Lock for sprites with a designed palette."),
    "preview": ("Preview", "Shows exactly what a model would receive for a sample prompt: your words first, the added style text highlighted, the negative prompt, and which reference images are attached. Nothing is generated, so it is instant. Pick a model and a task to see how they differ."),
    "test_subjects": ("Test on 4 subjects", "Generates one picture for each subject in the list with the saved guide, so you can judge whether the style holds on different things. It uses the selected model, runs in the queue, and needs a saved guide. The results open in the Library."),
    "history": ("History", "Every save keeps a copy. You can read what changed between versions and restore an old one, which saves it as a new version. Assets remember the version that made them, so you can always see which style produced an image."),
    "chip": ("Project style", "Shows that this project has a style guide and whether it applies to this job. Click to switch it off for the job, change the variant or strength, and see the exact text that will be sent. The setting is remembered per studio."),
    "chip_apply": ("Apply to this job", "Switch off to generate this job without the style, for example for a one-off experiment. The chip then turns amber, so you cannot forget. It switches back on by itself after the guide changes."),
    "chip_variant": ("Variant for this job", "The version of the style to use. It is picked from the kind of job (image, texture, video or audio), but you can choose another one, for example Sprite for an image that will become a game sprite."),
    "chip_strength": ("Strength for this job", "Light uses the written look only. Normal adds up to 2 reference images where the model supports them. Strong adds up to 4 and names the palette colours in the prompt. This changes only this job."),
    "describe": ("Describe from reference", "Asks the prompt language model to look at your first reference image and write down its style (never its subject), then fills the empty fields. It needs the prompt model, which may take a few seconds to load, and it never replaces text you wrote without asking."),
    "copy_project": ("Copy from another project", "Starts this project's guide from the one in another project. Reference images belong to their project, so they are not copied; the text, avoid list, palette and variants are."),
}
INFO_KEYS = tuple(INFO)

TEST_SUBJECTS = "a potion, a sword, a boot, a treasure chest"


class Conflict(Exception):
    def __init__(self, current: int):
        super().__init__(f"The style guide was changed elsewhere (now version {current}).")
        self.current = current


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


class StyleService:
    """Everything about guides: storage, versions, composition for one job. `core.style` (see register_task)."""

    def __init__(self, core):
        self.core = core
        self._locks: dict[str, threading.Lock] = {}
        self._mx = threading.Lock()
        self._scanned = -1

    # ---- storage
    def dir(self, project: str) -> Path:
        d = self.core.assets.project_dir(check_project(project))
        if not d.is_dir():
            raise AssetError(f"No such project '{project}'.", 404)
        return d / "style"

    def _lock(self, project: str) -> threading.Lock:
        with self._mx:
            return self._locks.setdefault(project, threading.Lock())

    def get(self, project: str | None) -> dict | None:
        """The current guide of a project, or None (no guide, unreadable file, unknown project)."""
        if not project:
            return None
        try:
            return json.loads((self.dir(project) / "guide.json").read_text())
        except (OSError, ValueError, AssetError):
            return None

    def load_version(self, project: str, n) -> dict | None:
        try:
            n = int(n)
            if not 1 <= n <= 9999:
                return None
            return json.loads((self.dir(project) / "versions" / f"v{n:04d}.json").read_text())
        except (OSError, ValueError, AssetError, TypeError):
            return None

    def _last_version(self, project: str) -> int:
        vd = self.dir(project) / "versions"
        nums = [int(m.group(1)) for p in vd.glob("v*.json") if (m := re.match(r"^v(\d{4})\.json$", p.name))] if vd.is_dir() else []
        return max(nums, default=0)

    def versions(self, project: str) -> list[dict]:
        out = []
        for n in range(self._last_version(project), 0, -1):
            g = self.load_version(project, n)
            if g:
                out.append({"version": n, "hash": g.get("hash"), "saved": g.get("saved"), "saved_by": g.get("saved_by"), "name": g.get("name"),
                            "summary": sc.summary(g)})
        return out

    def save(self, project: str, data: dict, *, base_version=None, by: str = "api", patch: bool = False) -> tuple[dict, bool, list[str]]:
        """Validate + store a guide. Returns (guide, created_version, warnings). A save that leaves the content hash unchanged
        returns the current guide and does not bump the version. Raises Conflict (stale base_version) and ValueError."""
        by = by if by in BY else "api"
        assets = self.core.assets
        with self._lock(project):
            self.dir(project)
            cur = self.get(project)
            if base_version not in (None, "") and int(base_version) != (cur["version"] if cur else 0):
                raise Conflict(cur["version"] if cur else 0)
            if patch:
                merged = sc.content_of(cur) if cur else sc.default_guide()
                config.apply_patch(merged, data or {})
            else:
                merged = data
            clean = sc.validate_guide(merged, project, assets, allow_missing={r["asset"] for r in (cur or {}).get("refs") or []})
            h = sc.guide_hash(clean)
            if cur and cur.get("hash") == h:
                return cur, False, sc.lint(cur)
            n = self._last_version(project) + 1
            rec = {**clean, "version": n, "hash": h, "saved": _now(), "saved_by": by}
            d = self.dir(project)
            _write(d / "versions" / f"v{n:04d}.json", rec)         # the snapshot first, then the current guide
            _write(d / "guide.json", rec)
        self._announce(project, n)
        return rec, True, sc.lint(rec)

    def restore(self, project: str, version: int) -> tuple[dict, bool, list[str]]:
        g = self.load_version(project, version)
        if not g:
            raise HTTPException(404, f"Style guide version {version} does not exist.")
        return self.save(project, sc.content_of(g), by="restore")

    def delete(self, project: str) -> None:
        with self._lock(project):
            p = self.dir(project) / "guide.json"
            if p.is_file():
                p.unlink()
        self._announce(project, None)

    def _announce(self, project: str, version) -> None:
        try:
            self.core.bus.publish("settings", {"scope": "project", "project": project, "style": {"version": version}})
        except Exception:  # noqa: BLE001
            log.exception("style: could not publish the settings event")

    def summary_all(self) -> dict:
        out = {}
        for p in self.core.assets.projects():
            g = self.get(p["id"])
            out[p["id"]] = {"name": g["name"], "version": g["version"], "enabled": g["enabled"]} if g else None
        return out

    # ---- composition for one job
    def register_task(self, task: str, variant: str, sep: str = ". ", prompt_key: str = "prompt", refs: bool = False) -> None:
        sc.register_task(task, variant, sep, prompt_key, refs)

    def _scan_plugins(self) -> None:
        """Pick up `STYLE_TASKS = {task: variant}` of every loaded plugin (cheap; re-run when more plugins appeared)."""
        plugins = getattr(self.core, "plugins", {}) or {}
        if len(plugins) == self._scanned:
            return
        self._scanned = len(plugins)
        for mod in list(plugins.values()):
            for t, v in (getattr(mod, "STYLE_TASKS", None) or {}).items():
                try:
                    sc.register_task(t, v)
                except ValueError:
                    log.warning("style: bad STYLE_TASKS entry %r", t)

    def lookup(self, aid: str) -> dict | None:
        try:
            return self.core.assets.get(aid)
        except Exception:  # noqa: BLE001
            return None

    def plan(self, project: str, model: dict | None, task: str, params: dict, inputs, meta: dict | None, req: dict | None,
             settings=None, guide: dict | None = None, use_current: bool = True) -> dict:
        self._scan_plugins()
        if guide is None and use_current:
            guide = self.get(project)
        label = (self.core.registry.tasks.get(task) or {}).get("label")
        return sc.compose(guide, model, task, params, inputs, req, meta=meta, lookup=self.lookup,
                          load_version=lambda n: self.load_version(project, n), commercial_only=licensing.settings_of(settings)[0], task_label=label)


def _req_of(meta: dict) -> dict:
    s = meta.get("style")
    if s is None or s == "auto":
        return {}
    if s == "off":
        return {"apply": False}
    if isinstance(s, dict):
        return {k: s[k] for k in ("apply", "variant", "strength", "version") if k in s}
    raise JobError("meta.style must be \"auto\", \"off\" or an object {apply, variant, strength, version}.", "input")


# ------------------------------------------------------------------ the hooks
def make_hooks(core, svc: StyleService):
    def submit_hook(sub: dict) -> None:
        if sub.get("jobs") is not core.jobs:
            return
        meta = sub["meta"]
        req = _req_of(meta)
        ae = dict(meta.get("asset_extra") or {})
        try:
            r = svc.plan(sub["project"], sub["model"], sub["task"], sub["params"], sub["inputs"], meta, req, settings=sub.get("settings"))
        except sc.StyleInputError as e:
            raise JobError(str(e), "input") from None
        except Exception as e:  # noqa: BLE001  a broken guide must not stop the job: it runs unstyled and the record says why
            log.exception("style hook failed")
            ae["x_style"] = {"applied": False, "reason": f"style error: {type(e).__name__}: {e}"}
            meta["asset_extra"] = ae
            return
        sub["params"], sub["inputs"] = r["params"], r["inputs"]
        if r["x_style"]:
            ae["x_style"] = r["x_style"]
        else:
            ae.pop("x_style", None)
        meta.pop("style_post", None)
        meta.pop("style_applied", None)
        if r["style_post"]:
            meta["style_post"] = r["style_post"]
        if r["summary"]:
            meta["style_applied"] = r["summary"]
        if r["applied"] and not meta.get("prompt"):
            meta["prompt"] = r["x_style"]["user_prompt"]           # the lineage shows what was typed, final_prompt what was sent
        if ae:
            meta["asset_extra"] = ae
        else:
            meta.pop("asset_extra", None)

    def output_hook(ctx, path, type, meta, attach):  # noqa: A002
        if ctx.mgr is not core.jobs:
            return None
        post = (ctx.job.get("meta") or {}).get("style_post")
        if not post or type != "image" or path.suffix.lower() not in IMG_SUFFIX:
            return None
        mp = (meta.get("x_material") or {}).get("map")
        if mp and mp not in ("basecolor", "albedo"):               # normal / roughness / ORM / height maps are data, never tinted
            return None
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            src_mode = im.mode
            fmt = im.format or "PNG"
            out = sc.apply_post(im, post["mode"], post["colors"], float(post.get("strength") or 0.35))
        raw = path.with_name(f"{path.stem}_raw{path.suffix}")
        shutil.copy2(path, raw)                                    # the untouched worker output stays as an attachment
        kw = {"quality": 95} if fmt.upper() in ("JPEG", "WEBP") else {}
        if fmt.upper() == "JPEG" or src_mode == "RGB":
            out = out.convert("RGB")
        out.save(path, format=fmt, **kw)
        meta = copy.deepcopy(meta)
        meta.setdefault("x_style", {})["post"] = {"palette": post["mode"], "strength": post.get("strength"), "colors": len(post["colors"]), "raw": raw.name}
        return path, meta, [*attach, str(raw)]

    def improve_context(project, model_id, task, style_req):
        if not project or not task:
            return ""
        return sc.improve_line(svc.get(project), task, style_req if isinstance(style_req, dict) else {})

    submit_hook._style_hook = output_hook._style_hook = improve_context._style_hook = True  # type: ignore[attr-defined]
    return submit_hook, output_hook, improve_context


def setup(core):
    global CORE, SVC
    CORE = core
    SVC = core.style = StyleService(core)
    submit_hook, output_hook, improve_context = make_hooks(core, SVC)
    # a second Core in one process (tests) must not leave the first one's hooks behind
    jobs.SUBMIT_HOOKS[:] = [f for f in jobs.SUBMIT_HOOKS if not getattr(f, "_style_hook", False)] + [submit_hook]
    jobs.OUTPUT_HOOKS[:] = [f for f in jobs.OUTPUT_HOOKS if not getattr(f, "_style_hook", False)] + [output_hook]
    hooks = [f for f in getattr(core, "improve_context_hooks", []) if not getattr(f, "_style_hook", False)]
    core.improve_context_hooks = hooks + [improve_context]
    SVC._scan_plugins()


def svc() -> StyleService:
    assert SVC is not None
    return SVC


# ------------------------------------------------------------------ the manual palette tool
@handler("tool:style.palette", gpu=False, label="Apply style palette")
def palette_tool(ctx: JobContext):
    """inputs: images. params {mode: tint | lock, strength?, colors?: [#rrggbb]} (default: the project guide's palette). Writes NEW assets."""
    from PIL import Image

    p = ctx.params
    mode = p.get("mode") or "tint"
    if mode not in ("tint", "lock"):
        raise JobError("mode must be tint or lock.", "input")
    colors = p.get("colors")
    guide = svc().get(ctx.job["project"])
    if not colors and guide:
        colors = (guide.get("palette") or {}).get("colors")
    try:
        colors = (sc.validate_guide({"palette": {"colors": colors or []}}, ctx.job["project"], None)["palette"]["colors"])
    except ValueError as e:
        raise JobError(str(e), "input") from None
    if not colors:
        raise JobError("No palette: give params.colors, or save a style guide with colours first.", "input")
    strength = float(p.get("strength") if p.get("strength") is not None else ((guide or {}).get("palette") or {}).get("tint_strength", 0.35))
    ins = [i for i in ctx.input_assets() if i["type"] == "image"]
    if not ins:
        raise JobError("Add one or more images.", "input")
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    for k, inp in enumerate(ins):
        ctx.check_cancel()
        ctx.progress(k / len(ins), f"Applying the palette to {inp['name']}")
        with Image.open(inp["path"]) as im:
            im.load()
            out = sc.apply_post(im, mode, colors, strength)
        dst = ctx.scratch / f"palette_{k}.png"
        out.save(dst, format="PNG")
        src = ctx.assets.get(inp["asset_id"])
        meta = ctx.asset_meta({"name": f"{inp['name']} ({mode})"}, k)
        meta["parents"] = [inp["asset_id"]]
        if src.get("license"):
            meta["license"] = src["license"]
        meta["x_style"] = {"applied": False, "reason": "manual palette step",
                           "post": {"palette": mode, "strength": strength, "colors": len(colors), "raw": None, "source": inp["asset_id"]}}
        ctx.add_output(dst, "image", meta)


# ------------------------------------------------------------------ REST: /api/x/style
class PutReq(BaseModel):
    project: str
    guide: dict
    base_version: int | None = None
    by: str = "api"


class PatchReq(BaseModel):
    project: str
    patch: dict
    base_version: int | None = None
    by: str = "api"


class RestoreReq(BaseModel):
    project: str
    version: int


class ComposeReq(BaseModel):
    project: str
    task: str = "t2i"
    model: str | None = None
    params: dict = Field(default_factory=dict)
    inputs: list = Field(default_factory=list)
    style: dict | None = None
    guide: dict | None = None        # an unsaved draft (the Style page previews while you type)


class ExtractReq(BaseModel):
    project: str
    asset: str
    n: int = 6
    ignore_background: bool = True


class DraftReq(BaseModel):
    project: str
    assets: list[str]
    describe: bool = False
    backend: str | None = None


def _resp_guide(project: str, g: dict | None) -> dict:
    s = svc()
    refs_info, missing = [], []
    for r in (g or {}).get("refs") or []:
        a = s.lookup(r["asset"])
        if not a or a.get("project") != project:
            missing.append(r["asset"])
            refs_info.append({"asset": r["asset"], "exists": False, "name": r.get("name")})
        else:
            refs_info.append({"asset": r["asset"], "exists": True, "name": a.get("name"), "model": a.get("model"), "model_name": a.get("model_name"),
                              "license": a.get("license"), "changed": bool(r.get("sha256") and _file_sha(a) not in ("", r["sha256"]))})
    return {"guide": g, "warnings": sc.lint(g) if g else [], "missing_refs": missing, "refs_info": refs_info}


def _file_sha(a: dict) -> str:
    try:
        return sc.sha256_file(CORE.assets.file_path(a["project"], a["id"], a["main"]))
    except Exception:  # noqa: BLE001
        return ""


@router.get("/guide")
def guide_get(project: str):
    return _resp_guide(project, svc().get(project))


@router.put("/guide")
def guide_put(req: PutReq):
    try:
        g, created, warn = svc().save(req.project, req.guide, base_version=req.base_version, by=req.by)
    except Conflict as e:
        return JSONResponse({"error": str(e), "kind": "conflict", "current_version": e.current}, status_code=409)
    return {**_resp_guide(req.project, g), "created_version": created, "warnings": warn}


@router.patch("/guide")
def guide_patch(req: PatchReq):
    try:
        g, created, warn = svc().save(req.project, req.patch, base_version=req.base_version, by=req.by, patch=True)
    except Conflict as e:
        return JSONResponse({"error": str(e), "kind": "conflict", "current_version": e.current}, status_code=409)
    return {**_resp_guide(req.project, g), "created_version": created, "warnings": warn}


@router.delete("/guide")
def guide_delete(project: str):
    svc().delete(project)
    return {"guide": None}


@router.get("/versions")
def versions_list(project: str):
    return {"versions": svc().versions(project)}


@router.get("/versions/{n}")
def version_get(n: int, project: str):
    g = svc().load_version(project, n)
    if not g:
        raise HTTPException(404, f"Style guide version {n} does not exist.")
    return g


@router.post("/restore")
def restore(req: RestoreReq):
    g, created, warn = svc().restore(req.project, req.version)
    return {**_resp_guide(req.project, g), "created_version": created, "warnings": warn}


@router.get("/summary")
def summary_all():
    """{project: {name, version, enabled} | null} for every project (MCP list_projects)."""
    return {"projects": svc().summary_all()}


@router.post("/compose")
def compose_preview(req: ComposeReq):
    s = svc()
    c = CORE
    task = req.task
    view = c.settings_for(req.project)
    mid = req.model or c.registry.default_for(task, settings=view)
    model = c.registry.models.get(mid) if mid else None
    params = dict(req.params or {})
    pkey = (sc.TASK_POLICY.get(task) or {}).get("prompt_key", "prompt")
    if not str(params.get(pkey) or "").strip():
        params[pkey] = "an iron short sword"
    guide = None
    if req.guide is not None:
        cur = s.get(req.project)
        guide = sc.validate_guide(req.guide, req.project, c.assets, allow_missing={r["asset"] for r in (cur or {}).get("refs") or []}, fill_refs=False)
        guide.update(version=(cur or {}).get("version", 0), hash=sc.guide_hash(guide))
    elif s.get(req.project) is None:
        return {"applied": False, "reason": "no style guide", "prompt": params[pkey], "negative": params.get("negative") or "", "refs": [], "skipped": [],
                "notes": [], "extra_seconds_est": 0, "variant": None, "strength": None, "version": None, "model": mid, "task": task}
    try:
        r = s.plan(req.project, model, task, params, req.inputs, {}, req.style or {}, settings=view, guide=guide)
    except sc.StyleInputError as e:
        raise ValueError(str(e)) from None
    x = r["x_style"] or {}
    return {"applied": r["applied"], "reason": r["reason"], "prompt": r["prompt"], "negative": r["negative"], "user_prompt": x.get("user_prompt"),
            "added": x.get("added"), "refs": r["refs"], "ref_clause": r["ref_clause"], "skipped": r["skipped"], "notes": r["notes"],
            "extra_seconds_est": r["extra_seconds_est"], "variant": r["variant"], "strength": r["strength"], "version": r["version"],
            "post": r["style_post"], "model": mid, "task": task}


def _open_image(project: str, aid: str):
    from PIL import Image

    a = CORE.assets.get(aid)
    if a.get("project") != project or a.get("type") != "image":
        raise ValueError("That is not an image of this project.")
    return Image.open(CORE.assets.file_path(project, aid, a["main"])), a


@router.post("/palette/extract")
def palette_extract(req: ExtractReq):
    im, _a = _open_image(req.project, req.asset)
    with im:
        colors, weights = sc.extract_palette(im, n=req.n, ignore_background=req.ignore_background)
    return {"colors": colors, "weights": weights}


@router.post("/draft")
def draft(req: DraftReq):
    """An UNSAVED guide built from up to 4 image assets: refs, the palette of the first, and (describe) the look written by the prompt LLM."""
    c = CORE
    if not 1 <= len(req.assets) <= sc.LIM["refs"]:
        raise ValueError(f"Pick 1 to {sc.LIM['refs']} images.")
    g = sc.default_guide()
    first = None
    for aid in req.assets:
        a = c.assets.get(aid)
        if a.get("project") != req.project or a.get("type") != "image":
            raise ValueError(f"{aid} is not an image of this project.")
        first = first or a
        g["refs"].append({"asset": aid, "name": a.get("name")})
    g["name"] = (f"Style of {first.get('name') or first['id']}")[:sc.LIM["name"]]
    im, _a = _open_image(req.project, first["id"])
    with im:
        g["palette"]["colors"], _w = sc.extract_palette(im, n=6)
    g["palette"]["source"] = first["id"]
    out = {"describe": None}
    if req.describe:
        view = c.settings_for(req.project)
        backend = req.backend or c.textgen.default_for("improve", settings=view)
        r = c.improver.improve("describe the art style", target_style="style_describe", image_path=c.assets.file_path(req.project, first["id"], first["main"]),
                               n=0, backend=backend)
        g["positive"] = (r.get("prompt") or "")[:sc.LIM["positive"]]
        g["background"] = (r.get("background") or "")[:sc.LIM["background"]]
        g["avoid"] = sc.dedupe_ci(sc.split_terms(r.get("negative")))[:sc.LIM["avoid_n"]]
        out["describe"] = {"llm": r.get("llm"), "seconds": r.get("seconds"), "notes": r.get("notes"), "warnings": r.get("warnings")}
    clean = sc.validate_guide(g, req.project, c.assets, fill_refs=False)
    return {"guide": clean, "warnings": sc.lint(clean), **out}


@router.get("/policy")
def policy():
    c = CORE
    svc()._scan_plugins()
    tasks = {t: (sc.TASK_POLICY.get(t) or {"variant": "skip"})["variant"] for t in c.registry.tasks}
    tasks.update({t: p["variant"] for t, p in sc.TASK_POLICY.items() if t in c.registry.tasks or p["variant"] != "skip"})
    models = {}
    for mid, m in c.registry.models.items():
        f = m.get("features") or {}
        if f.get("style_ref") or f.get("negative") not in (None, False):
            models[mid] = {"style_ref": list(f.get("style_ref") or []), "negative": f.get("negative", False), "max_refs": f.get("max_refs")}
    return {"tasks": tasks, "models": models, "variants": [{"id": v, "label": sc.VARIANT_LABELS[v]} for v in sc.VARIANTS],
            "labels": {t: v.get("label") for t, v in c.registry.tasks.items()}, "test_subjects": TEST_SUBJECTS,
            "limits": {k: v for k, v in sc.LIM.items() if k != "style_text"}}


@router.get("/info")
def info():
    return {k: {"title": t, "text": x} for k, (t, x) in INFO.items()}
