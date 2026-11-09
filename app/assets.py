"""Projects and assets on disk.

  data/projects/<project>/project.json
  data/projects/<project>/assets/<asset_id>/<files> + meta.json     (meta.json is written last)
  data/projects/<project>/.trash/<asset_id>/                         deleted assets (recoverable by hand)

Identifiers are validated by regex before they ever touch a path, every resolved path is checked to be inside the
project root, and file routes serve only names listed in the asset's meta. APIs never accept server-side paths.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable

from . import config, media

PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
ASSET_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TYPES = ("image", "audio", "mesh", "voice", "video", "text", "material", "other")
MAIN_BASENAME = {"material": "basecolor", "image": "image", "audio": "audio", "voice": "voice", "mesh": "mesh", "video": "video", "text": "text", "other": "file"}
META_VERSION = 1
# Per-type export hooks (plugins register here in setup(core)): fn(meta, asset_dir, convert, tmpdir, base) ->
# ([(source_path, archive_name)], report). A type with an exporter is zipped by it instead of the default main + attachments loop.
EXPORTERS: dict[str, Callable] = {}


class AssetError(Exception):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


def check_project(name: str) -> str:
    if not isinstance(name, str) or not PROJECT_RE.match(name):
        raise AssetError("Invalid project name (lowercase letters, digits, - and _; up to 48 characters).")
    return name


def check_asset_id(aid: str) -> str:
    if not isinstance(aid, str) or not ASSET_RE.match(aid):
        raise AssetError("Invalid asset id.", 404)
    return aid


def check_filename(name: str) -> str:
    if not isinstance(name, str) or not FILE_RE.match(name) or ".." in name:
        raise AssetError("Invalid file name.", 404)
    return name


def safe_name(name: str, default: str = "file") -> str:
    """Turn any user/worker-provided file name into a FILE_RE-valid one (keeps the extension)."""
    base = os.path.basename(str(name or ""))
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:80] or default
    ext = re.sub(r"[^A-Za-z0-9.]", "", ext)[:10].lower()
    out = f"{stem}{ext}"
    return out if FILE_RE.match(out) else default


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return s[:n].strip("_") or "asset"


def _write_json(path: Path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, path)


class AssetStore:
    def __init__(self, root: Path, on_change: Callable[[str, dict], None] | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.on_change = on_change or (lambda action, meta: None)
        self._lock = threading.RLock()
        self._proj_lock = threading.Lock()      # serialises project.json read-modify-write (title edits, settings)
        self.index: dict[str, dict] = {}
        self.scan()
        if not self.projects():
            self.create_project("default", "Default project")

    # ------------------------------------------------------------- paths
    def project_dir(self, project: str) -> Path:
        d = (self.root / check_project(project)).resolve()
        if d.parent != self.root:
            raise AssetError("Invalid project.")
        return d

    def asset_dir(self, project: str, aid: str) -> Path:
        d = (self.project_dir(project) / "assets" / check_asset_id(aid)).resolve()
        if not d.is_relative_to(self.root):
            raise AssetError("Invalid asset path.", 404)
        return d

    def file_path(self, project: str, aid: str, name: str) -> Path:
        """Path of a file the API may serve: listed in meta, inside the asset folder, not a symlink escape."""
        check_filename(name)
        meta = self.get(aid)
        if meta["project"] != project:
            raise AssetError("Not found.", 404)
        allowed = {f["name"] for f in meta.get("files", [])} | set((meta.get("preview") or {}).values())
        if name not in allowed:
            raise AssetError("Not found.", 404)
        d = self.asset_dir(project, aid)
        p = (d / name).resolve()
        if p.parent != d or not p.is_file():
            raise AssetError("Not found.", 404)
        return p

    # ---------------------------------------------------------- projects
    def projects(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            if d.is_dir() and PROJECT_RE.match(d.name):
                try:
                    info = json.loads((d / "project.json").read_text())
                except (OSError, ValueError):
                    info = {}
                n = sum(1 for m in self.index.values() if m["project"] == d.name)
                out.append({"id": d.name, "title": info.get("title") or d.name, "created": info.get("created"),
                            "description": info.get("description", ""), "assets": n,
                            "overrides": config.count_leaves(info.get("settings") or {})})
        return out

    def create_project(self, name: str, title: str = "", description: str = "") -> dict:
        d = self.project_dir(name)
        if d.exists():
            raise AssetError(f"Project '{name}' already exists.", 409)
        (d / "assets").mkdir(parents=True)
        info = {"id": name, "title": title or name, "description": description,
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        _write_json(d / "project.json", info)
        return info

    def _read_project(self, name: str) -> tuple[Path, dict]:
        d = self.project_dir(name)
        if not d.is_dir():
            raise AssetError("No such project.", 404)
        try:
            info = json.loads((d / "project.json").read_text())
            if not isinstance(info, dict):
                raise ValueError
        except (OSError, ValueError):
            info = {"id": name}
        return d, info

    def update_project(self, name: str, patch: dict) -> dict:
        with self._proj_lock:
            d, info = self._read_project(name)      # keeps every key it does not edit, including `settings`
            for k in ("title", "description"):
                if k in patch:
                    info[k] = str(patch[k])[:200]
            _write_json(d / "project.json", info)
            return info

    def project_settings(self, name: str) -> dict:
        """The project's setting overrides (same nested shape as settings.json). {} when there are none."""
        d = self.project_dir(name)
        try:
            s = json.loads((d / "project.json").read_text()).get("settings")
        except (OSError, ValueError, AttributeError):
            return {}
        return s if isinstance(s, dict) else {}

    def update_project_settings(self, name: str, patch: dict | None, *, replace: bool = False) -> dict:
        """Deep-apply `patch` to the project's overrides (None removes a key; `replace=True` with an empty patch
        resets them all). Returns the new overrides."""
        with self._proj_lock:
            d, info = self._read_project(name)
            cur = info.get("settings") if isinstance(info.get("settings"), dict) and not replace else {}
            config.apply_patch(cur, patch or {})
            if cur:
                info["settings"] = cur
            else:
                info.pop("settings", None)
            _write_json(d / "project.json", info)
            return cur

    # ------------------------------------------------------------- index
    def scan(self):
        idx = {}
        for pd in self.root.iterdir() if self.root.exists() else []:
            if not (pd.is_dir() and PROJECT_RE.match(pd.name)):
                continue
            ad = pd / "assets"
            if not ad.is_dir():
                continue
            for d in ad.iterdir():
                if not ASSET_RE.match(d.name):
                    continue
                try:
                    meta = json.loads((d / "meta.json").read_text())
                except (OSError, ValueError):
                    continue  # unfinished write or foreign folder
                meta["project"] = pd.name
                meta["id"] = d.name
                idx[d.name] = meta
        with self._lock:
            self.index = idx

    def get(self, aid: str) -> dict:
        check_asset_id(aid)
        m = self.index.get(aid)
        if m is None:
            raise AssetError("Asset not found.", 404)
        return m

    def list(self, project: str | None = None, *, type: str | None = None, model: str | None = None,  # noqa: A002
             tag: str | None = None, q: str | None = None, rating_min: int | None = None, favorite: bool | None = None,
             job: str | None = None, parent: str | None = None, ids: list[str] | None = None, sort: str = "new",
             task: str | None = None, studio: str | None = None,
             limit: int = 200, offset: int = 0) -> dict:
        items = list(self.index.values())
        if project:
            items = [m for m in items if m["project"] == project]
        if ids:
            want = set(ids)
            items = [m for m in items if m["id"] in want]
        if type:
            types = set(type.split(","))
            items = [m for m in items if m.get("type") in types]
        if model:
            items = [m for m in items if m.get("model") == model]
        if tag:
            items = [m for m in items if tag in (m.get("tags") or [])]
        if rating_min:
            items = [m for m in items if (m.get("rating") or 0) >= rating_min]
        if favorite:
            items = [m for m in items if m.get("favorite")]
        if job:
            items = [m for m in items if m.get("job") == job]
        if task:
            tasks = set(task.split(","))
            items = [m for m in items if m.get("task") in tasks]
        if studio:
            items = [m for m in items if m.get("studio") == studio]
        if parent:
            items = [m for m in items if parent in (m.get("parents") or [])]
        if q:
            terms = q.lower().split()

            def hay(m):
                return " ".join(str(x) for x in (m.get("name"), m.get("prompt"), m.get("final_prompt"), m.get("model"),
                                                 m.get("model_name"), m.get("notes"), m.get("type"), m.get("task"),
                                                 " ".join(m.get("tags") or []), m["id"])).lower()
            items = [m for m in items if all(t in hay(m) for t in terms)]
        keys = {"new": (lambda m: m.get("created", ""), True), "old": (lambda m: m.get("created", ""), False),
                "rating": (lambda m: (m.get("rating") or 0, m.get("created", "")), True),
                "name": (lambda m: str(m.get("name") or "").lower(), False)}
        k, rev = keys.get(sort, keys["new"])
        items.sort(key=k, reverse=rev)
        total = len(items)
        tags = sorted({t for m in items for t in (m.get("tags") or [])})
        models = sorted({m.get("model") for m in items if m.get("model")})
        return {"total": total, "items": items[offset:offset + max(1, min(limit, 1000))], "tags": tags, "models": models}

    # ------------------------------------------------------------ create
    def _new_id(self) -> str:
        while True:
            aid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
            if aid not in self.index:
                return aid

    def ingest(self, project: str, src: Path, type: str, meta: dict | None = None, *,  # noqa: A002
               move: bool = True, attach: list[Path] | None = None, main_name: str | None = None) -> dict:
        """Create one asset from a finished file (worker output, upload, tool result). Returns its meta."""
        src = Path(src)
        if type not in TYPES:
            raise AssetError(f"Unknown asset type {type!r}.")
        if not src.is_file():
            raise AssetError(f"Output file missing: {src.name}", 500)
        pd = self.project_dir(project)
        if not pd.is_dir():
            raise AssetError(f"No such project '{project}'.", 404)
        aid = self._new_id()
        d = pd / "assets" / aid
        d.mkdir(parents=True)
        try:
            ext = src.suffix.lower()
            main = safe_name(main_name or (MAIN_BASENAME.get(type, "file") + ext))
            (shutil.move if move else shutil.copy2)(str(src), str(d / main))
            files = [self._file_entry(d / main, "main")]
            for a in attach or []:
                a = Path(a)
                if a.is_file():
                    n = safe_name(a.name)
                    if n in {f["name"] for f in files} or n == "meta.json":
                        n = safe_name(f"x_{n}")
                    (shutil.move if move else shutil.copy2)(str(a), str(d / n))
                    files.append(self._file_entry(d / n, "attachment"))
            m = dict(meta or {})
            info, preview = self._analyze(d, d / main, type)
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            base = {"schema": META_VERSION, "id": aid, "project": project, "type": type,
                    "name": m.pop("name", None) or src.stem, "main": main, "files": files, "preview": preview,
                    "info": {**info, **(m.pop("info", None) or {})}, "created": now, "updated": now,
                    "tags": [], "rating": 0, "favorite": False, "notes": "", "parents": [], "source": "generated"}
            base.update({k: v for k, v in m.items() if k not in ("id", "project", "files", "preview", "main", "schema")})
            base["parents"] = [p for p in base.get("parents") or [] if isinstance(p, str) and ASSET_RE.match(p)]
            _write_json(d / "meta.json", base)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            raise
        with self._lock:
            self.index[aid] = base
        self.on_change("created", base)
        return base

    @staticmethod
    def _file_entry(p: Path, role: str) -> dict:
        return {"name": p.name, "role": role, "bytes": p.stat().st_size,
                "mime": media.MIME.get(p.suffix.lower(), "application/octet-stream")}

    def _analyze(self, d: Path, main: Path, type: str) -> tuple[dict, dict]:  # noqa: A002
        info, preview = {}, {}
        try:
            if type == "image":
                info = media.image_info(main)
                media.make_thumb(main, d / "thumb.webp")
                preview["thumb"] = "thumb.webp"
            elif type in ("audio", "voice"):
                info, wf = media.analyze_audio(main)
                _write_json(d / "waveform.json", wf)
                preview["waveform"] = "waveform.json"
            elif type == "material":      # the card shows the sphere preview when the material has one, else the base colour
                info = media.image_info(main)
                src = d / "preview.png"
                media.make_thumb(src if src.is_file() else main, d / "thumb.webp")
                preview["thumb"] = "thumb.webp"
            elif type == "mesh" and main.suffix.lower() == ".glb":
                info = media.glb_info(main)
            elif type == "video":
                info = media.video_info(main)
                if media.make_video_poster(main, d / "thumb.webp", info.get("duration")):
                    preview["thumb"] = "thumb.webp"
        except ValueError as e:
            info = {"analysis_error": str(e)}
        return info, preview

    def upload(self, project: str, filename: str, tmp_path: Path, meta: dict | None = None) -> dict:
        """Validate an uploaded file by content (not only by name) and ingest it."""
        ext = Path(filename or "").suffix.lower()
        t = media.type_of(filename)
        if t == "other":
            raise AssetError(f"Unsupported file type '{ext or '?'}'. Allowed: images (png, jpg, webp), audio (wav, flac, "
                             "ogg, mp3), video (mp4, mov, webm, mkv), meshes (glb).", 415)
        limits = {"image": 60, "audio": 300, "mesh": 500, "video": 500}
        size = tmp_path.stat().st_size
        if size > limits[t] * 2**20:
            raise AssetError(f"File too large ({size / 2**20:.0f} MB; the limit for {t} is {limits[t]} MB).", 413)
        if size == 0:
            raise AssetError("The file is empty.")
        try:
            if t == "image":
                media.verify_image(tmp_path)
            elif t == "audio":
                media.analyze_audio(tmp_path, buckets=10)
            elif t == "mesh":
                media.glb_info(tmp_path)
            elif t == "video":
                media.verify_video(tmp_path)
        except ValueError as e:
            raise AssetError(str(e), 415) from None
        m = dict(meta or {})
        m.setdefault("name", Path(safe_name(filename)).stem)
        m["source"] = "upload"
        m["original_filename"] = safe_name(filename)
        atype = m.pop("type", None) if m.get("type") in ("voice",) and t == "audio" else None
        return self.ingest(project, tmp_path, atype or t, m, move=True)

    # ------------------------------------------------------------ change
    EDITABLE = ("name", "tags", "rating", "notes", "favorite")

    def update(self, aid: str, patch: dict, *, extra_ok: bool = False) -> dict:
        """User edits (name/tags/rating/notes/favorite). `extra_ok` lets the core merge other keys (e.g. `x_*`)."""
        meta = self.get(aid)
        d = self.asset_dir(meta["project"], aid)
        new = dict(meta)
        for k, v in patch.items():
            if k == "tags":
                tags = [re.sub(r"\s+", " ", str(t)).strip()[:40] for t in (v or [])]
                new["tags"] = sorted({t for t in tags if t})[:50]
            elif k == "rating":
                new["rating"] = max(0, min(5, int(v or 0)))
            elif k == "favorite":
                new["favorite"] = bool(v)
            elif k == "notes":
                new["notes"] = str(v or "")[:5000]
            elif k == "name":
                new["name"] = str(v or "").strip()[:120] or meta["name"]
            elif k.startswith("x_") or extra_ok:
                new[k] = v
            else:
                raise AssetError(f"Field '{k}' cannot be edited.")
        new["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock:
            _write_json(d / "meta.json", new)
            self.index[aid] = new
        self.on_change("updated", new)
        return new

    def set_thumb(self, aid: str, tmp_path: Path) -> dict:
        """Store a client-rendered preview (e.g. a model-viewer snapshot of a GLB) as thumb.webp."""
        meta = self.get(aid)
        media.verify_image(tmp_path)
        d = self.asset_dir(meta["project"], aid)
        media.make_thumb(tmp_path, d / "thumb.webp")
        tmp_path.unlink(missing_ok=True)
        new = dict(meta)
        new["preview"] = {**(meta.get("preview") or {}), "thumb": "thumb.webp"}
        with self._lock:
            _write_json(d / "meta.json", new)
            self.index[aid] = new
        self.on_change("updated", new)
        return new

    def delete(self, aid: str) -> dict:
        meta = self.get(aid)
        d = self.asset_dir(meta["project"], aid)
        trash = self.project_dir(meta["project"]) / ".trash"
        trash.mkdir(exist_ok=True)
        shutil.move(str(d), str(trash / aid))
        with self._lock:
            self.index.pop(aid, None)
        self.on_change("deleted", {"id": aid, "project": meta["project"], "type": meta.get("type")})
        return {"id": aid, "deleted": True}

    # ------------------------------------------------------------ lineage
    def lineage(self, aid: str, depth: int = 12) -> dict:
        meta = self.get(aid)

        def brief(m):
            return {k: m.get(k) for k in ("id", "type", "name", "model", "task", "created", "preview", "project", "main")}

        ancestors, seen, frontier = [], {aid}, list(meta.get("parents") or [])
        for _ in range(depth):
            nxt = []
            for p in frontier:
                if p in seen:
                    continue
                seen.add(p)
                pm = self.index.get(p)
                if pm is None:
                    ancestors.append({"id": p, "missing": True})
                    continue
                ancestors.append(brief(pm))
                nxt += pm.get("parents") or []
            frontier = nxt
            if not frontier:
                break
        children = [brief(m) for m in self.index.values() if aid in (m.get("parents") or [])]
        children.sort(key=lambda m: m.get("created") or "")
        siblings = []
        if meta.get("batch", {}).get("id") if isinstance(meta.get("batch"), dict) else False:
            b = meta["batch"]["id"]
            siblings = [brief(m) for m in self.index.values()
                        if isinstance(m.get("batch"), dict) and m["batch"].get("id") == b and m["id"] != aid]
        return {"asset": brief(meta), "ancestors": ancestors, "children": children, "siblings": siblings}

    # ------------------------------------------------------------ export
    def convert_main(self, meta: dict, export: dict | None, out_dir: Path, stem: str) -> tuple[Path, dict] | None:
        """Derived copy of an asset's main file per an export spec ({audio: {...}, image: {...}}, see app/profiles.py):
        audio / voice -> media.export_audio (two-pass loudnorm, rate, bit depth, format), image -> media.export_image.
        Returns (new file, report), or None when the spec has nothing for this asset type (meshes, video, no options).
        The stored original is only read, never changed."""
        if not export:
            return None
        d = self.asset_dir(meta["project"], meta["id"])
        src = (d / meta["main"]).resolve()
        if src.parent != d or not src.is_file():
            raise AssetError("The asset's main file is missing.", 404)
        if meta["type"] in ("audio", "voice") and export.get("audio"):
            from .profiles import MUSIC_TASKS
            return media.export_audio(src, out_dir, stem, export["audio"], music=meta.get("task") in MUSIC_TASKS)
        if meta["type"] == "image" and export.get("image"):
            return media.export_image(src, out_dir, stem, export["image"])
        return None

    def render(self, aid: str, export: dict | None, out_root: Path) -> dict:
        """Convert one asset for download. Writes out_root/render-<token>/<name>.<ext> and returns
        {token, name, path, report}. Raises AssetError when the export spec does not apply to this asset."""
        meta = self.get(aid)
        out_root.mkdir(parents=True, exist_ok=True)
        for old in out_root.glob("render-*"):
            if old.is_dir() and time.time() - old.stat().st_mtime > 3600:
                shutil.rmtree(old, ignore_errors=True)
        token = secrets.token_hex(6)
        d = out_root / f"render-{token}"
        try:
            res = self.convert_main(meta, export, d, slug(meta.get("name") or meta["id"]))
        except ValueError as e:      # media.ExportError
            shutil.rmtree(d, ignore_errors=True)
            raise AssetError(str(e)) from None
        if res is None:
            shutil.rmtree(d, ignore_errors=True)
            raise AssetError(f"This export preset has no conversion for {meta['type']} assets (the original file is unchanged).")
        path, report = res
        return {"token": token, "name": path.name, "path": path, "report": report, "bytes": path.stat().st_size}

    def export(self, ids: list[str], out_dir: Path, *, naming: str = "name", include_meta: bool = True,
               include_attachments: bool = True, convert: dict | None = None) -> Path:
        """Zip of the selected assets: <type>/<name>.<ext> (+ attachments) and manifest.json.
        `convert` ({audio: {...}, image: {...}}) re-encodes audio / images into derived files inside the zip (loudness,
        sample rate, format, crop, size); without it the stored files are zipped as they are."""
        if not ids:
            raise AssetError("Select at least one asset to export.")
        metas = [self.get(i) for i in ids]
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("export-*.zip"):
            if time.time() - old.stat().st_mtime > 3600:
                old.unlink(missing_ok=True)
        zpath = out_dir / f"export-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}.zip"
        manifest = {"exported": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": "Open Asset Studio", "assets": []}
        if convert:
            manifest["export"] = convert
        used: set[str] = set()
        tmp = tempfile.TemporaryDirectory(prefix="oas-export-")
        with tmp, zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for m in metas:
                d = self.asset_dir(m["project"], m["id"])
                stem = slug(m.get("name") or m["id"]) if naming == "name" else m["id"]
                folder = {"voice": "voices", "audio": "audio", "image": "images", "mesh": "models", "video": "video",
                          "material": "materials"}.get(m["type"], "other")
                base = f"{folder}/{stem}"
                n = 2
                while base in used:
                    base = f"{folder}/{stem}_{n}"
                    n += 1
                used.add(base)
                entry = {"id": m["id"], "type": m["type"], "name": m.get("name"), "files": [],
                         "model": m.get("model"), "license": m.get("license"), "prompt": m.get("final_prompt") or m.get("prompt"),
                         "seed": m.get("seed"), "params": m.get("params"), "parents": m.get("parents"),
                         "tags": m.get("tags"), "rating": m.get("rating"), "info": m.get("info")}
                conv = None
                exporter = EXPORTERS.get(m["type"])
                if exporter is not None:       # e.g. materials: packed maps per engine layout, written as derived files
                    try:
                        files, report = exporter(m, d, (convert or {}).get(m["type"]) or {}, Path(tmp.name) / m["id"], base)
                    except ValueError as e:
                        raise AssetError(f"{m.get('name') or m['id']}: {e}") from None
                    for src, arc in files:
                        z.write(src, arc)
                        entry["files"].append(arc)
                    entry[m["type"]] = report
                    if include_meta:
                        z.writestr(f"{base}.meta.json", json.dumps(m, indent=1, ensure_ascii=False, default=str))
                    manifest["assets"].append(entry)
                    continue
                if convert:
                    try:
                        conv = self.convert_main(m, convert, Path(tmp.name) / m["id"], "converted")
                    except ValueError as e:
                        raise AssetError(f"{m.get('name') or m['id']}: {e}") from None
                for f in m.get("files", []):
                    if f["role"] != "main" and not include_attachments:
                        continue
                    src = (d / f["name"]).resolve()
                    if src.parent != d or not src.is_file():
                        continue
                    if f["role"] == "main" and conv:
                        src = conv[0]
                        arc = f"{base}{src.suffix}"
                        entry["export"] = {**conv[1], "original": f["name"]}
                    else:
                        arc = f"{base}{Path(f['name']).suffix}" if f["role"] == "main" else f"{base}_{f['name']}"
                    z.write(src, arc)
                    entry["files"].append(arc)
                if include_meta:
                    z.writestr(f"{base}.meta.json", json.dumps(m, indent=1, ensure_ascii=False, default=str))
                manifest["assets"].append(entry)
            z.writestr("manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
        return zpath
