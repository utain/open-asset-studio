"""Game-engine export: the job handler of `engine-export` plus the safe delivery of a finished package into a game project folder.

  handler engine.export   inputs: one or more mesh assets (GLB). Per mesh it runs the `engine` backend (workers/engine_worker.py, envs/retopo)
                          beside the GPU queue and writes ONE new mesh asset per input: `mesh.glb` (the preview: LOD0 + MSFT_lod levels + hidden
                          hull nodes) with attachments engine_<target>.zip, engine_report.json and engine_collision.glb. Meta: worker.engine.
                          A failing mesh of a batch is reported as a warning; the job fails only when every mesh failed.
  router /api/x/engine    GET  /destinations   the delivery folders configured in settings.json (the path only for same-machine callers)
                          POST /deliver        {asset_id, destination, overwrite?}  copy a package into <destination>/<subdir>/<asset name>/

Delivery security (docs/CONTRACTS.md section 12; every rule has a test in tests/test_engine_export.py):
  * the destinations are FILE-ONLY (settings.json `engine.destinations`): no API, no MCP tool and no UI can add or change one,
  * the API names a destination by its id, never by a path (the body model forbids extra fields),
  * same-machine callers only unless the destination sets `allow_lan`; a browser Origin of another site is refused; the MCP tool is gated on the
    MCP caller's locality (the HTTP MCP server calls this API from 127.0.0.1, so locality here says nothing about its caller),
  * zip member names are whitelisted (`[Textures|Editor/]name.ext`, ext fbx/glb/png/json/txt/cs), at most 64 files, total size <= max_mb, with a
    zip-bomb guard (declared sizes are checked, then the bytes actually read are counted),
  * writes walk from the destination root with O_DIRECTORY|O_NOFOLLOW directory descriptors (no symlink is ever followed), write a temporary file
    with O_EXCL and rename it into place,
  * files are only replaced when `.oas-delivery.json` (sha256 per file) says we wrote them AND they are unchanged since; a folder that exists
    without that ledger is never touched; nothing is ever deleted.
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import secrets
import stat
import sys
import threading
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .. import config, netguard
from ..assets import AssetError
from ..jobs import JobContext, JobError, handler

log = logging.getLogger("studio")
router = APIRouter()
CORE = None

LEDGER = ".oas-delivery.json"
LEDGER_MAX = 1 << 20
ID_RE = re.compile(r"[a-z0-9_-]{1,32}")
SUBDIR_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
MEMBER_RE = re.compile(r"(?:(?:Textures|Editor)/)?[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}")
MEMBER_EXT = {"fbx", "glb", "png", "json", "txt", "cs"}
MAX_FILES = 64
MAX_MB_CAP = 4096
DEFAULT_SUBDIR = "OpenAssetStudio"
ENGINES = ("unreal", "unity", "godot", "gltf", "any")
# a destination may not be (or be inside) any of these
FORBIDDEN = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/proc", "/sys", "/dev", "/var", "/run", "/root")
_BUSY = threading.Lock()


class DeliveryError(Exception):
    """A refusal with its HTTP status; `code` is a short machine-readable reason for the UI and the tests."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


# ================================================================== helpers shared with the export handler
def _common():
    if str(config.WORKERS) not in sys.path:
        sys.path.append(str(config.WORKERS))
    import engine_common
    return engine_common


def _lic(parents: list) -> dict | None:
    """The most restrictive licence (non-commercial wins); in a tie the first one, so the source asset comes first (same as retopo)."""
    lics = [x for x in parents if x]
    nc = [x for x in lics if not x.get("commercial", True)]
    return (nc or lics or [None])[0]


def _unique(stem: str, used: set) -> str:
    k, out = 2, stem
    while out.lower() in used:
        out, k = f"{stem}_{k}", k + 1
    used.add(out.lower())
    return out


def _settings_engine(ctx: JobContext) -> dict:
    try:
        return dict(ctx.settings.get("engine") or {}) if ctx.settings else {}
    except Exception:  # noqa: BLE001
        return {}


@handler("engine.export", gpu=False, label="Game engine export")
def export(ctx: JobContext):
    ec = _common()
    ins = [i for i in ctx.input_assets() if i["type"] == "mesh"]
    if not ins:
        raise JobError("Pick a mesh asset (GLB) to export.", "input")
    for i in ins:
        if not str(i["path"]).lower().endswith(".glb"):
            raise JobError(f"'{i.get('name')}' is not a GLB. Export the model as GLB first (3D > Export), then send it to the game engine step.", "input")
    es = _settings_engine(ctx)
    style = None
    try:
        g = CORE.style.get(ctx.job.get("project")) if CORE is not None and getattr(CORE, "style", None) else None
        if g and g.get("enabled", True):
            style = {"name": g.get("name"), "version": g.get("version"), "hash": g.get("hash")}
    except Exception:  # noqa: BLE001 - the style guide is optional context
        style = None
    n, used, outs, errors, warnings = len(ins), set(), 0, [], []
    for k, inp in enumerate(ins):
        src = ctx.assets.get(inp["asset_id"])
        p = dict(ctx.params)
        base = ec.safe_stem(p.get("name") or src.get("name") or "model") if (n == 1 and p.get("name")) else ec.safe_stem(src.get("name") or "model")
        p["name"] = _unique(base, used)
        p["_ctx"] = {"source": {"id": src["id"], "name": src.get("name"), "license": src.get("license"), "x_style": ec.x_style_ref(src)},
                     "prefix_static": es.get("prefix_static"), "prefix_texture": es.get("prefix_texture"), "style": style}
        lo, hi = 0.95 * k / n, 0.95 * (k + 1) / n
        try:
            res = ctx.run_worker(params=p, inputs=[{**inp, "role": "mesh"}], frac_range=(lo, hi))
        except JobError as e:
            if n == 1:
                raise
            errors.append(e)
            warnings.append(f"{src.get('name') or src['id']}: {e}")
            continue
        ctx.progress(hi, "Saving outputs", stage="saving")
        lic = _lic([src.get("license"), (ctx.model or {}).get("license")])
        for i, f in enumerate(res.get("files") or []):
            meta = ctx.asset_meta(f, k + i)
            meta["parents"] = [src["id"]]
            if lic:
                meta["license"] = lic
            meta["tags"] = sorted(set(meta.get("tags") or []) | {"engine", str(ctx.params.get("target") or "unreal")})
            att = [str(Path(f["path"]).parent / a) if not str(a).startswith("/") else a for a in f.get("attach") or []]
            ctx.add_output(f["path"], f.get("type") or "mesh", meta, attach=att)
            outs += 1
        warnings += [f"{src.get('name') or src['id']}: {w}" for w in (res.get("meta") or {}).get("warnings") or [] if n > 1]
    if not outs:
        raise errors[0] if errors else JobError("The engine export produced no file.", "internal")
    if n > 1 and warnings:
        ctx.job["result_meta"] = {"warnings": warnings}


def setup(core):
    global CORE
    CORE = core


# ================================================================== destinations (settings.json, file-only)
def validate_destinations(raw, *, studio=None, extra_forbidden=(), home=None) -> tuple[list[dict], list[str]]:
    """Check `engine.destinations` -> (usable destinations, problems). A bad entry is dropped (and named in `problems`), never repaired.

    Each usable entry: {id, label, path (as written), real (os.path.realpath), subdir, engine, max_mb, allow_lan}.
    Refused: a bad or duplicate id, a relative or non-existing path, `/`, the home directory, fewer than 3 path components, anything
    inside the studio folder (or `extra_forbidden`, e.g. the data and logs folders) and the system folders in FORBIDDEN."""
    problems: list[str] = []
    if raw in (None, ""):
        return [], problems
    if not isinstance(raw, list):
        return [], ["engine.destinations must be a list."]
    studio_real = os.path.realpath(studio if studio is not None else config.STUDIO)
    home_real = os.path.realpath(home if home is not None else os.path.expanduser("~"))
    blocked = [studio_real, *[os.path.realpath(str(x)) for x in extra_forbidden]]
    out, seen = [], set()
    for n, d in enumerate(raw):
        tag = f"destination #{n + 1}"
        if not isinstance(d, dict):
            problems.append(f"{tag}: must be an object.")
            continue
        did = d.get("id")
        tag = f"destination '{did}'" if isinstance(did, str) else tag
        if not isinstance(did, str) or not ID_RE.fullmatch(did):
            problems.append(f"{tag}: id must match [a-z0-9_-]{{1,32}}.")
            continue
        if did in seen:
            problems.append(f"{tag}: duplicate id.")
            continue
        path = d.get("path")
        if not isinstance(path, str) or not path.startswith("/") or "\0" in path:
            problems.append(f"{tag}: path must be an absolute folder path.")
            continue
        real = os.path.realpath(path)
        if not os.path.isdir(real):
            problems.append(f"{tag}: the folder does not exist.")
            continue
        parts = [x for x in real.split("/") if x]
        if real == "/" or real == home_real or len(parts) < 3:
            problems.append(f"{tag}: too broad (use the folder of the game project, not '/', your home folder or a top-level folder).")
            continue
        if any(real == b or real.startswith(b.rstrip("/") + "/") for b in blocked):
            problems.append(f"{tag}: inside the studio's own folders.")
            continue
        if any(real == f or real.startswith(f + "/") for f in FORBIDDEN):
            problems.append(f"{tag}: system folders are not allowed.")
            continue
        sub = d.get("subdir", DEFAULT_SUBDIR)
        if not isinstance(sub, str) or not SUBDIR_RE.fullmatch(sub):
            problems.append(f"{tag}: subdir must match [A-Za-z0-9_-]{{1,64}}.")
            continue
        eng = d.get("engine", "any")
        if eng not in ENGINES:
            problems.append(f"{tag}: engine must be one of {', '.join(ENGINES)}.")
            continue
        mb = d.get("max_mb", 512)
        if isinstance(mb, bool) or not isinstance(mb, (int, float)) or not 1 <= mb <= MAX_MB_CAP:
            problems.append(f"{tag}: max_mb must be between 1 and {MAX_MB_CAP}.")
            continue
        lan = d.get("allow_lan", False)
        if not isinstance(lan, bool):
            problems.append(f"{tag}: allow_lan must be true or false.")
            continue
        label = d.get("label") if isinstance(d.get("label"), str) and d.get("label").strip() else did
        seen.add(did)
        out.append({"id": did, "label": label.strip()[:80], "path": path, "real": real, "subdir": sub, "engine": eng,
                    "max_mb": int(mb), "allow_lan": lan})
    return out, problems


def configured_destinations() -> tuple[list[dict], list[str]]:
    raw = CORE.settings.get("engine", "destinations") if CORE is not None else []
    return validate_destinations(raw, extra_forbidden=(config.DATA, config.LOGS))


# ================================================================== package reading (zip bomb guard, name whitelist)
def read_package(zip_path, *, max_bytes: int) -> dict[str, bytes]:
    """The members of an engine package as {name: bytes}, after every check. DeliveryError(400) on the first violation."""
    try:
        z = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile):
        raise DeliveryError(400, "bad_package", "The package file cannot be read.") from None
    with z:
        infos = z.infolist()
        if not infos:
            raise DeliveryError(400, "bad_package", "The package is empty.")
        if len(infos) > MAX_FILES:
            raise DeliveryError(400, "too_many_files", f"The package has more than {MAX_FILES} files.")
        names, total = set(), 0
        for zi in infos:
            nm = zi.filename
            if zi.is_dir() or not MEMBER_RE.fullmatch(nm) or nm.rsplit("/", 1)[-1] in (".", "..") or ".." in nm.split("/"):
                raise DeliveryError(400, "bad_member", f"The package contains a file name that is not allowed ('{nm[:60]}').")
            if nm.rsplit(".", 1)[-1].lower() not in MEMBER_EXT or "." not in nm:
                raise DeliveryError(400, "bad_member", f"The package contains a file type that is not allowed ('{nm[:60]}').")
            if nm.lower() in names:
                raise DeliveryError(400, "bad_member", f"The package lists '{nm}' twice.")
            names.add(nm.lower())
            if stat.S_ISLNK((zi.external_attr >> 16) & 0xFFFF) or zi.flag_bits & 0x1:
                raise DeliveryError(400, "bad_member", f"'{nm}' is a link or encrypted.")
            total += zi.file_size
            if total > max_bytes:
                raise DeliveryError(413, "too_big", f"The package is larger than the {max_bytes // (1 << 20)} MB limit of this destination.")
        out, got = {}, 0
        for zi in infos:
            with z.open(zi) as fh:
                data = fh.read(zi.file_size + 1)
            got += len(data)
            if len(data) != zi.file_size or got > max_bytes:         # the header lied: a zip bomb
                raise DeliveryError(400, "bad_package", f"'{zi.filename}' does not match its declared size.")
            out[zi.filename] = data
    return out


# ================================================================== delivery (fd-based, no symlink is followed)
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _open_dir(parent_fd: int, name: str, create: bool) -> tuple[int, bool]:
    """Open (or create) the directory `name` inside `parent_fd` without following a symlink. -> (fd, created)."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    created = False
    try:
        return os.open(name, flags, dir_fd=parent_fd), created
    except FileNotFoundError:
        if not create:
            raise
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            raise DeliveryError(409, "unsafe_path", f"'{name}' is a link or not a folder; nothing was written.") from None
        raise
    try:
        os.mkdir(name, 0o755, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    try:
        return os.open(name, flags, dir_fd=parent_fd), created
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            raise DeliveryError(409, "unsafe_path", f"'{name}' is a link or not a folder; nothing was written.") from None
        raise


def _read_ledger(dfd: int) -> dict | None:
    """The delivery ledger of a folder: {relative name: sha256}, or None when the folder has none. A broken one is a refusal."""
    try:
        fd = os.open(LEDGER, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=dfd)
    except FileNotFoundError:
        return None
    except OSError:
        raise DeliveryError(409, "bad_ledger", "The delivery record in this folder is not readable; nothing was written.") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > LEDGER_MAX:
            raise DeliveryError(409, "bad_ledger", "The delivery record in this folder is not valid; nothing was written.")
        with os.fdopen(fd, "rb", closefd=False) as fh:
            j = json.loads(fh.read(LEDGER_MAX + 1))
    except ValueError:
        raise DeliveryError(409, "bad_ledger", "The delivery record in this folder is not valid; nothing was written.") from None
    finally:
        os.close(fd)
    files = j.get("files") if isinstance(j, dict) else None
    if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
        raise DeliveryError(409, "bad_ledger", "The delivery record in this folder is not valid; nothing was written.")
    return files


def _stat_file(dfd: int, name: str):
    """lstat of `name` inside dfd: None when missing; refuses a symlink or anything that is not a regular file."""
    try:
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode):
        raise DeliveryError(409, "unsafe_path", f"'{name}' exists and is not a plain file (link or folder); nothing was written.")
    return st


def _file_sha(dfd: int, name: str) -> str:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=dfd)
    h = hashlib.sha256()
    try:
        with os.fdopen(fd, "rb", closefd=False) as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    finally:
        os.close(fd)
    return h.hexdigest()


def _write_atomic(dfd: int, name: str, data: bytes) -> None:
    """Temp file (O_EXCL, never an existing path) in the same folder, then rename over `name`."""
    tmp = f".{name[:60]}.oas-{secrets.token_hex(4)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o644, dir_fd=dfd)
    try:
        with os.fdopen(fd, "wb", closefd=False) as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            os.unlink(tmp, dir_fd=dfd)
        except OSError:
            pass
        raise
    os.close(fd)
    try:
        os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
    except BaseException:
        try:
            os.unlink(tmp, dir_fd=dfd)
        except OSError:
            pass
        raise


def deliver_files(root: str, subdir: str, name: str, members: dict[str, bytes], *, overwrite: bool = False, asset_id: str = "") -> dict:
    """Copy `members` into <root>/<subdir>/<name>/. `root` must already be a real (resolved) directory path.

    Plan first, write second: any conflict raises DeliveryError(409) before the first byte is written.
      * the folder <name> exists but has no ledger  -> 409 (not ours, never touched)
      * a target exists and is not in the ledger     -> 409
      * a target is in the ledger but was edited      -> 409 (never overwritten, even with overwrite=True)
      * a target is ours and unchanged                -> replaced only with overwrite=True, otherwise 409 `exists`; identical content is skipped
    Files are never deleted; ledger entries of files that this package no longer contains are kept.
    -> {"written": [...], "unchanged": [...], "stale": [...], "bytes": n, "relpath": "<subdir>/<name>", "folder_created": bool}"""
    if not SUBDIR_RE.fullmatch(subdir or "") or not NAME_RE.fullmatch(name or ""):
        raise DeliveryError(400, "bad_name", "The asset name cannot be used as a folder name.")
    for m in members:
        if not MEMBER_RE.fullmatch(m) or ".." in m.split("/"):
            raise DeliveryError(400, "bad_member", f"File name not allowed: '{m[:60]}'.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    opened: list[int] = []
    try:
        try:
            rfd = os.open(root, flags)
        except OSError:
            raise DeliveryError(409, "unsafe_path", "The destination folder is missing or is a link; nothing was written.") from None
        opened.append(rfd)
        sfd, _ = _open_dir(rfd, subdir, True)
        opened.append(sfd)
        existed = True
        try:
            nfd, _ = _open_dir(sfd, name, False)
        except FileNotFoundError:
            existed = False
            nfd, _ = _open_dir(sfd, name, True)
        opened.append(nfd)
        if not existed:                    # claim the new folder at once, so a crash later never leaves a "foreign" folder behind
            _write_atomic(nfd, LEDGER, json.dumps({"tool": "Open Asset Studio", "version": 1, "asset": asset_id, "files": {}}).encode())
        ledger = _read_ledger(nfd)
        if existed and ledger is None:
            raise DeliveryError(409, "not_ours", f"The folder '{subdir}/{name}' already exists and was not created by Open Asset Studio; nothing was written.")
        ledger = dict(ledger or {})
        # ---- plan (no write yet)
        dirs: dict[str, int] = {"": nfd}
        todo, same = [], []
        for rel, data in members.items():
            d, _, base = rel.rpartition("/")
            if d not in dirs:
                try:
                    dfd, _ = _open_dir(nfd, d, False)
                except FileNotFoundError:
                    dfd = -1
                if dfd >= 0:
                    opened.append(dfd)
                dirs[d] = dfd
            dfd = dirs[d]
            st = _stat_file(dfd, base) if dfd >= 0 else None
            sha = _sha(data)
            if st is None:
                todo.append((rel, data, sha))
                continue
            cur = _file_sha(dfd, base)
            if rel not in ledger:
                raise DeliveryError(409, "exists_foreign", f"'{rel}' already exists in the destination and is not one of our files; nothing was written.")
            if cur != ledger[rel]:
                raise DeliveryError(409, "edited", f"'{rel}' was changed after it was delivered; it is not overwritten and nothing was written.")
            if cur == sha:
                same.append(rel)
            elif overwrite:
                todo.append((rel, data, sha))
            else:
                raise DeliveryError(409, "exists", f"'{rel}' was delivered before. Deliver again with overwrite to replace it.")
        # ---- write
        written, total = [], 0
        for rel, data, sha in todo:
            d, _, base = rel.rpartition("/")
            if dirs.get(d, -1) < 0:
                dfd, _ = _open_dir(nfd, d, True)
                opened.append(dfd)
                dirs[d] = dfd
            _write_atomic(dirs[d], base, data)
            ledger[rel] = sha
            written.append(rel)
            total += len(data)
        for rel in same:
            ledger[rel] = _sha(members[rel])
        led = {"tool": "Open Asset Studio", "version": 1, "asset": asset_id, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": ledger}
        _write_atomic(nfd, LEDGER, json.dumps(led, indent=1, sort_keys=True).encode())
        stale = sorted(set(ledger) - set(members))
        return {"written": written, "unchanged": same, "stale": stale, "bytes": total, "relpath": f"{subdir}/{name}", "folder_created": not existed}
    finally:
        for fd in opened:
            try:
                os.close(fd)
            except OSError:
                pass


# ================================================================== REST
class DeliverReq(BaseModel):
    model_config = ConfigDict(extra="forbid")     # a `path` (or anything else) in the body is a 422: destinations are named by id only
    asset_id: str
    destination: str
    overwrite: bool = False


def _public(d: dict, local: bool) -> dict:
    out = {k: d[k] for k in ("id", "label", "subdir", "engine", "max_mb", "allow_lan")}
    out["can_deliver"] = local or d["allow_lan"]
    if local:
        out["path"] = d["real"]
    return out


@router.get("/destinations")
def destinations(request: Request):
    local = netguard.is_local(request)
    dests, problems = configured_destinations()
    body = {"destinations": [_public(d, local) for d in dests], "local": local}
    if local and problems:
        body["problems"] = problems
    return body


def _package_of(aid: str) -> tuple[dict, dict, Path]:
    try:
        meta = CORE.assets.get(aid)
    except AssetError as e:
        raise DeliveryError(e.status if hasattr(e, "status") else 404, "no_asset", str(e)) from None
    eng = (meta.get("worker") or {}).get("engine") if meta.get("type") == "mesh" else None
    pkg = (eng or {}).get("package")
    if not isinstance(eng, dict) or not isinstance(pkg, str) or not pkg.startswith("engine_") or pkg not in {f["name"] for f in meta.get("files", [])}:
        raise DeliveryError(400, "not_a_package", "This asset is not a game-engine export. Run the Game engine step first.")
    try:
        path = CORE.assets.file_path(meta["project"], meta["id"], pkg)
    except AssetError as e:
        raise DeliveryError(404, "no_package", str(e)) from None
    return meta, eng, path


@router.post("/deliver")
def deliver(req: DeliverReq, request: Request):
    try:
        return _deliver(req, request)
    except DeliveryError as e:
        log.warning("engine delivery refused: asset=%s dest=%s code=%s client=%s", req.asset_id[:40], req.destination[:40], e.code,
                    request.client.host if request.client else "?")
        return JSONResponse({"error": str(e), "code": e.code, "kind": "input"}, status_code=e.status)


def _deliver(req: DeliverReq, request: Request):
    if not netguard.origin_ok(request):
        raise DeliveryError(403, "origin", "Origin not allowed.")
    local = netguard.is_local(request)
    dests, _ = configured_destinations()
    dest = next((d for d in dests if d["id"] == req.destination), None)
    if dest is None:
        raise DeliveryError(404, "no_destination", "That delivery destination is not configured in settings.json.")
    if not (local or dest["allow_lan"]):
        raise DeliveryError(403, "not_local", "Delivery into a game project is only allowed from the computer the studio runs on.")
    if not _BUSY.acquire(blocking=False):
        raise DeliveryError(409, "busy", "Another delivery is running; try again in a moment.")
    try:
        meta, eng, zpath = _package_of(req.asset_id)
        target = str(eng.get("target") or "")
        if dest["engine"] not in ("any", target):
            raise DeliveryError(400, "engine_mismatch", f"This destination is for {dest['engine']}; the package is for {target}.")
        name = str(eng.get("name") or "")
        members = read_package(zpath, max_bytes=dest["max_mb"] << 20)
        res = deliver_files(dest["real"], dest["subdir"], name, members, overwrite=req.overwrite, asset_id=meta["id"])
    except OSError as e:
        log.warning("engine delivery failed: asset=%s dest=%s error=%s", req.asset_id[:40], dest["id"], e)
        raise DeliveryError(500, "io", f"Could not write into the destination ({e.strerror or 'I/O error'}).") from None
    finally:
        _BUSY.release()
    log.info("engine delivery: asset=%s dest=%s written=%d unchanged=%d bytes=%d overwrite=%s client=%s", meta["id"], dest["id"],
             len(res["written"]), len(res["unchanged"]), res["bytes"], req.overwrite, request.client.host if request.client else "?")
    try:                                   # audit trail in the asset (destination id only: the path stays in settings.json)
        w = dict(meta.get("worker") or {})
        e2 = dict(w.get("engine") or {})
        e2["deliveries"] = (e2.get("deliveries") or [])[-19:] + [{"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "destination": dest["id"], "relpath": res["relpath"],
                                                                 "written": len(res["written"]), "unchanged": len(res["unchanged"]), "overwrite": req.overwrite}]
        w["engine"] = e2
        CORE.assets.update(meta["id"], {"worker": w}, extra_ok=True)
    except Exception:  # noqa: BLE001 - the files are delivered; the audit note is best effort
        log.exception("engine delivery: could not record it in the asset")
    out = {"ok": True, "destination": dest["id"], "relpath": res["relpath"], "written": res["written"], "unchanged": res["unchanged"],
           "stale": res["stale"], "bytes": res["bytes"], "folder_created": res["folder_created"]}
    if local:
        out["dir"] = os.path.join(dest["real"], dest["subdir"], name)
    return out
