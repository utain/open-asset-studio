"""Download runner: the ONLY studio process that goes online. Spawned by app/downloads.py, one per weight entry.

  python -m app.dl_runner plan --repo R [--include PAT ...]       -> one JSON line {total, present, files, gated, ...}
  python -m app.dl_runner hf   --repo R [--include PAT ...]       -> JSON lines: start / progress / done | error
  python -m app.dl_runner cmd  --watch DIR --size N -- argv...    -> runs argv (from the registry, never from users)

Resumes automatically: huggingface_hub keeps partial files as blobs/*.incomplete and continues them.
Progress = bytes already cached + growth of the repo's blobs/ directory (or of --watch for cmd downloads).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def emit(**kw):
    sys.stdout.write(json.dumps(kw) + "\n")
    sys.stdout.flush()


def dir_bytes(p: Path) -> int:
    n = 0
    if p.is_dir():
        for dp, _d, names in os.walk(p):
            for f in names:
                try:
                    n += os.lstat(os.path.join(dp, f)).st_size
                except OSError:
                    pass
    return n


def classify(e: Exception) -> tuple[str, str]:
    from huggingface_hub import errors as E
    name = type(e).__name__
    msg = str(e).splitlines()[0][:300] if str(e) else name
    if isinstance(e, getattr(E, "GatedRepoError", ())):
        return "gated", "This repo is gated: accept its licence on huggingface.co (logged in as the token's user), then retry."
    if isinstance(e, getattr(E, "RepositoryNotFoundError", ())):
        return "auth", "Repo not found or not accessible with the current token (run `hf auth login`)."
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return "disk", "Disk full."
    if "Connection" in name or "Timeout" in name or "connect" in msg.lower():
        return "network", f"Network problem: {msg}"
    return "internal", f"{name}: {msg}"


def plan_files(repo: str, include: list[str]):
    from huggingface_hub import HfApi, try_to_load_from_cache
    info = HfApi().model_info(repo, files_metadata=True)
    from huggingface_hub import constants
    sib = [s for s in info.siblings or [] if not include or any(fnmatch.fnmatch(s.rfilename, p) for p in include)]
    files = [(s.rfilename, s.size or 0) for s in sib]
    blobs = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "blobs"
    present = 0
    for s in sib:
        hit = try_to_load_from_cache(repo, s.rfilename)
        if isinstance(hit, str) and os.path.exists(hit):
            present += s.size or 0
        elif getattr(s, "lfs", None) and getattr(s.lfs, "sha256", None):  # resumable partial download
            part = blobs / f"{s.lfs.sha256}.incomplete"
            if part.exists():
                present += part.stat().st_size
    return info, files, present


def cmd_plan(a):
    try:
        info, files, present = plan_files(a.repo, a.include)
        emit(type="plan", repo=a.repo, total=sum(s for _, s in files), present=present, files=len(files),
             gated=bool(getattr(info, "gated", False)), sha=info.sha,
             largest=sorted(([n, s] for n, s in files), key=lambda x: -x[1])[:5])
    except Exception as e:  # noqa: BLE001
        kind, msg = classify(e)
        emit(type="error", kind=kind, message=msg)
        sys.exit(1)


def watch(total: int, base: int, root: Path, stop: threading.Event, start_bytes: int):
    t0, last_b, last_t, rate = time.time(), start_bytes, time.time(), 0.0
    while not stop.wait(1.0):
        done = min(total, base + max(0, dir_bytes(root) - start_bytes)) if total else base + max(0, dir_bytes(root) - start_bytes)
        now = time.time()
        inst = (done - last_b) / max(1e-6, now - last_t)
        rate = inst if rate == 0 else 0.7 * rate + 0.3 * inst
        last_b, last_t = done, now
        emit(type="progress", done=done, total=total, rate=max(0.0, rate), elapsed=now - t0)


def cmd_hf(a):
    from huggingface_hub import constants, snapshot_download
    try:
        _info, files, present = plan_files(a.repo, a.include)
    except Exception as e:  # noqa: BLE001
        kind, msg = classify(e)
        emit(type="error", kind=kind, message=msg)
        sys.exit(1)
    total = sum(s for _, s in files)
    blobs = Path(constants.HF_HUB_CACHE) / ("models--" + a.repo.replace("/", "--")) / "blobs"
    start = dir_bytes(blobs)
    emit(type="start", repo=a.repo, total=total, present=present, files=len(files))
    stop = threading.Event()
    th = threading.Thread(target=watch, args=(total, present, blobs, stop, start), daemon=True)
    th.start()
    try:
        snapshot_download(a.repo, allow_patterns=a.include or None, max_workers=8)
    except Exception as e:  # noqa: BLE001
        stop.set()
        kind, msg = classify(e)
        emit(type="error", kind=kind, message=msg)
        sys.exit(1)
    stop.set()
    emit(type="progress", done=total, total=total, rate=0.0)
    emit(type="done", repo=a.repo)


def cmd_run(a):
    root = Path(os.path.expanduser(a.watch)) if a.watch else None
    start = dir_bytes(root) if root else 0
    emit(type="start", total=a.size, present=0, files=1)
    stop = threading.Event()
    if root:
        threading.Thread(target=watch, args=(a.size, 0, root, stop, start), daemon=True).start()
    argv = [x for x in a.argv if x != "--"]
    def _pdeath():
        import ctypes, signal as _s
        try:
            ctypes.CDLL("libc.so.6").prctl(1, _s.SIGTERM, 0, 0, 0)
        except Exception:  # noqa: BLE001
            pass
    p = subprocess.run(argv, cwd=a.cwd or None, capture_output=True, text=True, preexec_fn=_pdeath)  # noqa: PLW1509
    stop.set()
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-1:] or ["failed"]
        emit(type="error", kind="network" if "connect" in tail[0].lower() else "internal", message=tail[0][:300])
        sys.exit(1)
    emit(type="progress", done=a.size, total=a.size, rate=0.0)
    emit(type="done")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op", required=True)
    for op in ("plan", "hf"):
        s = sub.add_parser(op)
        s.add_argument("--repo", required=True)
        s.add_argument("--include", action="append", default=[])
    s = sub.add_parser("cmd")
    s.add_argument("--watch", default="")
    s.add_argument("--size", type=int, default=0)
    s.add_argument("--cwd", default="")
    s.add_argument("argv", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    {"plan": cmd_plan, "hf": cmd_hf, "cmd": cmd_run}[a.op](a)


if __name__ == "__main__":
    main()
