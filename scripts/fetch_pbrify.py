#!/usr/bin/env python3
"""Download the PBRify_Remix 1.7.2 models (CC0-1.0, Kim2091) used by the seamless PBR materials, verify them, extract the four we use.

    python scripts/fetch_pbrify.py [--dest DIR] [--url URL] [--zip LOCAL.zip] [--keep]

Standard library only (urllib, hashlib, zipfile). The zip is downloaded next to the destination (so the Models page can watch the
folder grow), its sha256 is checked, ONLY the four members below are extracted (by base name, never by their path in the archive),
each is checked against its own sha256, and the zip is deleted. Files that are already present and correct are left alone.
Weights are loaded later with torch.load(..., weights_only=True) only.

Source: https://github.com/Kim2091/PBRify_Remix  (release v1.7.2_ComfyOnly, 72.5 MB; the repository LICENSE is CC0 1.0 Universal)
Exit codes: 0 done / already complete, 1 download or hash failure.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/Kim2091/PBRify_Remix/releases/download/v1.7.2_ComfyOnly/PBRify_Remix_1.7.2_ComfyUI_ONLY.zip"
ZIP_SHA256 = "d9755a1e97cd299eb61f8a9420f52478dc7df5da0fdc9e60890c763eae3a8c95"
FILES = {
    "1x-PBRify_NormalV3.pth": "b0a18270da765f02eaae3c228203bee0677fc28b2854a562515e4aae9c61223b",
    "1x-PBRify_Height.pth": "5b973ecb8bae9d96d14d77b8a8f1d88fb6a8580bcc1bb55d2872811acdc4277d",
    "1x-PBRify_RoughnessV2.pth": "7003c39041af64cdb77d5120bd560a2878538fbb14a07657d8fd12aac5773679",
    "4x-PBRify_UpscalerSPANV4.pth": "d1f1fa1af7a5ab23fcf143e75aac4951e6c52d56bf77d9ccd4d004a4831d0461",
}
DEFAULT_DEST = "~/.cache/pbrify/1.7.2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def missing(dest: Path) -> list[str]:
    return [n for n, d in FILES.items() if not (dest / n).is_file() or sha256(dest / n) != d]


def download(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "open-asset-studio/fetch_pbrify"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def extract(zpath: Path, dest: Path) -> list[str]:
    """Extract the four members (matched by base name) and verify each hash. Returns the names written."""
    done = []
    with zipfile.ZipFile(zpath) as z:
        by_base = {Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}
        for name, digest in FILES.items():
            info = by_base.get(name)
            if info is None:
                raise RuntimeError(f"{name} is not in the archive")
            tmp = dest / (name + ".part")
            h = hashlib.sha256()
            with z.open(info) as src, open(tmp, "wb") as out:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    h.update(chunk)
                    out.write(chunk)
            if h.hexdigest() != digest:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"{name}: sha256 mismatch ({h.hexdigest()[:12]}...)")
            os.replace(tmp, dest / name)
            done.append(name)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--zip", default=None, help="use this local archive instead of downloading (still hash-checked)")
    ap.add_argument("--zip-sha256", default=ZIP_SHA256)
    a = ap.parse_args(argv)
    dest = Path(os.path.expanduser(a.dest))
    dest.mkdir(parents=True, exist_ok=True)
    todo = missing(dest)
    if not todo:
        print(f"ok: the four PBRify files are already in {dest}")
        return 0
    work = dest / ".download"
    work.mkdir(exist_ok=True)
    zpath = Path(a.zip) if a.zip else work / "pbrify.zip"
    try:
        if not a.zip:
            print(f"downloading {a.url}", flush=True)
            download(a.url, zpath)
        got = sha256(zpath)
        if got != a.zip_sha256:
            raise RuntimeError(f"archive sha256 mismatch: expected {a.zip_sha256[:12]}..., got {got[:12]}...")
        names = extract(zpath, dest)
    except Exception as e:  # noqa: BLE001
        print(f"failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"ok: {', '.join(names)} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
