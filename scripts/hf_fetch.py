#!/usr/bin/env python3
"""Download ONE Hugging Face repo (optionally only some files) into the HF cache. Run by scripts/install.py with
the python of envs/core (it has huggingface_hub). One repo per call, one connection (max_workers=1), no retries.

    python scripts/hf_fetch.py <repo> [--include PATTERN ...]

Exit codes (install.py turns them into messages):
    0   done (or already complete in the cache)
    10  HTTP 401 / 403: the repo is gated or private (accept the terms on its page, then `hf auth login`)
    11  HTTP 429: rate limited (retry later)
    12  HTTP 404: repo or file not found
    13  anything else (network error, 5xx, disk full, ...)
"""
from __future__ import annotations

import argparse
import os
import sys


def _status_of(exc: BaseException) -> int | None:
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code
    text = str(exc)
    for c in (401, 403, 404, 429):
        if f" {c} " in text or f"({c})" in text or f"{c} Client Error" in text:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--include", nargs="*", default=None, help="file patterns (hf download --include)")
    a = ap.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        path = snapshot_download(a.repo, allow_patterns=a.include or None, max_workers=1)
    except GatedRepoError as e:
        print(f"HTTP 401/403: {a.repo} is gated: {e}", file=sys.stderr)
        return 10
    except RepositoryNotFoundError as e:
        code = _status_of(e)
        print(f"HTTP {code or 404}: {a.repo} not found, private or gated: {e}", file=sys.stderr)
        return 10 if code in (401, 403) else 12
    except BaseException as e:  # noqa: BLE001
        code = _status_of(e)
        print(f"{type(e).__name__}{f' (HTTP {code})' if code else ''}: {e}", file=sys.stderr)
        if code in (401, 403):
            return 10
        if code == 429:
            return 11
        if code == 404:
            return 12
        return 13
    print(f"ok: {a.repo} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
