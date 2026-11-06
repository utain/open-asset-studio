"""Image studio plugin.

* ComfyUI reachability: the optional `comfyui` backend (FLUX.2 klein via the ComfyUI container) is shown as ready only
  while ComfyUI answers GET /system_stats. A background check runs at startup and every 60 s and records the result
  as a probe on the registry (the documented `probe_failed` status), so pickers grey the model out with the reason.
  Set STUDIO_NO_COMFYUI=1 to never contact ComfyUI (the model then stays unavailable).
* GET /api/x/image/comfyui -> {reachable, url, checked, reason}.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import urllib.error
import urllib.request

from fastapi import APIRouter

log = logging.getLogger("studio.image")
router = APIRouter()
CORE = None
STATE = {"reachable": False, "url": None, "checked": None, "reason": "not checked yet"}
INTERVAL_S = 60


def _comfy_models(reg):
    return [m for m in reg.models.values() if m.get("backend") == "comfyui"]


def _url(reg) -> str:
    b = reg.backends.get("comfyui") or {}
    return str((b.get("env") or {}).get("COMFYUI_URL") or os.environ.get("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")


def check_comfyui(reg, fetch=None) -> bool:
    """Probe ComfyUI once and record the outcome on the registry. Returns True when the status changed."""
    url = _url(reg)
    if os.environ.get("STUDIO_NO_COMFYUI"):
        ok, reason = False, "ComfyUI checks are disabled (STUDIO_NO_COMFYUI)"
    else:
        try:
            (fetch or (lambda u: urllib.request.urlopen(u, timeout=2).read()))(url + "/system_stats")
            ok, reason = True, ""
        except (urllib.error.URLError, OSError, ValueError) as e:
            ok, reason = False, f"ComfyUI is not reachable at {url} ({getattr(e, 'reason', e)})"
    STATE.update(reachable=ok, url=url, checked=time.strftime("%Y-%m-%d %H:%M:%S"), reason=reason)
    b = reg.backends.get("comfyui")
    if not b:
        return False
    changed = False
    with reg._lock:
        stamp = reg._env_stamp(b)
        for m in _comfy_models(reg):
            before = reg.status.get(m["id"], {}).get("status")
            reg.probes[m["id"]] = {"ok": ok, "reason": reason, "stamp": stamp, "at": time.time(),
                                   "fix": "Start the ComfyUI server (COMFYUI_URL), then press Re-check."}
            reg.status[m["id"]] = reg._compute_status(m)
            changed |= before != reg.status[m["id"]]["status"]
    return changed


def _loop():
    while True:
        try:
            if CORE is not None and check_comfyui(CORE.registry):
                CORE.bus.publish("models", {"probed": "comfyui"})
        except Exception:  # noqa: BLE001
            log.exception("ComfyUI check failed")
        time.sleep(INTERVAL_S)


@router.get("/comfyui")
def comfyui_status(recheck: bool = False):
    if recheck and CORE is not None and check_comfyui(CORE.registry):
        CORE.bus.publish("models", {"probed": "comfyui"})
    return dict(STATE)


def setup(core):
    global CORE
    CORE = core
    if os.environ.get("STUDIO_NO_COMFYUI"):
        check_comfyui(core.registry)   # no network: mark it unavailable before the first page load
    threading.Thread(target=_loop, daemon=True, name="comfyui-check").start()
