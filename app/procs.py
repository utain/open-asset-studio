"""Child-process spawning that cannot leak processes.

PR_SET_PDEATHSIG fires when the *thread* that forked the child exits, not the process. Children are therefore
forked from one long-lived daemon thread ("spawner"): it lives exactly as long as the core process, so the
signal fires only when the core itself dies (also on SIGKILL). Only our own children are ever signalled.
"""
from __future__ import annotations

import ctypes
import queue
import signal
import subprocess
import threading

_PR_SET_PDEATHSIG = 1
_q: queue.Queue = queue.Queue()
_started = threading.Lock()
_thread: threading.Thread | None = None


def _pdeathsig():
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass


def _loop():
    while True:
        args, kwargs, box, ev = _q.get()
        try:
            box["proc"] = subprocess.Popen(*args, preexec_fn=_pdeathsig, **kwargs)  # noqa: PLW1509
        except BaseException as e:  # noqa: BLE001
            box["error"] = e
        ev.set()


def spawn(*args, **kwargs) -> subprocess.Popen:
    """subprocess.Popen(...) executed on the spawner thread, with PR_SET_PDEATHSIG(SIGTERM)."""
    global _thread
    with _started:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_loop, daemon=True, name="spawner")
            _thread.start()
    box: dict = {}
    ev = threading.Event()
    _q.put((args, kwargs, box, ev))
    ev.wait()
    if "error" in box:
        raise box["error"]
    return box["proc"]


def rss_gb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except (OSError, ValueError):
        pass
    return 0.0
