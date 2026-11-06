"""Model weight downloads for registry models (Models page "Download" button).

- Only weights declared in the registry are downloadable: `{hf: repo, files: [...]}` entries, or `{path: ..., download_cmd:
  [argv], size_bytes: N, watch: dir}` entries. The API takes a model id, never a repo name, URL or command.
- One download at a time (FIFO queue), separate from the GPU job queue. The runner subprocess (app/dl_runner.py) is the
  only process that goes online; the core and workers stay HF_HUB_OFFLINE=1.
- The user must accept the licence (accept_license=true) before a download starts. Disk space is checked first.
- Cancel kills the runner's process group; huggingface_hub keeps *.incomplete blobs, so the next start resumes.
- When a model finishes, the registry re-checks status and a "models" event refreshes every page.
Events: "download" {model, status, done, total, rate, eta, message, kind}.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque

from . import config, procs

log = logging.getLogger("studio.downloads")
ACTIVE = ("queued", "planning", "running")
MARGIN = 2 * 2**30


class DownloadError(Exception):
    def __init__(self, message: str, kind: str = "input", fix: str = ""):
        super().__init__(message)
        self.kind, self.fix = kind, fix


class Downloads:
    def __init__(self, registry, bus):
        self.registry, self.bus = registry, bus
        self.state: dict[str, dict] = {}
        self.queue: deque[str] = deque()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.stopping = False
        threading.Thread(target=self._loop, daemon=True, name="downloads").start()

    # ------------------------------------------------------------ helpers
    def _missing(self, m: dict) -> list[dict]:
        return [w for w in m.get("weights", []) if not self.registry._check_weight(w)[0]]

    @staticmethod
    def _label(w: dict) -> str:
        if "hf" in w:
            return w["hf"] + (f" ({', '.join(w['files'])})" if w.get("files") else "")
        return w.get("label") or os.path.basename(str(w.get("path") or w.get("lmstudio") or "file"))

    def _runner(self, args: list[str]) -> subprocess.Popen:
        env = {**os.environ, "HF_HUB_OFFLINE": "0", "HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1"}
        return procs.spawn([sys.executable, "-m", "app.dl_runner", *args], cwd=str(config.STUDIO), env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, start_new_session=True)

    @staticmethod
    def _hf_args(w: dict) -> list[str]:
        out = ["--repo", w["hf"]]
        for f in w.get("files") or []:
            out += ["--include", f]
        return out

    def _publish(self, mid: str):
        self.bus.publish("download", dict(self.state[mid]))

    # ------------------------------------------------------------ API
    def plan(self, model_id: str) -> dict:
        m = self.registry.get(model_id)
        missing = self._missing(m)
        items, blocked = [], []
        for w in missing:
            if "hf" in w:
                p = self._runner(["plan", *self._hf_args(w)])
                try:
                    out, _ = p.communicate(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    raise DownloadError("Hugging Face did not answer within 60 s.", "network", "Check the internet connection.")
                lines = [json.loads(x) for x in out.splitlines() if x.startswith("{")]
                r = lines[-1] if lines else {"type": "error", "kind": "internal", "message": "no answer from the runner"}
                if r["type"] == "error":
                    fix = f"https://huggingface.co/{w['hf']}" if r["kind"] in ("gated", "auth") else ""
                    raise DownloadError(r["message"], r["kind"], fix)
                items.append({"label": self._label(w), "kind": "hf", "repo": w["hf"], "total": r["total"],
                              "present": r["present"], "files": r["files"], "gated": r["gated"],
                              "url": f"https://huggingface.co/{w['hf']}"})
            elif w.get("download_cmd"):
                items.append({"label": self._label(w), "kind": "cmd", "total": int(w.get("size_bytes") or 0), "present": 0,
                              "files": 1, "gated": False})
            else:
                blocked.append({"label": self._label(w), "fix": self.registry._check_weight(w)[2]})
        total = sum(i["total"] for i in items)
        need = sum(max(0, i["total"] - i["present"]) for i in items)
        free = shutil.disk_usage(self.registry.hf_cache if self.registry.hf_cache.exists() else config.STUDIO).free
        return {"model": model_id, "name": m.get("name", model_id), "items": items, "blocked": blocked,
                "total": total, "need": need, "free": free, "enough_disk": free >= need + MARGIN,
                "license": m.get("license", {}), "note": m.get("download_note", ""),
                "downloadable": bool(items), "state": self.state.get(model_id)}

    def start(self, model_id: str, accept_license: bool) -> dict:
        m = self.registry.get(model_id)
        if not accept_license:
            raise DownloadError("Accept the model licence first.", "input")
        with self.lock:
            cur = self.state.get(model_id)
            if cur and cur["status"] in ACTIVE:
                return cur
            if not self._missing(m):
                raise DownloadError("All weights of this model are already on disk.", "input")
            self.state[model_id] = {"model": model_id, "name": m.get("name", model_id), "status": "queued", "done": 0,
                                    "total": 0, "rate": 0.0, "eta": None, "message": "", "kind": "", "item": "",
                                    "queued": time.time(), "license_accepted": time.time()}
            self.queue.append(model_id)
        log.info("download queued: %s (licence accepted)", model_id)
        self._publish(model_id)
        self.wake.set()
        return self.state[model_id]

    def cancel(self, model_id: str) -> dict:
        with self.lock:
            st = self.state.get(model_id)
            if not st or st["status"] not in ACTIVE:
                raise DownloadError("No active download for this model.", "input")
            st["cancel"] = True
            if model_id in self.queue:
                self.queue.remove(model_id)
                st.update(status="cancelled", message="Removed from the queue.")
            p = self.proc if st["status"] != "cancelled" else None
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self._publish(model_id)
        return st

    def list(self) -> list[dict]:
        with self.lock:
            return [{k: v for k, v in s.items() if k != "cancel"} for s in self.state.values()]

    def shutdown(self):
        self.stopping = True
        p = self.proc
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.wake.set()

    # ------------------------------------------------------------ worker thread
    def _loop(self):
        while not self.stopping:
            self.wake.wait(5)
            self.wake.clear()
            while not self.stopping:
                with self.lock:
                    if not self.queue:
                        break
                    mid = self.queue.popleft()
                try:
                    self._run(mid)
                except Exception as e:  # noqa: BLE001
                    log.exception("download %s failed", mid)
                    self.state[mid].update(status="error", kind="internal", message=str(e))
                    self._publish(mid)

    def _run(self, mid: str):
        st = self.state[mid]
        m = self.registry.get(mid)
        st.update(status="planning", started=time.time())
        self._publish(mid)
        try:
            p = self.plan(mid)
        except DownloadError as e:
            st.update(status="error", kind=e.kind, message=str(e), fix=e.fix)
            self._publish(mid)
            return
        if not p["items"]:
            st.update(status="error", kind="input", message="Nothing here can be downloaded automatically.",
                      fix=p["blocked"][0]["fix"] if p["blocked"] else "")
            self._publish(mid)
            return
        if not p["enough_disk"]:
            st.update(status="error", kind="disk", message=f"Not enough disk: needs {p['need'] / 2**30:.1f} GB, "
                      f"{p['free'] / 2**30:.1f} GB free.", fix="Free some disk space and retry.")
            self._publish(mid)
            return
        grand = max(1, p["total"])
        offset = 0
        auto = [w for w in self._missing(m) if "hf" in w or w.get("download_cmd")]
        for w, item in zip(auto, p["items"]):
            if st.get("cancel"):
                break
            args = ["hf", *self._hf_args(w)] if "hf" in w else \
                ["cmd", "--watch", str(w.get("watch") or os.path.dirname(os.path.expanduser(str(w["path"])))),
                 "--size", str(item["total"]), "--cwd", str(config.STUDIO), "--", *w["download_cmd"]]
            st.update(status="running", item=item["label"], total=p["total"])
            self._publish(mid)
            proc = self._runner(args)
            with self.lock:
                self.proc = proc
            err, last_pub = None, 0.0
            for line in proc.stdout:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev["type"] == "progress":
                    done = offset + ev["done"]
                    rate = ev.get("rate") or 0.0
                    st.update(done=done, rate=rate, eta=(grand - done) / rate if rate > 1 else None)
                    if time.time() - last_pub > 0.9:
                        last_pub = time.time()
                        self._publish(mid)
                elif ev["type"] == "error":
                    err = ev
            proc.wait()
            with self.lock:
                self.proc = None
            if st.get("cancel"):
                break
            if err or proc.returncode != 0:
                err = err or {"kind": "internal", "message": f"runner exited with code {proc.returncode}"}
                fix = item.get("url", "") if err["kind"] in ("gated", "auth") else "Press Download again to resume."
                st.update(status="error", kind=err["kind"], message=err["message"], fix=fix)
                self._publish(mid)
                return
            offset += item["total"]
        if st.get("cancel"):
            st.update(status="cancelled", message="Cancelled. Press Download again to resume where it stopped.", rate=0.0, eta=None)
            self._publish(mid)
            return
        self.registry.refresh()
        status = self.registry.status.get(mid, {})
        ok = status.get("status") == "ready"
        st.update(status="done", done=p["total"], rate=0.0, eta=None, finished=time.time(),
                  message="Ready to use." if ok else f"Downloaded, but the model is not ready yet: {status.get('reason', '')}",
                  fix="" if ok else status.get("fix", ""))
        log.info("download finished: %s -> %s", mid, status.get("status"))
        self._publish(mid)
        self.bus.publish("models", {"refreshed": True, "downloaded": mid})
