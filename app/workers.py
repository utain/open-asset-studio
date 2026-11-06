"""Core side of the worker protocol: one WorkerProcess per backend (= per venv), owned by the WorkerPool.

The process is started lazily by the first request, unloads models / exits by itself when idle, and dies with the
core (see procs.py and workers/studio_worker.py). All requests carry an id; events for that id are routed to the
caller, unsolicited ones (unloaded on idle, bye) update the loaded-model table.
"""
from __future__ import annotations

import collections
import contextlib
import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from . import config, procs
from .registry import Registry, _expand, worker_env

TERMINAL = ("result", "error", "cancelled")


class WorkerError(Exception):
    def __init__(self, message: str, kind: str = "internal", fix: str = ""):
        super().__init__(message)
        self.kind = kind
        self.fix = fix


class WorkerCancelled(WorkerError):
    def __init__(self, files=None):
        super().__init__("Cancelled", "cancelled")
        self.files = files or []


class WorkerProcess:
    def __init__(self, backend: dict, settings: config.Settings | None = None,
                 on_loaded: Callable | None = None, on_unloaded: Callable | None = None, log_dir: Path | None = None):
        self.backend = backend
        self.name = backend["name"]
        self.settings = settings
        self.on_loaded = on_loaded or (lambda *a: None)
        self.on_unloaded = on_unloaded or (lambda *a: None)
        self.log_path = Path(log_dir or config.LOGS) / f"worker-{self.name}.log"
        self.proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._waiters: dict[str, queue.Queue] = {}
        self.loaded: dict[str, dict] = {}      # model id -> {since, gpu_gb}
        self.busy: str | None = None           # job id of the running request
        self.stderr_tail: collections.deque = collections.deque(maxlen=40)
        self.exit_reason: str | None = None
        self.started_at: float | None = None
        self.generation = 0

    # ---------------------------------------------------------------- lifecycle
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _cmd(self) -> list[str]:
        py, script = _expand(self.backend.get("python")), _expand(self.backend.get("script"))
        if py is None or script is None:
            raise WorkerError(f"Backend '{self.name}' needs `python` and `script` in models.yaml.", "env")
        if not py.exists():
            raise WorkerError(f"The '{self.name}' environment is missing ({py}).", "env",
                              f"Recreate it from env-lock/{self.backend.get('lock', self.name)}.txt")
        if not script.exists():
            raise WorkerError(f"The '{self.name}' worker is not implemented ({script.name}).", "env")
        s = self.settings
        idle = s.get("workers", "model_idle_s", default=600) if s else 600
        exit_idle = s.get("workers", "exit_idle_s", default=900) if s else 900
        idle = self.backend.get("model_idle_s", idle)
        return [str(py), "-u", str(script), "--model-idle", str(idle), "--exit-idle", str(exit_idle)]

    def start(self):
        with self._lock:
            if self.alive():
                return
            cmd = self._cmd()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists() and self.log_path.stat().st_size > 20 * 2**20:
                self.log_path.replace(self.log_path.with_suffix(".log.1"))
            self.exit_reason = None
            self.loaded.clear()
            self.generation += 1
            # own session: a Ctrl-C in the core's terminal does not hit the worker mid-step; the core stops it
            self.proc = procs.spawn(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, bufsize=1, cwd=str(config.STUDIO), env=worker_env(self.backend),
                                    start_new_session=True)
            self.started_at = time.time()
            p, gen = self.proc, self.generation
            threading.Thread(target=self._read_out, args=(p, gen), daemon=True, name=f"w-{self.name}-out").start()
            threading.Thread(target=self._read_err, args=(p,), daemon=True, name=f"w-{self.name}-err").start()
            self._log(f"=== worker started pid {p.pid}: {' '.join(cmd)}")

    def _log(self, line: str):
        try:
            with self.log_path.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
        except OSError:
            pass

    def _read_err(self, p):
        with contextlib.suppress(Exception), self.log_path.open("a") as f:
            for line in p.stderr:
                line = line.rstrip("\n")
                if not line:
                    continue
                self.stderr_tail.append(line)
                if "it/s]" in line or "s/it]" in line:
                    continue
                f.write(line + "\n")
                f.flush()

    def _read_out(self, p, gen):
        try:
            for line in p.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                self._dispatch(msg)
        finally:
            rc = p.wait()
            self._log(f"=== worker exited code {rc} ({self.exit_reason or 'unexpected'})")
            if gen == self.generation:
                for mid in list(self.loaded):
                    self.loaded.pop(mid, None)
                    self.on_unloaded(self.name, mid, "exit")
                for q in list(self._waiters.values()):
                    q.put({"ev": "_exit", "code": rc})

    def _dispatch(self, msg: dict):
        ev = msg.get("ev")
        if ev == "loaded":
            self.loaded[msg["model"]] = {"since": time.time(), "gpu_gb": msg.get("gpu_gb"), "seconds": msg.get("seconds")}
            self.on_loaded(self.name, msg["model"], msg)
        elif ev == "unloaded":
            self.loaded.pop(msg.get("model"), None)
            self.on_unloaded(self.name, msg.get("model"), msg.get("reason", ""))
        elif ev == "bye":
            self.exit_reason = msg.get("reason")
        q = self._waiters.get(msg.get("id") or "")
        if q is not None:
            q.put(msg)

    def stop(self, wait: float = 10.0):
        with self._lock:
            p = self.proc
            if p is None or p.poll() is not None:
                self.proc = None
                return
            self.exit_reason = "stopped"
            with contextlib.suppress(Exception):
                p.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                p.stdin.flush()
            try:
                p.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                p.kill()
                with contextlib.suppress(Exception):
                    p.wait(timeout=5)
            self.proc = None

    def kill(self):
        p = self.proc
        if p is not None and p.poll() is None:
            self.exit_reason = "killed"
            p.kill()

    # ---------------------------------------------------------------- requests
    def _send(self, msg: dict):
        with self._lock:
            if not self.alive():
                raise BrokenPipeError
            self.proc.stdin.write(json.dumps(msg, default=str) + "\n")
            self.proc.stdin.flush()

    def request(self, op: str, payload: dict | None = None, *, on_event: Callable | None = None,
                cancel: threading.Event | None = None, timeout: float | None = None, req_id: str | None = None) -> dict:
        """Send one request and block until its terminal event. Raises WorkerError / WorkerCancelled."""
        rid = req_id or uuid.uuid4().hex[:12]
        msg = {"op": op, "id": rid, **(payload or {})}
        q: queue.Queue = queue.Queue()
        self._waiters[rid] = q
        cancel_sent_at = None
        grace = float(self.settings.get("workers", "cancel_grace_s", default=20) if self.settings else 20)
        t0 = time.time()
        try:
            for attempt in (0, 1):
                self.start()
                accepted = False
                try:
                    self._send(msg)
                except (BrokenPipeError, OSError, ValueError):
                    if attempt == 0:
                        self.stop(wait=2)
                        continue
                    raise WorkerError(f"The '{self.name}' worker is not reachable.") from None
                if op == "run":
                    self.busy = rid
                while True:
                    try:
                        ev = q.get(timeout=0.5)
                    except queue.Empty:
                        ev = None
                    if cancel is not None and cancel.is_set():
                        if cancel_sent_at is None:
                            cancel_sent_at = time.time()
                            with contextlib.suppress(Exception):
                                self._send({"op": "cancel", "id": rid})
                        elif time.time() - cancel_sent_at > grace and self.backend.get("kill_on_cancel", True):
                            self._log(f"cancel not honoured within {grace:.0f}s: killing the worker")
                            self.kill()
                            cancel_sent_at = float("inf")  # kill once
                    if ev is None:
                        if timeout and time.time() - t0 > timeout:
                            raise WorkerError(f"'{self.name}' did not answer within {timeout:.0f}s", "timeout")
                        continue
                    e = ev.get("ev")
                    if e == "_exit":
                        if cancel_sent_at is not None:
                            raise WorkerCancelled()
                        if not accepted and attempt == 0 and self.exit_reason in ("idle", "stopped", None):
                            break  # exited right before taking the job (idle race): one fresh start
                        tail = " | ".join(list(self.stderr_tail)[-4:])
                        kind = "oom" if "out of memory" in tail.lower() or ev.get("code") in (-9, 137) else "internal"
                        raise WorkerError(f"The '{self.name}' worker exited unexpectedly (code {ev.get('code')}). {tail}"[:900], kind)
                    if e == "accepted":
                        accepted = True
                    if on_event is not None and e not in TERMINAL:
                        on_event(ev)
                    if e == "result":
                        return ev
                    if e == "cancelled":
                        raise WorkerCancelled(ev.get("files"))
                    if e == "error":
                        raise WorkerError(ev.get("message", "unknown error"), ev.get("kind", "internal"))
                    if e in ("status", "probe"):
                        return ev
            raise WorkerError(f"The '{self.name}' worker could not be started.")
        finally:
            self._waiters.pop(rid, None)
            if op == "run":
                self.busy = None

    def run(self, job_id: str, model: dict, task: str, params: dict, inputs: list, out_dir: str, *,
            on_event: Callable | None = None, cancel: threading.Event | None = None) -> dict:
        public = {k: v for k, v in model.items() if not k.startswith("_")}
        return self.request("run", {"model": public, "task": task, "params": params, "inputs": inputs,
                                    "out_dir": out_dir}, on_event=on_event, cancel=cancel, req_id=job_id)

    def unload(self, model_id: str | None = None):
        if not self.alive():
            return
        if model_id is None and not self.busy:
            self.stop()  # exiting is the only way to give back *every* byte (CUDA context, allocator cache)
            return
        self.request("unload", {"model": model_id}, timeout=60)

    def status(self) -> dict:
        alive = self.alive()
        return {"backend": self.name, "running": alive, "pid": self.proc.pid if alive else None,
                "rss_gb": round(procs.rss_gb(self.proc.pid), 2) if alive else 0.0, "busy": self.busy,
                "loaded": {k: dict(v) for k, v in self.loaded.items()}, "last_exit": self.exit_reason,
                "started_at": self.started_at, "log": self.log_path.name}


class WorkerPool:
    def __init__(self, registry: Registry, settings: config.Settings | None = None, memory=None):
        self.registry = registry
        self.settings = settings
        self.memory = memory
        self.workers: dict[str, WorkerProcess] = {}
        self._lock = threading.Lock()

    def get(self, backend_name: str) -> WorkerProcess:
        with self._lock:
            w = self.workers.get(backend_name)
            b = self.registry.backends.get(backend_name)
            if b is None:
                raise WorkerError(f"Unknown backend '{backend_name}'", "env")
            if w is None:
                w = WorkerProcess(b, self.settings, on_loaded=self._loaded, on_unloaded=self._unloaded)
                self.workers[backend_name] = w
            else:
                w.backend = b
            return w

    def _loaded(self, backend, model_id, msg):
        if self.memory is not None:
            gb = self.registry.models.get(model_id, {}).get("memory_gb", 0)
            key = f"{backend}:{model_id}"
            self.memory.register(key, gb, kind="worker", label=self.registry.models.get(model_id, {}).get("name", model_id),
                                 evict=lambda b=backend, m=model_id: self.unload(b, m))
            w = self.workers.get(backend)
            if w is not None and w.busy:
                self.memory.set_busy(key, True)

    def _unloaded(self, backend, model_id, reason):
        if self.memory is not None and model_id:
            self.memory.unregister(f"{backend}:{model_id}")

    def unload(self, backend: str, model_id: str | None = None):
        w = self.workers.get(backend)
        if w is None:
            return
        if model_id and len(w.loaded) <= 1 and not w.busy:
            w.stop()
            for mid in list(w.loaded):
                self._unloaded(backend, mid, "stopped")
            w.loaded.clear()
            if self.memory is not None:
                self.memory.unregister(f"{backend}:{model_id}")
            return
        w.unload(model_id)

    def loaded(self) -> list[dict]:
        out = []
        for w in self.workers.values():
            for mid, info in w.loaded.items():
                out.append({"backend": w.name, "model": mid, **info})
        return out

    def stop_all(self):
        for w in list(self.workers.values()):
            with contextlib.suppress(Exception):
                w.stop(wait=5)
