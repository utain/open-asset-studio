"""Studio worker framework (worker side). Standard library only: it runs inside every backend venv.

A backend is one subclass of `Backend` in `workers/<name>_worker.py` ending with `serve(MyBackend())`. The core
starts it as `<venv python> -u workers/<name>_worker.py` and talks JSON lines over its stdin/stdout
(protocol v1, docs/CONTRACTS.md section 2). The worker never opens a socket.

    core -> worker  {"op":"run","id","model":{registry entry},"task","params","inputs":[{...}],"out_dir"}
                    {"op":"load","id","model":{...}}      {"op":"unload","id","model":"<id>"|null}
                    {"op":"cancel","id"}   {"op":"status","id"}   {"op":"shutdown"}
    worker -> core  {"ev":"hello","pid","backend","protocol":1}
                    {"ev":"accepted","id"}
                    {"ev":"progress","id","step","total","stage","msg","interruptible"}
                    {"ev":"log","id","msg"}
                    {"ev":"file","id","file":{path,type,...}}         an output is finished (partial results)
                    {"ev":"loaded","id","model","seconds","gpu_gb"}  {"ev":"unloaded","model","reason"}
                    {"ev":"result","id","files":[...],"meta":{},"timings":{}}
                    {"ev":"cancelled","id","files":[...]}            {"ev":"error","id","kind","message"}
                    {"ev":"status","id",...}   {"ev":"bye","reason"}

Parent death: PR_SET_PDEATHSIG, stdin EOF and a parent-pid watch each make the worker exit at once, so it can
never hold GPU memory on its own. Everything printed by libraries goes to stderr (fd 1 is re-pointed), which the
core writes to logs/worker-<backend>.log.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

PROTOCOL = 1

# ---------------------------------------------------------------- protocol channel
_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)            # stray prints / C libraries writing to fd 1 can never corrupt the protocol stream
sys.stdout = sys.stderr
_OUT_LOCK = threading.Lock()


def send(**msg):
    with _OUT_LOCK:
        try:
            _OUT.write(json.dumps(msg, default=str, ensure_ascii=False) + "\n")
            _OUT.flush()
        except (BrokenPipeError, ValueError, OSError):
            os._exit(0)  # the core is gone


def log(*a):
    try:
        print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


def _set_pdeathsig():
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG = 1
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- errors
class Cancelled(Exception):
    """Raised by ctx.check() once the core asked to cancel the running job."""


class InputError(ValueError):
    """Bad parameters or inputs: the message is shown to the user as is."""


def classify(e: BaseException) -> str:
    if isinstance(e, Cancelled):
        return "cancelled"
    text = f"{type(e).__name__}: {e}".lower()
    if "out of memory" in text or type(e).__name__ in ("OutOfMemoryError", "MemoryError"):
        return "oom"
    if isinstance(e, (InputError, ValueError, FileNotFoundError, KeyError)):
        return "input"
    return "internal"


# ---------------------------------------------------------------- job + context
class Job:
    def __init__(self, msg: dict):
        self.id: str = msg["id"]
        self.model: dict = msg.get("model") or {}
        self.model_id: str = self.model.get("id", "")
        self.task: str = msg.get("task", "")
        self.params: dict = msg.get("params") or {}
        self.inputs: list[dict] = msg.get("inputs") or []   # [{asset_id, type, path, name, role, meta}]
        self.out_dir = Path(msg.get("out_dir") or ".")

    def input(self, role: str | None = None, type: str | None = None) -> dict | None:  # noqa: A002
        for i in self.inputs:
            if (role is None or i.get("role") == role) and (type is None or i.get("type") == type):
                return i
        return None


class Context:
    """Handed to Backend.load/run: progress, cancel checks, outputs."""

    def __init__(self, worker: "_Worker", job_id: str | None, out_dir: Path | None = None):
        self._w = worker
        self.job_id = job_id
        self.out_dir = out_dir
        self.files: list[dict] = []
        self.t0 = time.time()

    @property
    def cancelled(self) -> bool:
        return self._w.cancel.is_set()

    def check(self):
        if self._w.cancel.is_set():
            raise Cancelled()

    def progress(self, step: float = 0, total: float = 0, stage: str = "", msg: str = "", interruptible: bool = True):
        send(ev="progress", id=self.job_id, step=step, total=total, stage=stage, msg=msg, interruptible=interruptible)

    def log(self, msg: str):
        log(msg)
        send(ev="log", id=self.job_id, msg=str(msg)[:2000])

    def output(self, path, type: str, **meta) -> dict:  # noqa: A002
        """Declare one finished output file (inside out_dir). type: image|audio|mesh|voice|text|other.
        Extra keyword args land in the asset's meta (e.g. seed=, index=, name=, sample_rate=)."""
        p = Path(path)
        if not p.is_absolute():
            p = (self.out_dir or Path(".")) / p
        f = {"path": str(p), "type": type, **meta}
        self.files.append(f)
        send(ev="file", id=self.job_id, file=f)
        return f

    def step_callback(self, total: int, stage: str = "denoising", label: str = ""):
        """A diffusers `callback_on_step_end` that reports progress and raises Cancelled between steps."""
        def cb(_pipe, step, _t, kw):
            if self._w.cancel.is_set():
                raise Cancelled()
            self.progress(step + 1, total, stage, f"{label}step {step + 1}/{total}")
            return kw
        return cb

    @staticmethod
    def gpu_gb() -> float | None:
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_available():
                return round(torch.cuda.memory_allocated() / 2**30, 2)
        except Exception:  # noqa: BLE001
            pass
        return None


# ---------------------------------------------------------------- backend base
class Backend:
    """Subclass per backend venv. Override load/run (and probe / unload when useful).

    name          backend name (matches `backends:` in models.yaml)
    max_loaded    models resident at once in this process (others are unloaded first); 0 = unlimited
    """
    name = "backend"
    max_loaded = 1

    def __init__(self):
        self.models: dict[str, object] = {}     # model id -> whatever load() returned
        self.last_used: dict[str, float] = {}

    # --- override these
    def probe(self, model: dict) -> tuple[bool, str]:
        """Cheap check that the model can run here (imports, files). Runs with `--probe`, not during jobs."""
        return True, ""

    def load(self, model: dict, ctx: Context):
        """Load the model and return the handle (stored in self.models[model['id']])."""
        return True

    def unload(self, model_id: str):
        """Drop the handle; the framework then runs gc + torch.cuda.empty_cache()."""
        self.models.pop(model_id, None)

    def run(self, job: Job, ctx: Context) -> dict | None:
        """Do the work. Call ctx.progress()/ctx.check() often, ctx.output() for each file. Return {"meta": {...}}."""
        raise NotImplementedError

    # --- helpers
    def handle(self, model_id: str):
        return self.models.get(model_id)


ASPECTS = {"1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3), "16:9": (16, 9), "9:16": (9, 16),
           "21:9": (21, 9), "2:1": (2, 1), "1:2": (1, 2)}


def size_from_aspect(aspect: str, long_side, multiple: int = 16) -> tuple[int, int]:
    """Shared image-size convention of the registry: `aspect` ("16:9") + `size` (long side in px) -> (w, h),
    both rounded to `multiple` (16 for most DiTs, 32 for Qwen-Image)."""
    a, b = ASPECTS.get(str(aspect), (1, 1))
    L = int(long_side)
    w, h = (L, L * b / a) if a >= b else (L * a / b, L)
    r = lambda v: max(multiple, int(round(v / multiple)) * multiple)  # noqa: E731
    return r(w), r(h)


def _free_memory():
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- the loop
class _Worker:
    def __init__(self, backend: Backend, model_idle_s: float, exit_idle_s: float):
        self.b = backend
        self.q: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.current: str | None = None
        self.parent = os.getppid()
        self.model_idle_s = model_idle_s
        self.exit_idle_s = exit_idle_s
        self.last_activity = time.time()

    def _reader(self):
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    log("bad command ignored:", line[:200])
                    continue
                op = msg.get("op")
                if op == "cancel":
                    if msg.get("id") in (None, self.current):
                        self.cancel.set()
                elif op == "shutdown":
                    self.cancel.set()
                    self.q.put(msg)
                else:
                    self.q.put(msg)
        finally:
            os._exit(0)  # stdin EOF: the core exited (even SIGKILL closes the pipe)

    def _watch_parent(self):
        while True:
            time.sleep(1.5)
            if os.getppid() != self.parent:
                os._exit(0)

    # --- model management
    def _unload(self, mid: str, reason: str):
        if mid not in self.b.models:
            return
        try:
            self.b.unload(mid)
        except Exception as e:  # noqa: BLE001
            log(f"unload {mid} failed: {e}")
        self.b.models.pop(mid, None)
        self.b.last_used.pop(mid, None)
        _free_memory()
        send(ev="unloaded", model=mid, reason=reason, gpu_gb=Context.gpu_gb())

    def _ensure(self, model: dict, ctx: Context, job_id: str | None):
        mid = model["id"]
        if mid in self.b.models:
            self.b.last_used[mid] = time.time()
            return False
        if self.b.max_loaded:
            while len(self.b.models) >= self.b.max_loaded:
                old = min(self.b.models, key=lambda k: self.b.last_used.get(k, 0))
                self._unload(old, "make-room")
        t = time.time()
        ctx.progress(0, 0, "loading", f"Loading {model.get('name', mid)}", interruptible=False)
        handle = self.b.load(model, ctx)
        self.b.models[mid] = handle if handle is not None else True
        self.b.last_used[mid] = time.time()
        send(ev="loaded", id=job_id, model=mid, seconds=round(time.time() - t, 2), gpu_gb=Context.gpu_gb())
        return True

    def _error(self, job_id, e: BaseException):
        kind = classify(e)
        if kind not in ("cancelled", "input"):
            traceback.print_exc(file=sys.stderr)
        send(ev="error", id=job_id, kind=kind, message=f"{e}" if kind == "input" else f"{type(e).__name__}: {e}"[:2000])
        _free_memory()

    def _run(self, msg: dict):
        job = Job(msg)
        self.current = job.id
        self.cancel.clear()
        send(ev="accepted", id=job.id)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        ctx = Context(self, job.id, job.out_dir)
        timings = {}
        try:
            t = time.time()
            if self._ensure(job.model, ctx, job.id):
                timings["load"] = round(time.time() - t, 2)
            ctx.check()
            t = time.time()
            ret = self.b.run(job, ctx) or {}
            timings["run"] = round(time.time() - t, 2)
            if self.cancel.is_set() and not ret.get("ignore_cancel"):
                send(ev="cancelled", id=job.id, files=ctx.files)
            else:
                send(ev="result", id=job.id, files=ctx.files, meta=ret.get("meta", {}), timings=timings,
                     gpu_gb=Context.gpu_gb())
        except Cancelled:
            _free_memory()
            send(ev="cancelled", id=job.id, files=ctx.files)
        except BaseException as e:  # noqa: BLE001
            self._error(job.id, e)
        finally:
            if job.model_id in self.b.models:
                self.b.last_used[job.model_id] = time.time()
            self.current = None
            self.cancel.clear()

    def _idle_tick(self):
        now = time.time()
        for mid in list(self.b.models):
            if self.model_idle_s and now - self.b.last_used.get(mid, now) > self.model_idle_s:
                log(f"{mid} idle for {self.model_idle_s:.0f}s: unloading")
                self._unload(mid, "idle")
        if not self.b.models and self.exit_idle_s and now - self.last_activity > self.exit_idle_s:
            return True
        return False

    def main(self):
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._watch_parent, daemon=True).start()
        send(ev="hello", pid=os.getpid(), backend=self.b.name, protocol=PROTOCOL)
        reason = "shutdown"
        while True:
            try:
                msg = self.q.get(timeout=1.0)
            except queue.Empty:
                if self._idle_tick():
                    reason = "idle"
                    break
                continue
            op = msg.get("op")
            mid = msg.get("id")
            if op == "shutdown":
                break
            try:
                if op == "run":
                    self._run(msg)
                elif op == "load":
                    ctx = Context(self, mid)
                    try:
                        self._ensure(msg["model"], ctx, mid)
                        send(ev="result", id=mid, files=[], meta={"loaded": msg["model"]["id"]})
                    except BaseException as e:  # noqa: BLE001
                        self._error(mid, e)
                elif op == "unload":
                    targets = [msg["model"]] if msg.get("model") else list(self.b.models)
                    for t in targets:
                        self._unload(t, "request")
                    send(ev="result", id=mid, files=[], meta={"unloaded": targets})
                elif op == "status":
                    send(ev="status", id=mid, loaded=list(self.b.models), busy=self.current is not None,
                         gpu_gb=Context.gpu_gb(), pid=os.getpid())
                elif op == "probe":
                    send(ev="probe", id=mid, probe=_probe_all(self.b, msg.get("models") or []))
                else:
                    send(ev="error", id=mid, kind="input", message=f"unknown op {op!r}")
            finally:
                self.last_activity = time.time()
        for m in list(self.b.models):
            self._unload(m, "shutdown")
        send(ev="bye", reason=reason)
        os._exit(0)


def _probe_all(b: Backend, models: list[dict]) -> dict:
    out = {}
    for m in models:
        try:
            ok, reason = b.probe(m)
            out[m["id"]] = {"ok": bool(ok), "reason": reason or ""}
        except BaseException as e:  # noqa: BLE001
            out[m["id"]] = {"ok": False, "reason": f"{type(e).__name__}: {e}"[:300]}
    return out


def serve(backend: Backend):
    """Entry point of every worker script."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="read registry entries (JSON list) on stdin, print probe results")
    ap.add_argument("--model-idle", type=float, default=float(os.environ.get("STUDIO_MODEL_IDLE_S", 600)))
    ap.add_argument("--exit-idle", type=float, default=float(os.environ.get("STUDIO_EXIT_IDLE_S", 900)))
    a = ap.parse_args()
    if a.probe:
        models = json.loads(sys.stdin.read() or "[]")
        send(probe=_probe_all(backend, models))
        os._exit(0)
    _set_pdeathsig()
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    signal.signal(signal.SIGHUP, lambda *_: os._exit(0))
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C in the core's terminal: the core shuts us down cleanly
    if os.environ.get("STUDIO_MAX_LOADED"):
        backend.max_loaded = int(os.environ["STUDIO_MAX_LOADED"])
    _Worker(backend, a.model_idle, a.exit_idle).main()
