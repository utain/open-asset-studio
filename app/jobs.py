"""Job system: one global queue, ONE GPU job at a time, CPU jobs alongside, cooperative cancel, retry, history.

A job is executed by a *handler* (see `handler()` below). The default handler for model jobs ("worker") sends the
job to the model's backend worker and turns every output file into an asset. Studios register extra handlers
(CPU tools, multi-step pipelines) in app/plugins/*.py -- docs/CONTRACTS.md section 3.

Job lifecycle: queued -> running -> done | failed | cancelled. Every change is published on the event bus
("job" events, streamed to the browser via SSE) and terminal states are appended to data/jobs.jsonl.
"""
from __future__ import annotations

import collections
import json
import secrets
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from . import licensing
from .assets import AssetError, AssetStore
from .config import SettingsView
from .memory import InsufficientMemory, MemoryManager
from .registry import Registry
from .workers import WorkerCancelled, WorkerError, WorkerPool

TERMINAL = ("done", "failed", "cancelled")
HANDLERS: dict[str, dict] = {}


def handler(name: str, *, gpu: bool | None = None, label: str = "", precheck: Callable | None = None,
            pool: str | None = None):
    """Register a job handler: `@handler("tool:image.resize", gpu=False)` / `@handler("sound.sheet", gpu=True)`.

    The function receives a JobContext. gpu=None means "GPU if the job's model runs on the GPU".
    precheck(spec, model, params, inputs) runs in submit() before the job exists: it may raise JobError to refuse
    (e.g. kind "confirm" / "budget") or return a dict of extra job fields (e.g. {"cost": {...}}).
    pool names the executor of non-GPU jobs ("cpu" default; "cloud" = network-bound jobs, jobs.cloud_concurrency)."""
    def deco(fn: Callable[["JobContext"], None]):
        HANDLERS[name] = {"fn": fn, "gpu": gpu, "label": label or name, "precheck": precheck, "pool": pool or "cpu"}
        return fn
    return deco


class JobError(Exception):
    def __init__(self, message: str, kind: str = "internal", fix: str = "", data: dict | None = None):
        super().__init__(message)
        self.kind, self.fix = kind, fix
        self.data = data or {}   # extra JSON fields for the API error body (e.g. a cost estimate)


class JobCancelled(Exception):
    pass


ERROR_FIX = {
    "oom": "Out of memory: use a smaller size / fewer takes, unload other models (Queue & System), or pick a lighter model.",
    "memory": "Free memory used by other programs, or pick a smaller model.",
    "env": "See the Models page for the fix command.",
    "input": "",
    "timeout": "The worker stopped answering; it was restarted. Retry the job.",
}


class JobContext:
    """What a handler gets. Wraps one job with progress / cancel / worker / asset helpers."""

    def __init__(self, mgr: "JobManager", job: dict):
        self.mgr = mgr
        self.job = job
        self.cancel_event = threading.Event()
        self.registry: Registry = mgr.registry
        self.assets: AssetStore = mgr.assets
        # the effective settings the job was submitted with (system + its project's overrides); later edits do not
        # change a job that is already queued or running
        self.settings = job.get("_settings") or mgr.settings
        self.scratch = mgr.scratch_root / job["id"]
        self._outputs_lock = threading.Lock()

    # ---- state
    @property
    def params(self) -> dict:
        return self.job["params"]

    @property
    def model(self) -> dict | None:
        return self.registry.models.get(self.job.get("model") or "")

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise JobCancelled()

    def progress(self, frac: float | None = None, msg: str | None = None, *, stage: str | None = None,
                 step: float | None = None, total: float | None = None, interruptible: bool | None = None):
        p = self.job["progress"]
        if stage is not None:
            p["stage"] = stage
        if msg is not None:
            p["msg"] = msg
        if step is not None:
            p["step"] = step
        if total is not None:
            p["total"] = total
        if interruptible is not None:
            p["interruptible"] = interruptible
        if frac is None and step is not None and total:
            frac = step / total
        if frac is not None:
            p["frac"] = round(max(0.0, min(1.0, float(frac))), 4)
        self.mgr._changed(self.job, progress=True)

    def log(self, msg: str):
        lines = self.job.setdefault("log", [])
        lines.append(f"{time.strftime('%H:%M:%S')} {msg}")
        del lines[:-200]
        self.mgr._changed(self.job, progress=True)

    def input_assets(self) -> list[dict]:
        """Inputs resolved to {asset_id, role, type, name, path, meta} (paths only ever produced server-side)."""
        out = []
        for inp in self.job.get("inputs") or []:
            m = self.assets.get(inp["asset"])
            p = self.assets.file_path(m["project"], m["id"], m["main"])
            out.append({"asset_id": m["id"], "role": inp.get("role", "input"), "type": m["type"], "name": m.get("name"),
                        "path": str(p), "meta": {k: m.get(k) for k in ("info", "prompt", "final_prompt", "model", "x_voice")}})
        return out

    # ---- worker
    def run_worker(self, model_id: str | None = None, params: dict | None = None, inputs: list[dict] | None = None,
                   task: str | None = None, frac_range: tuple[float, float] = (0.0, 1.0), out_dir: Path | None = None) -> dict:
        """Run one request on the model's backend worker: memory check + eviction, progress mapping, cancel.
        Returns the worker's result event ({files, meta, timings}); raises JobError / JobCancelled."""
        mid = model_id or self.job["model"]
        m = self.registry.get(mid)
        if model_id and model_id != self.job.get("model"):
            licensing.check(m, self.settings)        # chained sub-models obey the same licence filter as the job's model
        backend = m["backend"]
        key = f"{backend}:{mid}"
        lo, hi = frac_range
        self.check_cancel()
        try:
            self.progress(lo, "Checking memory", stage="memory")
            self.mgr.memory.ensure(float(m.get("memory_gb") or 0), key=key, label=m["name"],
                                   on_status=lambda s: self.progress(None, s, stage="memory"))
        except InsufficientMemory as e:
            raise JobError(str(e), "memory", ERROR_FIX["memory"]) from None
        w = self.mgr.pool.get(backend)
        od = Path(out_dir or self.scratch / "out")
        od.mkdir(parents=True, exist_ok=True)

        def on_event(ev):
            e = ev.get("ev")
            if e == "progress":
                total = ev.get("total") or 0
                step = ev.get("step") or 0
                rel = (step / total) if total else None
                if ev.get("stage") == "loading":
                    rel = None
                self.progress(lo + (hi - lo) * rel if rel is not None else None, ev.get("msg") or ev.get("stage"),
                              stage=ev.get("stage"), step=step, total=total, interruptible=ev.get("interruptible", True))
            elif e == "log":
                self.log(ev.get("msg", ""))
            elif e == "loaded":
                self.log(f"{ev.get('model')} loaded in {ev.get('seconds')} s")
            elif e == "file":
                self.job.setdefault("partial_files", []).append(ev.get("file"))
        self.mgr.memory.set_busy(key, True)
        try:
            res = w.run(self.job["id"] if model_id is None else f"{self.job['id']}-{secrets.token_hex(2)}", m,
                        task or self.job["task"], params if params is not None else self.params,
                        inputs if inputs is not None else self.input_assets(), str(od),
                        on_event=on_event, cancel=self.cancel_event)
        except WorkerCancelled as e:
            self._partial = e.files
            raise JobCancelled() from None
        except WorkerError as e:
            raise JobError(str(e), e.kind, e.fix or ERROR_FIX.get(e.kind, "")) from None
        finally:
            self.mgr.memory.set_busy(key, False)
        self.mgr.memory.touch(key)
        self.job.setdefault("timings", {}).update({f"worker_{k}": v for k, v in (res.get("timings") or {}).items()})
        return res

    # ---- outputs
    def asset_meta(self, file_meta: dict | None = None, index: int = 0) -> dict:
        """Standard meta for an output of this job (DESIGN 4.4); file_meta = what the worker said about the file."""
        j, fm = self.job, dict(file_meta or {})
        m = self.model or {}
        p = dict(j["params"])
        jm = j.get("meta") or {}
        seed = fm.pop("seed", p.get("seed"))
        prompt = jm.get("prompt") or p.get("prompt")
        meta = {"name": fm.pop("name", None) or j.get("title") or (str(prompt)[:60] if prompt else None) or m.get("name"),
                "task": j["task"], "model": j.get("model"), "model_name": m.get("name"), "params": p,
                "seed": seed, "prompt": prompt, "improved_prompt": jm.get("improved_prompt"),
                "final_prompt": fm.pop("final_prompt", None) or p.get("prompt"),
                "negative": p.get("negative") or p.get("negative_prompt"),
                "parents": [i["asset"] for i in j.get("inputs") or []] + list(jm.get("parents") or []),
                "job": j["id"], "license": m.get("license"), "source": "generated" if m else "tool",
                "batch": {"id": j["id"], "index": fm.pop("index", index)}, "studio": jm.get("studio"),
                "tags": list(jm.get("tags") or [])}
        for k in ("path", "type", "attach"):
            fm.pop(k, None)
        if fm:
            meta["worker"] = fm
        for k, v in (jm.get("asset_extra") or {}).items():
            if k.startswith("x_"):
                meta[k] = v
        return meta

    def add_output(self, path, type: str, meta: dict | None = None, attach: list | None = None) -> dict:  # noqa: A002
        with self._outputs_lock:
            a = self.assets.ingest(self.job["project"], Path(path), type, meta if meta is not None else self.asset_meta(),
                                   attach=[Path(x) for x in attach or []])
            self.job["outputs"].append(a["id"])
            self.mgr._changed(self.job)
            return a

    def ingest_worker_files(self, files: list[dict]) -> list[dict]:
        out = []
        for i, f in enumerate(files or []):
            t = f.get("type") or "other"
            if t == "audio" and self.job["task"] in ("voice_design", "voice_clone") and f.get("as_voice"):
                t = "voice"
            att = [str(Path(f["path"]).parent / a) if not str(a).startswith("/") else a for a in f.get("attach") or []]
            out.append(self.add_output(f["path"], t, self.asset_meta(f, i), attach=att))
        return out


# ------------------------------------------------------------------ default handlers
@handler("worker")
def _worker_handler(ctx: JobContext):
    """Default for model jobs: run on the backend worker, ingest every output file."""
    try:
        res = ctx.run_worker()
    except JobCancelled:
        ctx.ingest_worker_files(getattr(ctx, "_partial", None) or [])
        raise
    ctx.progress(0.99, "Saving outputs", stage="saving")
    ctx.ingest_worker_files(res.get("files") or [])
    if res.get("meta"):
        ctx.job["result_meta"] = res["meta"]


class JobManager:
    def __init__(self, registry: Registry, assets: AssetStore, pool: WorkerPool, memory: MemoryManager, bus,
                 settings, history_path: Path, scratch_root: Path):
        self.registry, self.assets, self.pool, self.memory, self.bus = registry, assets, pool, memory, bus
        self.settings = settings
        self.history_path = Path(history_path)
        self.scratch_root = Path(scratch_root)
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = collections.OrderedDict()
        self.ctx: dict[str, JobContext] = {}
        self.gpu_queue: collections.deque[str] = collections.deque()
        self.gpu_current: str | None = None
        self._cv = threading.Condition()
        self._last_pub: dict[str, float] = {}
        self._stop = False
        n = int(settings.get("jobs", "cpu_concurrency", default=3)) if settings else 3
        self.cpu_pool = ThreadPoolExecutor(max_workers=max(1, n), thread_name_prefix="cpu-job")
        self.pools: dict[str, ThreadPoolExecutor] = {"cpu": self.cpu_pool}   # named executors of non-GPU jobs
        self._load_history()
        self.gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True, name="gpu-jobs")
        self.gpu_thread.start()

    # ------------------------------------------------------------ history
    def _load_history(self):
        if not self.history_path.exists():
            return
        recs: dict[str, dict] = {}
        try:
            for line in self.history_path.read_text().splitlines():
                try:
                    j = json.loads(line)
                    recs[j["id"]] = j
                except (ValueError, KeyError):
                    continue
        except OSError:
            return
        for j in list(recs.values())[-int(self.settings.get("jobs", "history_keep", default=2000) if self.settings else 2000):]:
            if j.get("status") not in TERMINAL:
                j["status"] = "failed"
                j["error"] = {"message": "Interrupted: the studio was restarted while this job was queued or running.",
                              "kind": "interrupted", "fix": "Retry it."}
            self.jobs[j["id"]] = j
        if len(recs) > 3 * len(self.jobs) + 100:  # compact
            tmp = self.history_path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(j, default=str, ensure_ascii=False) + "\n" for j in self.jobs.values()))
            tmp.replace(self.history_path)

    def _persist(self, job: dict):
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {k: v for k, v in job.items() if k not in ("partial_files",) and not k.startswith("_")}
            with self.history_path.open("a") as f:
                f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------ events
    def public(self, job: dict) -> dict:
        out = {k: v for k, v in job.items() if k not in ("partial_files",) and not k.startswith("_")}
        out["log"] = (job.get("log") or [])[-30:]
        if job["status"] == "queued" and job["id"] in self.gpu_queue:
            out["queue_position"] = list(self.gpu_queue).index(job["id"]) + 1
        return out

    def _changed(self, job: dict, progress: bool = False):
        now = time.time()
        if progress and now - self._last_pub.get(job["id"], 0) < 0.2:
            return
        self._last_pub[job["id"]] = now
        if self.bus:
            self.bus.publish("job", self.public(job))

    def _set(self, job: dict, status: str, **kw):
        job["status"] = status
        job.update(kw)
        t = job.setdefault("timings", {})
        if status == "running":
            job["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            t["queued_s"] = round(time.time() - job["_t_created"], 2)
            job["_t_started"] = time.time()
        if status in TERMINAL:
            job["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            if job.get("_t_started"):
                t["run_s"] = round(time.time() - job["_t_started"], 2)
            self._persist(job)
            self._last_pub.pop(job["id"], None)
        self._changed(job)
        if status in TERMINAL and self.bus:  # queue positions of the others changed
            for jid in list(self.gpu_queue):
                if jid in self.jobs:
                    self._changed(self.jobs[jid])

    # ------------------------------------------------------------ submit
    def submit(self, spec: dict) -> dict:
        """spec: {task, model?, tool?, params, inputs:[{asset, role}], project, title?, meta?}. Raises JobError(input)."""
        tool = spec.get("tool")
        model_id = spec.get("model")
        task = spec.get("task") or (f"tool:{tool}" if tool else None)
        params = dict(spec.get("params") or {})
        project = spec.get("project") or "default"
        try:
            self.assets.project_dir(project)
        except AssetError as e:
            raise JobError(str(e), "input") from None
        view = self._settings_for(project)       # system settings + this project's overrides, frozen for this job
        if tool:
            hname = f"tool:{tool}"
            if hname not in HANDLERS:
                raise JobError(f"Unknown tool '{tool}'.", "input")
            h = HANDLERS[hname]
            gpu = bool(h["gpu"])
            if model_id and model_id not in self.registry.models:
                raise JobError(f"Unknown model '{model_id}'.", "input")
        else:
            if not model_id:
                model_id = self.registry.default_for(task, settings=view) if task else None
            if not model_id or model_id not in self.registry.models:
                raise JobError(f"Unknown model '{model_id}'.", "input")
            m = self.registry.get(model_id)
            if task is None:
                task = m["tasks"][0]
            if task not in m["tasks"]:
                raise JobError(f"{m['name']} cannot do '{task}' (it supports {', '.join(m['tasks'])}).", "input")
            licensing.check(m, view)        # "Commercial-safe models only" (settings license.commercial_only)
            st = self.registry.status.get(model_id, {})
            if st.get("status") != "ready" and not spec.get("force"):
                raise JobError(f"{m['name']} is not ready: {st.get('reason')}", st.get("kind") or "env", st.get("fix", ""))
            try:
                params = self.registry.validate_params(model_id, params)
            except ValueError as e:
                raise JobError(str(e), "input") from None
            hname = (spec.get("handler") or m.get("handler")
                     or self.registry.backends.get(m.get("backend", ""), {}).get("handler") or "worker")
            if hname not in HANDLERS:
                raise JobError(f"Unknown handler '{hname}'.", "input")
            g = HANDLERS[hname]["gpu"]
            gpu = self.registry.is_gpu(model_id) if g is None else bool(g)
        inputs = []
        for inp in spec.get("inputs") or []:
            aid = inp.get("asset") if isinstance(inp, dict) else inp
            try:
                self.assets.get(aid)
            except AssetError:
                raise JobError(f"Input asset {aid} not found.", "input") from None
            inputs.append({"asset": aid, "role": (inp.get("role") if isinstance(inp, dict) else None) or "input"})
        extra: dict = {}
        pre = HANDLERS[hname].get("precheck")
        if pre:
            extra = pre(spec, self.registry.models.get(model_id) if model_id else None, params, inputs) or {}
        now = time.time()
        jid = "j" + time.strftime("%Y%m%d%H%M%S") + secrets.token_hex(2)
        job = {"id": jid, "task": task, "model": model_id, "tool": tool, "handler": hname, "gpu": gpu,
               "params": params, "inputs": inputs, "project": project, "title": spec.get("title") or "",
               "meta": dict(spec.get("meta") or {}), "status": "queued", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "progress": {"frac": 0.0, "stage": "queued", "msg": "Waiting", "step": 0, "total": 0, "interruptible": True},
               "outputs": [], "log": [], "error": None, "timings": {}, "retry_of": spec.get("retry_of"), "_t_created": now,
               "_settings": view}
        job.update({k: v for k, v in extra.items() if k not in job})
        if model_id and model_id in self.registry.models:
            job["model_name"] = self.registry.models[model_id]["name"]
        ctx = JobContext(self, job)
        with self._cv:
            self.jobs[jid] = job
            self.ctx[jid] = ctx
            if gpu:
                self.gpu_queue.append(jid)
                self._cv.notify_all()
        self._persist(job)
        self._changed(job)
        if not gpu:
            self._pool(HANDLERS[hname].get("pool") or "cpu").submit(self._execute, jid)
        return self.public(job)

    def _settings_for(self, project: str | None):
        """Effective settings of a project: a read-only SettingsView (system + the project's overrides)."""
        if not self.settings or not project:
            return self.settings
        try:
            return SettingsView(self.settings, self.assets.project_settings(project), project)
        except AssetError:
            return self.settings

    def _pool(self, name: str) -> ThreadPoolExecutor:
        """Named executor for non-GPU jobs; created on first use with jobs.<name>_concurrency workers (default 4)."""
        with self._cv:
            if name not in self.pools:
                n = int(self.settings.get("jobs", f"{name}_concurrency", default=4)) if self.settings else 4
                self.pools[name] = ThreadPoolExecutor(max_workers=max(1, n), thread_name_prefix=f"{name}-job")
            return self.pools[name]

    # ------------------------------------------------------------ run
    def _gpu_loop(self):
        while not self._stop:
            with self._cv:
                while not self.gpu_queue and not self._stop:
                    self._cv.wait(timeout=1.0)
                if self._stop:
                    return
                jid = self.gpu_queue.popleft()
                self.gpu_current = jid
            try:
                self._execute(jid)
            finally:
                self.gpu_current = None

    def _execute(self, jid: str):
        job = self.jobs.get(jid)
        ctx = self.ctx.get(jid)
        if job is None or ctx is None or job["status"] != "queued":
            return
        if ctx.cancel_event.is_set():
            self._set(job, "cancelled")
            return
        self._set(job, "running")
        ctx.progress(0.0, "Starting", stage="starting")
        try:
            HANDLERS[job["handler"]]["fn"](ctx)
            ctx.check_cancel()
            job["progress"].update(frac=1.0, stage="done", msg=f"{len(job['outputs'])} output(s)")
            self._set(job, "done")
        except JobCancelled:
            job["progress"].update(stage="cancelled", msg="Cancelled")
            self._set(job, "cancelled")
        except JobError as e:
            self._set(job, "failed", error={"message": str(e), "kind": e.kind, "fix": e.fix})
        except AssetError as e:
            self._set(job, "failed", error={"message": str(e), "kind": "input", "fix": ""})
        except Exception as e:  # noqa: BLE001
            ctx.log(traceback.format_exc()[-1500:])
            self._set(job, "failed", error={"message": f"{type(e).__name__}: {e}", "kind": "internal",
                                            "fix": "See the job log and logs/studio.log."})
        finally:
            shutil.rmtree(ctx.scratch, ignore_errors=True)
            self.ctx.pop(jid, None)

    # ------------------------------------------------------------ control
    def get(self, jid: str) -> dict:
        j = self.jobs.get(jid)
        if j is None:
            raise KeyError(jid)
        return j

    def cancel(self, jid: str) -> dict:
        job = self.get(jid)
        ctx = self.ctx.get(jid)
        if job["status"] in TERMINAL:
            return self.public(job)
        with self._cv:
            if job["status"] == "queued":
                if jid in self.gpu_queue:
                    self.gpu_queue.remove(jid)
                if ctx:
                    ctx.cancel_event.set()
                job["progress"].update(stage="cancelled", msg="Cancelled before it started")
                self._set(job, "cancelled")
                self.ctx.pop(jid, None)
                return self.public(job)
        if ctx:
            ctx.cancel_event.set()
            if not job["progress"].get("interruptible", True):
                ctx.progress(None, "Cancelling: the current stage cannot be interrupted, stopping right after it")
            else:
                ctx.progress(None, "Cancelling...")
        return self.public(job)

    def retry(self, jid: str, confirm_cost: bool = False, confirm_upload: bool = False) -> dict:
        old = self.get(jid)
        spec = {k: old.get(k) for k in ("task", "model", "tool", "params", "inputs", "project", "title", "meta")}
        if old.get("tool"):
            spec["task"] = None
        spec["retry_of"] = jid
        if (old.get("cost") or {}).get("confirmed") or confirm_cost:   # the user already confirmed this cost once
            spec["confirm_cost"] = True
        spec["confirm_upload"] = confirm_upload
        return self.submit(spec)

    def list(self, status: str | None = None, limit: int = 200, project: str | None = None) -> list[dict]:
        items = list(self.jobs.values())
        if status:
            want = set(status.split(","))
            items = [j for j in items if j["status"] in want]
        if project:
            items = [j for j in items if j.get("project") == project]
        items = items[-limit:][::-1]
        return [self.public(j) for j in items]

    def counts(self) -> dict:
        c = collections.Counter(j["status"] for j in self.jobs.values())
        return {"queued": c.get("queued", 0), "running": c.get("running", 0), "gpu_current": self.gpu_current,
                "gpu_queue": len(self.gpu_queue)}

    def clear_finished(self):
        with self._cv:
            for jid in [j for j, v in self.jobs.items() if v["status"] in TERMINAL]:
                self.jobs.pop(jid, None)
        try:
            self.history_path.write_text("")
        except OSError:
            pass

    def shutdown(self):
        self._stop = True
        for ctx in list(self.ctx.values()):
            ctx.cancel_event.set()
        with self._cv:
            self._cv.notify_all()
        for p in self.pools.values():
            p.shutdown(wait=False, cancel_futures=True)
