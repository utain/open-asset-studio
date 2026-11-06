"""Job queue: one GPU job at a time, CPU jobs alongside, cancel (queued + running), retry; memory eviction order."""
import textwrap
import threading
import time

import pytest

from app import config
from app.assets import AssetStore
from app.config import Settings
from app.jobs import TERMINAL, JobContext, JobManager, handler
from app.memory import InsufficientMemory, MemoryManager
from app.registry import Registry
from app.workers import WorkerPool

ROOT = config.STUDIO


# ------------------------------------------------------------------ memory policy (pure unit tests)
class FakeMem:
    def __init__(self, avail, total=120.0):
        self.avail, self.total = avail, total

    def __call__(self):
        return self.avail, self.total


def _mm(avail, margin=4.0):
    s = Settings(config.DATA / "mm-settings.json")
    s.update({"memory": {"margin_gb": margin, "llm_yield_gb": 40, "evict_wait_s": 0.01}})
    t = [1000.0]

    def clock():
        t[0] += 0.001
        return t[0]
    mem = FakeMem(avail)
    return MemoryManager(s, meminfo=mem, sleep=lambda _s: None, clock=clock), mem


def test_eviction_order_llm_first_then_lru_never_busy():
    mm, mem = _mm(10)
    order = []

    def ev(key, gb):
        def f():
            order.append(key)
            mem.avail += gb
        return f
    mm.register("w:old", 20, kind="worker", evict=ev("w:old", 20))
    mm.register("w:new", 20, kind="worker", evict=ev("w:new", 20))
    mm.register("w:busy", 30, kind="worker", evict=ev("w:busy", 30))
    mm.register("llm:prompt", 22, kind="llm", evict=ev("llm:prompt", 22))
    mm.touch("w:new")
    mm.set_busy("w:busy", True)
    assert mm.eviction_order() == ["llm:prompt", "w:old", "w:new"]
    evicted = mm.ensure(30, key="w:target", label="target")  # needs 30 + 4 margin, has 10
    assert evicted == ["llm:prompt", "w:old"] == order
    assert "w:busy" in mm.entries and "w:new" in mm.entries
    # resident model: nothing to do
    assert mm.ensure(999, key="w:new") == []


def test_big_job_always_stops_llm_and_insufficient_memory():
    mm, mem = _mm(100)
    stopped = []
    mm.register("llm:prompt", 22, kind="llm", evict=lambda: stopped.append(1))
    assert mm.ensure(56, key="qwen:x") == ["llm:prompt"] and stopped  # >= llm_yield_gb
    mm2, _ = _mm(10)
    mm2.register("w:busy", 30, kind="worker", evict=lambda: None)
    mm2.set_busy("w:busy", True)
    with pytest.raises(InsufficientMemory) as e:
        mm2.ensure(50, key="big")
    assert "50 GB" in str(e.value) and "w:busy" in mm2.entries


# ------------------------------------------------------------------ queue with the real echo worker
@pytest.fixture
def mgr(tmp_path):
    yaml = textwrap.dedent(f"""
    backends:
      echo_gpu: {{python: {ROOT}/envs/core/bin/python, script: {ROOT}/workers/echo_worker.py, device: gpu}}
      echo_cpu: {{python: {ROOT}/envs/core/bin/python, script: {ROOT}/workers/echo_worker.py, device: cpu}}
    models:
      - {{id: test-tone, backend: echo_gpu, tasks: [sfx], memory_gb: 0, weights: [],
          params: [{{name: steps, type: int, min: 1, max: 400, default: 5}}, {{name: step_ms, type: int, default: 20}}]}}
      - {{id: test-swatch, backend: echo_cpu, tasks: [t2i], memory_gb: 0, weights: []}}
    """)
    (tmp_path / "models.yaml").write_text(yaml)
    s = Settings(tmp_path / "settings.json")
    s.update({"workers": {"cancel_grace_s": 5}})
    reg = Registry(tmp_path / "models.yaml", s, probe_cache=tmp_path / "p.json")
    assets = AssetStore(tmp_path / "projects")
    mem = MemoryManager(s)
    pool = WorkerPool(reg, s, mem)
    events = []

    class Bus:
        def publish(self, t, d):
            events.append((t, dict(d) if isinstance(d, dict) else d))
    m = JobManager(reg, assets, pool, mem, Bus(), s, tmp_path / "jobs.jsonl", tmp_path / "scratch")
    m.events = events
    yield m
    m.shutdown()
    pool.stop_all()


def wait(m, jid, timeout=30):
    t = time.time()
    while time.time() - t < timeout:
        if m.get(jid)["status"] in TERMINAL:
            return m.get(jid)
        time.sleep(0.05)
    raise AssertionError(f"job {jid} still {m.get(jid)['status']}")


def test_gpu_jobs_are_serialized_and_cpu_runs_alongside(mgr):
    a = mgr.submit({"model": "test-tone", "task": "sfx", "params": {"steps": 15, "step_ms": 30, "batch": 1}})
    b = mgr.submit({"model": "test-tone", "task": "sfx", "params": {"steps": 5, "step_ms": 10, "batch": 1}})
    c = mgr.submit({"model": "test-swatch", "task": "t2i", "params": {"steps": 3, "step_ms": 10, "size": 32, "batch": 1}})
    assert a["gpu"] and b["gpu"] and not c["gpu"]
    assert mgr.get(b["id"])["status"] == "queued"
    ja, jb, jc = wait(mgr, a["id"]), wait(mgr, b["id"]), wait(mgr, c["id"])
    assert ja["status"] == jb["status"] == jc["status"] == "done", (ja["error"], jb["error"], jc["error"])
    # b started only after a finished
    assert jb["started"] >= ja["finished"]
    assert len(ja["outputs"]) == 1 and len(jc["outputs"]) == 1
    meta = mgr.assets.get(ja["outputs"][0])
    assert meta["job"] == a["id"] and meta["model"] == "test-tone" and meta["type"] == "audio"
    # progress went through the bus
    assert any(t == "job" and d["id"] == a["id"] and d["status"] == "running" for t, d in mgr.events)
    assert (mgr.history_path.read_text().count(a["id"])) >= 2


def test_cancel_running_and_queued_then_retry(mgr):
    long = mgr.submit({"model": "test-tone", "task": "sfx", "params": {"steps": 400, "step_ms": 20, "batch": 3}})
    queued = mgr.submit({"model": "test-tone", "task": "sfx", "params": {"steps": 2}})
    t0 = time.time()
    while mgr.get(long["id"])["progress"].get("stage") != "generating":
        time.sleep(0.05)
        assert time.time() - t0 < 20
    assert mgr.cancel(queued["id"])["status"] == "cancelled"
    mgr.cancel(long["id"])
    j = wait(mgr, long["id"], timeout=10)
    assert j["status"] == "cancelled"
    assert time.time() - t0 < 15  # cooperative cancel, not the 24 s the job would take
    r = mgr.retry(queued["id"])
    assert r["retry_of"] == queued["id"]
    assert wait(mgr, r["id"])["status"] == "done"


def test_failures_are_reported(mgr):
    j = wait(mgr, mgr.submit({"model": "test-tone", "task": "sfx", "params": {"fail": "boom"}})["id"])
    assert j["status"] == "failed" and j["error"]["kind"] == "input" and "boom" in j["error"]["message"]
    with pytest.raises(Exception):
        mgr.submit({"model": "test-tone", "task": "t2i"})  # task not supported


def test_custom_handler(mgr):
    seen = threading.Event()

    @handler("tool:test.noop", gpu=False)
    def _noop(ctx: JobContext):
        ctx.progress(0.5, "half")
        seen.set()
    j = wait(mgr, mgr.submit({"tool": "test.noop"})["id"])
    assert j["status"] == "done" and seen.is_set() and not j["gpu"]
