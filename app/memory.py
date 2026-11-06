"""Memory manager for the unified-memory GB10: what we have loaded, what is free, and who to evict.

On the DGX Spark the GPU allocates from the same 121 GB as the CPU, so the truth is /proc/meminfo MemAvailable
(CUDA's free-memory numbers miss the page cache and other tenants). Before a job starts, `ensure()` makes sure
MemAvailable >= need + margin by unloading OUR loaded models: the prompt LLM first (cheap to reload), then the
least recently used worker model. Busy entries and the job's own model are never evicted. Other users'
processes (ComfyUI, other Hunyuan3D engine instances, ...) are never touched: if they hold the memory, the job fails with a clear
message instead.
"""
from __future__ import annotations

import threading
import time
from typing import Callable


def read_meminfo() -> tuple[float, float]:
    """(MemAvailable, MemTotal) in GiB."""
    vals = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                if k in ("MemAvailable", "MemTotal"):
                    vals[k] = int(rest.split()[0]) / 2**20
    except OSError:
        return 0.0, 0.0
    return vals.get("MemAvailable", 0.0), vals.get("MemTotal", 0.0)


class InsufficientMemory(Exception):
    def __init__(self, need: float, available: float, margin: float):
        self.need, self.available, self.margin = need, available, margin
        super().__init__(
            f"Not enough free memory: this model needs about {need:.0f} GB (+{margin:.0f} GB margin) but only "
            f"{available:.0f} GB is available after unloading every studio model. Other programs on this machine "
            f"(ComfyUI, another Hunyuan3D engine instance, other users) hold the rest. Free memory there, or pick a smaller model.")


class MemoryManager:
    KIND_ORDER = {"llm": 0, "worker": 1}

    def __init__(self, settings=None, meminfo: Callable[[], tuple[float, float]] = read_meminfo,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time):
        self.settings = settings
        self.meminfo = meminfo
        self.sleep = sleep
        self.clock = clock
        self.entries: dict[str, dict] = {}
        self._lock = threading.RLock()
        self.events: list[dict] = []   # recent evictions (System page)

    def _s(self, key, default):
        return float(self.settings.get("memory", key, default=default)) if self.settings else default

    # ------------------------------------------------------------ table
    def register(self, key: str, gb: float, *, kind: str = "worker", label: str = "", evict: Callable | None = None):
        with self._lock:
            e = self.entries.get(key) or {"loaded_at": self.clock(), "busy": False}
            e.update(key=key, gb=float(gb or 0), kind=kind, label=label or key, evict=evict, last_used=self.clock())
            self.entries[key] = e

    def unregister(self, key: str):
        with self._lock:
            self.entries.pop(key, None)

    def touch(self, key: str):
        with self._lock:
            if key in self.entries:
                self.entries[key]["last_used"] = self.clock()

    def set_busy(self, key: str, busy: bool):
        with self._lock:
            if key in self.entries:
                self.entries[key]["busy"] = busy
                self.entries[key]["last_used"] = self.clock()

    def set_gb(self, key: str, gb: float):
        with self._lock:
            if key in self.entries:
                self.entries[key]["gb"] = float(gb)

    def snapshot(self) -> dict:
        avail, total = self.meminfo()
        with self._lock:
            loaded = [{k: v for k, v in e.items() if k != "evict"} for e in self.entries.values()]
        loaded.sort(key=lambda e: -e["gb"])
        return {"available_gb": round(avail, 1), "total_gb": round(total, 1),
                "used_gb": round(total - avail, 1), "ours_gb": round(sum(e["gb"] for e in loaded), 1),
                "loaded": loaded, "margin_gb": self._s("margin_gb", 6.0), "recent_evictions": self.events[-10:]}

    # ------------------------------------------------------------ policy
    def eviction_order(self, exclude: set[str] | frozenset = frozenset()) -> list[str]:
        """LLM first, then least recently used. Busy and excluded entries are not candidates."""
        with self._lock:
            cands = [e for e in self.entries.values() if e["key"] not in exclude and not e["busy"] and e.get("evict")]
            cands.sort(key=lambda e: (self.KIND_ORDER.get(e["kind"], 1), e["last_used"]))
            return [e["key"] for e in cands]

    def evict(self, key: str, reason: str = "") -> bool:
        with self._lock:
            e = self.entries.get(key)
        if e is None or e.get("evict") is None:
            return False
        try:
            e["evict"]()
        finally:
            self.unregister(key)
            self.events.append({"key": key, "label": e["label"], "gb": e["gb"], "reason": reason,
                                "at": time.strftime("%H:%M:%S")})
            del self.events[:-50]
        return True

    def ensure(self, need_gb: float, key: str | None = None, label: str = "", on_status: Callable | None = None) -> list[str]:
        """Make room for a model that needs `need_gb` (0 = nothing to check). Returns the evicted keys."""
        evicted: list[str] = []
        if key and key in self.entries:
            self.touch(key)
            return evicted   # already resident: its memory is already taken
        if not need_gb:
            return evicted
        margin = self._s("margin_gb", 6.0)
        wait_s = self._s("evict_wait_s", 12.0)
        exclude = {key} if key else set()
        if need_gb >= self._s("llm_yield_gb", 40.0):
            for k in [k for k in self.eviction_order(exclude) if self.entries.get(k, {}).get("kind") == "llm"]:
                if on_status:
                    on_status(f"Stopping the prompt LLM to make room for {label or key}")
                self.evict(k, f"big job ({label or key})")
                evicted.append(k)
        avail, _ = self.meminfo()
        while avail < need_gb + margin:
            order = self.eviction_order(exclude)
            if not order:
                raise InsufficientMemory(need_gb, avail, margin)
            victim = order[0]
            if on_status:
                on_status(f"Unloading {self.entries.get(victim, {}).get('label', victim)} to free memory")
            self.evict(victim, f"needed {need_gb:.0f} GB for {label or key}")
            evicted.append(victim)
            deadline = self.clock() + wait_s
            while True:
                avail, _ = self.meminfo()
                if avail >= need_gb + margin or self.clock() >= deadline:
                    break
                self.sleep(0.5)
        return evicted
