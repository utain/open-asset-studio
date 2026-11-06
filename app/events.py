"""In-process event bus -> Server-Sent Events.

publish(type, data) may be called from any thread; every SSE client gets its own bounded asyncio queue (a slow
client drops events instead of blocking the job threads). Event types: job, asset, models, memory, llm, toast.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time


class EventBus:
    def __init__(self, maxsize: int = 500):
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue, dict]] = []
        self._lock = threading.Lock()
        self.maxsize = maxsize
        self.seq = 0

    def subscribe(self, filters: dict | None = None) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(self.maxsize)
        with self._lock:
            self._subs.append((loop, q, filters or {}))
        return q

    def unsubscribe(self, q: asyncio.Queue):
        with self._lock:
            self._subs = [s for s in self._subs if s[1] is not q]

    @staticmethod
    def _put(q: asyncio.Queue, item):
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def publish(self, type_: str, data: dict):
        with self._lock:
            self.seq += 1
            item = {"type": type_, "data": data, "seq": self.seq, "t": time.time()}
            subs = list(self._subs)
        for loop, q, f in subs:
            if f.get("types") and type_ not in f["types"]:
                continue
            if f.get("job") and type_ == "job" and data.get("id") != f["job"]:
                continue
            try:
                loop.call_soon_threadsafe(self._put, q, item)
            except RuntimeError:  # loop closed
                self.unsubscribe(q)

    @staticmethod
    def format(item: dict) -> str:
        return f"id: {item['seq']}\nevent: {item['type']}\ndata: {json.dumps(item['data'], default=str, ensure_ascii=False)}\n\n"
