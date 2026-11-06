"""Reference backend: CPU-only test models (no ML, standard library only).

  test-tone    task sfx/tone : writes a WAV (sine / square / saw / noise burst with an envelope)
  test-swatch  task t2i      : writes a PNG gradient swatch (seeded colours)

It exists to exercise the whole path (queue -> worker -> progress -> SSE -> assets -> library) and is the
template to copy for a real backend. `steps` x `step_ms` simulates work so progress and cancel can be seen.
"""
from __future__ import annotations

import math
import random
import struct
import time
import wave
import zlib

from studio_worker import Backend, InputError, serve, size_from_aspect


def _png(path, w: int, h: int, rows: list[bytes]):
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + r for r in rows)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


class EchoBackend(Backend):
    name = "echo"
    max_loaded = 2

    def probe(self, model):
        return True, ""

    def load(self, model, ctx):
        delay = float(model.get("features", {}).get("load_delay_s", 0.3))
        for i in range(5):
            ctx.progress(i + 1, 5, "loading", f"Warming up {model['name']}", interruptible=False)
            time.sleep(delay / 5)
        return {"loaded_at": time.time()}

    def _work(self, ctx, steps: int, step_ms: int, label: str):
        for s in range(steps):
            ctx.check()
            time.sleep(step_ms / 1000)
            ctx.progress(s + 1, steps, "generating", f"{label}: step {s + 1}/{steps}")

    def run(self, job, ctx):
        p = job.params
        n = max(1, min(8, int(p.get("batch", 1))))
        steps = max(1, min(200, int(p.get("steps", 10))))
        step_ms = max(0, min(2000, int(p.get("step_ms", 60))))
        seed0 = int(p.get("seed", -1))
        if seed0 < 0:
            seed0 = random.randrange(2**31)
        if p.get("fail"):
            raise InputError(f"Requested failure: {p['fail']}")
        for i in range(n):
            seed = seed0 + i
            rnd = random.Random(seed)
            label = f"take {i + 1}/{n}" if n > 1 else "generating"
            self._work(ctx, steps, step_ms, label)
            if job.model_id == "test-swatch":
                w, h = size_from_aspect(p.get("aspect", "1:1"), p.get("size", 384))
                if not (16 <= w <= 2048 and 16 <= h <= 2048):
                    raise InputError("width and height must be 16..2048")
                c1 = [rnd.randrange(256) for _ in range(3)]
                c2 = [rnd.randrange(256) for _ in range(3)]
                alpha = bool(p.get("transparent", False))
                rows = []
                for y in range(h):
                    row = bytearray()
                    for x in range(w):
                        t = (x + y) / (w + h)
                        px = [int(c1[k] * (1 - t) + c2[k] * t) for k in range(3)]
                        a = 255
                        if alpha:
                            dx, dy = (x - w / 2) / (w / 2), (y - h / 2) / (h / 2)
                            a = 255 if dx * dx + dy * dy < 0.6 else 0
                        row += bytes(px + [a])
                    rows.append(bytes(row))
                f = job.out_dir / f"swatch_{i + 1}.png"
                _png(f, w, h, rows)
                ctx.output(f, "image", seed=seed, index=i, name=f"Swatch {seed}")
            else:
                sr = int(p.get("sample_rate", 44100))
                secs = max(0.05, min(30.0, float(p.get("seconds", 1.5))))
                freq = float(p.get("frequency", 440)) * (1 + 0.05 * (rnd.random() - 0.5) * (1 if n > 1 else 0))
                shape = p.get("waveform", "sine")
                nframes = int(sr * secs)
                frames = bytearray()
                for k in range(nframes):
                    t = k / sr
                    ph = (freq * t) % 1.0
                    if shape == "square":
                        v = 1.0 if ph < 0.5 else -1.0
                    elif shape == "saw":
                        v = 2 * ph - 1
                    elif shape == "noise":
                        v = rnd.uniform(-1, 1)
                    else:
                        v = math.sin(2 * math.pi * ph)
                    env = min(1.0, t / 0.01) * math.exp(-3.0 * t / secs)
                    frames += struct.pack("<h", int(max(-1, min(1, v * env * 0.7)) * 32767))
                f = job.out_dir / f"tone_{i + 1}.wav"
                with wave.open(str(f), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(bytes(frames))
                ctx.output(f, "audio", seed=seed, index=i, name=f"{shape} {freq:.0f} Hz")
        return {"meta": {"seed": seed0, "backend": "echo"}}


if __name__ == "__main__":
    serve(EchoBackend())
