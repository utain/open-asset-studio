"""Tiny shared helper for the snippets: timing, peak GPU memory, basic audio stats.
Every snippet prints one JSON line `{"model":..., "load_s":..., "gen_s":..., ...}` at the end,
which is also the shape a JSON-lines worker would return."""
import json, os, time
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def audio_stats(wav, sr):
    """wav: np.ndarray (T,) or (T, C) float in [-1, 1]."""
    w = np.asarray(wav, dtype=np.float32)
    mono = w.mean(axis=1) if w.ndim == 2 else w
    frame = max(1, int(sr * 0.02))
    n = len(mono) // frame
    rms = np.sqrt((mono[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12) if n else np.zeros(1)
    db = 20 * np.log10(rms + 1e-12)
    return {
        "sr": int(sr),
        "channels": int(w.shape[1]) if w.ndim == 2 else 1,
        "duration_s": round(len(mono) / sr, 2),
        "peak": round(float(np.abs(w).max()), 3) if w.size else 0.0,
        "clip_frac": round(float((np.abs(w) >= 0.999).mean()), 5) if w.size else 0.0,
        "silence_frac": round(float((db < -45).mean()), 3),
        "rms_db": round(float(20 * np.log10(np.sqrt((mono ** 2).mean() + 1e-12))), 1),
        "nan": bool(np.isnan(w).any()),
    }


class Bench:
    def __init__(self, model):
        import torch
        self.torch = torch
        self.r = {"model": model}
        self.t = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def loaded(self):
        self.r["load_s"] = round(time.time() - self.t, 1)
        self.t = time.time()
        if self.torch.cuda.is_available():
            self.r["gpu_after_load_gb"] = round(self.torch.cuda.memory_allocated() / 2**30, 2)

    def done(self, wav, sr, out):
        import soundfile as sf
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        self.r["gen_s"] = round(time.time() - self.t, 2)
        if self.torch.cuda.is_available():
            self.r["peak_gpu_gb"] = round(self.torch.cuda.max_memory_allocated() / 2**30, 2)
        w = np.asarray(wav, dtype=np.float32)
        sf.write(out, w, sr)
        self.r.update(audio_stats(w, sr))
        self.r["rtf"] = round(self.r["gen_s"] / max(self.r["duration_s"], 1e-3), 3)
        self.r["out"] = out
        print("RESULT " + json.dumps(self.r), flush=True)
        return self.r
