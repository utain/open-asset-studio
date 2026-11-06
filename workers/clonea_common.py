"""Shared helpers of the `clonea_*` workers (voice cloning group A). Imports only numpy / stdlib at module level, so it
works in every clonea venv. Modelled on the helpers of audio_worker.py (kept as a copy so those two stay independent).
"""
from __future__ import annotations

import math
import random
import re
from pathlib import Path

import numpy as np

from studio_worker import InputError

THAI = re.compile(r"[฀-๿]")
CJK = re.compile(r"[぀-ヿ一-鿿]")


def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def seed_all(s: int):
    import torch

    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def tidy_voice(w: np.ndarray, sr: int, pad_s: float = 0.12, floor_db: float = -50.0) -> np.ndarray:
    """Trim leading / trailing silence of a speech take (keeps `pad_s` around the voice)."""
    mono = np.abs(w).max(axis=1) if w.ndim == 2 else np.abs(w)
    idx = np.flatnonzero(mono > 10 ** (floor_db / 20))
    if idx.size == 0:
        return w
    pad = int(pad_s * sr)
    a, b = max(0, idx[0] - pad), min(len(mono), idx[-1] + pad + 1)
    return w[a:b]


def save_wav(path: Path, wav, sr: int) -> dict:
    """Write float audio (T,) or (T, C) as 24-bit WAV. Brings overs to -1 dBFS. Returns file meta."""
    import soundfile as sf

    w = np.asarray(wav, dtype=np.float32)
    if w.ndim == 2 and w.shape[0] <= 8 and w.shape[0] < w.shape[1]:
        w = w.T
    if w.ndim == 2 and w.shape[1] == 1:
        w = w[:, 0]
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if w.size == 0 or len(w) < int(0.02 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    peak = float(np.abs(w).max())
    meta = {"sample_rate": int(sr)}
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def read_mono(path: str, sr: int | None = None) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, s = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr and s != sr:
        import librosa

        y = librosa.resample(y, orig_sr=s, target_sr=sr)
        s = sr
    return y, s


def patch_torchaudio_load():
    """torchaudio 2.9 load() needs torchcodec (no aarch64 wheel for cu130): read with soundfile instead."""
    import soundfile as sf
    import torch
    import torchaudio

    def _sf_load(path, *a, **k):
        w, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(w.T.copy()), sr
    torchaudio.load = _sf_load


def ref_input(job) -> dict | None:
    for role in ("ref", "voice", "reference"):
        r = job.input(role=role)
        if r:
            return r
    return next((i for i in job.inputs if i.get("type") in ("voice", "audio")), None)


def ref_text_of(job, ref: dict | None) -> str:
    t = (job.params.get("ref_text") or "").strip()
    if not t and ref:
        t = (((ref.get("meta") or {}).get("x_voice") or {}).get("ref_text") or "").strip()
    return t


def text_of(job) -> str:
    t = (job.params.get("text") or job.params.get("prompt") or "").strip()
    if not t:
        raise InputError("Write the line to speak first.")
    return t


def take_name(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def ref_wav_to_file(ref_path: str, out, seconds: float = 30.0, sr: int = 24000) -> str:
    """Convert any reference clip to a mono wav (<= `seconds`, `sr` Hz) at `out`; returns the path."""
    import soundfile as sf

    y, s = read_mono(ref_path, sr)
    y = y[: int(seconds * s)]
    if y.size < int(0.5 * s):
        raise InputError("The reference clip is too short (need at least 0.5 s, better 3-10 s of clean speech).")
    sf.write(str(out), y, s, subtype="PCM_16")
    return str(out)


class Asr:
    """Lazy Whisper (large-v3-turbo) for reference transcripts; freed with the backend."""
    pipe = None

    @classmethod
    def transcribe(cls, path: str, ctx) -> str:
        import torch
        from transformers import pipeline

        ctx.progress(0, 0, "transcribing", "No reference transcript: transcribing the clip with Whisper", interruptible=False)
        if cls.pipe is None:
            cls.pipe = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo", dtype=torch.float16, device="cuda")
        y, _ = read_mono(path, 16000)
        text = cls.pipe({"raw": y[: 16000 * 30], "sampling_rate": 16000})["text"].strip()
        ctx.log(f"reference transcript (Whisper): {text}")
        return text

    @classmethod
    def free(cls):
        cls.pipe = None
