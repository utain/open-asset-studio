"""Shared helpers of the video workers (runs in envs/video; plain numpy + the ffmpeg binary, no torch import here).

  * dims_from_aspect()   aspect ratio + short side -> (width, height) rounded to the model grid
  * frame_count()        seconds * fps -> a legal frame count (4k+1 for Wan / Hunyuan / Kandinsky, 8k+1 for LTX)
  * fit_image()          load + centre-crop/resize an input image to the exact output size
  * write_mp4()          numpy frames (+ optional audio) -> h264 yuv420p mp4 with faststart through ffmpeg
  * Bar / hook_progress  make a diffusers pipeline report one progress event per denoising step and honour cancel
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

ASPECTS = {"1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "16:9": (16, 9), "9:16": (9, 16), "21:9": (21, 9), "3:2": (3, 2), "2:3": (2, 3)}


def round_to(v: float, mult: int) -> int:
    return max(mult, int(round(v / mult)) * mult)


def dims_from_aspect(aspect: str, short_side: int, mult: int = 32, image_size: tuple[int, int] | None = None) -> tuple[int, int]:
    """(width, height): `short_side` is the shorter edge in px; `image_size` (w, h) overrides the aspect (match input image)."""
    if image_size:
        a, b = image_size
    else:
        a, b = ASPECTS.get(str(aspect), (16, 9))
    s = int(short_side)
    if a >= b:
        w, h = s * a / b, s
    else:
        w, h = s, s * b / a
    return round_to(w, mult), round_to(h, mult)


def frame_count(seconds: float, fps: float, step: int = 4, max_frames: int = 121, min_frames: int | None = None) -> int:
    """A legal number of frames (k * step + 1) closest to seconds * fps, capped at max_frames."""
    n = max(1.0, float(seconds) * float(fps))
    k = max(1, int(round((n - 1) / step)))
    k = min(k, (int(max_frames) - 1) // step)
    if min_frames:
        k = max(k, (int(min_frames) - 1 + step - 1) // step)
    return k * step + 1


def fit_image(path: str, size: tuple[int, int]):
    """Open `path` as RGB and cover-crop it to exactly `size` (w, h) with Lanczos resampling."""
    from PIL import Image, ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    return ImageOps.fit(im, size, method=Image.LANCZOS, centering=(0.5, 0.5))


def image_size(path: str) -> tuple[int, int]:
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.size


def to_uint8(frames) -> np.ndarray:
    """(T, H, W, 3) float [0,1] / uint8 / list of PIL images -> uint8 array."""
    if isinstance(frames, (list, tuple)):
        if frames and hasattr(frames[0], "convert"):
            return np.stack([np.asarray(f.convert("RGB")) for f in frames])
        frames = np.stack([np.asarray(f) for f in frames])
    a = np.asarray(frames)
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 1.0) * 255.0 + 0.5
        a = a.astype(np.uint8)
    return a


def _write_wav(path: str, audio: np.ndarray, sr: int):
    """audio: float32 (S,) or (C, S) in [-1, 1] -> 16-bit PCM wav (stdlib only)."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[0] > a.shape[1]:   # (S, C) given
        a = a.T
    pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2").T.copy()
    with wave.open(path, "wb") as w:
        w.setnchannels(a.shape[0])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def write_mp4(path, frames: np.ndarray, fps: float, audio: np.ndarray | None = None, sample_rate: int = 24000, crf: int = 17) -> dict:
    """Encode (T, H, W, 3) uint8 frames to h264 / yuv420p / +faststart (and AAC audio when given)."""
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg is not installed (needed to write the mp4)")
    frames = to_uint8(frames)
    t, h, w, _ = frames.shape
    if w % 2 or h % 2:   # yuv420p needs even sides
        frames = frames[:, : h - h % 2, : w - w % 2]
        t, h, w, _ = frames.shape
    wav = None
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{float(fps):g}", "-i", "-"]
    if audio is not None and np.asarray(audio).size:
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        _write_wav(wav, audio, sample_rate)
        cmd += ["-i", wav]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-movflags", "+faststart", "-r", f"{float(fps):g}"]
    if wav:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [str(path)]
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        chunk = max(1, (64 << 20) // (w * h * 3))
        try:
            for i in range(0, t, chunk):
                p.stdin.write(np.ascontiguousarray(frames[i:i + chunk]).tobytes())
            p.stdin.close()
        except BrokenPipeError:
            pass
        err = p.stderr.read().decode("utf-8", "replace")
        if p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err[-500:]}")
    finally:
        if wav:
            try:
                os.unlink(wav)
            except OSError:
                pass
    return {"width": w, "height": h, "frames": t, "fps": float(fps), "has_audio": bool(wav), "bytes": Path(path).stat().st_size}


def read_video_frames(path: str, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Decode a video to (T, H, W, 3) uint8 + fps through ffmpeg / ffprobe (used by the RIFE interpolation)."""
    import json
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                            "-of", "json", str(path)], capture_output=True, text=True, check=True)
    st = json.loads(probe.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    num, den = (st.get("r_frame_rate") or "24/1").split("/")
    fps = float(num) / float(den or 1)
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    if max_frames:
        cmd[3:3] = ["-frames:v", str(int(max_frames))]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype=np.uint8)
    n = arr.size // (w * h * 3)
    return arr[: n * w * h * 3].reshape(n, h, w, 3), fps


class Bar:
    """Stand-in for diffusers' tqdm bar: `with pipe.progress_bar(total=n) as bar: ... bar.update()`."""

    def __init__(self, ctx, total, label=""):
        self.ctx, self.total, self.n, self.label = ctx, int(total or 0), 0, label

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.ctx.progress(self.total, self.total, "decoding", "Decoding the video (VAE)…", interruptible=False)
        return False

    def update(self, n=1):
        self.n += n
        self.ctx.check()
        self.ctx.progress(self.n, self.total, "denoising", f"{self.label}step {self.n}/{self.total}")

    def set_description(self, *a, **k):
        pass

    def close(self):
        pass


def hook_progress(pipe, ctx, label=""):
    """Route every pipeline progress bar to ctx.progress and raise Cancelled between steps."""
    def progress_bar(iterable=None, total=None):
        if iterable is not None:
            return _iter_bar(iterable, ctx, label)
        return Bar(ctx, total, label)
    pipe.progress_bar = progress_bar


def _iter_bar(iterable, ctx, label):
    items = list(iterable)
    for i, x in enumerate(items):
        ctx.check()
        ctx.progress(i + 1, len(items), "denoising", f"{label}step {i + 1}/{len(items)}")
        yield x
