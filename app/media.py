"""Media helpers shared by the core and the studios (CPU only: PIL, numpy, soundfile, scipy).

  image_info / make_thumb       size, alpha, a webp thumbnail (alpha kept: the UI draws the checkerboard)
  analyze_audio                 duration, sample rate, channels, peak dBFS, RMS dBFS, integrated LUFS, waveform peaks
  integrated_lufs               ITU-R BS.1770-4 (K-weighting + gating)
  glb_info                      vertex / face / material counts read from the glTF JSON chunk (no mesh library)
  verify_video / video_info     magic bytes (always) + ffprobe (when installed): duration, size, fps, audio track
  make_video_poster             a webp poster frame through ffmpeg (None when ffmpeg is missing)
"""
from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MAX_IMAGE_PIXELS = 80_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

IMAGE_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
AUDIO_EXT = {".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg", ".mp3": "audio/mpeg"}
MESH_EXT = {".glb": "model/gltf-binary"}
VIDEO_EXT = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska"}
OTHER_EXT = {".json": "application/json", ".txt": "text/plain", ".obj": "text/plain", ".mtl": "text/plain",
             ".fbx": "application/octet-stream", ".npz": "application/octet-stream", ".csv": "text/csv"}
MIME = {**IMAGE_EXT, **AUDIO_EXT, **MESH_EXT, **VIDEO_EXT, **OTHER_EXT}


def type_of(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in MESH_EXT:
        return "mesh"
    if ext in VIDEO_EXT:
        return "video"
    return "other"


# ------------------------------------------------------------------ images
def alpha_stats(img: Image.Image) -> tuple[bool, float]:
    if img.mode not in ("RGBA", "LA", "PA") and "transparency" not in img.info:
        return False, 0.0
    a = np.asarray(img.convert("RGBA").getchannel("A"))
    return bool((a < 250).mean() > 0.002), float((a < 128).mean())


def image_info(path: Path) -> dict:
    with Image.open(path) as im:
        has, frac = alpha_stats(im)
        return {"width": im.width, "height": im.height, "mode": im.mode, "has_alpha": has,
                "transparent_fraction": round(frac, 4), "format": im.format}


def make_thumb(src: Path, dst: Path, side: int = 384) -> Path:
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((side, side), Image.LANCZOS)
        im = im.convert("RGBA") if alpha_stats(im)[0] else im.convert("RGB")
        im.save(dst, "WEBP", quality=86, method=4)
    return dst


def verify_image(path: Path) -> dict:
    """Raise ValueError unless `path` is a real, sane image."""
    try:
        with Image.open(path) as im:
            if im.width * im.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"Image too large ({im.width}x{im.height}).")
            im.verify()
        with Image.open(path) as im:
            im.load()
    except (OSError, SyntaxError, Image.DecompressionBombError) as e:
        raise ValueError(f"Not a valid image: {e}") from None
    return image_info(path)


# ------------------------------------------------------------------ audio
def _biquad_high_shelf(fs, G=3.99984385397, Q=0.7071752369554193, fc=1681.9744509555319):
    A = 10 ** (G / 40.0)
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2 * Q)
    c = math.cos(w0)
    sa = 2 * math.sqrt(A) * alpha
    b = [A * ((A + 1) + (A - 1) * c + sa), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sa)]
    a = [(A + 1) - (A - 1) * c + sa, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sa]
    return np.array(b) / a[0], np.array(a) / a[0]


def _biquad_high_pass(fs, Q=0.5003270373253953, fc=38.13547087613982):
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2 * Q)
    c = math.cos(w0)
    b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
    a = [1 + alpha, -2 * c, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def integrated_lufs(x: np.ndarray, sr: int) -> float | None:
    """x: (frames, channels) float in [-1, 1]. Returns integrated loudness (LUFS) or None for silence."""
    from scipy.signal import lfilter

    if x.ndim == 1:
        x = x[:, None]
    x = x[:, :2] if x.shape[1] <= 2 else x[:, :3]  # L, R (+C); surround weighting is not needed here
    b1, a1 = _biquad_high_shelf(sr)
    b2, a2 = _biquad_high_pass(sr)
    y = lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)
    block = int(0.4 * sr)
    hop = int(0.1 * sr)
    if len(y) < block:
        z = np.mean(y ** 2, axis=0)[None, :]
    else:
        n = 1 + (len(y) - block) // hop
        z = np.stack([np.mean(y[i * hop:i * hop + block] ** 2, axis=0) for i in range(n)])
    s = z.sum(axis=1)
    with np.errstate(divide="ignore"):
        lj = -0.691 + 10 * np.log10(np.maximum(s, 1e-12))
    gated = s[lj > -70]
    if gated.size == 0:
        return None
    rel = -0.691 + 10 * math.log10(gated.mean()) - 10
    final = s[(lj > -70) & (lj > rel)]
    if final.size == 0:
        return None
    return round(-0.691 + 10 * math.log10(final.mean()), 2)


def waveform_peaks(x: np.ndarray, buckets: int = 1000) -> list[float]:
    mono = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    if len(mono) == 0:
        return []
    buckets = min(buckets, len(mono))
    edges = np.linspace(0, len(mono), buckets + 1).astype(int)
    return [round(float(mono[edges[i]:max(edges[i] + 1, edges[i + 1])].max()), 3) for i in range(buckets)]


def analyze_audio(path: Path, buckets: int = 1000) -> tuple[dict, dict]:
    """(info, waveform). info: duration, sample_rate, channels, peak_db, rms_db, lufs, silence_fraction."""
    import soundfile as sf

    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except RuntimeError as e:
        raise ValueError(f"Not a readable audio file: {e}") from None
    if x.size == 0:
        raise ValueError("The audio file is empty.")
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x ** 2)))
    env = np.abs(x).max(axis=1)
    info = {"duration": round(len(x) / sr, 3), "sample_rate": int(sr), "channels": int(x.shape[1]),
            "frames": int(len(x)), "peak_db": round(20 * math.log10(peak), 2) if peak > 0 else None,
            "rms_db": round(20 * math.log10(rms), 2) if rms > 0 else None,
            "lufs": integrated_lufs(x, sr) if len(x) > sr * 0.05 else None,
            "silence_fraction": round(float((env < 10 ** (-50 / 20)).mean()), 3),
            "clipped": bool(peak >= 0.999)}
    return info, {"version": 1, "peaks": [waveform_peaks(x, buckets)], "duration": info["duration"],
                  "sample_rate": info["sample_rate"]}


# ------------------------------------------------------------------ meshes
def glb_info(path: Path) -> dict:
    """Read the JSON chunk of a .glb; raise ValueError if it is not a glTF binary."""
    with open(path, "rb") as f:
        head = f.read(20)
        if len(head) < 20 or head[:4] != b"glTF":
            raise ValueError("Not a GLB file (bad magic).")
        _magic, version, length, clen, ctype = struct.unpack("<4sIIII", head)
        if ctype != 0x4E4F534A or clen > 64 * 2**20:
            raise ValueError("Not a GLB file (no JSON chunk).")
        doc = json.loads(f.read(clen))
    acc = doc.get("accessors", [])
    verts = faces = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is not None and pos < len(acc):
                verts += int(acc[pos].get("count", 0))
            idx = prim.get("indices")
            if idx is not None and idx < len(acc):
                faces += int(acc[idx].get("count", 0)) // 3
    return {"glb_version": version, "bytes": length, "vertices": verts, "faces": faces,
            "meshes": len(doc.get("meshes", [])), "materials": len(doc.get("materials", [])),
            "textures": len(doc.get("textures", []))}


# ------------------------------------------------------------------ video
# ffprobe is told the container explicitly (never auto-detected): playlists / concat files / network URLs are
# demuxers too, and a "video" upload must not be able to make ffprobe open other files.
_FFPROBE_FORMAT = {".mp4": "mov,mp4,m4a,3gp,3g2,mj2", ".mov": "mov,mp4,m4a,3gp,3g2,mj2",
                   ".webm": "matroska,webm", ".mkv": "matroska,webm"}
_MP4_BOXES = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot", b"junk", b"uuid"}
_EBML = b"\x1a\x45\xdf\xa3"


def video_container_ok(path: Path) -> bool:
    """Magic bytes only: ISO-BMFF (mp4 / mov: a known box type at offset 4) or EBML (webm / mkv)."""
    ext = Path(path).suffix.lower()
    with open(path, "rb") as f:
        head = f.read(16)
    if ext in (".webm", ".mkv"):
        return head[:4] == _EBML
    if ext in (".mp4", ".mov"):
        return len(head) >= 12 and head[4:8] in _MP4_BOXES
    return False


def ffprobe_video(path: Path, timeout: float = 20.0) -> dict | None:
    """ffprobe summary {duration, width, height, fps, codec, has_audio, audio_codec}; None when ffprobe is missing.
    Raises ValueError when the file has no decodable video stream."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    fmt = _FFPROBE_FORMAT.get(Path(path).suffix.lower())
    cmd = [exe, "-v", "error", "-protocol_whitelist", "file", "-print_format", "json", "-show_format", "-show_streams"]
    if fmt:
        cmd += ["-f", fmt]
    try:
        r = subprocess.run(cmd + [str(path)], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise ValueError("Not a readable video (ffprobe timed out).") from None
    if r.returncode != 0:
        raise ValueError("Not a readable video file (ffprobe could not parse it).")
    try:
        d = json.loads(r.stdout or "{}")
    except ValueError:
        raise ValueError("Not a readable video file (ffprobe output).") from None
    vs = [s for s in d.get("streams", []) if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic") != 1]
    if not vs:
        raise ValueError("The file has no video stream.")
    v = vs[0]
    a = [s for s in d.get("streams", []) if s.get("codec_type") == "audio"]
    dur = v.get("duration")
    if dur in (None, "N/A"):
        dur = d.get("format", {}).get("duration")
    try:
        dur = round(float(dur), 3)
    except (TypeError, ValueError):
        dur = None
    fps = None
    try:
        n, _, den = str(v.get("avg_frame_rate", "0/1")).partition("/")
        fps = round(float(n) / float(den or 1), 3) if float(den or 1) else None
    except ValueError:
        pass
    w, h = v.get("width"), v.get("height")
    if not (isinstance(w, int) and isinstance(h, int) and 0 < w <= 16384 and 0 < h <= 16384):
        raise ValueError("The video has no valid frame size.")
    return {"duration": dur, "width": w, "height": h, "fps": fps, "codec": v.get("codec_name"),
            "has_audio": bool(a), "audio_codec": a[0].get("codec_name") if a else None}


def verify_video(path: Path) -> dict:
    """Raise ValueError unless `path` is a real video (magic bytes, then ffprobe when available). Returns video_info."""
    if not video_container_ok(path):
        raise ValueError("Not a valid video file (unknown container; mp4, mov, webm and mkv are supported).")
    return video_info(path)


def video_info(path: Path) -> dict:
    """info block of a video asset: duration, size in px and bytes, fps, codecs. Works without ffprobe (bytes only)."""
    info = ffprobe_video(path) or {}
    info = {k: v for k, v in info.items() if v is not None}
    info["bytes"] = Path(path).stat().st_size
    return info


def make_video_poster(src: Path, dst: Path, duration: float | None = None, side: int = 384, timeout: float = 30.0) -> Path | None:
    """One frame (about 10% in, capped at 1 s) scaled to `side` px as a webp. None when ffmpeg is missing / fails."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    at = min(1.0, max(0.0, (duration or 0) * 0.1))
    vf = f"scale='if(gt(iw,ih),{side},-2)':'if(gt(iw,ih),-2,{side})'"
    for ss in (at, 0.0):
        try:
            r = subprocess.run([exe, "-y", "-v", "error", "-protocol_whitelist", "file", "-ss", f"{ss:.3f}", "-i", str(src), "-frames:v", "1",
                                "-vf", vf, "-c:v", "libwebp", "-quality", "80", str(dst)],
                               capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode == 0 and Path(dst).is_file() and Path(dst).stat().st_size > 0:
            return Path(dst)
    Path(dst).unlink(missing_ok=True)
    return None
