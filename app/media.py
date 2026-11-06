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
import re
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
             ".fbx": "application/octet-stream", ".npz": "application/octet-stream", ".csv": "text/csv",
             ".srt": "application/x-subrip", ".vtt": "text/vtt"}
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


# ------------------------------------------------------------------ export (derived files; CPU only)
# The originals are never touched: these write a new file next to a scratch / export path chosen by the caller.
AUDIO_EXPORT_FORMATS = ("wav", "mp3", "ogg", "flac")
IMAGE_EXPORT_FORMATS = ("png", "jpeg", "webp")
_IMAGE_EXT = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}
_ASPECT_RE = re.compile(r"^(\d{1,3}(?:\.\d+)?):(\d{1,3}(?:\.\d+)?)$")


class ExportError(ValueError):
    pass


def _ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise ExportError("ffmpeg is not installed (needed for audio export): install it with your package manager.")
    return p


def _run_ffmpeg(args: list[str], timeout: float = 900.0) -> str:
    try:
        r = subprocess.run([_ffmpeg(), "-hide_banner", "-nostdin", "-nostats", "-protocol_whitelist", "file", *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ExportError("ffmpeg took too long converting this audio file.") from None
    if r.returncode != 0:
        tail = [ln for ln in r.stderr.strip().splitlines() if ln.strip()][-2:]
        raise ExportError("ffmpeg failed: " + " ".join(tail)[:300])
    return r.stderr


def _loudnorm_json(stderr: str) -> dict | None:
    """The JSON block ffmpeg's loudnorm prints at the end of stderr."""
    i = stderr.rfind("{")
    j = stderr.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        return json.loads(stderr[i:j + 1])
    except ValueError:
        return None


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def audio_export_options(opts: dict) -> dict:
    """Validate / normalise audio export options (see app/profiles.py for the keys)."""
    o = dict(opts or {})
    fmt = str(o.get("format") or "wav").lower().lstrip(".")
    fmt = {"ogg_vorbis": "ogg", "vorbis": "ogg"}.get(fmt, fmt)
    if fmt not in AUDIO_EXPORT_FORMATS:
        raise ExportError(f"Audio format must be one of {', '.join(AUDIO_EXPORT_FORMATS)}.")
    sr = o.get("sample_rate")
    sr = int(sr) if sr not in (None, "", 0) else None
    if sr is not None and not 8000 <= sr <= 192000:
        raise ExportError("sample_rate must be 8000..192000 Hz.")
    bd = o.get("bit_depth")
    bd = int(bd) if bd not in (None, "", 0) else None
    if bd is not None and bd not in (16, 24, 32):
        raise ExportError("bit_depth must be 16, 24 or 32.")
    if fmt == "flac" and bd == 32:
        bd = 24
    lufs = _num(o.get("lufs")) if o.get("lufs") not in (None, "") else None
    if lufs is not None and not -40.0 <= lufs <= -5.0:
        raise ExportError("lufs must be between -40 and -5.")
    tp = _num(o.get("true_peak"))
    tp = -1.0 if tp is None else tp
    if not -9.0 <= tp <= 0.0:
        raise ExportError("true_peak must be between -9 and 0 dBTP.")
    ch = o.get("channels")
    ch = int(ch) if ch not in (None, "", 0) else None
    if ch not in (None, 1, 2):
        raise ExportError("channels must be 1 or 2.")
    br = int(o.get("bitrate_kbps") or 192)
    if not 32 <= br <= 320:
        raise ExportError("bitrate_kbps must be 32..320.")
    return {"format": fmt, "sample_rate": sr, "bit_depth": bd, "lufs": lufs, "true_peak": tp, "channels": ch, "bitrate_kbps": br}


def export_audio(src: Path, dst_dir: Path, stem: str, opts: dict, *, music: bool = False) -> tuple[Path, dict]:
    """Convert `src` into `dst_dir/<stem>.<ext>` per `opts`; returns (path, report).

    Loudness: two-pass ffmpeg loudnorm (pass 1 measures, pass 2 applies the measured values with linear=true so the
    dynamics are kept whenever the target is reachable). Silent files are converted without a gain change. Sample
    rate / bit depth / channels are applied after the loudness stage (loudnorm works internally at 192 kHz)."""
    o = {**(opts or {})}
    if music and o.get("music_sample_rate"):
        o["sample_rate"] = o["music_sample_rate"]
    o = audio_export_options(o)
    src, dst_dir = Path(src), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    fmt = o["format"]
    out = dst_dir / f"{stem}.{fmt}"
    report: dict = {"format": fmt, "sample_rate": o["sample_rate"], "bit_depth": o["bit_depth"], "channels": o["channels"],
                    "target_lufs": o["lufs"], "true_peak": o["true_peak"], "normalized": False}
    filters = []
    if o["lufs"] is not None:
        base = f"loudnorm=I={o['lufs']}:TP={o['true_peak']}:LRA=11"
        err = _run_ffmpeg(["-i", str(src), "-af", base + ":print_format=json", "-vn", "-f", "null", "-"])
        m = _loudnorm_json(err) or {}
        mi, mtp, mlra, mth, off = (_num(m.get(k)) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset"))
        report["input_lufs"] = mi
        if None in (mi, mtp, mlra, mth, off) or mi < -69.0:
            report["note"] = "Input is silent or too quiet to measure: converted without a gain change."
        else:
            filters.append(f"{base}:measured_I={mi}:measured_TP={mtp}:measured_LRA={mlra}:measured_thresh={mth}:offset={off}:linear=true:print_format=json")
            report["normalized"] = True
    args = ["-y", "-i", str(src), "-vn", "-map_metadata", "-1"]
    if filters:
        args += ["-af", ",".join(filters)]
    sr_out = o["sample_rate"]
    if filters and not sr_out:      # loudnorm resamples to 192 kHz internally: bring it back to the source rate
        try:
            import soundfile as sf

            sr_out = int(sf.info(str(src)).samplerate)
        except (RuntimeError, OSError):
            sr_out = 48000
    if sr_out:
        args += ["-ar", str(sr_out)]
        report["sample_rate"] = sr_out
    if o["channels"]:
        args += ["-ac", str(o["channels"])]
    bd = o["bit_depth"] or 16
    if fmt == "wav":
        args += ["-c:a", {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_f32le"}[bd]]
        report["bit_depth"] = bd
    elif fmt == "flac":
        args += ["-c:a", "flac", "-sample_fmt", "s32" if bd == 24 else "s16"] + (["-bits_per_raw_sample", "24"] if bd == 24 else [])
        report["bit_depth"] = bd
    elif fmt == "mp3":
        args += ["-c:a", "libmp3lame", "-b:a", f"{o['bitrate_kbps']}k"]
        report["bit_depth"] = None
    else:
        args += ["-c:a", "libvorbis", "-q:a", "6"]
        report["bit_depth"] = None
    args.append(str(out))
    err = _run_ffmpeg(args)
    if filters:
        res = _loudnorm_json(err) or {}
        report["output_lufs"] = _num(res.get("output_i"))
        report["output_true_peak"] = _num(res.get("output_tp"))
        if res.get("normalization_type") == "dynamic":
            report["note"] = "The target was not reachable linearly; ffmpeg used dynamic normalisation."
    try:
        info, _ = analyze_audio(out, buckets=10)
        report["result"] = {k: info.get(k) for k in ("duration", "sample_rate", "channels", "peak_db", "lufs")}
    except ValueError:
        pass
    return out, report


def image_export_options(opts: dict) -> dict:
    o = dict(opts or {})
    fmt = str(o.get("format") or "png").lower().lstrip(".")
    fmt = {"jpg": "jpeg"}.get(fmt, fmt)
    if fmt not in IMAGE_EXPORT_FORMATS:
        raise ExportError("Image format must be png, jpeg or webp.")
    out = {"format": fmt, "fit": "pad" if o.get("fit") == "pad" else "crop"}
    aspect = o.get("aspect")
    if aspect not in (None, "", "original"):
        mm = _ASPECT_RE.match(str(aspect))
        if not mm or float(mm.group(1)) <= 0 or float(mm.group(2)) <= 0:
            raise ExportError("aspect must look like 16:9.")
        out["aspect"] = (float(mm.group(1)), float(mm.group(2)))
    for k, lo, hi in (("width", 16, 16384), ("height", 16, 16384), ("max_side", 16, 16384)):
        v = o.get(k)
        if v not in (None, "", 0):
            v = int(v)
            if not lo <= v <= hi:
                raise ExportError(f"{k} must be {lo}..{hi}.")
            out[k] = v
    q = int(o.get("quality") or 90)
    out["quality"] = max(1, min(100, q))
    bg = str(o.get("background") or "#ffffff")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", bg):
        raise ExportError("background must look like #rrggbb.")
    out["background"] = bg
    return out


def export_image(src: Path, dst_dir: Path, stem: str, opts: dict) -> tuple[Path, dict]:
    """Crop (or pad) to the aspect, resize and re-encode `src` into `dst_dir/<stem>.<ext>`; returns (path, report).

    `width`+`height` give an exact size (the aspect defaults to width:height; a different `aspect` is cropped first);
    `max_side` scales the longest side; neither keeps the pixel size of the crop. Images are never upscaled unless an
    exact size is requested. The crop is centred. PNG / WebP keep alpha, JPEG is flattened onto `background`."""
    o = image_export_options(opts)
    src, dst_dir = Path(src), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im0:
        im = ImageOps.exif_transpose(im0)
        im.load()
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if (im.mode in ("LA", "PA") or "transparency" in im.info) else "RGB")
    w0, h0 = im.size
    if o.get("aspect"):
        ar = o["aspect"][0] / o["aspect"][1]
    elif o.get("width") and o.get("height"):
        ar = o["width"] / o["height"]
    else:
        ar = None
    if ar and abs(ar - w0 / h0) > 1e-3:
        if o["fit"] == "crop":
            if w0 / h0 > ar:
                nw = max(1, round(h0 * ar))
                x0 = (w0 - nw) // 2
                im = im.crop((x0, 0, x0 + nw, h0))
            else:
                nh = max(1, round(w0 / ar))
                y0 = (h0 - nh) // 2
                im = im.crop((0, y0, w0, y0 + nh))
        else:
            if w0 / h0 > ar:
                canvas = (w0, max(1, round(w0 / ar)))
            else:
                canvas = (max(1, round(h0 * ar)), h0)
            pad = Image.new("RGBA", canvas, (0, 0, 0, 0))
            pad.paste(im.convert("RGBA"), ((canvas[0] - w0) // 2, (canvas[1] - h0) // 2))
            im = pad
    if o.get("width") and o.get("height"):
        if im.size != (o["width"], o["height"]):
            im = im.resize((o["width"], o["height"]), Image.LANCZOS)
    elif o.get("max_side"):
        side = max(im.size)
        if side > o["max_side"]:
            f = o["max_side"] / side
            im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
    elif o.get("width") or o.get("height"):
        f = (o.get("width") / im.width) if o.get("width") else (o["height"] / im.height)
        im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
    fmt = o["format"]
    out = dst_dir / f"{stem}{_IMAGE_EXT[fmt]}"
    if fmt == "jpeg":
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, o["background"])
        bg.paste(rgba, mask=rgba.getchannel("A"))
        bg.save(out, "JPEG", quality=o["quality"], optimize=True)
    elif fmt == "webp":
        im.save(out, "WEBP", quality=o["quality"], method=4)
    else:
        im.save(out, "PNG", optimize=True)
    report = {"format": fmt, "width": im.width, "height": im.height, "source_width": w0, "source_height": h0,
              "alpha": im.mode == "RGBA" and fmt != "jpeg"}
    return out, report
