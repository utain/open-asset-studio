"""Video studio plugin: CPU post tools built on ffmpeg (no model needed, they run beside GPU jobs).

Every tool takes video assets as `inputs` (role `video`; `audio` for mux), writes NEW assets (parents = the inputs, the
input's task / prompt / licence carried over, `x_post` describing the operation) and never touches the originals.

  tool:video.frame     {at: "last" | "first" | <seconds>}      one still frame as a PNG image asset (continue a clip: image to video)
  tool:video.trim      {start: 0, end: 0}                       cut a range (end 0 = to the end), re-encoded frame-accurately
  tool:video.loop      {mode: pingpong | repeat, times: 2}      forward + backward loop, or the clip repeated
  tool:video.speed     {speed: 0.25..4}                         slow motion / fast forward (audio follows)
  tool:video.reframe   {aspect: 9:16|1:1|16:9|4:5, fit: crop | blur | pad, short_side: 1080}   social formats
  tool:video.convert   {format: mp4 | webm, max_side: 0, fps: 0, crf: 20}
  tool:video.concat    several clips (in input order) joined into one, unified size / fps / audio
  tool:video.mux       inputs: one `video` + one `audio` asset  {volume: 1.0, keep_original: false, fade_out: 0.5}
                       puts a Sound-studio track (music, voice-over, foley) under a clip

Routes (/api/x/video): GET /formats  (aspect / resolution presets the studio and the docs use)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter

from .. import media
from ..jobs import JobContext, JobError, handler

router = APIRouter()
CORE = None

FORMATS = {
    "aspects": {"16:9": [1920, 1080], "9:16": [1080, 1920], "1:1": [1080, 1080], "4:5": [1080, 1350], "4:3": [1440, 1080], "3:4": [1080, 1440]},
    "use": {"youtube": "16:9", "reels_shorts_tiktok": "9:16", "feed_square": "1:1", "feed_portrait": "4:5"},
}
ASPECT_RATIOS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5), "4:3": (4, 3), "3:4": (3, 4)}
X264 = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p", "-profile:v", "high"]


def setup(core):
    global CORE
    CORE = core


@router.get("/formats")
def formats():
    return FORMATS


# ===================================================================== ffmpeg plumbing
def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise JobError("ffmpeg is not installed.", "env", "Install ffmpeg (apt install ffmpeg) and restart the studio.")
    return exe


def _probe(path: str) -> dict:
    try:
        info = media.ffprobe_video(Path(path))
    except ValueError as e:
        raise JobError(f"{Path(path).name}: {e}", "input") from None
    return info or {}


def run_ffmpeg(ctx: JobContext, args: list[str], duration: float | None = None, label: str = "Encoding", base: float = 0.0, span: float = 1.0):
    """Run ffmpeg with progress (`-progress pipe:1`) and cooperative cancel. `args` = everything after `-y`."""
    cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file", "-progress", "pipe:1", "-nostats", *args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
    err: list[str] = []
    t = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
    t.start()
    last = 0.0
    try:
        for line in p.stdout:
            if ctx.cancel_event.is_set():
                p.kill()
                ctx.check_cancel()
            if line.startswith("out_time_us=") and duration:
                try:
                    done = int(line.split("=", 1)[1]) / 1e6
                except ValueError:
                    continue
                if time.time() - last > 0.25:
                    last = time.time()
                    ctx.progress(base + span * min(0.99, done / duration), label, stage="encoding")
        rc = p.wait()
    finally:
        if p.poll() is None:
            p.kill()
    t.join(timeout=2)
    if rc != 0:
        msg = (err[0] if err else "").strip().splitlines()[-1:] or ["ffmpeg failed"]
        raise JobError(f"ffmpeg failed: {msg[0][:300]}", "internal")


def _videos(ctx: JobContext, role: str | None = "video", min_n: int = 1) -> list[dict]:
    ins = [i for i in ctx.input_assets() if i["type"] == "video" and (role is None or i["role"] in (role, "input"))]
    if len(ins) < min_n:
        raise JobError("Pick a video asset first." if min_n == 1 else f"Pick at least {min_n} video assets.", "input")
    return ins


def _lic(parents: list[dict]) -> dict | None:
    """Licence of a derived file: the most restrictive of its parents (non-commercial wins)."""
    lics = [p.get("license") for p in parents if p.get("license")]
    nc = [x for x in lics if not x.get("commercial", True)]
    return (nc or lics or [None])[0]


def _derived_meta(ctx: JobContext, inputs: list[dict], i: int, name: str, op: dict) -> dict:
    parents = [ctx.assets.get(x["asset_id"]) for x in inputs]
    par = parents[0]
    meta = ctx.asset_meta({"name": name}, i)
    meta["parents"] = [x["asset_id"] for x in inputs]
    meta["source"] = "tool"
    for k in ("task", "studio", "prompt", "improved_prompt", "final_prompt", "negative", "model", "model_name", "seed"):
        if par.get(k) is not None:
            meta[k] = par[k]
    lic = _lic(parents)
    if lic:
        meta["license"] = lic
    meta["batch"] = {"id": ctx.job["id"], "index": i}
    meta["studio"] = "video"
    for k, v in par.items():
        if k.startswith("x_") and k not in ("x_post", "x_qa"):
            meta[k] = v
    meta["x_post"] = {"tool": ctx.job.get("tool"), "op": op, "from": [x["asset_id"] for x in inputs]}
    meta["tags"] = sorted(set(par.get("tags") or []) | {"video-tool"})
    return meta


def _num(v, lo, hi, default, what):
    try:
        x = float(v if v not in (None, "") else default)
    except (TypeError, ValueError):
        raise JobError(f"{what} must be a number.", "input") from None
    if not lo <= x <= hi:
        raise JobError(f"{what} must be between {lo:g} and {hi:g}.", "input")
    return x


def _base(inp: dict) -> str:
    return (inp.get("name") or "video").rsplit(" · ", 1)[0]


# ===================================================================== tools
@handler("tool:video.frame", gpu=False, label="Extract a frame")
def video_frame(ctx: JobContext):
    at = str(ctx.params.get("at", "last")).strip().lower()
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    for i, inp in enumerate(_videos(ctx)):
        ctx.check_cancel()
        info = _probe(inp["path"])
        out = ctx.scratch / f"frame_{i + 1}.png"
        if at == "last":
            args = ["-sseof", "-0.5", "-i", inp["path"], "-update", "1", "-frames:v", "1000", "-an", str(out)]
            label = "last frame"
        elif at == "first":
            args = ["-i", inp["path"], "-frames:v", "1", "-an", str(out)]
            label = "first frame"
        else:
            t = _num(at, 0, float(info.get("duration") or 3600), 0, "Time")
            args = ["-ss", f"{t:.3f}", "-i", inp["path"], "-frames:v", "1", "-an", str(out)]
            label = f"frame at {t:g}s"
        ctx.progress(i / 1, f"Extracting the {label}", stage="encoding")
        run_ffmpeg(ctx, args)
        if not out.exists() or out.stat().st_size == 0:
            raise JobError("Could not read a frame from this video.", "input")
        meta = _derived_meta(ctx, [inp], i, f"{_base(inp)} · {label}", {"at": at})
        meta["tags"] = sorted(set(meta["tags"]) | {"frame"})
        meta.pop("task", None)           # a still is not a video-task result: keep it out of the Video studio grid
        meta["studio"] = "image"
        ctx.add_output(out, "image", meta)


@handler("tool:video.trim", gpu=False, label="Trim")
def video_trim(ctx: JobContext):
    start = _num(ctx.params.get("start"), 0, 36000, 0, "Start")
    end = _num(ctx.params.get("end"), 0, 36000, 0, "End")
    if end and end <= start:
        raise JobError("End must be after start (0 = to the end).", "input")
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    ins = _videos(ctx)
    for i, inp in enumerate(ins):
        ctx.check_cancel()
        info = _probe(inp["path"])
        dur = float(info.get("duration") or 0)
        if start >= dur > 0:
            raise JobError(f"{inp['name']} is only {dur:.1f} s long.", "input")
        stop = min(end, dur) if end else dur
        out = ctx.scratch / f"trim_{i + 1}.mp4"
        args = ["-ss", f"{start:.3f}", "-i", inp["path"], *(["-to", f"{stop - start:.3f}"] if end else []), *X264, "-crf", "17",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
        run_ffmpeg(ctx, args, (stop - start) or dur, f"Trimming {inp['name']}", i / len(ins), 1 / len(ins))
        meta = _derived_meta(ctx, [inp], i, f"{_base(inp)} · {start:g}-{stop:g}s", {"start": start, "end": end})
        ctx.add_output(out, "video", meta)


@handler("tool:video.loop", gpu=False, label="Loop")
def video_loop(ctx: JobContext):
    mode = str(ctx.params.get("mode", "pingpong"))
    times = int(_num(ctx.params.get("times"), 2, 8, 2, "Repeats"))
    if mode not in ("pingpong", "repeat"):
        raise JobError("mode must be pingpong or repeat", "input")
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    ins = _videos(ctx)
    for i, inp in enumerate(ins):
        ctx.check_cancel()
        info = _probe(inp["path"])
        dur = float(info.get("duration") or 0)
        if dur > 60 and mode == "pingpong":
            raise JobError("Ping-pong needs the whole clip in memory: use clips under 60 s.", "input")
        out = ctx.scratch / f"loop_{i + 1}.mp4"
        if mode == "pingpong":
            # forward + reversed, audio dropped (a reversed soundtrack is not useful); the last frame of each half is not repeated
            fc = "[0:v]split[a][b];[b]reverse,trim=start_frame=1[r];[a][r]concat=n=2:v=1:a=0[v]"
            args = ["-i", inp["path"], "-filter_complex", fc, "-map", "[v]", *X264, "-crf", "17", "-movflags", "+faststart", str(out)]
            total = dur * 2
        else:
            args = ["-stream_loop", str(times - 1), "-i", inp["path"], *X264, "-crf", "17", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
            total = dur * times
        run_ffmpeg(ctx, args, total, f"Looping {inp['name']}", i / len(ins), 1 / len(ins))
        meta = _derived_meta(ctx, [inp], i, f"{_base(inp)} · {'ping-pong' if mode == 'pingpong' else f'x{times}'} loop", {"mode": mode, "times": times})
        meta["tags"] = sorted(set(meta["tags"]) | {"loop"})
        ctx.add_output(out, "video", meta)


def _atempo(speed: float) -> str:
    parts = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.4f}")
    return ",".join(parts)


@handler("tool:video.speed", gpu=False, label="Change speed")
def video_speed(ctx: JobContext):
    speed = _num(ctx.params.get("speed"), 0.25, 4.0, 1.0, "Speed")
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    ins = _videos(ctx)
    for i, inp in enumerate(ins):
        ctx.check_cancel()
        info = _probe(inp["path"])
        out = ctx.scratch / f"speed_{i + 1}.mp4"
        args = ["-i", inp["path"], "-filter_complex", f"[0:v]setpts=PTS/{speed:.4f}[v]" + (f";[0:a]{_atempo(speed)}[a]" if info.get("has_audio") else ""),
                "-map", "[v]", *(["-map", "[a]"] if info.get("has_audio") else []), *X264, "-crf", "17", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
        run_ffmpeg(ctx, args, float(info.get("duration") or 0) / speed, f"Speed x{speed:g}: {inp['name']}", i / len(ins), 1 / len(ins))
        meta = _derived_meta(ctx, [inp], i, f"{_base(inp)} · x{speed:g}", {"speed": speed})
        ctx.add_output(out, "video", meta)


@handler("tool:video.reframe", gpu=False, label="Reframe (aspect ratio)")
def video_reframe(ctx: JobContext):
    aspect = str(ctx.params.get("aspect", "9:16"))
    fit = str(ctx.params.get("fit", "crop"))
    if aspect not in ASPECT_RATIOS or fit not in ("crop", "blur", "pad"):
        raise JobError("aspect must be 9:16, 1:1, 16:9, 4:5, 4:3 or 3:4 and fit crop, blur or pad.", "input")
    short = int(_num(ctx.params.get("short_side"), 360, 2160, 1080, "Short side"))
    a, b = ASPECT_RATIOS[aspect]
    W, H = (short * a // b, short) if a >= b else (short, short * b // a)
    W, H = W - W % 2, H - H % 2
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    ins = _videos(ctx)
    for i, inp in enumerate(ins):
        ctx.check_cancel()
        info = _probe(inp["path"])
        if fit == "crop":
            vf = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[v]"
        elif fit == "pad":
            vf = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v]"
        else:   # blurred, enlarged copy behind the whole picture (the usual 16:9 -> 9:16 social treatment)
            vf = (f"[0:v]split[bg][fg];[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:5[b];"
                  f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[f];[b][f]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
        out = ctx.scratch / f"reframe_{i + 1}.mp4"
        args = ["-i", inp["path"], "-filter_complex", vf, "-map", "[v]", *(["-map", "0:a?"] if info.get("has_audio") else []), *X264, "-crf", "17",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
        run_ffmpeg(ctx, args, float(info.get("duration") or 0), f"Reframing {inp['name']}", i / len(ins), 1 / len(ins))
        meta = _derived_meta(ctx, [inp], i, f"{_base(inp)} · {aspect} {fit}", {"aspect": aspect, "fit": fit, "size": [W, H]})
        meta["tags"] = sorted(set(meta["tags"]) | {aspect})
        ctx.add_output(out, "video", meta)


@handler("tool:video.convert", gpu=False, label="Convert video")
def video_convert(ctx: JobContext):
    fmt = str(ctx.params.get("format", "mp4")).lower()
    if fmt not in ("mp4", "webm"):
        raise JobError("format must be mp4 or webm", "input")
    max_side = int(_num(ctx.params.get("max_side"), 0, 7680, 0, "Max side"))
    fps = _num(ctx.params.get("fps"), 0, 120, 0, "Frame rate")
    crf = int(_num(ctx.params.get("crf"), 10, 40, 20, "Quality (CRF)"))
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    ins = _videos(ctx)
    for i, inp in enumerate(ins):
        ctx.check_cancel()
        info = _probe(inp["path"])
        vf = []
        if max_side:
            vf.append(f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'")
        if fps:
            vf.append(f"fps={fps:g}")
        out = ctx.scratch / f"convert_{i + 1}.{fmt}"
        codec = (X264 + ["-crf", str(crf), "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]) if fmt == "mp4" else \
            ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf + 8), "-row-mt", "1", "-pix_fmt", "yuv420p", "-c:a", "libopus", "-b:a", "128k"]
        args = ["-i", inp["path"], *(["-vf", ",".join(vf)] if vf else []), *codec, str(out)]
        run_ffmpeg(ctx, args, float(info.get("duration") or 0), f"Converting {inp['name']}", i / len(ins), 1 / len(ins))
        bits = [fmt] + ([f"{max_side}px"] if max_side else []) + ([f"{fps:g}fps"] if fps else [])
        meta = _derived_meta(ctx, [inp], i, f"{_base(inp)} · {' '.join(bits)}", {"format": fmt, "max_side": max_side, "fps": fps, "crf": crf})
        ctx.add_output(out, "video", meta)


@handler("tool:video.concat", gpu=False, label="Join clips")
def video_concat(ctx: JobContext):
    ins = _videos(ctx, role=None, min_n=2)
    probes = [_probe(i["path"]) for i in ins]
    W = int(probes[0].get("width") or 1280)
    H = int(probes[0].get("height") or 720)
    W, H = W - W % 2, H - H % 2
    fps = float(probes[0].get("fps") or 24)
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    args, parts, labels = [], [], []
    for k, (inp, pr) in enumerate(zip(ins, probes)):
        args += ["-i", inp["path"]]
        parts.append(f"[{k}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps:g},format=yuv420p[v{k}]")
        if pr.get("has_audio"):
            parts.append(f"[{k}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{k}]")
        else:
            parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={float(pr.get('duration') or 0):.3f}[a{k}]")
        labels.append(f"[v{k}][a{k}]")
    fc = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(ins)}:v=1:a=1[v][a]"
    out = ctx.scratch / "joined.mp4"
    total = sum(float(p.get("duration") or 0) for p in probes)
    run_ffmpeg(ctx, args + ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", *X264, "-crf", "17", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)],
               total, f"Joining {len(ins)} clips")
    meta = _derived_meta(ctx, ins, 0, f"{_base(ins[0])} + {len(ins) - 1} more", {"clips": len(ins), "size": [W, H], "fps": fps})
    ctx.add_output(out, "video", meta)


@handler("tool:video.mux", gpu=False, label="Add a soundtrack")
def video_mux(ctx: JobContext):
    allin = ctx.input_assets()
    vids = [i for i in allin if i["type"] == "video"]
    auds = [i for i in allin if i["type"] in ("audio", "voice")]
    if not vids or not auds:
        raise JobError("Pick one video and one audio asset (music, voice-over or foley from the Sound studio).", "input")
    video, audio = vids[0], auds[0]
    volume = _num(ctx.params.get("volume"), 0, 4, 1.0, "Volume")
    keep = bool(ctx.params.get("keep_original", False))
    fade = _num(ctx.params.get("fade_out"), 0, 10, 0.5, "Fade-out")
    info = _probe(video["path"])
    dur = float(info.get("duration") or 0)
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    af = f"[1:a]volume={volume:g},apad" + (f",afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade:g}" if fade and dur else "") + "[m]"
    if keep and info.get("has_audio"):
        fc = af + ";[0:a][m]amix=inputs=2:duration=first:normalize=0[a]"
    else:
        fc = af.replace("[m]", "[a]")
    out = ctx.scratch / "mux.mp4"
    run_ffmpeg(ctx, ["-i", video["path"], "-i", audio["path"], "-filter_complex", fc, "-map", "0:v:0", "-map", "[a]", "-c:v", "copy",
                     "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}" if dur else "3600", "-movflags", "+faststart", str(out)],
               dur, "Muxing the soundtrack")
    meta = _derived_meta(ctx, [video, audio], 0, f"{_base(video)} + {audio.get('name') or 'sound'}", {"volume": volume, "keep_original": keep, "fade_out": fade})
    meta["tags"] = sorted(set(meta["tags"]) | {"soundtrack"})
    ctx.add_output(out, "video", meta)
