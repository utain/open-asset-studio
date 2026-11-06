"""Sound studio plugin: post tools (CPU), QA handler, voice library, sheets, engine-ready export.

CPU tools (`POST /api/jobs {"tool": "...", "inputs": [{asset}], "params": {...}}`); each input -> one derived asset
with lineage (parents = [input]), the input's task / licence / prompt carried over, and `x_post` describing the ops:

  tool:audio.process     chain of ops: {trim, loop, fade, sample_rate, channels, normalize, format} (order fixed below)
  tool:audio.trim        {threshold_db=-50, pad_ms=30}
  tool:audio.fade        {in_ms=10, out_ms=200}
  tool:audio.normalize   {lufs=-16, ceiling_db=-1}
  tool:audio.loop        {xfade_ms=500}                  seamless loop: the tail is crossfaded into the head
  tool:audio.convert     {format: wav|ogg|mp3, bit_depth: 16|24, quality: 0..1}
  tool:audio.channels    {channels: mono|stereo}
  tool:audio.resample    {sample_rate: 22050|32000|44100|48000}
  tool:audio.make_voice  {name, ref_text, language, design_prompt} -> a `voice` asset (clone reference)

GPU handler `sound.qa` (registry models clap-htsat / whisper-v3-turbo): runs the scorer on the input takes and writes
`x_qa: {clap: {...}, asr: {...}}` onto each take (asset update event -> cards re-render).

Routes (/api/x/sound): GET /voices · GET|PUT /sheets · POST /export · GET /export/{name}
"""
from __future__ import annotations

import json
import math
import re
import secrets
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import config, media
from ..assets import check_project, slug
from ..jobs import JobContext, JobError, handler

router = APIRouter()
CORE = None
SR_CHOICES = (16000, 22050, 24000, 32000, 44100, 48000)
FORMATS = {"wav": ".wav", "ogg": ".ogg", "mp3": ".mp3", "flac": ".flac"}
_SHEET_LOCK = threading.Lock()


def setup(core):
    global CORE
    CORE = core


# ===================================================================== DSP (pure numpy; tests/test_sound_dsp.py)
def as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x[:, None] if x.ndim == 1 else x


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -50.0, pad_ms: float = 30.0) -> np.ndarray:
    """Cut leading and trailing parts quieter than threshold_db (peak, any channel), keeping pad_ms around."""
    x = as2d(x)
    env = np.abs(x).max(axis=1)
    idx = np.flatnonzero(env > 10 ** (threshold_db / 20))
    if idx.size == 0:
        return x[: max(1, int(0.05 * sr))] * 0
    pad = int(sr * pad_ms / 1000)
    return x[max(0, idx[0] - pad): min(len(x), idx[-1] + 1 + pad)]


def fade(x: np.ndarray, sr: int, in_ms: float = 0.0, out_ms: float = 0.0) -> np.ndarray:
    """Equal-power (sine) fades in and out."""
    x = as2d(x).copy()
    n = len(x)
    a = min(n, int(sr * in_ms / 1000))
    b = min(n, int(sr * out_ms / 1000))
    if a > 1:
        x[:a] *= np.sin(np.linspace(0, np.pi / 2, a, dtype=np.float32))[:, None]
    if b > 1:
        x[n - b:] *= np.cos(np.linspace(0, np.pi / 2, b, dtype=np.float32))[:, None]
    return x


def make_loop(x: np.ndarray, sr: int, xfade_ms: float = 500.0) -> tuple[np.ndarray, int]:
    """Seamless loop: the last L samples are crossfaded (equal power) into the first L, the tail is dropped.
    The result ends on the sample just before the original tail, so end -> start is continuous. Returns (y, L)."""
    x = as2d(x)
    n = len(x)
    L = int(sr * xfade_ms / 1000)
    L = max(0, min(L, n // 3))
    if L < 2:
        return x.copy(), 0
    t = np.linspace(0, np.pi / 2, L, dtype=np.float32)[:, None]
    y = x[: n - L].copy()
    y[:L] = x[n - L:] * np.cos(t) + x[:L] * np.sin(t)
    return y, L


def lufs_of(x: np.ndarray, sr: int) -> float | None:
    return media.integrated_lufs(as2d(x), sr)


def limit(x: np.ndarray, sr: int, ceiling_db: float = -1.0, release_ms: float = 60.0) -> np.ndarray:
    """Peak limiter: per-sample gain = min(1, ceiling / peak in a 3 ms look-around window), released smoothly."""
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import lfilter

    x = as2d(x)
    ceil = 10 ** (ceiling_db / 20)
    env = np.abs(x).max(axis=1)
    if env.max() <= ceil:
        return x
    win = max(1, int(sr * 0.003))
    peak = maximum_filter1d(env, size=2 * win + 1)
    g = np.minimum(1.0, ceil / np.maximum(peak, 1e-9))
    # instant attack, exponential release: smooth (1-g) with a one-pole and keep the larger reduction
    a = math.exp(-1.0 / max(1.0, sr * release_ms / 1000))
    red = 1.0 - g
    sm = lfilter([1 - a], [1, -a], red)
    g2 = 1.0 - np.maximum(red, sm)
    y = x * g2[:, None].astype(np.float32)
    return np.clip(y, -ceil, ceil)


def normalize_lufs(x: np.ndarray, sr: int, target: float = -16.0, ceiling_db: float = -1.0) -> tuple[np.ndarray, dict]:
    """Gain to the target integrated loudness, then a peak limiter at ceiling_db. Short clips (<0.4 s) fall back to
    RMS loudness. Returns (y, report)."""
    x = as2d(x)
    cur = lufs_of(x, sr)
    if cur is None:
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms <= 1e-9:
            return x.copy(), {"lufs_in": None, "gain_db": 0.0, "note": "silent"}
        cur = 20 * math.log10(rms) - 0.7
    gain_db = float(np.clip(target - cur, -40, 40))
    y = limit(x * 10 ** (gain_db / 20), sr, ceiling_db)
    return y, {"lufs_in": round(cur, 2), "gain_db": round(gain_db, 2), "lufs_out": lufs_of(y, sr)}


def resample(x: np.ndarray, sr: int, new_sr: int) -> np.ndarray:
    from scipy.signal import resample_poly

    x = as2d(x)
    if sr == new_sr:
        return x
    g = math.gcd(int(sr), int(new_sr))
    return resample_poly(x, int(new_sr) // g, int(sr) // g, axis=0).astype(np.float32)


def set_channels(x: np.ndarray, mode: str | None) -> np.ndarray:
    x = as2d(x)
    if mode == "mono" and x.shape[1] > 1:
        return x.mean(axis=1, keepdims=True)
    if mode == "stereo" and x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x


def write_audio(path: Path, x: np.ndarray, sr: int, fmt: str = "wav", bit_depth: int = 16, quality: float = 0.6) -> Path:
    import soundfile as sf

    x = np.clip(as2d(x), -1.0, 1.0)
    if x.shape[1] == 1:
        x = x[:, 0]
    if fmt == "wav":
        sf.write(str(path), x, sr, format="WAV", subtype="PCM_24" if int(bit_depth) == 24 else "PCM_16")
    elif fmt == "flac":
        sf.write(str(path), x, sr, format="FLAC", subtype="PCM_24" if int(bit_depth) == 24 else "PCM_16")
    elif fmt == "ogg":
        if sr not in (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000):
            raise JobError(f"OGG Vorbis cannot use {sr} Hz; resample first.", "input")
        sf.write(str(path), x, sr, format="OGG", subtype="VORBIS", compression_level=float(np.clip(1 - quality, 0, 1)))
    elif fmt == "mp3":
        if sr not in (16000, 22050, 24000, 32000, 44100, 48000):
            raise JobError(f"MP3 cannot use {sr} Hz; resample first.", "input")
        sf.write(str(path), x, sr, format="MP3", subtype="MPEG_LAYER_III", compression_level=float(np.clip(1 - quality, 0, 1)),
                 bitrate_mode="VARIABLE")
    else:
        raise JobError(f"Unknown format '{fmt}' (wav, ogg, mp3, flac).", "input")
    return path


def read_audio(path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, int(sr)


OP_ORDER = ("trim", "loop", "fade", "sample_rate", "channels", "normalize")


def process(x: np.ndarray, sr: int, ops: dict) -> tuple[np.ndarray, int, list[str], dict]:
    """Apply the ops in OP_ORDER. Returns (y, sr, short labels, report)."""
    labels, rep = [], {}
    y = as2d(x)
    if ops.get("trim") is not None:
        o = ops["trim"] or {}
        n0 = len(y)
        y = trim_silence(y, sr, float(o.get("threshold_db", -50)), float(o.get("pad_ms", 30)))
        rep["trimmed_s"] = round((n0 - len(y)) / sr, 3)
        labels.append("trim")
    if ops.get("loop") is not None:
        o = ops["loop"] or {}
        y, L = make_loop(y, sr, float(o.get("xfade_ms", 500)))
        rep["loop_xfade_s"] = round(L / sr, 3)
        labels.append("loop")
    if ops.get("fade") is not None and ops.get("loop") is None:
        o = ops["fade"] or {}
        y = fade(y, sr, float(o.get("in_ms", 10)), float(o.get("out_ms", 200)))
        labels.append("fade")
    if ops.get("sample_rate"):
        nsr = int(ops["sample_rate"])
        if nsr not in SR_CHOICES:
            raise JobError(f"Sample rate must be one of {', '.join(map(str, SR_CHOICES))}.", "input")
        if nsr != sr:
            y = resample(y, sr, nsr)
            sr = nsr
            labels.append(f"{nsr // 1000 if nsr % 1000 == 0 else nsr / 1000}k")
    if ops.get("channels") in ("mono", "stereo"):
        before = y.shape[1]
        y = set_channels(y, ops["channels"])
        if y.shape[1] != before:
            labels.append(ops["channels"])
    if ops.get("normalize") is not None:
        o = ops["normalize"] or {}
        y, r = normalize_lufs(y, sr, float(o.get("lufs", -16)), float(o.get("ceiling_db", -1)))
        rep["normalize"] = r
        labels.append(f"{float(o.get('lufs', -16)):g} LUFS")
    return y, sr, labels, rep


# ===================================================================== post tools
def _inputs(ctx: JobContext) -> list[dict]:
    inputs = [i for i in ctx.input_assets() if i["type"] in ("audio", "voice")]
    if not inputs:
        raise JobError("Pick at least one audio take.", "input")
    return inputs


def _derived_meta(ctx: JobContext, inp: dict, i: int, labels: list[str], ops: dict, rep: dict, fmt: str) -> dict:
    parent = ctx.assets.get(inp["asset_id"])
    base = re.sub(r"\s+·\s+[^·]*$", "", parent.get("name") or inp["name"] or "audio") if parent.get("x_post") else (parent.get("name") or "audio")
    meta = ctx.asset_meta({"name": f"{base} · {' '.join(labels) or fmt}"}, i)
    meta["parents"] = [inp["asset_id"]]
    meta["source"] = "tool"
    # a processed take belongs where its parent lives: same task / studio / prompt / licence (NC stays NC)
    for k in ("task", "studio", "prompt", "improved_prompt", "final_prompt", "negative", "license", "model", "model_name", "seed"):
        if parent.get(k) is not None:
            meta[k] = parent[k]
    meta["batch"] = {"id": ctx.job["id"], "index": i}
    for k, v in parent.items():
        if k.startswith("x_") and k not in ("x_qa", "x_post"):
            meta[k] = v
    meta["x_post"] = {"ops": ops, "labels": labels, "report": rep, "from": inp["asset_id"], "format": fmt,
                      "tool": ctx.job.get("tool")}
    meta["tags"] = sorted(set(parent.get("tags") or []) | set(meta.get("tags") or []))
    return meta


def _run_process(ctx: JobContext, ops: dict, fmt: str = "wav", bit_depth: int = 16, quality: float = 0.6):
    fmt = (fmt or "wav").lower()
    if fmt not in FORMATS:
        raise JobError("format must be wav, ogg, mp3 or flac", "input")
    inputs = _inputs(ctx)
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    for i, inp in enumerate(inputs):
        ctx.check_cancel()
        ctx.progress(i / len(inputs), f"Processing {inp['name']}", stage="processing")
        x, sr = read_audio(inp["path"])
        y, sr2, labels, rep = process(x, sr, ops)
        if fmt != "wav" or not labels:
            labels.append(fmt)
        out = ctx.scratch / f"post_{i + 1}{FORMATS[fmt]}"
        write_audio(out, y, sr2, fmt, bit_depth, quality)
        meta = _derived_meta(ctx, inp, i, labels, ops, rep, fmt)
        a = ctx.add_output(out, "voice" if inp["type"] == "voice" else "audio", meta)
        ctx.log(f"{inp['name']} -> {a['name']}" + (f" ({rep['normalize']['lufs_in']} -> {rep['normalize']['lufs_out']} LUFS)" if rep.get("normalize") else ""))


def _ops_from_params(p: dict) -> dict:
    """tool:audio.process params: {ops: {...}} or flat keys (trim, fade_in_ms, lufs, ...)."""
    if isinstance(p.get("ops"), dict):
        return p["ops"]
    ops = {}
    if p.get("trim"):
        ops["trim"] = {"threshold_db": p.get("threshold_db", -50), "pad_ms": p.get("pad_ms", 30)}
    if p.get("fade"):
        ops["fade"] = {"in_ms": p.get("fade_in_ms", 10), "out_ms": p.get("fade_out_ms", 200)}
    if p.get("loop"):
        ops["loop"] = {"xfade_ms": p.get("xfade_ms", 500)}
    if p.get("normalize"):
        ops["normalize"] = {"lufs": p.get("lufs", -16), "ceiling_db": p.get("ceiling_db", -1)}
    if p.get("sample_rate"):
        ops["sample_rate"] = int(p["sample_rate"])
    if p.get("channels") in ("mono", "stereo"):
        ops["channels"] = p["channels"]
    return ops


def _fmt_args(p: dict) -> dict:
    return {"fmt": p.get("format", "wav"), "bit_depth": int(p.get("bit_depth", 16)), "quality": float(p.get("quality", 0.6))}


@handler("tool:audio.process", gpu=False, label="Process audio (chain)")
def t_process(ctx: JobContext):
    ops = _ops_from_params(ctx.params)
    if not ops and ctx.params.get("format", "wav") == "wav":
        raise JobError("Choose at least one operation.", "input")
    _run_process(ctx, ops, **_fmt_args(ctx.params))


@handler("tool:audio.trim", gpu=False, label="Trim silence")
def t_trim(ctx: JobContext):
    p = ctx.params
    _run_process(ctx, {"trim": {"threshold_db": p.get("threshold_db", -50), "pad_ms": p.get("pad_ms", 30)}}, **_fmt_args(p))


@handler("tool:audio.fade", gpu=False, label="Fade in / out")
def t_fade(ctx: JobContext):
    p = ctx.params
    _run_process(ctx, {"fade": {"in_ms": p.get("in_ms", 10), "out_ms": p.get("out_ms", 200)}}, **_fmt_args(p))


@handler("tool:audio.normalize", gpu=False, label="Normalize loudness")
def t_norm(ctx: JobContext):
    p = ctx.params
    _run_process(ctx, {"normalize": {"lufs": p.get("lufs", -16), "ceiling_db": p.get("ceiling_db", -1)}}, **_fmt_args(p))


@handler("tool:audio.loop", gpu=False, label="Seamless loop")
def t_loop(ctx: JobContext):
    p = ctx.params
    _run_process(ctx, {"loop": {"xfade_ms": p.get("xfade_ms", 500)}}, **_fmt_args(p))


@handler("tool:audio.convert", gpu=False, label="Convert format")
def t_convert(ctx: JobContext):
    _run_process(ctx, {}, **_fmt_args(ctx.params))


@handler("tool:audio.channels", gpu=False, label="Mono / stereo")
def t_channels(ctx: JobContext):
    p = ctx.params
    if p.get("channels") not in ("mono", "stereo"):
        raise JobError("channels must be mono or stereo", "input")
    _run_process(ctx, {"channels": p["channels"]}, **_fmt_args(p))


@handler("tool:audio.resample", gpu=False, label="Resample")
def t_resample(ctx: JobContext):
    p = ctx.params
    _run_process(ctx, {"sample_rate": int(p.get("sample_rate", 44100))}, **_fmt_args(p))


@handler("tool:audio.make_voice", gpu=False, label="Save as voice")
def t_make_voice(ctx: JobContext):
    """Copy a take into the voice library: trimmed, -20 LUFS, mono, with its transcript (the clone reference)."""
    p = ctx.params
    inp = _inputs(ctx)[0]
    parent = ctx.assets.get(inp["asset_id"])
    x, sr = read_audio(inp["path"])
    y, sr, _labels, rep = process(x, sr, {"trim": {"threshold_db": -50, "pad_ms": 60}, "channels": "mono",
                                          "normalize": {"lufs": -20, "ceiling_db": -1}})
    if len(y) / sr > 30:
        y = y[: sr * 30]
        ctx.log("reference clip cut to 30 s (cloners use the first 10-20 s)")
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    out = ctx.scratch / "voice.wav"
    write_audio(out, y, sr, "wav", 24)
    old = parent.get("x_voice") or {}
    pw = parent.get("params") or {}
    ref_text = (p.get("ref_text") or old.get("ref_text") or pw.get("text") or "").strip()
    meta = _derived_meta(ctx, inp, 0, ["voice"], {"make_voice": True}, rep, "wav")
    meta["name"] = (p.get("name") or "").strip()[:80] or f"Voice · {parent.get('name') or 'reference'}"
    meta["x_voice"] = {"ref_text": ref_text, "language": p.get("language") or old.get("language") or pw.get("language"),
                       "design_prompt": p.get("design_prompt") or old.get("design_prompt") or pw.get("instruction") or pw.get("instruct"),
                       "character": p.get("character") or old.get("character"), "from": inp["asset_id"]}
    meta["task"] = parent.get("task") if parent.get("task") in ("voice_design", "voice_clone") else "voice_design"
    meta["tags"] = sorted(set(meta.get("tags") or []) | {"voice"})
    ctx.add_output(out, "voice", meta)


# ===================================================================== QA (GPU, via the registry models)
@handler("sound.qa", gpu=True, label="Audio QA")
def qa(ctx: JobContext):
    """Run CLAP or Whisper on the input takes; write the scores onto each take as x_qa."""
    res = ctx.run_worker(frac_range=(0.0, 0.95))
    meta = res.get("meta") or {}
    kind = meta.get("kind") or ("clap" if ctx.job["task"] == "qa_clap" else "asr")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    flagged = 0
    for aid, sc in (meta.get("scores") or {}).items():
        try:
            a = ctx.assets.get(aid)
        except Exception:  # noqa: BLE001 - deleted meanwhile
            continue
        q = dict(a.get("x_qa") or {})
        entry = {**sc, "model": ctx.job.get("model"), "at": stamp}
        if kind == "asr" and sc.get("recall") is not None:
            entry["flag"] = sc["recall"] < 0.8
            flagged += bool(entry["flag"])
        if kind == "clap" and sc.get("clap") is not None:
            entry["flag"] = sc["clap"] < max(sc.get("clap_silence", -1), sc.get("clap_speech", -1)) + 0.05
            flagged += bool(entry["flag"])
        q[kind] = entry
        ctx.assets.update(aid, {"x_qa": q})
    ctx.job["result_meta"] = {"kind": kind, "scores": meta.get("scores"), "flagged": flagged}
    ctx.progress(1.0, f"{len(meta.get('scores') or {})} take(s) scored, {flagged} flagged", stage="done")


# ===================================================================== routes
def _store():
    if CORE is None:
        raise HTTPException(503, "sound plugin not initialised")
    return CORE.assets


@router.get("/voices")
def voices(project: str = "default"):
    """Voice library of a project: voice assets (clone references) newest first."""
    check_project(project)
    r = _store().list(project, type="voice", sort="new", limit=500)
    return {"total": r["total"], "items": r["items"]}


def _sheet_path(project: str) -> Path:
    check_project(project)
    return config.DATA / "sound" / project / "sheets.json"


@router.get("/sheets")
def sheets_get(project: str = "default"):
    p = _sheet_path(project)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {"sheets": {}}


class SheetsReq(BaseModel):
    project: str = "default"
    sheets: dict = Field(default_factory=dict)


@router.put("/sheets")
def sheets_put(req: SheetsReq):
    p = _sheet_path(req.project)
    data = {"sheets": req.sheets, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    raw = json.dumps(data, ensure_ascii=False)
    if len(raw) > 4 * 2**20:
        raise HTTPException(413, "Sheets are too large (4 MB max).")
    with _SHEET_LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(raw)
        tmp.replace(p)
    return {"ok": True, "updated": data["updated"]}


# ------------------------------------------------------------------ export
CATEGORY = {"sfx": "sfx", "ambience": "ambience", "music": "music", "song": "music", "sing": "music", "tts": "voice",
            "voice_clone": "voice", "voice_design": "voice", "voice_convert": "voice", "s2s": "voice", "v2a": "foley",
            "test": "sfx"}


def event_of(a: dict) -> str:
    """Event key for file names: x_event (event list / sheet row) > the take's name without ' #n' / tool suffix."""
    sh = a.get("x_sheet") or {}
    ev = a.get("x_event") or sh.get("event")
    if not ev:
        name = re.sub(r"\s+#\d+$", "", re.sub(r"\s+·\s+.*$", "", str(a.get("name") or a.get("id"))))
        ev = name
    return slug(ev, 48)


def plan_export(metas: list[dict], fmt: str = "ogg", by_character: bool = True) -> list[dict]:
    """Deterministic file plan: <category>[/<character>]/<event>_<take:02d>.<ext>. Takes are numbered per event in
    the order (rating desc, favourite, created). Pure function (tests/test_sound_export.py)."""
    ext = FORMATS.get(fmt, ".ogg")
    groups: dict[tuple, list[dict]] = {}
    for a in metas:
        cat = "voices" if a.get("type") == "voice" else CATEGORY.get(a.get("task") or "", "audio")
        sub = ""
        ch = (a.get("x_sheet") or {}).get("character") or (a.get("x_voice") or {}).get("character")
        if by_character and cat in ("voice", "voices") and ch:
            sub = slug(ch, 32)
        groups.setdefault((cat, sub, event_of(a)), []).append(a)
    plan = []
    for (cat, sub, ev), items in sorted(groups.items()):
        items.sort(key=lambda a: (-(a.get("rating") or 0), not a.get("favorite"), a.get("created") or "", a["id"]))
        for k, a in enumerate(items, 1):
            folder = f"{cat}/{sub}" if sub else cat
            plan.append({"asset": a, "category": cat, "character": sub or None, "event": ev, "take": k,
                         "file": f"{folder}/{ev}_{k:02d}{ext}"})
    return plan


class ExportReq(BaseModel):
    ids: list[str]
    format: str = "ogg"
    folder: str = ""
    sample_rate: int | None = None
    channels: str | None = None
    normalize_lufs: float | None = None
    bit_depth: int = 16
    by_character: bool = True


@router.post("/export")
def export(req: ExportReq):
    """Engine-ready folder (zip): <folder>/<category>/<event>_<take>.<ext> + manifest.json (+ licences summary)."""
    store = _store()
    if not req.ids:
        raise HTTPException(400, "Select at least one take to export.")
    if len(req.ids) > 2000:
        raise HTTPException(400, "Too many assets in one export (2000 max).")
    fmt = req.format.lower()
    if fmt not in FORMATS:
        raise HTTPException(400, "format must be ogg, wav, mp3 or flac")
    metas = [store.get(i) for i in dict.fromkeys(req.ids)]
    metas = [m for m in metas if m.get("type") in ("audio", "voice")]
    if not metas:
        raise HTTPException(400, "No audio among the selected assets.")
    project = metas[0]["project"]
    folder = slug(req.folder or f"{project}_sound", 40)
    plan = plan_export(metas, fmt, req.by_character)
    out_dir = config.DATA / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sound-export-*.zip"):
        if time.time() - old.stat().st_mtime > 3600:
            old.unlink(missing_ok=True)
    name = f"sound-export-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}.zip"
    zpath = out_dir / name
    manifest = {"tool": "Open Asset Studio · Sound", "exported": time.strftime("%Y-%m-%dT%H:%M:%S"), "project": project,
                "format": fmt, "sample_rate": req.sample_rate, "channels": req.channels, "normalize_lufs": req.normalize_lufs,
                "naming": "<category>/[<character>/]<event>_<take>.<ext>", "items": [], "licenses": {}}
    ops = {}
    if req.sample_rate:
        ops["sample_rate"] = int(req.sample_rate)
    if req.channels in ("mono", "stereo"):
        ops["channels"] = req.channels
    if req.normalize_lufs is not None:
        ops["normalize"] = {"lufs": float(req.normalize_lufs), "ceiling_db": -1.0}
    with tempfile.TemporaryDirectory(dir=out_dir) as td, zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for k, item in enumerate(plan):
            a = item["asset"]
            src = store.file_path(a["project"], a["id"], a["main"])
            x, sr = read_audio(src)
            y, sr2, _l, _r = process(x, sr, ops)
            tmp = Path(td) / f"{k}{FORMATS[fmt]}"
            try:
                write_audio(tmp, y, sr2, fmt, req.bit_depth)
            except JobError as e:
                raise HTTPException(400, str(e)) from None
            z.write(tmp, f"{folder}/{item['file']}")
            info, _wf = media.analyze_audio(tmp, buckets=10)
            lic = a.get("license") or {}
            manifest["items"].append({
                "file": item["file"], "category": item["category"], "character": item["character"], "event": item["event"],
                "take": item["take"], "asset_id": a["id"], "name": a.get("name"), "model": a.get("model_name") or a.get("model"),
                "license": lic.get("name"), "commercial": lic.get("commercial"), "prompt": a.get("final_prompt") or a.get("prompt"),
                "line": (a.get("x_sheet") or {}).get("line") or (a.get("params") or {}).get("text"), "seed": a.get("seed"),
                "rating": a.get("rating"), "duration": info.get("duration"), "sample_rate": info.get("sample_rate"),
                "channels": info.get("channels"), "lufs": info.get("lufs"), "peak_db": info.get("peak_db"),
                "loop": bool((a.get("x_post") or {}).get("report", {}).get("loop_xfade_s")), "qa": a.get("x_qa")})
            if lic.get("name"):
                L = manifest["licenses"].setdefault(lic["name"], {"commercial": lic.get("commercial"), "note": lic.get("note"), "files": 0})
                L["files"] += 1
        nc = [n for n, v in manifest["licenses"].items() if not v["commercial"]]
        if nc:
            manifest["warning"] = f"Contains non-commercial material ({', '.join(nc)}): not for a shipped product."
        z.writestr(f"{folder}/manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
    return {"url": f"/api/x/sound/export/{name}", "name": name, "bytes": zpath.stat().st_size, "count": len(plan),
            "folder": folder, "warning": manifest.get("warning"), "files": [i["file"] for i in plan][:50]}


@router.get("/export/{name}")
def export_get(name: str):
    if not re.fullmatch(r"sound-export-\d{8}-\d{6}-[0-9a-f]{6}\.zip", name):
        raise HTTPException(404)
    p = config.DATA / "exports" / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/zip", filename=name)
