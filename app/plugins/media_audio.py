"""Media audio plugin: the dubbing pipeline (handler `media.dub`) and its helper routes.

Dubbing = video or audio in -> dubbed audio (+ remuxed MP4) out. One GPU job, several workers in turn:

  1. separate   Demucs / RoFormer (workers/media_audio_worker.py): vocals + instrumental
  2. transcribe Whisper (same worker, task `subtitles`) on the vocals: sentence-sized lines with timestamps
  3. translate  the prompt LLM (core.textgen: local llama-server by default, any text backend by id). When no text
                backend is available the job ends as a *draft*: the lines are saved for hand translation
  4. speak      a voice_clone model (registry task `voice_clone`, configurable) in the voice of the original speaker
                (a 6-12 s excerpt of the separated vocals, with its transcript), one call per line
  5. fit + mix  each line goes into its slot (silence after it first, then ffmpeg atempo up to `max_speedup`), the
                instrumental is ducked under speech, the mix is loudness-normalised (media.export_audio when present)
  6. outputs    dubbed audio, translated subtitles (SRT + VTT), and for a video input the MP4 with the new audio

`stage: draft` stops after step 3 and saves a `dub plan` (text asset, x_dub = the editable lines) plus the stems; a
later job with `params.segments` (the edited lines) and the stems as inputs (roles `bg` and `vocals`) renders it.

Routes (/api/x/media_audio): GET /dub/options · POST /dub/translate
Pure helpers are unit-tested without a GPU in tests/test_media_audio_dub.py.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import config, licensing, media
from ..jobs import JobCancelled, JobContext, JobError, handler
from ..textgen import TextGenError

router = APIRouter()
CORE = None


def setup(core):
    global CORE
    CORE = core


def _common():
    """workers/media_audio_common.py (pure Python, shared with the worker); loaded by path, it is not a package."""
    name = "media_audio_common"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, config.WORKERS / "media_audio_common.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _common()
MIX_SR = 44100


# ===================================================================== audio helpers (numpy + ffmpeg)
def ffmpeg_bin() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise JobError("ffmpeg is not installed (needed to read video and write the MP4).", "env",
                       "Install ffmpeg with your package manager.")
    return p


def read_audio(path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """File -> float32 [channels, samples] (and resampled to `sr` when given)."""
    import soundfile as sf

    x, s = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.T
    if mono:
        x = x.mean(axis=0, keepdims=True)
    if sr and s != sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), int(s))
        x = resample_poly(x, int(sr) // g, int(s) // g, axis=1).astype(np.float32)
        s = sr
    return x, int(s)


def write_wav(path, x: np.ndarray, sr: int, subtype: str = "PCM_24"):
    import soundfile as sf

    sf.write(str(path), np.clip(x, -1.0, 1.0).T, sr, subtype=subtype, format="WAV")


def trim_silence(x: np.ndarray, sr: int, thr_db: float = -45.0, pad_s: float = 0.03) -> np.ndarray:
    """Cut leading / trailing silence of a mono signal: frames more than 40 dB under the loudest frame (and under
    `thr_db`) count as silence; `pad_s` of margin is kept."""
    hop = max(1, int(0.01 * sr))
    n = len(x) // hop
    if n == 0:
        return x
    e = 20 * np.log10(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-9)
    idx = np.nonzero(e > max(thr_db, float(e.max()) - 40.0))[0]
    if idx.size == 0:
        return x
    a = max(0, int(idx[0] * hop - pad_s * sr))
    b = min(len(x), int((idx[-1] + 1) * hop + pad_s * sr))
    return x[a:b]


def ffmpeg_tempo(x: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time-stretch a mono signal by `rate` (>1 faster) with ffmpeg atempo (pitch is kept)."""
    if abs(rate - 1.0) < 0.01:
        return x
    p = subprocess.run([ffmpeg_bin(), "-v", "error", "-nostdin", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                        "-af", C.atempo_chain(rate), "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"],
                       input=np.ascontiguousarray(x, dtype="<f4").tobytes(), capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise JobError("ffmpeg could not time-fit a line: " + p.stderr.decode("utf8", "replace")[-200:], "internal")
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float32)


def duck_curve(speech: np.ndarray, sr: int, duck_db: float, attack_s: float = 0.08, release_s: float = 0.35) -> np.ndarray:
    """Gain curve (1.0 = untouched) that lowers the background by `duck_db` while the speech track is active."""
    if duck_db <= 0 or speech.size == 0:
        return np.ones(speech.shape[-1], dtype=np.float32)
    mono = np.abs(speech).max(axis=0) if speech.ndim == 2 else np.abs(speech)
    hop = int(0.02 * sr)
    n = max(1, len(mono) // hop)
    env = mono[: n * hop].reshape(n, hop).max(axis=1)
    act = (env > 10 ** (-48 / 20)).astype(np.float32)
    a, r = np.exp(-0.02 / attack_s), np.exp(-0.02 / release_s)
    sm = np.zeros(n, dtype=np.float32)
    v = 0.0
    for i in range(n):
        t = act[i]
        v = t + (v - t) * (a if t > v else r)
        sm[i] = v
    # smooth the 20 ms steps back to sample rate
    g = 10 ** (-duck_db / 20)
    curve = 1.0 - (1.0 - g) * sm
    full = np.interp(np.arange(speech.shape[-1]), (np.arange(n) + 0.5) * hop, curve).astype(np.float32)
    return full


def place_lines(lines: list[tuple[float, np.ndarray]], total_s: float, sr: int) -> np.ndarray:
    n = int(round(total_s * sr))
    out = np.zeros(n, dtype=np.float32)
    for start, x in lines:
        i0 = max(0, int(round(start * sr)))
        if i0 >= n:
            continue
        seg = x[: n - i0]
        out[i0:i0 + len(seg)] += seg
    return out


def soft_limit(x: np.ndarray, ceiling_db: float = -1.0, knee: float = 0.6) -> np.ndarray:
    """Gentle tanh knee above `knee` so a loudness gain does not clip; then scale to the ceiling if needed."""
    c = 10 ** (ceiling_db / 20)
    a = np.abs(x)
    over = a > knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    pk = float(np.abs(y).max())
    return y * (c / pk) if pk > c else y


def normalize_loudness(path: Path, lufs: float, tmp: Path, log) -> tuple[Path, dict]:
    """Loudness-normalise a wav. Uses the core's two-pass loudnorm export (media.export_audio) when it exists, else a
    private gain + soft limiter. Returns (path, report)."""
    exp = getattr(media, "export_audio", None)
    if exp is not None:
        try:
            out, rep = exp(path, tmp, "dub_norm", {"format": "wav", "bit_depth": 24, "lufs": lufs, "true_peak": -1.0})
            return Path(out), {"method": "media.export_audio", **(rep or {})}
        except Exception as e:  # noqa: BLE001 - fall back to the private helper
            log(f"loudnorm export failed ({e}); using the built-in gain")
    x, sr = read_audio(path)
    cur = media.integrated_lufs(x.T, sr)
    if cur is None:
        return path, {"method": "none", "note": "silent mix"}
    gain = max(-24.0, min(24.0, lufs - cur))
    y = soft_limit(x * 10 ** (gain / 20), -1.0)
    out = tmp / "dub_norm.wav"
    tmp.mkdir(parents=True, exist_ok=True)
    write_wav(out, y, sr)
    return out, {"method": "gain+softlimit", "input_lufs": cur, "gain_db": round(gain, 2),
                 "output_lufs": media.integrated_lufs(y.T, sr), "target_lufs": lufs}


# ===================================================================== lines, translation
def parse_segments(raw) -> list[dict]:
    """params.segments (JSON text or list) -> [{id, start, end, text, translation}] validated and ordered."""
    if raw in (None, "", []):
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError as e:
        raise JobError(f"The lines are not valid JSON: {e}", "input") from None
    if isinstance(data, dict):
        data = data.get("segments")
    if not isinstance(data, list) or not data:
        raise JobError("The lines must be a JSON list of {start, end, text, translation}.", "input")
    out = []
    for i, s in enumerate(data[:2000]):
        try:
            a, b = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            raise JobError(f"Line {i + 1} needs numeric start and end (seconds).", "input") from None
        if not (0 <= a < b < 36000):
            raise JobError(f"Line {i + 1}: start/end out of range.", "input")
        out.append({"start": round(a, 3), "end": round(b, 3), "text": str(s.get("text") or "").strip(),
                    "translation": str(s.get("translation") or "").strip(), "skip": bool(s.get("skip"))})
    out.sort(key=lambda s: s["start"])
    for i, s in enumerate(out):
        s["id"] = i
    return out


TRANSLATE_SCHEMA = {"type": "object", "required": ["lines"], "properties": {"lines": {"type": "array", "minItems": 1, "items": {
    "type": "object", "required": ["id", "text"], "properties": {"id": {"type": "integer"}, "text": {"type": "string"}}}}}}


def translation_messages(batch: list[dict], src: str, tgt: str, context: list[str]) -> list[dict]:
    system = (
        f"You are a professional dubbing translator. Translate spoken lines from {src} into {tgt} for voice-over.\n"
        f"- Write natural, speakable {tgt} with the same meaning and tone. Keep names.\n"
        "- Each line has a duration in seconds. The translation will be spoken in that time: keep it about as long when "
        "spoken, and shorten rather than lengthen when unsure.\n"
        "- Spell numbers, symbols and abbreviations out as words, the way they are spoken, so a text-to-speech engine reads "
        "them correctly. No digits, no emoji, no stage directions, no quotation marks that are not in the original.\n"
        "- Translate every line, one output per input id, same ids. Do not merge or split lines.\n"
        'Reply with JSON only: {"lines": [{"id": 0, "text": "..."}]}')
    user = ""
    if context:
        user += "Previous lines of the same speaker (context only, do not translate):\n" + "\n".join(f"- {c}" for c in context[-4:]) + "\n\n"
    user += "Lines to translate:\n" + json.dumps(
        [{"id": s["id"], "dur": round(s["end"] - s["start"], 1), "text": s["text"]} for s in batch], ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def translate_lines(textgen, lines: list[dict], src_name: str, tgt_name: str, backend: str | None, cancel=None,
                    on_batch=None, batch_size: int = 12) -> dict:
    """Fill `translation` of the lines that have none. Returns {"backend": label, "seconds": s}. Raises TextGenError."""
    todo = [s for s in lines if s["text"] and not s["translation"] and not s.get("skip")]
    t0 = time.time()
    label = None
    for b in range(0, len(todo), batch_size):
        batch = todo[b:b + batch_size]
        ids = {s["id"] for s in batch}
        ctx_lines = [s["translation"] or s["text"] for s in lines if s["id"] < batch[0]["id"]][-4:]

        def check(data, ids=ids):
            got = {int(x["id"]) for x in data.get("lines", []) if isinstance(x, dict) and "id" in x}
            errs = [f"missing id {i}" for i in sorted(ids - got)]
            errs += [f"empty text for id {x['id']}" for x in data.get("lines", []) if not str(x.get("text", "")).strip()]
            return errs
        r = textgen.generate(messages=translation_messages(batch, src_name, tgt_name, ctx_lines), schema=TRANSLATE_SCHEMA,
                             backend=backend or None, purpose="improve", temperature=0.3, max_tokens=4000, cancel=cancel,
                             validate=check)
        label = (r.get("backend") or {}).get("label") or label
        by_id = {int(x["id"]): str(x["text"]).strip() for x in r["data"]["lines"] if isinstance(x, dict) and "id" in x}
        for s in batch:
            s["translation"] = by_id.get(s["id"], "")
        if on_batch:
            on_batch(min(b + batch_size, len(todo)), len(todo))
    return {"backend": label, "seconds": round(time.time() - t0, 1), "translated": len(todo)}


def tts_lang_name(code: str) -> str:
    return C.LANG_NAMES.get(code, code)


# ===================================================================== the handler
def _pick(inputs: list[dict], role: str, types: tuple = ()) -> dict | None:
    for i in inputs:
        if i["role"] == role and (not types or i["type"] in types):
            return i
    return None


def _ready(ctx: JobContext, mid: str | None, task: str, what: str) -> dict:
    m = ctx.registry.models.get(mid or "")
    if not m or task not in (m.get("tasks") or []):
        raise JobError(f"No {what} model '{mid}' (task {task}).", "input")
    st = ctx.registry.status.get(mid, {})
    if st.get("status") != "ready":
        raise JobError(f"{m['name']} is not ready: {st.get('reason')}", st.get("kind") or "env", st.get("fix", ""))
    licensing.check(m, ctx.settings)
    return m


def _worker_input(path: Path, asset_id: str, name: str, role: str = "media", type_: str = "audio", meta: dict | None = None) -> dict:  # noqa: A002
    return {"asset_id": asset_id, "role": role, "type": type_, "name": name, "path": str(path), "meta": meta or {}}


@handler("media.dub", gpu=True, label="Dub")
def dub(ctx: JobContext):
    P = ctx.params
    ins = ctx.input_assets()
    src = _pick(ins, "media") or next((i for i in ins if i["type"] in ("video", "audio", "voice")), None)
    if not src:
        raise JobError("Pick the video or audio to dub.", "input")
    is_video = src["type"] == "video"
    src_dur = float((src["meta"].get("info") or {}).get("duration") or 0)
    stage = P.get("stage") or "all"
    tts_id = (P.get("tts_model") or "").strip() or ctx.registry.default_for("voice_clone", settings=ctx.settings)
    tts = _ready(ctx, tts_id, "voice_clone", "voice-clone")
    tgt = C.norm_lang(P.get("target_language"))
    if not tgt:
        raise JobError("Choose the language to dub into (an ISO code such as en, th, ja).", "input")
    langs = C.model_languages(tts)
    if langs and tgt not in langs and stage != "draft":
        raise JobError(f"{tts['name']} cannot speak '{tgt}'. It speaks: {', '.join(langs)}. Pick another voice model.", "input")
    src_lang = C.norm_lang(P.get("source_language"))
    sep_id = (P.get("separate_model") or "").strip() or ctx.registry.default_for("separate", settings=ctx.settings)
    asr_id = ctx.registry.default_for("subtitles", settings=ctx.settings)
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    work = ctx.scratch / "dub"
    work.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    seed = int(P.get("seed", -1))
    seed = seed if seed >= 0 else int(time.time()) % 100000

    lines = parse_segments(P.get("segments"))
    bg_in, voc_in, ref_in = _pick(ins, "bg"), _pick(ins, "vocals"), _pick(ins, "ref")
    stem_assets: dict = {}

    # ---- 1. separation
    ctx.progress(0.01, "Preparing", stage="separating")
    if bg_in and bg_in["type"] in ("audio", "voice"):
        bg_path = Path(bg_in["path"])
        voc_path = Path(voc_in["path"]) if voc_in else None
        stem_assets = {"instrumental": bg_in["asset_id"], "vocals": voc_in["asset_id"] if voc_in else None}
        ctx.log("using the given instrumental" + (" and vocals" if voc_in else ""))
    else:
        _ready(ctx, sep_id, "separate", "separation")
        res = ctx.run_worker(model_id=sep_id, task="separate", params={"stems": "vocals_instrumental", "format": "wav", "shifts": 1, "overlap": 0.25},
                             inputs=[_worker_input(Path(src["path"]), src["asset_id"], src["name"] or "media", "media", src["type"])],
                             frac_range=(0.02, 0.2), out_dir=work / "sep")
        files = {f.get("stem"): Path(f["path"]) for f in res.get("files") or []}
        bg_path, voc_path = files.get("instrumental"), files.get("vocals")
        if not bg_path or not bg_path.is_file():
            raise JobError("The separation produced no instrumental.", "internal")
    ctx.check_cancel()

    # ---- 2. transcription of the vocals
    detected = src_lang
    if not lines:
        _ready(ctx, asr_id, "subtitles", "transcription")
        asr_src = voc_path if voc_path and voc_path.is_file() else Path(src["path"])
        res = ctx.run_worker(model_id=asr_id, task="subtitles",
                             params={"language": src_lang or "auto", "main_format": "srt", "max_chars": 220, "max_lines": 1,
                                     "max_cue_s": 11.0, "pause_s": 0.6, "min_cue_s": 0.4, "split_sentences": True},
                             inputs=[_worker_input(asr_src, src["asset_id"], src["name"] or "media", "media", "audio")],
                             frac_range=(0.2, 0.38), out_dir=work / "asr")
        jf = next((Path(a) if str(a).startswith("/") else Path(res["files"][0]["path"]).parent / a
                   for a in (res["files"][0].get("attach") or []) if str(a).endswith(".json")), None)
        if jf is None or not jf.is_file():
            raise JobError("The transcription produced no timing file.", "internal")
        doc = json.loads(jf.read_text(encoding="utf-8"))
        detected = C.norm_lang(doc.get("source_language")) or src_lang
        lines = [{"id": i, "start": s["start"], "end": s["end"], "text": s["text"].replace("\n", " ").strip(), "translation": "", "skip": False}
                 for i, s in enumerate(doc.get("segments") or []) if s["text"].strip()]
        # drop hallucinations over silence: nothing audible in the vocals under the line
        if voc_path and voc_path.is_file() and lines:
            v, vsr = read_audio(voc_path, 16000, mono=True)
            keep = []
            for s in lines:
                seg = v[0, int(s["start"] * 16000):int(s["end"] * 16000)]
                db = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9) if seg.size else -99
                if db < -50:
                    ctx.log(f"dropped a line with no audible speech ({db:.0f} dB): {s['text'][:50]}")
                    continue
                keep.append(s)
            lines = [{**s, "id": i} for i, s in enumerate(keep)]
        if not lines:
            raise JobError("No speech was found in this file.", "input")
    ctx.check_cancel()
    if detected and detected == tgt and stage != "draft":
        raise JobError(f"The speech is already in '{tgt}'. Choose another dub language.", "input")
    src_name = tts_lang_name(detected) if detected else "the source language"

    # ---- 3. translation
    tr_info: dict = {}
    needs_translation = False
    ctx.progress(0.4, "Translating", stage="translating")
    try:
        if any(s["text"] and not s["translation"] and not s.get("skip") for s in lines):
            tr_info = translate_lines(CORE.textgen, lines, src_name, tts_lang_name(tgt), (P.get("translate_backend") or "").strip() or None,
                                      cancel=ctx.cancel_event,
                                      on_batch=lambda a, b: ctx.progress(0.4 + 0.08 * a / max(b, 1), f"Translating {a}/{b} lines", stage="translating"))
            ctx.log(f"translated {tr_info.get('translated')} line(s) with {tr_info.get('backend')} in {tr_info.get('seconds')} s")
    except TextGenError as e:
        if e.kind == "cancelled":
            raise JobCancelled() from None
        needs_translation = True
        warnings.append(f"Translation unavailable ({e}). Edit the lines by hand, then render.")
        ctx.log(warnings[-1])
    ctx.check_cancel()
    if needs_translation:
        stage = "draft"

    meta_x = {"source_language": detected, "target_language": tgt, "tts_model": tts_id, "separate_model": sep_id,
              "translate": tr_info or None, "warnings": warnings}

    # ---- draft: save the editable plan + stems and stop
    if stage == "draft":
        ctx.progress(0.9, "Saving the draft", stage="saving")
        for label, p in (("vocals", voc_path), ("instrumental", bg_path)):
            if p and Path(p).is_file() and not stem_assets.get(label):
                m = ctx.asset_meta({"name": f"{src['name']} - {label}"}, 0)
                m["parents"] = [src["asset_id"]]
                m["tags"] = sorted({*(m.get("tags") or []), "stem", label})
                m["x_stem"] = label
                stem_assets[label] = ctx.add_output(Path(p), "audio", m)["id"]
        plan = {"version": 1, "media": src["asset_id"], "stems": stem_assets, **meta_x, "segments": lines}
        pf = ctx.scratch / "dub_plan.json"
        pf.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
        m = ctx.asset_meta({"name": f"{src['name']} - dub plan ({tgt})"}, 0)
        m["parents"] = [src["asset_id"]]
        m["x_dub"] = plan
        m["tags"] = sorted({*(m.get("tags") or []), "dub", "draft"})
        m["params"] = {k: v for k, v in (m.get("params") or {}).items() if k != "segments"}
        ctx.add_output(pf, "text", m)
        ctx.job["result_meta"] = {"draft": True, "needs_translation": needs_translation, "lines": len(lines), "stems": stem_assets}
        ctx.progress(1.0, "Draft ready: check the lines, then render", stage="done")
        return

    # ---- 4. voice reference of the original speaker (or the picked voice)
    ctx.progress(0.48, "Preparing the voice reference", stage="speaking")
    if ref_in:
        ref_path = Path(ref_in["path"])
        ref_text = (((ref_in.get("meta") or {}).get("x_voice") or {}).get("ref_text") or "").strip()
        ref_meta = ref_in.get("meta") or {}
        ref_asset = ref_in["asset_id"]
    else:
        if not voc_path or not Path(voc_path).is_file():
            raise JobError("Cloning the original speaker needs the vocals stem (role `vocals`) or a voice to use (role `ref`).", "input")
        a, b, ref_text = C.speech_ref_window([{"start": s["start"], "end": s["end"], "text": s["text"]} for s in lines if s["text"] and not s.get("skip")])
        v, vsr = read_audio(voc_path, 24000, mono=True)
        clip = v[0, int(a * 24000):int(b * 24000)]
        if clip.size < 24000 * 2:
            clip = v[0, : 24000 * 12]
        ref_path = work / "speaker_ref.wav"
        write_wav(ref_path, clip[None, :], 24000, "PCM_16")
        ref_meta = {"x_voice": {"ref_text": ref_text, "language": detected}}
        ref_asset = src["asset_id"]
        ctx.log(f"voice reference: {a:.1f}-{b:.1f} s of the vocals ({len(clip) / 24000:.1f} s)")
    ref_input = _worker_input(ref_path, ref_asset, "original speaker", "ref", "voice",
                              {**ref_meta, "x_voice": {**(ref_meta.get("x_voice") or {}), "ref_text": ref_text}})
    lang_value = C.language_value(tts, tgt)

    # ---- 5. speak every line
    todo = [s for s in lines if s["translation"] and not s.get("skip")]
    if not todo:
        raise JobError("No line has a translation: nothing to speak.", "input")
    audio: dict[int, np.ndarray] = {}
    tts_sr = 24000
    for k, s in enumerate(todo):
        ctx.check_cancel()
        a, b = 0.5 + 0.35 * k / len(todo), 0.5 + 0.35 * (k + 1) / len(todo)
        ctx.progress(a, f"Speaking line {k + 1}/{len(todo)}", stage="speaking")
        slot = s["end"] - s["start"]
        best = None
        for attempt in range(2):
            params = {"text": re.sub(r"\s+", " ", s["translation"]).strip(), "batch": 1, "seed": seed + k * 7 + attempt * 1000, "ref_text": ref_text}
            if lang_value is not None:
                params["language"] = lang_value
            try:
                params = ctx.registry.validate_params(tts_id, params)
            except ValueError as e:
                raise JobError(str(e), "input") from None
            try:
                res = ctx.run_worker(model_id=tts_id, task="voice_clone", params=params, inputs=[ref_input], frac_range=(a, b),
                                     out_dir=work / f"tts_{s['id']}_{attempt}")
            except JobError as e:
                if attempt == 0 and e.kind in ("input", "internal", "timeout"):
                    ctx.log(f"line {k + 1}: {e}; retrying once")
                    continue
                raise
            f = next((x for x in res.get("files") or [] if x.get("type") in ("audio", "voice")), None)
            if not f:
                continue
            x, sr = read_audio(f["path"], None, mono=True)
            x = trim_silence(x[0], sr)
            tts_sr = sr
            dur = len(x) / sr
            if best is None or abs(dur - slot) < abs(len(best) / sr - slot):
                best = x
            if dur <= max(slot * 3.0, slot + 4.0):      # a loop / runaway take is much longer than its slot: try another seed
                break
        if best is None or best.size < 100:
            warnings.append(f"line {k + 1} could not be spoken and was left silent")
            continue
        audio[s["id"]] = best
    if not audio:
        raise JobError("The voice model produced no audio.", "internal")
    ctx.check_cancel()

    # ---- 6. time-fit, mix, loudness
    ctx.progress(0.86, "Fitting lines and mixing", stage="mixing")
    spoken = [s for s in todo if s["id"] in audio]
    bg, bsr = read_audio(bg_path, MIX_SR)
    if bg.shape[0] == 1:
        bg = np.repeat(bg, 2, axis=0)
    total = bg.shape[1] / MIX_SR
    if src_dur:
        total = max(src_dur, min(total, src_dur + 1.0)) if is_video else total
    max_speed = float(P.get("max_speedup") or 1.3)
    plan = C.fit_plan([{"start": s["start"], "end": s["end"]} for s in spoken], [len(audio[s["id"]]) / tts_sr for s in spoken],
                      total=total, max_speedup=max_speed)
    placed = []
    for s, pl in zip(spoken, plan):
        x = audio[s["id"]]
        if tts_sr != MIX_SR:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(MIX_SR, tts_sr)
            x = resample_poly(x, MIX_SR // g, tts_sr // g).astype(np.float32)
        x = ffmpeg_tempo(x, MIX_SR, pl["rate"]) if pl["rate"] > 1.01 else x
        fade = min(len(x) // 2, int(0.015 * MIX_SR))
        if fade > 0:
            x = x.copy()
            x[:fade] *= np.linspace(0, 1, fade)
            x[-fade:] *= np.linspace(1, 0, fade)
        placed.append((pl["start"], x))
        s.update(start_out=pl["start"], end_out=round(pl["start"] + len(x) / MIX_SR, 3), rate=pl["rate"], overflow=pl["overflow"])
        if pl["overflow"] > 0.25:
            warnings.append(f"line {s['id'] + 1} runs {pl['overflow']:.1f} s into the next line (shorten its translation)")
    end_speech = max(s["end_out"] for s in spoken)
    if not is_video:
        total = max(total, end_speech + 0.3)
    speech = place_lines(placed, total, MIX_SR)
    n = len(speech)
    bgm = bg[:, :n] if bg.shape[1] >= n else np.pad(bg, ((0, 0), (0, n - bg.shape[1])))
    gain_bg = 10 ** (float(P.get("bg_gain_db") or 0.0) / 20)
    curve = duck_curve(speech, MIX_SR, float(P.get("duck_db") if P.get("duck_db") is not None else 4.0))
    mix = bgm * gain_bg * curve[None, :] + speech[None, :]
    raw = work / "mix_raw.wav"
    write_wav(raw, mix, MIX_SR, "PCM_24")
    lufs = float(P.get("lufs") if P.get("lufs") is not None else -16.0)
    final, norm = normalize_loudness(raw, lufs, work / "norm", ctx.log)

    # ---- 7. outputs
    ctx.progress(0.92, "Saving", stage="saving")
    tgt_name = tts_lang_name(tgt)
    base_name = src["name"] or "media"
    x_dub = {**meta_x, "media": src["asset_id"], "segments": [{k: s.get(k) for k in ("id", "start", "end", "text", "translation", "skip", "start_out", "end_out", "rate", "overflow")} for s in lines],
             "loudness": norm, "voice_reference": {"from": "ref" if ref_in else "original", "text": ref_text}, "stems": stem_assets,
             "warnings": warnings, "max_speedup": max_speed}
    lic = tts.get("license") or {}

    def derive(name, extra_tags=()):
        m = ctx.asset_meta({"name": name}, 0)
        m["parents"] = [src["asset_id"]] + ([ref_in["asset_id"]] if ref_in else [])
        m["license"] = {**lic, "name": f"{lic.get('name', 'unknown')} ({tts['name']} voice)"} if lic else m.get("license")
        m["tags"] = sorted({*(m.get("tags") or []), "dub", tgt, *extra_tags})
        m["params"] = {k: v for k, v in (m.get("params") or {}).items() if k != "segments"}
        m["model"] = "dub-pipeline"
        m["x_dub"] = x_dub
        return m

    out_wav = ctx.scratch / "dub_audio.wav"
    shutil.copy2(final, out_wav)           # add_output moves its file; the MP4 mux below still needs `final`
    audio_asset = ctx.add_output(out_wav, "audio", derive(f"{base_name} - dub ({tgt_name})"))
    cues = [{"start": s["start_out"], "end": s["end_out"], "text": s["translation"]} for s in spoken]
    for a_, b_ in zip(cues, cues[1:]):     # overrunning lines are logged as warnings; subtitles must not overlap
        a_["end"] = max(a_["start"] + 0.2, min(a_["end"], b_["start"]))
    (ctx.scratch / "dub_subtitles.srt").write_text(C.to_srt(cues), encoding="utf-8")
    (ctx.scratch / "dub_subtitles.vtt").write_text(C.to_vtt(cues), encoding="utf-8")
    ms = derive(f"{base_name} - dub subtitles ({tgt})", ("subtitles",))
    ms["parents"] = [audio_asset["id"]]
    ctx.add_output(ctx.scratch / "dub_subtitles.srt", "text", ms, attach=[ctx.scratch / "dub_subtitles.vtt"])
    video_asset = None
    if is_video:
        ctx.progress(0.95, "Muxing the MP4", stage="muxing")
        mp4 = ctx.scratch / "dub.mp4"
        mux_video(Path(src["path"]), final, mp4, keep_original=bool(P.get("keep_original_track")), lang=tgt)
        mv = derive(f"{base_name} - dub video ({tgt_name})")
        mv["parents"] = [src["asset_id"], audio_asset["id"]]
        video_asset = ctx.add_output(mp4, "video", mv)
    ctx.job["result_meta"] = {"audio": audio_asset["id"], "video": video_asset["id"] if video_asset else None, "lines": len(spoken),
                              "warnings": warnings, "loudness": norm, "translate": tr_info or None}
    ctx.progress(1.0, "Done", stage="done")


def mux_video(video: Path, audio: Path, out: Path, keep_original: bool = False, lang: str = "und"):
    """Replace the audio track of `video` with `audio` (video stream copied; re-encoded to H.264 when the container refuses)."""
    iso3 = {"en": "eng", "th": "tha", "ja": "jpn", "zh": "zho", "ko": "kor", "fr": "fra", "de": "deu", "es": "spa", "it": "ita",
            "ru": "rus", "pt": "por", "vi": "vie", "id": "ind"}.get(lang, "und")
    base = [ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-protocol_whitelist", "file", "-i", str(video), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0"]
    if keep_original:
        base += ["-map", "0:a:0?"]
    tail = ["-c:a", "aac", "-b:a", "192k", "-metadata:s:a:0", f"language={iso3}", "-shortest", "-movflags", "+faststart", str(out)]
    for vcodec in (["-c:v", "copy"], ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]):
        r = subprocess.run(base + vcodec + tail, capture_output=True, text=True)
        if r.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            return
    raise JobError("ffmpeg could not write the MP4: " + " ".join(r.stderr.strip().splitlines()[-2:])[:300], "internal")


# ===================================================================== routes
@router.get("/dub/options")
def dub_options():
    """Voice-clone models usable for dubbing, with the languages each can speak, and the text backends for translation."""
    reg = CORE.registry
    out = []
    for m in reg.models.values():
        if "voice_clone" not in (m.get("tasks") or []) or m.get("test") or m.get("backend") == "cloud":
            continue
        st = reg.status.get(m["id"], {})
        out.append({"id": m["id"], "name": m["name"], "status": st.get("status"), "reason": st.get("reason"), "fix": st.get("fix"),
                    "languages": C.model_languages(m), "license": m.get("license"), "order": m.get("order", 100),
                    "thai": (m.get("features") or {}).get("thai")})
    out.sort(key=lambda m: (m["status"] != "ready", m["order"], m["name"]))
    try:
        tb = CORE.textgen.backends(check_auth=False)
    except Exception:  # noqa: BLE001
        tb = []
    return {"tts_models": out, "default_tts": reg.default_for("voice_clone"),
            "language_names": C.LANG_NAMES,
            "translate_backends": [{"id": b.get("id"), "label": b.get("label"), "available": b.get("available"), "reason": b.get("reason"),
                                    "cloud": b.get("kind") in ("api",)} for b in tb],
            "tasks_ready": {t: [mid for mid, mm in reg.models.items() if t in (mm.get("tasks") or []) and reg.status.get(mid, {}).get("status") == "ready"]
                            for t in ("subtitles", "separate", "dub")}}


class TranslateReq(BaseModel):
    segments: list[dict] = Field(default_factory=list)
    source_language: str | None = None
    target_language: str
    backend: str | None = None
    only_missing: bool = True


@router.post("/dub/translate")
def dub_translate(req: TranslateReq):
    """Translate (or re-translate) the lines of a dub draft with the chosen text backend; returns the lines."""
    tgt = C.norm_lang(req.target_language)
    if not tgt:
        raise HTTPException(400, "Unknown target language.")
    lines = parse_segments([{**s, "translation": (s.get("translation") if req.only_missing else "")} for s in req.segments])
    if not lines:
        raise HTTPException(400, "No lines.")
    src = C.norm_lang(req.source_language)
    try:
        info = translate_lines(CORE.textgen, lines, tts_lang_name(src) if src else "the source language", tts_lang_name(tgt), req.backend or None)
    except TextGenError as e:
        raise HTTPException(503, detail={"error": str(e), "kind": "improve", "fix": e.fix or ""}) from None
    return {"segments": lines, **info}
