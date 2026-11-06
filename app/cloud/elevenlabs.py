"""ElevenLabs (xi-api-key): sound effects, text to speech, voice design, music. All synchronous (audio bytes back).

POST /sound-generation?output_format=mp3_44100_128   {text, duration_seconds, prompt_influence, loop, model_id}
POST /text-to-speech/{voice_id}?output_format=...     {text, model_id, language_code?}
POST /text-to-voice/design                            {voice_description, model_id, text | auto_generate_text, seed?}
     -> previews[] {audio_base_64, generated_voice_id, duration_secs}   (saved as `voice` assets)
POST /music                                           {prompt, music_length_ms, model_id, force_instrumental}
API usage is billed in USD per 1k characters / per minute: the registry estimate is recorded.
"""
from __future__ import annotations

import base64

from .base import CloudRun
from .common import batch_count
from ..providers import ProviderError

FMT = "mp3_44100_128"


def _cost_header(r: CloudRun, resp):
    for h in ("character-cost", "x-character-count"):
        if resp.headers.get(h):
            r.log(f"ElevenLabs {h}: {resp.headers[h]}")
    rid = resp.headers.get("request-id") or resp.headers.get("x-request-id")
    if rid:
        r.request_ids.append(rid)


def sfx(r: CloudRun):
    p = r.params
    n = batch_count(r)
    for i in range(n):
        r.check()
        body = {"text": r.prompt, "model_id": r.remote(), "prompt_influence": float(p.get("prompt_influence", 0.3))}
        if p.get("seconds"):
            body["duration_seconds"] = max(0.5, min(30.0, float(p["seconds"])))
        if p.get("loop") or r.task == "ambience":
            body["loop"] = bool(p.get("loop", True))
        r.progress(i / n + 0.05, f"Take {i + 1}/{n} on ElevenLabs", stage="generating", interruptible=False)
        r.submitted = True
        resp = r.request("POST", r.url(f"/sound-generation?output_format={FMT}"), json=body, timeout=180)
        _cost_header(r, resp)
        r.add_audio_bytes(resp.content, f"el_sfx_{i + 1}", ".mp3", index=i)


def tts(r: CloudRun):
    p = r.params
    text = str(p.get("text") or r.prompt)
    if not text.strip():
        raise ProviderError("Write the line to speak.", "input")
    voice = p.get("voice") or "JBFqnCBsd6RMkjVDRZzb"
    model = p.get("tts_model") or r.remote()
    n = batch_count(r)
    for i in range(n):
        r.check()
        body = {"text": text, "model_id": model}
        if p.get("language") and p["language"] != "auto":
            body["language_code"] = p["language"]
        if int(p.get("seed", -1)) >= 0:
            body["seed"] = int(p["seed"]) + i
        r.progress(i / n + 0.05, f"Take {i + 1}/{n} on ElevenLabs ({model})", stage="generating", interruptible=False)
        r.submitted = True
        resp = r.request("POST", r.url(f"/text-to-speech/{voice}?output_format={FMT}"), json=body, timeout=180)
        _cost_header(r, resp)
        r.add_audio_bytes(resp.content, f"el_tts_{i + 1}", ".mp3", index=i, voice=voice)


def voice_design(r: CloudRun):
    p = r.params
    desc = str(p.get("instruct") or p.get("instruction") or r.prompt)
    if len(desc.strip()) < 20:
        raise ProviderError("Describe the voice in at least 20 characters (age, gender, accent, tone, pace).", "input")
    body = {"voice_description": desc[:1000], "model_id": r.remote()}
    line = str(p.get("text") or "")
    if len(line) >= 100:
        body["text"] = line[:1000]
    else:
        body["auto_generate_text"] = True
    if int(p.get("seed", -1)) >= 0:
        body["seed"] = int(p["seed"])
    r.progress(0.1, "Designing voices on ElevenLabs (3 previews)", stage="generating", interruptible=False)
    r.submitted = True
    resp = r.request("POST", r.url("/text-to-voice/design"), json=body, timeout=240)
    _cost_header(r, resp)
    d = resp.json()
    prev = d.get("previews") or []
    if not prev:
        raise ProviderError("ElevenLabs returned no voice preview.", "provider")
    for i, pv in enumerate(prev[:3]):
        raw = base64.b64decode(pv.get("audio_base_64") or "")
        r.add_audio_bytes(raw, f"el_voice_{i + 1}", ".mp3", type="voice", index=i,
                          generated_voice_id=pv.get("generated_voice_id"), spoken_text=d.get("text") or line)
    r.extra_meta["x_voice"] = {"ref_text": d.get("text") or line, "design_prompt": desc,
                               "language": p.get("language") if p.get("language") not in (None, "", "auto") else None}


def music(r: CloudRun):
    p = r.params
    secs = max(3.0, min(600.0, float(p.get("seconds") or 30)))
    prompt = r.prompt
    if p.get("lyrics"):
        prompt += "\n\nLyrics:\n" + str(p["lyrics"])
    n = batch_count(r)
    for i in range(n):
        r.check()
        body = {"prompt": prompt[:4000], "music_length_ms": int(secs * 1000), "model_id": r.remote()}
        if r.task == "music" or p.get("instrumental"):
            body["force_instrumental"] = True
        r.progress(i / n + 0.05, f"Composing take {i + 1}/{n} on ElevenLabs ({int(secs)} s)", stage="generating", interruptible=False)
        r.submitted = True
        resp = r.request("POST", r.url(f"/music?output_format={FMT}"), json=body, timeout=600)
        _cost_header(r, resp)
        r.add_audio_bytes(resp.content, f"el_music_{i + 1}", ".mp3", index=i)


def run(r: CloudRun):
    return {"el_sfx": sfx, "el_tts": tts, "el_voice_design": voice_design, "el_music": music}[r.m.get("api", "el_sfx")](r)
