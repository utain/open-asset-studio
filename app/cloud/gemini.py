"""Google Gemini API (x-goog-api-key): Nano Banana image generation + editing, Gemini TTS, Lyria music.

Images / TTS: POST /models/{model}:generateContent
  image: generationConfig {responseModalities: [TEXT, IMAGE], imageConfig: {aspectRatio, imageSize}}; input images
         as inlineData parts; output candidates[0].content.parts[].inlineData {mimeType, data}
  tts:   generationConfig {responseModalities: [AUDIO], speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName};
         output inlineData audio (raw PCM s16le 24 kHz, wrapped as WAV here; a RIFF answer is kept as is)
Music (Lyria 3): POST /interactions (header Api-Revision) {model, input: [{type: text, text}]}; the audio is the
  first base64 audio content found in the answer (shape not verified with a real key).
Imagen 4 is shut down on the Gemini API (use Nano Banana).
"""
from __future__ import annotations

import base64
import re

from .base import CloudRun, nearest_aspect, pcm_to_wav
from .common import batch_count
from ..providers import ProviderError

ASPECTS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


def _inline_parts(d) -> list[dict]:
    """Every {mimeType|mime_type, data} blob in an answer (generateContent or Interactions)."""
    out = []
    if isinstance(d, dict):
        blob = d.get("inlineData") or d.get("inline_data")
        if isinstance(blob, dict) and blob.get("data"):
            out.append({"mime": blob.get("mimeType") or blob.get("mime_type") or "", "data": blob["data"]})
        elif isinstance(d.get("data"), str) and len(d["data"]) > 64 and (d.get("mime_type") or d.get("mimeType") or d.get("type") in ("audio", "image")):
            out.append({"mime": d.get("mime_type") or d.get("mimeType") or d.get("type", ""), "data": d["data"]})
        for k, v in d.items():
            if k not in ("inlineData", "inline_data", "data"):
                out += _inline_parts(v)
    elif isinstance(d, list):
        for v in d:
            out += _inline_parts(v)
    return out


def _blocked(d: dict) -> str | None:
    fb = d.get("promptFeedback") or {}
    if fb.get("blockReason"):
        return f"blocked: {fb['blockReason']}"
    c = (d.get("candidates") or [{}])[0]
    if c.get("finishReason") and c["finishReason"] not in ("STOP", "MAX_TOKENS"):
        return f"finish reason {c['finishReason']}"
    return None


def images(r: CloudRun):
    p = r.params
    n = batch_count(r)
    model = r.remote()
    for i in range(n):
        r.check()
        parts = [{"text": r.prompt}]
        for x in r.images_in()[: int((r.m.get("features") or {}).get("max_refs", 14))]:
            data, mime = r.read_input(x, max_side=2048, fmt="png")
            parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}})
        ic = {"aspectRatio": nearest_aspect(p.get("aspect") or "1:1", ASPECTS)}
        if p.get("resolution"):
            ic["imageSize"] = str(p["resolution"]).upper()
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": ic}}
        r.progress((i + 0.1) / n, f"Image {i + 1}/{n} on Gemini ({model})", stage="generating", interruptible=False)
        r.submitted = True
        d = r.post_json(r.url(f"/models/{model}:generateContent"), body, timeout=240)
        if d.get("responseId"):
            r.request_ids.append(d["responseId"])
        imgs = [b for b in _inline_parts(d.get("candidates") or d) if b["mime"].startswith("image/")]
        if not imgs:
            why = _blocked(d) or "no image in the answer"
            text = " ".join(pt.get("text", "") for c in d.get("candidates") or [] for pt in (c.get("content") or {}).get("parts") or [])
            raise ProviderError(f"Gemini returned no image ({why}). {text[:200]}".strip(), "input",
                                "Rephrase the prompt; Gemini refuses some content.")
        for j, b in enumerate(imgs[:1] if n > 1 else imgs):
            r.add_image(base64.b64decode(b["data"]), f"gemini_{i + 1}_{j + 1}", index=i)


def tts(r: CloudRun):
    p = r.params
    text = str(p.get("text") or r.prompt)
    if not text.strip():
        raise ProviderError("Write the line to speak.", "input")
    say = f"{p['instruct'].strip()}: {text}" if p.get("instruct") else text
    voice = p.get("voice") or "Kore"
    for i in range(batch_count(r)):
        r.check()
        body = {"contents": [{"role": "user", "parts": [{"text": say}]}],
                "generationConfig": {"responseModalities": ["AUDIO"],
                                     "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
        r.progress(0.2, f"Speaking take {i + 1} on Gemini", stage="generating", interruptible=False)
        r.submitted = True
        d = r.post_json(r.url(f"/models/{r.remote()}:generateContent"), body, timeout=180)
        aud = [b for b in _inline_parts(d.get("candidates") or d) if b["mime"].startswith("audio") or "pcm" in b["mime"].lower()]
        if not aud:
            raise ProviderError(f"Gemini TTS returned no audio ({_blocked(d) or 'empty answer'}).", "provider")
        raw = base64.b64decode(aud[0]["data"])
        rate = int((re.search(r"rate=(\d+)", aud[0]["mime"]) or [0, 24000])[1])
        r.add_audio_bytes(raw if raw[:4] == b"RIFF" else pcm_to_wav(raw, rate), f"gemini_tts_{i + 1}", ".wav", index=i, voice=voice)


def music(r: CloudRun):
    text = r.prompt
    if r.params.get("lyrics"):
        text += "\n\nLyrics:\n" + str(r.params["lyrics"])
    if r.params.get("instrumental"):
        text += "\n\nInstrumental only, no vocals."
    for i in range(batch_count(r)):
        r.check()
        body = {"model": r.remote(), "input": [{"type": "text", "text": text}]}
        r.progress(0.15, f"Composing take {i + 1} with Lyria (up to a few minutes)", stage="generating", interruptible=False)
        r.submitted = True
        d = r.post_json(r.url("/interactions"), body, headers={"Api-Revision": "2026-05-20"}, timeout=600)
        if d.get("id"):
            r.request_ids.append(str(d["id"]))
        aud = [b for b in _inline_parts(d) if "audio" in b["mime"] or b["mime"] == "audio"]
        if aud:
            raw = base64.b64decode(aud[0]["data"])
            r.add_audio_bytes(raw, f"lyria_{i + 1}", ".mp3", index=i)
        elif not r.fetch_outputs(d, "audio", prefix=f"lyria{i + 1}"):
            raise ProviderError("Lyria returned no audio.", "provider")


def run(r: CloudRun):
    api = r.m.get("api")
    if api == "gemini_tts":
        return tts(r)
    if api == "gemini_music":
        return music(r)
    return images(r)
