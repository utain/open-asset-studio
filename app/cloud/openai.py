"""OpenAI (and OpenAI-compatible endpoints): gpt-image generate + edit, text to speech.

POST /images/generations {model, prompt, n, size, quality, background, output_format} -> data[].b64_json (or url)
POST /images/edits       multipart: image[] (repeat), prompt, model, n, size, quality, background
POST /audio/speech       {model, input, voice, instructions, response_format: wav} -> audio bytes
Actual cost of images from `usage` tokens x the entry's `usage_pricing` {text_in, image_in, image_out} (USD / 1M).
"""
from __future__ import annotations

import base64

from .base import CloudRun, nearest_aspect
from .common import batch_count
from ..providers import ProviderError

SIZES = {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536"}


def _size(r: CloudRun) -> str:
    p = r.params
    s = p.get("size")
    if isinstance(s, str) and ("x" in s or s == "auto"):
        return s
    return SIZES[nearest_aspect(p.get("aspect") or "1:1", list(SIZES))]


def _images_out(r: CloudRun, d: dict, prefix: str) -> int:
    k = 0
    for i, item in enumerate(d.get("data") or []):
        if item.get("b64_json"):
            r.add_image(base64.b64decode(item["b64_json"]), f"{prefix}_{i + 1}", index=i,
                        revised_prompt=item.get("revised_prompt"))
            k += 1
        elif item.get("url"):
            p = r.download(item["url"], f"{prefix}_{i + 1}.img")
            r.add_image(p, f"{prefix}_{i + 1}", index=i)
            k += 1
    return k


def _usage_cost(r: CloudRun, usage: dict | None):
    up = r.m.get("usage_pricing")
    if not usage or not up:
        return
    det = usage.get("input_tokens_details") or {}
    t_in = det.get("text_tokens", usage.get("input_tokens", 0) - det.get("image_tokens", 0))
    i_in = det.get("image_tokens", 0)
    out = usage.get("output_tokens", 0)
    usd = (t_in * float(up.get("text_in", 0)) + i_in * float(up.get("image_in", 0)) + out * float(up.get("image_out", 0))) / 1e6
    r.cost_usd = round((r.cost_usd or 0) + usd, 5)
    r.cost_note = f"{t_in} text + {i_in} image input tokens, {out} output tokens"


def images(r: CloudRun):
    p = r.params
    n = batch_count(r)
    model = r.remote()
    body = {"model": model, "prompt": r.prompt, "n": n, "size": _size(r)}
    if p.get("quality"):
        body["quality"] = p["quality"]
    transparent = bool(p.get("transparent")) or r.task == "rgba"
    if transparent:
        body["background"] = "transparent"
    if not r.m.get("user_endpoint"):
        body["output_format"] = "png"
    imgs = r.images_in()
    r.submitted = True
    if r.task in ("edit", "multi_ref") or imgs:
        if not imgs:
            raise ProviderError("Pick the image to edit.", "input")
        files = []
        for x in imgs[: int((r.m.get("features") or {}).get("max_refs", 10))]:
            data, mime = r.read_input(x, max_side=2048, fmt="png")
            files.append(("image[]", (f"{x['asset_id']}.png", data, mime)))
        form = {k: str(v) for k, v in body.items()}
        r.progress(0.1, f"Editing on {r.label} ({model}) · usually 20-90 s", stage="generating", interruptible=False)
        resp = r.request("POST", r.url("/images/edits"), data=form, files=files, timeout=300)
    else:
        r.progress(0.1, f"Generating on {r.label} ({model}) · usually 15-60 s", stage="generating", interruptible=False)
        resp = r.request("POST", r.url("/images/generations"), json=body, timeout=300)
    d = resp.json()
    rid = resp.headers.get("x-request-id")
    if rid:
        r.request_ids.append(rid)
    _usage_cost(r, d.get("usage"))
    if not _images_out(r, d, "openai"):
        raise ProviderError(f"{r.label} returned no image.", "provider")


def tts(r: CloudRun):
    p = r.params
    text = str(p.get("text") or r.prompt)
    if not text.strip():
        raise ProviderError("Write the line to speak.", "input")
    n = batch_count(r)
    for i in range(n):
        r.check()
        body = {"model": r.remote(), "input": text[:4096], "voice": p.get("voice") or "marin", "response_format": "wav"}
        if p.get("instruct"):
            body["instructions"] = str(p["instruct"])[:1000]
        if p.get("speed"):
            body["speed"] = float(p["speed"])
        r.progress((i + 0.2) / n, f"Speaking take {i + 1}/{n} on {r.label}", stage="generating", interruptible=False)
        r.submitted = True
        resp = r.request("POST", r.url("/audio/speech"), json=body, timeout=180)
        rid = resp.headers.get("x-request-id")
        if rid:
            r.request_ids.append(rid)
        r.add_audio_bytes(resp.content, f"tts_{i + 1}", ".wav", index=i, voice=body["voice"])


def run(r: CloudRun):
    if r.m["modality"] == "audio":
        return tts(r)
    return images(r)
