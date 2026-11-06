"""OpenRouter image generation (usage.cost is the actual cost; always returned).

api: or_images -> POST /images {model, prompt, aspect_ratio, n, output_format, input_references?} -> data[].b64_json
api: or_chat   -> POST /chat/completions {model, messages, modalities: [image, text], image_config: {aspect_ratio}}
                  -> choices[0].message.images[].image_url.url (data URL)
"""
from __future__ import annotations

import base64

from .base import CloudRun, nearest_aspect
from .common import batch_count
from ..providers import ProviderError

ASPECTS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
HDR = {"X-Title": "Open Asset Studio"}


def _cost(r: CloudRun, d: dict):
    u = d.get("usage") or {}
    if u.get("cost") is not None:
        r.cost_usd = round((r.cost_usd or 0) + float(u["cost"]), 6)
        r.cost_note = "usage.cost reported by OpenRouter"
    if d.get("id"):
        r.request_ids.append(str(d["id"]))


def run(r: CloudRun):
    p = r.params
    n = batch_count(r)
    aspect = nearest_aspect(p.get("aspect") or "1:1", ASPECTS)
    refs = [r.data_url(x) for x in r.images_in()[:8]]
    r.submitted = True
    if r.m.get("api") == "or_chat":
        for i in range(n):
            r.check()
            content = [{"type": "text", "text": r.prompt}] + [{"type": "image_url", "image_url": {"url": u}} for u in refs]
            body = {"model": r.remote(), "messages": [{"role": "user", "content": content}], "modalities": ["image", "text"],
                    "image_config": {"aspect_ratio": aspect}}
            r.progress((i + 0.1) / n, f"Image {i + 1}/{n} via OpenRouter ({r.remote()})", stage="generating", interruptible=False)
            d = r.post_json(r.url("/chat/completions"), body, headers=HDR, timeout=300)
            _cost(r, d)
            msg = ((d.get("choices") or [{}])[0].get("message") or {})
            ims = msg.get("images") or []
            if not ims:
                raise ProviderError(f"OpenRouter returned no image. {str(msg.get('content') or '')[:200]}", "input")
            url = (ims[0].get("image_url") or {}).get("url") or ""
            path = r.download(url, f"or_{i + 1}.img")
            r.add_image(path, f"or_{i + 1}", index=i)
        return
    body = {"model": r.remote(), "prompt": r.prompt, "aspect_ratio": aspect, "n": n, "output_format": "png"}
    if refs:
        body["input_references"] = refs
    if p.get("quality"):
        body["quality"] = p["quality"]
    r.progress(0.1, f"Generating via OpenRouter ({r.remote()})", stage="generating", interruptible=False)
    d = r.post_json(r.url("/images"), body, headers=HDR, timeout=300)
    _cost(r, d)
    k = 0
    for i, item in enumerate(d.get("data") or []):
        if item.get("b64_json"):
            r.add_image(base64.b64decode(item["b64_json"]), f"or_{i + 1}", index=i)
            k += 1
        elif item.get("url"):
            r.add_image(r.download(item["url"], f"or_{i + 1}.img"), f"or_{i + 1}", index=i)
            k += 1
    if not k:
        raise ProviderError("OpenRouter returned no image.", "provider")
