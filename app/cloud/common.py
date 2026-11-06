"""Declarative request building shared by the queue-style providers (fal, Replicate).

A registry entry describes how studio params become the provider's input JSON:

    request:
      prompt: prompt              # input field of the prompt (null = none)
      size: wh                    # wh: <size_field> = {width, height} | wh_flat: width + height fields
                                  # aspect: aspect_ratio (nearest of `aspects`) | none
      size_field: image_size
      aspects: ["1:1", "16:9", ...]
      images: image_urls          # input images: list field (…_urls / images) or single (image_url / input_image_url)
      n: num_images               # batch field; absent = one request per image (the adapter loops)
      map: {seed: seed, seconds: seconds_total}     # studio param -> input field (values copied as is)
      values: {quality: {fast: TURBO}}             # optional value translation per studio param
      fixed: {output_format: png} # always sent
      task_fixed: {music: {is_instrumental: true}}  # sent for one task only
      negative: negative_prompt   # where the negative prompt goes (if supported)
"""
from __future__ import annotations

from .base import CloudRun, nearest_aspect, size_wh


def build_input(r: CloudRun, cfg: dict, *, n: int | None = None, image_mode: str = "data") -> dict:
    p = r.params
    body: dict = {}
    pf = cfg.get("prompt", "prompt")
    if pf and (r.prompt or cfg.get("prompt_required")):
        body[pf] = r.prompt
    if cfg.get("negative") and p.get("negative"):
        body[cfg["negative"]] = p["negative"]
    size = cfg.get("size", "none")
    if size == "wh":
        w, h = size_wh(p.get("aspect"), p.get("size") or 1024)
        body[cfg.get("size_field", "image_size")] = {"width": w, "height": h}
    elif size == "wh_flat":
        w, h = size_wh(p.get("aspect"), p.get("size") or 1024)
        body["width"], body["height"] = w, h
    elif size == "aspect" and p.get("aspect"):
        body[cfg.get("size_field", "aspect_ratio")] = nearest_aspect(p["aspect"], cfg.get("aspects") or ["1:1"])
    imgs = r.images_in()
    if cfg.get("images") and imgs:
        urls = [r.data_url(x) for x in imgs[: int(cfg.get("max_images", 10))]]
        f = cfg["images"]
        body[f] = urls if (f.endswith(("urls", "images", "s")) and not f.endswith("_url")) else urls[0]
    if cfg.get("n") and n:
        body[cfg["n"]] = n
    values = cfg.get("values") or {}
    for src, dst in (cfg.get("map") or {}).items():
        v = p.get(src)
        if v is None or v == "" or (src == "seed" and int(v) < 0):
            continue
        if src in values and str(v) in {str(k) for k in values[src]}:
            v = next(val for k, val in values[src].items() if str(k) == str(v))
        _put(body, dst, v)
    body.update(cfg.get("fixed") or {})
    for k, v in ((cfg.get("task_fixed") or {}).get(r.task) or {}).items():
        _put(body, k, v)
    return body


def _put(body: dict, dst: str, v):
    """dst may be dotted ("voice_setting.voice_id") for nested input objects."""
    cur = body
    *head, last = dst.split(".")
    for k in head:
        cur = cur.setdefault(k, {})
    cur[last] = v


def batch_count(r: CloudRun) -> int:
    return max(1, int(r.params.get("batch") or 1))
