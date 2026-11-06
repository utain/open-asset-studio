"""Knobs, presets, sizes and prompt templates of the Image Studio (Qwen-Image-2.1).

Same philosophy as params.py: one place with defaults and ranges; `normalize()` is applied to whatever arrives
(UI, API client, params.json from the history), so an out-of-range value can never reach the worker.
"""
from __future__ import annotations

from typing import Any

from params import MAX_SEED, Spec, _coerce, _s

# ---------------------------------------------------------------------------- sizes
# Official sizes of the model card (long side 2400-2752): used as-is by the "Official" size.
OFFICIAL_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400), "3:2": (2528, 1696),
    "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752),
}
ASPECTS = tuple(OFFICIAL_SIZES)
# name -> long side in px (None = the official size of the chosen aspect ratio)
SIZE_LONG_SIDE: dict[str, int | None] = {"Draft (768)": 768, "Fast (1024)": 1024, "Detailed (1536)": 1536, "Official (2K)": None}
SIZES = tuple(SIZE_LONG_SIDE)
EDIT_SIZES = {"Draft (768)": 768, "Fast (1024)": 1024, "Detailed (1536)": 1536, "Official (2K)": 2048}
MAX_REFERENCES = 10
GRID = 32  # width / height must be multiples of 32 (16x VAE compression, 2x2 grouping)


def _r32(v: float) -> int:
    return max(GRID, int(round(v / GRID)) * GRID)


def gen_size(aspect: str, size: str) -> tuple[int, int]:
    """(width, height) for an aspect ratio and size name."""
    aspect = aspect if aspect in OFFICIAL_SIZES else "1:1"
    ow, oh = OFFICIAL_SIZES[aspect]
    long_side = SIZE_LONG_SIDE.get(size, 1024)
    if long_side is None:
        return ow, oh
    s = long_side / max(ow, oh)
    return _r32(ow * s), _r32(oh * s)


def edit_size(src_w: int, src_h: int, size: str) -> tuple[int, int]:
    """Output size of an edit: the aspect ratio of the main image at the area of a square with side `size`
    (what the pipeline itself does with `output_resolution`)."""
    side = EDIT_SIZES.get(size, 1024)
    ratio = src_w / max(1, src_h)
    w = (side * side * ratio) ** 0.5
    return _r32(w), _r32(w / ratio)


# ------------------------------------------------------------------- prompt templates
RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = " The image has alpha channel and the background is transparent."
THREED_DEFAULT = ("A single centered object, completely inside the frame with some empty margin around it, not cropped, "
                  "shown in a three-quarter front view, clean even soft lighting, no cast shadow, no text, no watermark, "
                  "no other objects, clear silhouette, consistent materials.")
THREED_OPAQUE_BG = " Plain pure white background."
EDIT_MASK_HINT = ("The red marks drawn on image 1 show the region to change; apply the edit only there and remove the marks "
                  "in the result. Everything else stays as it is.")
EDIT_MASKIMG_HINT = ("Image 2 is a black and white mask for image 1: change only the white area of image 1 and keep "
                     "everything outside it unchanged.")
EXTRACT_PROMPT = "Extract {what} from the image and place it on its own, cut out cleanly with all of it visible and nothing else."
EXTRACT_WHAT_DEFAULT = "the main subject"

EDIT_PRESETS: dict[str, str] = {
    "Change background": "Replace the background with a soft studio backdrop. Keep the subject exactly as it is.",
    "Plain background": "Replace the background with a plain pure white background. Keep the subject exactly as it is.",
    "Change color": "Change the main color of the subject to deep red. Keep its shape, details and material unchanged.",
    "Change material": "Change the material of the subject to brushed metal. Keep its shape and proportions unchanged.",
    "Remove object": "Remove the object marked or mentioned and fill the area naturally with the surrounding background.",
    "Cartoon style": "Redraw the image in a clean cartoon style with bold outlines and flat colors. Keep the composition.",
    "Low-poly style": "Redraw the subject as a low-poly 3D render with visible flat-shaded facets. Keep the composition.",
    "Neutral pose": ("Turn the character into a clean neutral T-pose reference: full body, facing the camera, arms "
                     "stretched sideways, legs straight, plain background, same outfit and identity."),
    "3D-ready cleanup": ("Show the subject alone as a single centered object, fully visible, in a three-quarter front "
                         "view, clean even lighting, no cast shadow, plain white background."),
    "Clay render": "Turn the subject into an untextured matte grey clay render with soft even lighting.",
}


def compose_generate_prompt(prompt: str, transparent: bool, threed: bool, threed_text: str) -> str:
    """The exact text sent to the model (also shown to the user in the UI)."""
    body = (prompt or "").strip()
    if threed:
        extra = (threed_text or "").strip() or THREED_DEFAULT
        body = f"{body.rstrip('. ')}. {extra}" if body else extra
        if not transparent:
            body += THREED_OPAQUE_BG
    if transparent:
        body = f"{RGBA_PREFIX}{body}{RGBA_SUFFIX}"
    return body


def compose_edit_prompt(instruction: str, transparent: bool, mask_mode: str) -> str:
    body = (instruction or "").strip()
    if mask_mode == "annotate":
        body = f"{body} {EDIT_MASK_HINT}"
    elif mask_mode == "mask":
        body = f"{body} {EDIT_MASKIMG_HINT}"
    if transparent:
        body = f"{RGBA_PREFIX}{body}{RGBA_SUFFIX}"
    return body


# ---------------------------------------------------------------------------- specs
SPECS: dict[str, Spec] = {
    "aspect": _s("1:1", choices=ASPECTS),
    "size": _s("Fast (1024)", choices=SIZES),
    "steps": _s(28, 1, 100),
    "true_cfg": _s(1.0, 1.0, 10.0),
    "seed": _s(1234, 0, MAX_SEED),
    "randomize_seed": _s(True),
    "batch": _s(1, 1, 8),
    "kv_cache": _s(True),
    "alpha_clean": _s(8, 0, 64),
    "transparent": _s(False),
    "threed_ready": _s(False),
    "edit_size": _s("Fast (1024)", choices=tuple(EDIT_SIZES)),
    "mask_mode": _s("none", choices=("none", "annotate", "mask")),
    "paste_back": _s(False),
    "preset": _s("Balanced", choices=("Fast", "Balanced", "Quality", "Custom")),
}
STR_SPECS = {"negative_prompt": 2000, "threed_text": 1500}
DEFAULTS: dict[str, Any] = {k: s.default for k, s in SPECS.items()}
DEFAULTS.update(negative_prompt="", threed_text=THREED_DEFAULT)
PROMPT_MAX = 4000

# shared sampling knobs per preset (sizes are chosen separately; Quality means a larger size too)
PRESETS: dict[str, dict[str, Any]] = {
    "Fast": dict(size="Draft (768)", edit_size="Draft (768)", steps=16),
    "Balanced": dict(size="Fast (1024)", edit_size="Fast (1024)", steps=28),
    "Quality": dict(size="Detailed (1536)", edit_size="Detailed (1536)", steps=40),
}
GEN_KEYS = ["aspect", "size", "steps", "true_cfg", "negative_prompt", "seed", "randomize_seed", "batch", "kv_cache",
            "transparent", "alpha_clean", "threed_ready", "threed_text", "preset"]
EDIT_KEYS = ["edit_size", "steps", "true_cfg", "negative_prompt", "seed", "randomize_seed", "batch", "kv_cache",
             "transparent", "alpha_clean", "mask_mode", "paste_back", "preset"]
STR_CHOICE_KEYS: tuple = ()


def normalize(raw: dict | None) -> dict:
    """Complete, clamped parameter dict (unknown keys dropped)."""
    raw = raw or {}
    out = {k: _coerce(s, raw.get(k), s.default) for k, s in SPECS.items()}
    for k, limit in STR_SPECS.items():
        v = raw.get(k)
        out[k] = (v if isinstance(v, str) else DEFAULTS[k])[:limit]
    return out


def from_values(keys: list[str], values) -> dict:
    return normalize(dict(zip(keys, values)))


def clean_prompt(text: Any) -> str:
    return (text if isinstance(text, str) else "").strip()[:PROMPT_MAX]
