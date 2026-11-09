"""Project style guide: the pure part (schema, validation, hashing, prompt composition, palette maths).

No I/O except what the caller passes in (`assets` for validation, `lookup` / `load_version` for composition) and the
image paths handed to the palette helpers. numpy + PIL only (the core env). The plugin `app/plugins/style.py` owns the
files, the REST API and the job hooks; docs/CONTRACTS.md section 11 is the contract.

Rules that matter (spec: the "Application pipeline"):
  * the user's text always comes first; the style is a SUFFIX (a prefix would break the RGBA trigger sentence of the
    Qwen worker, and a truncating text encoder would then drop the subject instead of the style),
  * the avoid list goes only into a real negative prompt, and only when the model uses one at the current settings,
  * reference images are attached (role `style_ref`) only to models that declare `features.style_ref: [tasks]`,
  * `compose` is deterministic and idempotent: running it on its own output (a retry, a regenerate) gives the same
    params and inputs, because the first thing it does is to restore the user's own text from `x_style`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

SCHEMA = 1
VARIANTS = ("image", "texture", "sprite", "ui_icon", "video", "audio")
VARIANT_LABELS = {"image": "Image", "texture": "Texture & materials", "sprite": "Sprite", "ui_icon": "UI icon", "video": "Video", "audio": "Audio mood"}
STRENGTHS = ("light", "normal", "strong")
POSTS = ("off", "tint", "lock")
MODES = ("extend", "replace")
SERVER_KEYS = ("version", "hash", "saved", "saved_by")        # not part of the content hash
TOP_KEYS = {"schema", "name", "enabled", "strength", "positive", "background", "avoid", "palette", "refs", "variants", *SERVER_KEYS}
PALETTE_KEYS = {"colors", "source", "in_prompt", "post", "tint_strength", "weights"}
VARIANT_KEYS = ("enabled", "positive", "background", "avoid", "mode", "refs", "palette_post")
REF_KEYS = {"asset", "sha256", "name", "note"}

LIM = {"name": 60, "positive": 400, "background": 160, "avoid_n": 16, "avoid_len": 40, "colors": 12, "refs": 4, "ref_name": 80, "ref_note": 160,
       "variant_positive": 300, "style_text": 600}
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
NEG_RE = re.compile(r"\b(no|without|never|avoid|don'?t|dont)\b", re.I)
REF_CAP = {"light": 0, "normal": 2, "strong": 4}
SEC_PER_REF_IMAGE = 20          # measured on Qwen-Image 2.1 at 1024 px / 20 steps with 1 reference: +20 s per image (docs/research/style_guide.md)
PALETTE_VARIANTS = ("image", "texture", "sprite", "ui_icon")

# Base negative prompts: a worker uses its own built-in negative ONLY when `params.negative` is empty, so adding our avoid list
# would silently drop it. These are quality-only bases (keyed by the model's prompt style). The WAN worker default also lists
# 风格/作品/画作 ("style, artwork, painting"), which works against a stylised guide, so it is NOT reused here.
BASE_NEGATIVE = {
    "video_wan": "worst quality, low quality, blurry, static, subtitles, watermark, jpeg artifacts, deformed",
    "video_ltx": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "lumina": "blurry, low quality, watermark",
}

# task -> how it is styled. `variant: skip` = never (edits, cut-outs, upscales, 3D shape / texture, voices, tools ...).
# Other plugins add theirs with register_task() (the seamless-materials plugin registers its task as "texture").
TASK_POLICY: dict[str, dict] = {}


def _pol(tasks, variant, sep=". ", refs=False, prompt_key="prompt"):
    for t in tasks:
        TASK_POLICY[t] = {"variant": variant, "sep": sep, "refs": refs, "prompt_key": prompt_key}


_pol(("t2i", "rgba"), "image", refs=True)
_pol(("multi_ref",), "image", refs=True)
_pol(("text3d",), "texture")
_pol(("t2v", "i2v", "flf2v"), "video")
_pol(("sfx", "ambience", "music", "song"), "audio", sep=", ")
SKIPPED_TASKS = ("edit", "extract", "inpaint", "rmbg", "upscale", "recenter", "shape", "texture", "multiview", "retopo", "rig", "reskin", "motion",
                 "mesh_export", "interpolate", "tts", "voice_design", "voice_clone", "sing", "s2s", "v2a", "qa_clap", "qa_asr", "transcribe",
                 "subtitles", "dub", "separate", "audio_ask", "improve", "write", "test")
_pol(SKIPPED_TASKS, "skip")


def register_task(task: str, variant: str, sep: str = ". ", prompt_key: str = "prompt", refs: bool = False) -> None:
    """Make `task` styled with `variant` (or "skip"). Used by plugins, e.g. register_task("material", "texture")."""
    if variant != "skip" and variant not in VARIANTS:
        raise ValueError(f"Unknown style variant {variant!r}.")
    _pol((task,), variant, sep=sep, refs=refs, prompt_key=prompt_key)


class StyleInputError(Exception):
    """A bad per-job style request (e.g. a version that does not exist). The hook turns it into JobError(kind input)."""


# ------------------------------------------------------------------ text helpers
def _s(v) -> str:
    return re.sub(r"\s+", " ", str(v if v is not None else "")).strip()


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def clean_text(text: str, cap: int = LIM["style_text"]) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip(" .,;")
    t = re.sub(r"\s*,(\s*,)+", ",", t)
    if len(t) > cap:
        cut = t[:cap]
        i = cut.rfind(", ")
        t = (cut[:i] if i > cap // 2 else cut).strip(" .,;")
    return t


def _join(*parts: str, sep: str = ", ") -> str:
    return sep.join(p for p in (_s(x).strip(" ,;") for x in parts) if p)


def split_terms(text) -> list[str]:
    if isinstance(text, (list, tuple)):
        items = list(text)
    else:
        items = str(text or "").split(",")
    return [_s(x) for x in items if _s(x)]


def merge_terms(base, extra) -> tuple[str, list[str]]:
    """Comma list `base` + `extra` terms, case-insensitive dedupe, existing terms first. Returns (text, new terms only)."""
    out, seen, new = [], set(), []
    for t in split_terms(base):
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    for t in split_terms(extra):
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
            new.append(t)
    return ", ".join(out), new


def dedupe_ci(items) -> list[str]:
    out, seen = [], set()
    for t in split_terms(items):
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


# ------------------------------------------------------------------ schema
def variant_defaults(vid: str) -> dict:
    d = {"enabled": True, "positive": "", "background": None, "avoid": [], "mode": "extend", "refs": True, "palette_post": None}
    if vid == "texture":
        d.update(background="", avoid=["perspective", "shadows"], refs=False)      # a picture of a scene breaks tileability
    elif vid == "video":
        d["refs"] = False
    elif vid == "audio":
        d.update(enabled=False, mode="replace", refs=False)                        # a visual look makes no sense in a sound prompt
    return d


def default_guide() -> dict:
    return {"schema": SCHEMA, "name": "Project style", "enabled": True, "strength": "normal", "positive": "", "background": "", "avoid": [],
            "palette": {"colors": [], "source": None, "in_prompt": False, "post": "off", "tint_strength": 0.35}, "refs": [], "variants": {}}


def effective_variant(guide: dict, vid: str) -> dict:
    return {**variant_defaults(vid), **((guide.get("variants") or {}).get(vid) or {})}


def content_of(guide: dict) -> dict:
    return {k: v for k, v in guide.items() if k not in SERVER_KEYS}


def guide_hash(guide: dict) -> str:
    blob = json.dumps(content_of(guide), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


_SHA: dict[tuple, str] = {}


def sha256_file(path: Path) -> str:
    st = Path(path).stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if len(_SHA) > 256:
            _SHA.clear()
        _SHA[key] = h.hexdigest()
    return _SHA[key]


def _text(v, cap: int, label: str) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        raise ValueError(f"{label} must be text.")
    t = _s(v)
    if len(t) > cap:
        raise ValueError(f"{label} is too long ({len(t)} > {cap} characters).")
    return t


def _avoid(v, label: str) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = split_terms(v)
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"{label} must be a list of short phrases.")
    items = dedupe_ci([_s(x) for x in v if x is not None])
    if len(items) > LIM["avoid_n"]:
        raise ValueError(f"{label} has {len(items)} items (at most {LIM['avoid_n']}).")
    for t in items:
        if len(t) > LIM["avoid_len"]:
            raise ValueError(f"{label} item '{t[:20]}...' is longer than {LIM['avoid_len']} characters.")
    return items


def _num(v, lo: float, hi: float, label: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{label} must be a number.")
    return round(min(hi, max(lo, float(v))), 3)


def _bool(v, label: str) -> bool:
    if not isinstance(v, bool):
        raise ValueError(f"{label} must be true or false.")
    return v


def _unknown(d: dict, allowed, label: str):
    bad = sorted(set(d) - set(allowed))
    if bad:
        raise ValueError(f"Unknown {label} key(s): {', '.join(map(str, bad))}.")


def _palette(p, label="palette") -> dict:
    out = default_guide()["palette"]
    if p is None:
        return out
    if not isinstance(p, dict):
        raise ValueError("palette must be an object.")
    _unknown(p, PALETTE_KEYS, label)
    if "colors" in p:
        cols = p["colors"] or []
        if not isinstance(cols, (list, tuple)):
            raise ValueError("palette.colors must be a list of #rrggbb colours.")
        seen = []
        for c in cols:
            if not isinstance(c, str) or not HEX_RE.match(c.strip()):
                raise ValueError(f"'{str(c)[:20]}' is not a #rrggbb colour.")
            c = c.strip().lower()
            if c not in seen:
                seen.append(c)
        if len(seen) > LIM["colors"]:
            raise ValueError(f"palette.colors has {len(seen)} colours (at most {LIM['colors']}).")
        out["colors"] = seen
    if "source" in p:
        src = p["source"]
        if src not in (None, "") and not (isinstance(src, str) and re.match(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$", src)):
            raise ValueError("palette.source must be an asset id or null.")
        out["source"] = src or None
    if "in_prompt" in p:
        out["in_prompt"] = _bool(p["in_prompt"], "palette.in_prompt")
    if "post" in p:
        if p["post"] not in POSTS:
            raise ValueError(f"palette.post must be one of {', '.join(POSTS)}.")
        out["post"] = p["post"]
    if "tint_strength" in p:
        out["tint_strength"] = _num(p["tint_strength"], 0.1, 0.8, "palette.tint_strength")
    if "weights" in p and p["weights"]:
        w = p["weights"]
        if not isinstance(w, list) or len(w) != len(out["colors"]) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in w):
            raise ValueError("palette.weights must be one number per colour.")
        out["weights"] = [round(float(x), 3) for x in w]
    return out


def _variant(vid: str, v) -> dict:
    out = variant_defaults(vid)
    if v is None:
        return out
    if not isinstance(v, dict):
        raise ValueError(f"variants.{vid} must be an object.")
    _unknown(v, VARIANT_KEYS, f"variants.{vid}")
    if "enabled" in v:
        out["enabled"] = _bool(v["enabled"], f"variants.{vid}.enabled")
    if "positive" in v:
        out["positive"] = _text(v["positive"], LIM["variant_positive"], f"variants.{vid}.positive")
    if "background" in v:
        out["background"] = None if v["background"] is None else _text(v["background"], LIM["background"], f"variants.{vid}.background")
    if "avoid" in v:
        out["avoid"] = _avoid(v["avoid"], f"variants.{vid}.avoid")
    if "mode" in v:
        if v["mode"] not in MODES:
            raise ValueError(f"variants.{vid}.mode must be extend or replace.")
        out["mode"] = v["mode"]
    if "refs" in v:
        out["refs"] = _bool(v["refs"], f"variants.{vid}.refs")
    if "palette_post" in v:
        if v["palette_post"] is not None and v["palette_post"] not in POSTS:
            raise ValueError(f"variants.{vid}.palette_post must be null, off, tint or lock.")
        out["palette_post"] = v["palette_post"]
    return out


def validate_guide(g, project: str, assets=None, *, allow_missing=(), fill_refs: bool = True) -> dict:
    """Return the cleaned guide (defaults filled, server keys dropped). Raises ValueError (HTTP 400) on anything wrong.

    `assets` is an AssetStore (refs must be image assets of `project`; their sha256 and name are filled in).
    `allow_missing`: asset ids that may stay in `refs` although the asset is gone (so a deleted ref does not block every later save).
    fill_refs=False skips the sha256 (previews of unsaved drafts)."""
    if not isinstance(g, dict):
        raise ValueError("The style guide must be an object.")
    _unknown(g, TOP_KEYS, "guide")
    if g.get("schema") not in (None, SCHEMA):
        raise ValueError(f"Unsupported style guide schema {g.get('schema')!r}.")
    out = default_guide()
    if "name" in g:
        out["name"] = _text(g["name"], LIM["name"], "name") or out["name"]
    if "enabled" in g:
        out["enabled"] = _bool(g["enabled"], "enabled")
    if "strength" in g:
        if g["strength"] not in STRENGTHS:
            raise ValueError(f"strength must be one of {', '.join(STRENGTHS)}.")
        out["strength"] = g["strength"]
    if "positive" in g:
        out["positive"] = _text(g["positive"], LIM["positive"], "The look (positive)")
    if "background" in g:
        out["background"] = _text(g["background"], LIM["background"], "background")
    if "avoid" in g:
        out["avoid"] = _avoid(g["avoid"], "avoid")
    out["palette"] = _palette(g.get("palette"))
    refs, seen = [], set()
    rin = g.get("refs") or []
    if not isinstance(rin, (list, tuple)):
        raise ValueError("refs must be a list.")
    for r in rin:
        if isinstance(r, str):
            r = {"asset": r}
        if not isinstance(r, dict) or not r.get("asset"):
            raise ValueError("Each reference needs an asset id.")
        _unknown(r, REF_KEYS, "reference")
        aid = r["asset"]
        if aid in seen:
            continue
        seen.add(aid)
        entry = {"asset": aid, "sha256": str(r.get("sha256") or ""), "name": _text(r.get("name"), LIM["ref_name"], "reference name"),
                 "note": _text(r.get("note"), LIM["ref_note"], "reference note")}
        meta = None
        if assets is not None:
            try:
                meta = assets.get(aid)
            except Exception:  # noqa: BLE001  AssetError / KeyError
                meta = None
            if meta is None:
                if aid not in allow_missing:
                    raise ValueError(f"Reference {aid}: asset not found.")
            else:
                if meta.get("project") != project:
                    raise ValueError(f"Reference {aid} belongs to another project.")
                if meta.get("type") != "image":
                    raise ValueError(f"Reference {aid} is not an image.")
                entry["name"] = entry["name"] or _text(meta.get("name"), LIM["ref_name"], "reference name")
                if fill_refs:
                    try:
                        entry["sha256"] = sha256_file(assets.file_path(project, aid, meta["main"]))
                    except Exception:  # noqa: BLE001  file vanished
                        pass
        refs.append(entry)
    if len(refs) > LIM["refs"]:
        raise ValueError(f"At most {LIM['refs']} reference images.")
    out["refs"] = refs
    vin = g.get("variants") or {}
    if not isinstance(vin, dict):
        raise ValueError("variants must be an object.")
    _unknown(vin, VARIANTS, "variant")
    out["variants"] = {vid: _variant(vid, v) for vid, v in vin.items()}
    return out


def lint(guide: dict) -> list[str]:
    """Warnings (never errors): negative phrasing in a positive field. Some models draw what you name."""
    w = []
    fields = [("The look", guide.get("positive")), ("Background", guide.get("background"))]
    fields += [(f"{VARIANT_LABELS.get(k, k)} look", (v or {}).get("positive")) for k, v in (guide.get("variants") or {}).items()]
    for label, text in fields:
        m = NEG_RE.search(text or "")
        if m:
            w.append(f"{label}: \"{m.group(0)}\" in a positive text. Some models draw what you name: move it to Avoid.")
    return w


def summary(guide: dict) -> str:
    pos = guide.get("positive") or ""
    bits = [pos[:70] + ("..." if len(pos) > 70 else "") or "(no text)"]
    if guide.get("refs"):
        bits.append(f"{len(guide['refs'])} ref")
    if (guide.get("palette") or {}).get("colors"):
        bits.append(f"{len(guide['palette']['colors'])} colours")
    if not guide.get("enabled", True):
        bits.append("off")
    return " | ".join(bits)


# ------------------------------------------------------------------ colour names (nearest CSS name, in Lab)
CSS_COLORS = {
    "black": "000000", "dimgray": "696969", "gray": "808080", "silver": "c0c0c0", "gainsboro": "dcdcdc", "white": "ffffff", "ivory": "fffff0",
    "beige": "f5f5dc", "cream": "fff5d6", "linen": "faf0e6", "wheat": "f5deb3", "tan": "d2b48c", "burlywood": "deb887", "khaki": "f0e68c",
    "gold": "ffd700", "goldenrod": "daa520", "yellow": "ffff00", "orange": "ffa500", "darkorange": "ff8c00", "coral": "ff7f50",
    "salmon": "fa8072", "tomato": "ff6347", "orangered": "ff4500", "red": "ff0000", "crimson": "dc143c", "firebrick": "b22222",
    "brown": "a52a2a", "maroon": "800000", "darkred": "8b0000", "saddlebrown": "8b4513", "sienna": "a0522d", "chocolate": "d2691e",
    "peru": "cd853f", "rosybrown": "bc8f8f", "pink": "ffc0cb", "hotpink": "ff69b4", "deeppink": "ff1493", "magenta": "ff00ff",
    "orchid": "da70d6", "plum": "dda0dd", "violet": "ee82ee", "purple": "800080", "indigo": "4b0082", "darkviolet": "9400d3",
    "slateblue": "6a5acd", "lavender": "e6e6fa", "navy": "000080", "darkblue": "00008b", "blue": "0000ff", "royalblue": "4169e1",
    "dodgerblue": "1e90ff", "steelblue": "4682b4", "skyblue": "87ceeb", "lightblue": "add8e6", "cyan": "00ffff", "turquoise": "40e0d0",
    "teal": "008080", "cadetblue": "5f9ea0", "aquamarine": "7fffd4", "mintcream": "f5fffa", "lightgreen": "90ee90", "palegreen": "98fb98",
    "springgreen": "00ff7f", "seagreen": "2e8b57", "mediumseagreen": "3cb371", "green": "008000", "limegreen": "32cd32", "lime": "00ff00",
    "forestgreen": "228b22", "darkgreen": "006400", "olive": "808000", "olivedrab": "6b8e23", "yellowgreen": "9acd32", "chartreuse": "7fff00",
    "darkkhaki": "bdb76b", "darkolivegreen": "556b2f", "slategray": "708090", "darkslategray": "2f4f4f", "lightgray": "d3d3d3",
    "lightyellow": "ffffe0", "peachpuff": "ffdab9", "moccasin": "ffe4b5", "sandybrown": "f4a460", "darksalmon": "e9967a", "indianred": "cd5c5c",
    "mistyrose": "ffe4e1", "thistle": "d8bfd8", "mediumpurple": "9370db", "darkorchid": "9932cc", "darkmagenta": "8b008b",
    "midnightblue": "191970", "lightseagreen": "20b2aa", "darkcyan": "008b8b", "darkslateblue": "483d8b", "powderblue": "b0e0e6",
}
_CSS_LAB: list | None = None


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(x) for x in c)


def color_names(colors: list[str]) -> list[str]:
    """Nearest CSS colour name for each hex colour (CIE Lab distance), deduped, in order."""
    import numpy as np

    global _CSS_LAB
    if _CSS_LAB is None:
        names = list(CSS_COLORS)
        _CSS_LAB = [names, srgb2lab(np.array([hex_to_rgb(CSS_COLORS[n]) for n in names], dtype=float))]
    names, lab = _CSS_LAB
    out = []
    for c in colors:
        v = srgb2lab(np.array([hex_to_rgb(c)], dtype=float))[0]
        n = names[int(((lab - v) ** 2).sum(1).argmin())]
        if n not in out:
            out.append(n)
    return out


# ------------------------------------------------------------------ composition
def _prompt_style(model: dict | None, task: str) -> str | None:
    ps = (model or {}).get("prompt_style")
    if isinstance(ps, dict):
        return ps.get(task) if task in ps else next(iter(ps.values()), None)
    return ps


def cfg_active(model: dict | None, params: dict) -> bool:
    """features.negative == "cfg": the negative prompt only matters when the model's own CFG scale is above 1."""
    for name in ("true_cfg", "cfg", "guidance_scale"):
        decl = next((p for p in (model or {}).get("params") or [] if p.get("name") == name), None)
        if decl is None:
            continue
        v = params.get(name, decl.get("default", 1.0))
        try:
            return float(v) > 1.0
        except (TypeError, ValueError):
            return False
    return False


def _norm_inputs(inputs) -> list[dict]:
    out = []
    for i in inputs or []:
        if isinstance(i, str):
            out.append({"asset": i, "role": "input"})
        elif isinstance(i, dict) and i.get("asset"):
            out.append({**i, "role": i.get("role") or "input"})
    return out


def style_parts(guide: dict, variant: str, task: str, params: dict, strength: str, policy: dict | None = None) -> tuple[str, list[dict]]:
    """(the style text, skipped) for one variant: the part that is appended to the user's prompt."""
    v = effective_variant(guide, variant)
    skipped = []
    pos = v["positive"] if v["mode"] == "replace" else _join(guide.get("positive"), v["positive"])
    bg = guide.get("background") if v["background"] is None else v["background"]
    if variant == "audio" or (v["mode"] == "replace" and v["background"] is None):
        bg = ""                  # a sound has no background; "replace" ignores the main look, and the inherited background is part of it
    if bg and (task == "rgba" or params.get("transparent") or params.get("threed_ready")):
        bg = ""
        skipped.append({"what": "background", "why": "a transparent or 3D-ready image has no background"})
    parts = [pos, bg]
    pal = guide.get("palette") or {}
    if pal.get("colors") and (pal.get("in_prompt") or strength == "strong") and variant != "audio":
        parts.append("colour palette: " + ", ".join(color_names(pal["colors"][:5])))
    return clean_text(_join(*parts)), skipped


def compose(guide: dict | None, model: dict | None, task: str, params: dict, inputs, req: dict | None = None, *, meta: dict | None = None,
            lookup=None, load_version=None, commercial_only: bool = False, task_label: str | None = None) -> dict:
    """Plan what to add to one job. Pure and idempotent. `lookup(asset_id) -> asset meta | None`, `load_version(n) -> guide | None`.

    Returns {applied, reason, params, inputs, x_style (record or None), style_post, summary, prompt, negative, refs: [{asset, index}],
    ref_clause, skipped, notes, variant, strength, version, extra_seconds_est}. params / inputs are new copies."""
    req = dict(req or {})
    meta = meta or {}
    lookup = lookup or (lambda _aid: None)
    params = copy.deepcopy(dict(params or {}))
    inputs = _norm_inputs(inputs)
    pol = TASK_POLICY.get(task) or {"variant": "skip", "sep": ". ", "refs": False, "prompt_key": "prompt"}
    pkey = pol["prompt_key"]
    # 1. RESTORE: a retry / regenerate carries the already composed prompt; go back to what the user wrote
    prev = (meta.get("asset_extra") or {}).get("x_style") or {}
    if isinstance(prev.get("user_prompt"), str):
        params[pkey] = prev["user_prompt"]
        if prev.get("user_negative"):
            params["negative"] = prev["user_negative"]
        elif "negative" in params:
            params["negative"] = ""
        inputs = [i for i in inputs if i.get("role") != "style_ref"]
    res = {"applied": False, "reason": None, "params": params, "inputs": inputs, "x_style": None, "style_post": None, "summary": None,
           "prompt": params.get(pkey), "negative": params.get("negative") or "", "refs": [], "ref_clause": None, "skipped": [], "notes": [],
           "variant": None, "strength": None, "version": None, "extra_seconds_est": 0}
    if not guide:
        return res
    if req.get("version") not in (None, "", guide.get("version")):
        try:
            n = int(req["version"])
        except (TypeError, ValueError):
            raise StyleInputError(f"Style guide version {req['version']!r} is not a number.") from None
        g2 = load_version(n) if load_version else None
        if not g2:
            raise StyleInputError(f"Style guide version {n} does not exist.")
        guide = g2
    res["version"] = guide.get("version")
    base_rec = {"name": guide.get("name"), "version": guide.get("version"), "hash": guide.get("hash")}

    def off(reason: str):
        res["reason"] = reason
        res["x_style"] = {"applied": False, "version": guide.get("version"), "hash": guide.get("hash"), "reason": reason}
        res["summary"] = {**base_rec, "variant": None, "applied": False}
        return res

    if not guide.get("enabled", True):
        return off("guide disabled")
    if req.get("apply") is False:
        return off("off for this job")
    if pol["variant"] == "skip":
        return off(f"{task_label or task} is not styled")
    variant = req.get("variant") or pol["variant"]
    if variant not in VARIANTS:
        raise StyleInputError(f"Unknown style variant {variant!r}.")
    strength = req.get("strength") or guide.get("strength") or "normal"
    if strength not in STRENGTHS:
        raise StyleInputError(f"Unknown style strength {strength!r}.")
    res["variant"], res["strength"] = variant, strength
    v = effective_variant(guide, variant)
    if not v["enabled"]:
        return off(f"the {VARIANT_LABELS[variant]} variant is off")
    user = str(params.get(pkey) or "").strip()
    if not user:
        return off("no prompt to add the style to")
    skipped: list[dict] = []
    added = {"prompt": "", "ref_clause": None, "negative": [], "refs": [], "base_negative": None}
    feats = (model or {}).get("features") or {}
    mname = (model or {}).get("name") or (model or {}).get("id") or "This model"
    # 3-4. STYLE TEXT (+ dedupe)
    style_text, sk = style_parts(guide, variant, task, params, strength, pol)
    skipped += sk
    if style_text and norm(style_text) in norm(user):
        skipped.append({"what": "prompt", "why": "already in the prompt"})
        style_text = ""
    # 5. REFS
    ref_clause = None
    take: list[dict] = []
    refs_in = guide.get("refs") or []
    if refs_in:
        why = None
        if strength == "light":
            why = "Light strength uses the text only"
        elif v["refs"] is False:
            why = f"references are off for the {VARIANT_LABELS[variant]} variant"
        elif (model or {}).get("backend") == "cloud":
            why = "cloud models never get reference images (nothing is uploaded without your consent)"
        elif not pol.get("refs") or task not in (feats.get("style_ref") or []):
            why = f"{mname} cannot take style references for {task_label or task}" if model else "no model to check references against"
        if why:
            skipped.append({"what": "refs", "why": why})
        else:
            n_user = sum(1 for i in inputs if (lookup(i["asset"]) or {}).get("type") == "image")
            room = int(feats.get("max_refs") or 1) - n_user
            usable = []
            for r in refs_in:
                a = lookup(r["asset"])
                if not a or a.get("type") != "image":
                    skipped.append({"what": "refs", "why": f"reference {r.get('name') or r['asset']} is missing (deleted?)"})
                elif commercial_only and not (a.get("license") or {}).get("commercial", True):
                    skipped.append({"what": "refs", "why": f"reference {r.get('name') or r['asset']} is non-commercial and the commercial-only filter is on"})
                else:
                    usable.append((r, a))
            take = [r for r, _a in usable][:max(0, min(REF_CAP[strength], room))]
            if usable and not take:
                skipped.append({"what": "refs", "why": f"{mname} has no room for references ({n_user} input image(s), at most {feats.get('max_refs') or 1})"})
            if take:
                a0, b0 = n_user + 1, n_user + len(take)
                ref_clause = (f"Use image {a0}{'-' + str(b0) if b0 > a0 else ''} only as a style reference: match its line work, shading, colours, "
                              "lighting and background; do not draw its subject.")
                for k, r in enumerate(take):
                    inputs.append({"asset": r["asset"], "role": "style_ref"})
                    res["refs"].append({"asset": r["asset"], "index": a0 + k, "name": r.get("name")})
                added["refs"] = [r["asset"] for r in take]
                res["extra_seconds_est"] = SEC_PER_REF_IMAGE
                made = {(lookup(r["asset"]) or {}).get("model_name") or (lookup(r["asset"]) or {}).get("model") for r in take}
                made.discard(None)
                if made and (model or {}).get("name") not in made and (model or {}).get("id") not in made:
                    res["notes"].append(f"References were made with {', '.join(sorted(made))}; {mname} may look different.")
    # 6. PROMPT = user + sep + [ref clause] + style (never a prefix)
    tail = " ".join(x for x in (ref_clause, style_text) if x)
    if tail:
        params[pkey] = user.rstrip(" .,;") + pol["sep"] + tail
    added["prompt"], added["ref_clause"] = style_text, ref_clause
    # 7. NEGATIVE
    avoid = dedupe_ci([*(guide.get("avoid") or []), *v["avoid"]])
    neg_feat = feats.get("negative")
    active = neg_feat is True or (neg_feat == "cfg" and cfg_active(model, params))
    user_neg = str(params.get("negative") or "").strip()
    if avoid:
        if active:
            base = user_neg or BASE_NEGATIVE.get(_prompt_style(model, task) or "", "")
            merged, new = merge_terms(base, avoid)
            if new:
                params["negative"] = merged
                added["negative"] = new
                if not user_neg and base:
                    added["base_negative"] = base
        else:
            why = (f"{mname} uses the negative prompt only with CFG above 1 (that doubles the time)" if neg_feat == "cfg"
                   else f"{mname} has no negative prompt")
            skipped.append({"what": "avoid", "why": why})
    # 8. POST (palette)
    pal = guide.get("palette") or {}
    post = v["palette_post"] or pal.get("post") or "off"
    style_post = None
    if post != "off" and variant in PALETTE_VARIANTS and pal.get("colors"):
        style_post = {"mode": post, "strength": pal.get("tint_strength", 0.35), "colors": list(pal["colors"])}
    changed = bool(tail or added["negative"] or style_post)
    if not changed:
        return off("nothing to add for this variant") if not skipped else _off_with_skips(off, skipped)
    # 9. RECORD
    res.update(applied=True, params=params, inputs=inputs, prompt=params[pkey], negative=params.get("negative") or "", skipped=skipped,
               ref_clause=ref_clause, style_post=style_post)
    res["x_style"] = {"applied": True, **base_rec, "variant": variant, "strength": strength, "user_prompt": user, "user_negative": user_neg,
                      "added": added, "skipped": skipped, "post": None}
    res["summary"] = {**base_rec, "variant": variant, "applied": True}
    return res


def _off_with_skips(off, skipped):
    r = off("nothing to add for this variant")
    r["skipped"] = skipped
    r["x_style"]["skipped"] = skipped
    return r


def improve_line(guide: dict | None, task: str, req: dict | None = None, limit: int = 450) -> str:
    """One line for the prompt improver ('PROJECT STYLE ...'), or "" when the style is not applied to this job."""
    req = req or {}
    if not guide or not guide.get("enabled", True) or req.get("apply") is False:
        return ""
    pol = TASK_POLICY.get(task)
    if not pol or pol["variant"] == "skip":
        return ""
    variant = req.get("variant") if req.get("variant") in VARIANTS else pol["variant"]
    v = effective_variant(guide, variant)
    if not v["enabled"]:
        return ""
    pos = v["positive"] if v["mode"] == "replace" else _join(guide.get("positive"), v["positive"])
    bg = guide.get("background") if v["background"] is None else v["background"]
    avoid = dedupe_ci([*(guide.get("avoid") or []), *v["avoid"]])
    head = ("PROJECT STYLE (the studio appends it after your prompt: do NOT repeat these words; write the subject so it suits this style; "
            "follow it over a conflicting style hint): ")
    tail_av = (" AVOID: " + ", ".join(avoid) + ".") if avoid else ""
    tail_hint = " (no negative prompt on some models: describe the wanted look positively.)" if avoid else ""
    room = limit - len(head) - len(tail_av) - len(tail_hint)
    if room < 60 and tail_hint:
        tail_hint, room = "", limit - len(head) - len(tail_av)
    if room < 40:
        tail_av = (" AVOID: " + ", ".join(avoid[:3]) + ".") if avoid else ""
        room = limit - len(head) - len(tail_av) - len(tail_hint)
    body = _join(pos, f"background: {bg}" if bg else "", sep="; ")
    if len(body) > room:
        body = body[:max(0, room - 3)].rstrip(" ,;") + "..."
    line = head + body + "." + tail_av + tail_hint
    return line[:limit]


# ------------------------------------------------------------------ palette: Lab maths, extraction, tint / lock
def srgb2lab(a):
    import numpy as np

    a = np.asarray(a, dtype=float) / 255.0
    a = np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = a @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2srgb(lab):
    import numpy as np

    lab = np.asarray(lab, dtype=float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    Mi = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = xyz @ Mi.T
    rgb = np.where(rgb > 0.0031308, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055, 12.92 * rgb)
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


def kmeans_palette(im, n: int = 6, min_de: float = 15.0, iters: int = 12, seed: int = 0, ignore_background: bool = True) -> tuple[list[str], list[float]]:
    """Distinct colours of a PIL image: k-means (k = 16) in Lab on a 128 px thumbnail, then greedily keep clusters by weight that are at
    least `min_de` apart. Seeded (deterministic). With `ignore_background` the clusters that look like the corner colour are dropped,
    except the biggest one, which comes first (the background colour stays as one swatch). Returns (#rrggbb list, weights)."""
    import numpy as np
    from PIL import Image

    rgba = im.convert("RGBA")
    rgba.thumbnail((128, 128), Image.BILINEAR)
    arr = np.asarray(rgba)
    opaque = arr[..., 3] >= 128
    px = arr[..., :3][opaque] if opaque.sum() >= 64 else arr[..., :3].reshape(-1, 3)
    a = srgb2lab(px.astype(float)).reshape(-1, 3)
    k = min(16, len(a))
    rng = np.random.default_rng(seed)
    c = a[rng.choice(len(a), k, replace=False)]
    lab = np.zeros(len(a), dtype=int)
    for _ in range(iters):
        lab = ((a[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([a[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    w = np.bincount(lab, minlength=k)
    order = [int(j) for j in np.argsort(-w, kind="stable") if w[j] > 0]
    bg_set: list[int] = []
    h, wd = arr.shape[:2]
    cs = max(1, min(16, h // 4, wd // 4))
    corners = np.concatenate([arr[:cs, :cs].reshape(-1, 4), arr[:cs, -cs:].reshape(-1, 4), arr[-cs:, :cs].reshape(-1, 4), arr[-cs:, -cs:].reshape(-1, 4)])
    if ignore_background and (corners[:, 3] >= 128).all():
        cm = srgb2lab(corners[:, :3].astype(float)).mean(0)
        bg_set = [j for j in order if np.linalg.norm(c[j] - cm) < 10.0]
    keep: list[int] = []
    if bg_set:
        keep.append(bg_set[0])
    for j in order:
        if j in bg_set:
            continue
        if all(np.linalg.norm(c[j] - c[i]) >= min_de for i in keep):
            keep.append(j)
        if len(keep) >= n:
            break
    keep = keep[:n]
    rgb = lab2srgb(c[keep][None])[0]
    total = float(w.sum()) or 1.0
    return [rgb_to_hex(x) for x in rgb], [round(float(w[j]) / total, 3) for j in keep]


def extract_palette(img, n: int = 6, min_de: float = 15.0, ignore_background: bool = True) -> tuple[list[str], list[float]]:
    return kmeans_palette(img, n=max(1, min(int(n), LIM["colors"])), min_de=min_de, ignore_background=ignore_background)


def _split_alpha(im):
    """(RGB array float, alpha array uint8 | None) of a PIL image."""
    import numpy as np

    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        r = im.convert("RGBA")
        arr = np.asarray(r)
        return arr[..., :3].astype(float), arr[..., 3].copy()
    return np.asarray(im.convert("RGB")).astype(float), None


def _join_alpha(rgb, alpha):
    import numpy as np
    from PIL import Image

    out = Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB")
    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(Image.fromarray(alpha, "L"))
    return out


def tint(im, colors: list[str], strength: float = 0.35, weights: list[float] | None = None):
    """Move the image's colour statistics toward the palette (Reinhard mean / std transfer in Lab, spread clipped so a 2-colour palette
    cannot crush the contrast), blended by `strength`. Alpha is untouched."""
    import numpy as np

    rgb, alpha = _split_alpha(im)
    if not colors:
        return _join_alpha(rgb, alpha)
    src = srgb2lab(rgb)
    flat = src.reshape(-1, 3)
    mask = (alpha.reshape(-1) >= 8) if alpha is not None and (alpha >= 8).any() else np.ones(len(flat), bool)
    ms, ss = flat[mask].mean(0), flat[mask].std(0) + 1e-6
    pal = srgb2lab(np.array([hex_to_rgb(c) for c in colors], dtype=float))
    w = np.asarray(weights if weights and len(weights) == len(colors) else [1.0] * len(colors), dtype=float)
    w = w / (w.sum() or 1.0)
    mt = (pal * w[:, None]).sum(0)
    st = np.sqrt((((pal - mt) ** 2) * w[:, None]).sum(0))
    ratio = np.clip(st / ss, 0.75, 1.33)
    ratio[0] = 1.0                               # keep the lightness contrast of the picture; colour (a, b) and mean lightness move
    t = (src - ms) * ratio + mt
    out = src + (t - src) * float(strength)
    return _join_alpha(lab2srgb(out), alpha)


def lock(im, colors: list[str]):
    """Hard quantise to the palette (nearest colour in Lab, no dither). Alpha is kept."""
    import numpy as np

    rgb, alpha = _split_alpha(im)
    if not colors:
        return _join_alpha(rgb, alpha)
    pal_rgb = np.array([hex_to_rgb(c) for c in colors], dtype=np.uint8)
    pal = srgb2lab(pal_rgb.astype(float))
    flat = rgb.reshape(-1, 3)
    out = np.empty_like(flat, dtype=np.uint8)
    for i in range(0, len(flat), 1 << 17):
        lab = srgb2lab(flat[i:i + (1 << 17)])
        out[i:i + (1 << 17)] = pal_rgb[((lab[:, None, :] - pal[None]) ** 2).sum(-1).argmin(1)]
    return _join_alpha(out.reshape(rgb.shape), alpha)


def apply_post(im, mode: str, colors: list[str], strength: float = 0.35, weights=None):
    if mode == "off":
        return im
    if mode == "tint":
        return tint(im, colors, strength, weights)
    if mode == "lock":
        return lock(im, colors)
    raise ValueError(f"Unknown palette mode {mode!r}.")
