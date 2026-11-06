"""Single source of truth for every tunable knob: defaults, ranges, presets.

The UI builds its controls with these defaults and the engine calls `normalize()`
on whatever it receives (UI, API client, params.json from history), so an
out-of-range or missing value can never reach a pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_SEED = 2**32 - 1
APP_VERSION = "0.1.0"


@dataclass(frozen=True)
class Spec:
    default: Any
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


def _s(default, lo=None, hi=None, choices=None) -> Spec:
    return Spec(default, lo, hi, tuple(choices) if choices else None)


# Order matters: it is the order of the UI controls passed to handlers.
SPECS: dict[str, Spec] = {
    # ---- input / shape -------------------------------------------------
    "remove_bg": _s(True),
    "border_ratio": _s(0.15, 0.0, 0.3),
    "seed": _s(1234, 0, MAX_SEED),
    "randomize_seed": _s(True),
    "steps": _s(30, 1, 100),
    "guidance": _s(5.0, -1.0, 20.0),
    "octree": _s(256, 64, 512),
    "num_chunks": _s(200_000, 1000, 2_000_000),
    "box_v": _s(1.01, 0.9, 1.5),
    "mc_level": _s(0.0, -0.1, 0.1),
    "sigma_shift": _s(1.0, 0.5, 6.0),
    "flashvdm": _s(False),
    "shape_cleanup": _s(True),
    # ---- texture -------------------------------------------------------
    "tex_faces": _s(40_000, 2000, 150_000),
    "tex_views": _s(8, 6, 9),
    "tex_resolution": _s(512, choices=(512, 768)),
    "tex_steps": _s(15, 4, 50),
    "tex_guidance": _s(3.0, 1.5, 8.0),
    "tex_seed": _s(0, 0, MAX_SEED),
    "tex_randomize_seed": _s(False),
    "tex_render_size": _s("auto", choices=("auto", 1024, 2048, 3072, 4096)),
    "tex_size": _s(4096, choices=(2048, 4096, 8192)),
    "tex_full_res": _s(False),
    "bake_exp": _s(4, 1, 10),
    "bake_angle": _s(75, 50, 89),
    "super_res": _s(True),
    "sr_tile": _s(0, choices=(0, 256, 512)),
    "vertex_inpaint": _s(True),
    "uv_padding": _s(0, 0, 8),
    "tex_reference": _s("cleaned", choices=("cleaned", "original", "current")),
    "metallic_scale": _s(1.0, 0.0, 1.0),
    "roughness_scale": _s(1.0, 0.0, 1.0),
    # ---- post-process / export ----------------------------------------
    "post_floaters": _s(True),
    "floater_ratio": _s(0.005, 0.0005, 0.1),
    "post_degenerate": _s(True),
    "post_simplify": _s(False),
    "post_target_faces": _s(10_000, 100, 2_000_000),
    "post_smooth": _s(0, 0, 50),
    "post_fix_normals": _s(False),
    "export_type": _s("glb", choices=("glb", "obj", "ply", "stl")),
    "include_texture": _s(False),
    # ---- misc ----------------------------------------------------------
    "preset": _s("Balanced", choices=("Fast", "Balanced", "Quality", "Custom")),
}
PARAM_KEYS: list[str] = list(SPECS)
DEFAULTS: dict[str, Any] = {k: s.default for k, s in SPECS.items()}

SHAPE_KEYS = ["remove_bg", "border_ratio", "seed", "randomize_seed", "steps", "guidance", "octree",
              "num_chunks", "box_v", "mc_level", "sigma_shift", "flashvdm", "shape_cleanup", "floater_ratio"]

PRESETS: dict[str, dict[str, Any]] = {
    "Fast": dict(steps=20, guidance=5.0, octree=196, num_chunks=200_000,
                 tex_faces=20_000, tex_views=6, tex_resolution=512, tex_steps=15, tex_guidance=3.0,
                 tex_render_size="auto", tex_size=2048, tex_full_res=False, super_res=True, sr_tile=0),
    "Balanced": dict(steps=30, guidance=5.0, octree=256, num_chunks=200_000,
                     tex_faces=40_000, tex_views=8, tex_resolution=512, tex_steps=15, tex_guidance=3.0,
                     tex_render_size="auto", tex_size=4096, tex_full_res=False, super_res=True, sr_tile=0),
    "Quality": dict(steps=50, guidance=5.0, octree=384, num_chunks=200_000,
                    tex_faces=60_000, tex_views=9, tex_resolution=768, tex_steps=25, tex_guidance=3.0,
                    tex_render_size="auto", tex_size=4096, tex_full_res=True, super_res=True, sr_tile=512),
}


def _coerce(spec: Spec, value: Any, default: Any):
    """Cast to the type of the default and clamp / validate against the spec."""
    if value is None:
        return default
    if spec.choices is not None:
        for c in spec.choices:  # also accepts "512" for 512 (dropdowns return strings)
            if value == c or str(value) == str(c):
                return c
        return default
    if isinstance(default, bool):
        return bool(value)
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if spec.lo is not None:
        num = max(spec.lo, num)
    if spec.hi is not None:
        num = min(spec.hi, num)
    return int(round(num)) if isinstance(default, int) else float(num)


def normalize(raw: dict | None) -> dict:
    """Return a complete, clamped parameter dict (unknown keys are dropped)."""
    raw = raw or {}
    out = {k: _coerce(s, raw.get(k), s.default) for k, s in SPECS.items()}
    if out["guidance"] < 1.0:  # the vendor enables CFG for any value >= 0, and 0..1 ignores the image: snap to "off"
        out["guidance"] = -1.0
    return out


# Radio / Dropdown controls whose choices are numbers are exposed to the UI and API as strings
# (Gradio validates against the advertised choices; normalize() maps them back to numbers).
STR_CHOICE_KEYS = ("tex_resolution", "tex_size", "sr_tile")


def ui_value(key: str, value):
    """Value to put into a UI control (numeric choice controls use string choices)."""
    return str(value) if key in STR_CHOICE_KEYS and value is not None else value


def from_values(values) -> dict:
    """Positional UI values -> normalized dict."""
    return normalize(dict(zip(PARAM_KEYS, values)))


def to_values(params: dict) -> list:
    p = normalize(params)
    return [p[k] for k in PARAM_KEYS]


def effective_render_size(p: dict) -> int:
    """'auto' = 4x multiview resolution, which sidesteps the vendor's partial-resize quirk."""
    rs = p["tex_render_size"]
    return 4 * int(p["tex_resolution"]) if rs == "auto" else int(rs)


def set_default(key: str, value) -> None:
    """Change a default at start-up (used by CLI flags, before the UI is built)."""
    from dataclasses import replace

    SPECS[key] = replace(SPECS[key], default=value)
    DEFAULTS[key] = value
