"""Pure numpy + PIL + stdlib helpers of the seamless PBR materials (workers/material_worker.py, app/plugins/material.py,
workers/qwen_worker.py, tests/test_material.py). No torch, no scipy: they import in envs/core.

Everything here treats an image as a torus (wrap-around), so a tileable input stays tileable. Arrays are float32 in [0, 1],
HxW (one channel) or HxWxC. Never use PIL.resize or scipy.ndimage with their default (clamping) edge modes on a tileable image:
use the periodic helpers below (FFT blur, np.roll, np.pad(mode="wrap"), `box_down`, `resize_wrap`).

Groups:
  basics        to_float, to_u8, to_u16, luminance, srgb_to_linear, linear_to_srgb, to_pil
  periodic      gauss_blur_wrap, box_down, resize_wrap, upscale_wrap, tile_grid, ramp, wrap_pad
  seams         seam_score, SEAM_OK, SEAM_WARN
  tileable      make_tileable_blend, wrap_fix
  maps          delight, height_levels, normal_from_height, height_from_normal, flip_green, renormalize_normal, normal_scale,
                ao_from_height, pack_orm, roughness_from_rgb
  SPAN weights  reparam_conv3xc (the PBRify checkpoints store stale eval_conv tensors: rebuild them)
  packing       LAYOUTS, layout_info, pack_layout (reuses engine_common.pack_textures: same channel conventions as the engine export)
  geometry      sample_mesh, gltf_document, write_gltf, write_glb (preview / sample meshes with tangents, analytic)
  preview       render_sphere (the thumbnail, also the normal-convention test)
Conventions (docs/CONTRACTS.md, "Seamless PBR materials"): normal maps are OpenGL (Y+) in storage, ORM = R occlusion, G roughness,
B metallic (linear), base colour is sRGB, height is 16-bit (0 = deepest).
"""
from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path

import numpy as np

SEAM_OK = 1.05          # <= seamless
SEAM_WARN = 1.25        # > visible seam; in between is acceptable


# ================================================================== basics
def to_float(im) -> np.ndarray:
    a = np.asarray(im)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    return a.astype(np.float32)


def to_u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def to_u16(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0, 1) * 65535.0 + 0.5).astype(np.uint16)


def to_pil(a: np.ndarray, bits: int = 8):
    """float array -> PIL image: 8-bit L / RGB, or 16-bit grayscale ('I;16') when bits == 16 and the array is 2-D."""
    from PIL import Image

    if bits == 16 and a.ndim == 2:
        return Image.fromarray(to_u16(a))
    return Image.fromarray(to_u8(a))


def luminance(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


# ================================================================== periodic filters (FFT = wrap-around by construction)
def gauss_blur_wrap(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with periodic boundaries (sigma in pixels). Works on HxW or HxWxC."""
    if sigma <= 0:
        return a.copy()
    h, w = a.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    g = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fy ** 2 + fx ** 2)).astype(np.float32)
    if a.ndim == 2:
        return np.fft.irfft2(np.fft.rfft2(a) * g, s=(h, w)).astype(np.float32)
    return np.stack([np.fft.irfft2(np.fft.rfft2(a[..., c]) * g, s=(h, w)) for c in range(a.shape[2])], -1).astype(np.float32)


def box_down(a: np.ndarray, f: int) -> np.ndarray:
    """Exact periodic box downscale by an integer factor (mean of f x f blocks). Keeps a seamless image seamless."""
    f = int(f)
    if f <= 1:
        return a
    h, w = a.shape[:2]
    if h % f or w % f:
        raise ValueError(f"box_down: {w}x{h} is not divisible by {f}.")
    if a.ndim == 2:
        return a.reshape(h // f, f, w // f, f).mean((1, 3)).astype(np.float32)
    return a.reshape(h // f, f, w // f, f, a.shape[2]).mean((1, 3)).astype(np.float32)


def wrap_pad(a: np.ndarray, p: int) -> np.ndarray:
    return np.pad(a, ((p, p), (p, p)) + ((0, 0),) * (a.ndim - 2), mode="wrap")


def resize_wrap(img, size: tuple[int, int], resample=None, pad: int = 8):
    """Wrap-safe resize of a PIL image: pad by wrapping, resize the padded image with the box that maps exactly the original
    area (PIL `box` takes float source coordinates), so the result is as periodic as the input. size = (w, h)."""
    from PIL import Image

    resample = Image.Resampling.BICUBIC if resample is None else resample
    a = np.asarray(img)
    w, h = img.size
    p = int(pad)
    padded = Image.fromarray(np.pad(a, ((p, p), (p, p)) + ((0, 0),) * (a.ndim - 2), mode="wrap"))
    return padded.resize((int(size[0]), int(size[1])), resample, box=(p, p, p + w, p + h))


def upscale_wrap(a: np.ndarray, f: int = 2, pad: int = 4) -> np.ndarray:
    """Wrap-safe bicubic upscale of a float map by an integer factor (per channel, PIL mode F)."""
    from PIL import Image

    h, w = a.shape[:2]
    chans = [a] if a.ndim == 2 else [a[..., c] for c in range(a.shape[2])]
    outs = []
    for ch in chans:
        p = np.pad(np.ascontiguousarray(ch, dtype=np.float32), pad, mode="wrap")
        im = Image.fromarray(p, "F").resize((w * f, h * f), Image.Resampling.BICUBIC, box=(pad, pad, pad + w, pad + h))
        outs.append(np.asarray(im, dtype=np.float32))
    out = outs[0] if a.ndim == 2 else np.stack(outs, -1)
    return np.clip(out, 0, 1)


def tile_grid(img: np.ndarray, n: int = 3) -> np.ndarray:
    a = np.asarray(img)
    return np.tile(a, (n, n, 1) if a.ndim == 3 else (n, n))


# ================================================================== seam metric
def seam_score(img) -> dict:
    """How visible the wrap-around seam is. Per axis: the mean |difference| across the wrap edge (last -> first column / row)
    divided by the 95th percentile of the same statistic over all interior column / row boundaries (p95, not the median, so strong but
    legitimate edges such as grout or plank lines do not flag). <= 1.05 seamless, 1.05-1.25 acceptable, > 1.25 a visible seam.
    `x_inner_max` / `y_inner_max` are the worst interior boundary (a blend cross shows up there).
    A pattern-sensitive hint (grout can score 1.2-1.4 without a visible seam): confirm with the 3x3 view."""
    a = to_float(img)
    if a.ndim == 3:
        a = luminance(a[..., :3])
    out = {}
    for axis, name in ((1, "x"), (0, "y")):
        b = a if axis == 1 else a.T
        inner = np.abs(np.diff(b, axis=1)).mean(axis=0)              # W-1 interior boundary strengths
        wrap = float(np.abs(b[:, 0] - b[:, -1]).mean())
        ref = float(np.percentile(inner, 95)) + 1e-6
        out[name] = round(wrap / ref, 3)
        out[f"{name}_inner_max"] = round(float(inner.max()) / ref, 3)
    out["score"] = max(out["x"], out["y"])
    return out


def seam_class(score: float) -> str:
    return "ok" if score <= SEAM_OK else "warn" if score <= SEAM_WARN else "bad"


# ================================================================== make an existing image tileable (no model)
def ramp(n: int, band: float = 0.25) -> np.ndarray:
    """1-D weight: 1 in the middle, smoothstep down to 0 at both ends over `band` (fraction of n)."""
    i = np.arange(n) + 0.5
    t = np.clip(np.minimum(i, n - i) / max(band * n, 1e-6), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _blend_axis(a: np.ndarray, b: np.ndarray, wt: np.ndarray, keep_contrast: bool) -> np.ndarray:
    if a.ndim == 3:
        wt = wt[..., None]
    if not keep_contrast:
        return a * wt + b * (1 - wt)
    mean = a.reshape(-1, a.shape[-1]).mean(0) if a.ndim == 3 else a.mean()
    num = (a - mean) * wt + (b - mean) * (1 - wt)
    return np.clip(mean + num / np.sqrt(wt ** 2 + (1 - wt) ** 2), 0, 1)


def make_tileable_blend(img, band: float = 0.25, keep_contrast: bool = True) -> np.ndarray:
    """Offset blend, one axis at a time (a 2-D product weight would leave the rolled copy's seam visible at the middle of each
    border). Pass x: blend with the copy rolled by half a width (continuous across the left / right wrap), weight ramp(w) per column.
    Pass y: the same with rows. A blend of two x-seamless images stays x-seamless. `keep_contrast`: variance-preserving blend
    (Heitz-Neyret style) instead of a washed-out linear blend. Keeps the original pixels in the middle of the image."""
    a = to_float(img)
    h, w = a.shape[:2]
    a = _blend_axis(a, np.roll(a, w // 2, 1), np.broadcast_to(ramp(w, band)[None, :], (h, w)), keep_contrast)
    a = _blend_axis(a, np.roll(a, h // 2, 0), np.broadcast_to(ramp(h, band)[:, None], (h, w)), keep_contrast)
    return a.astype(np.float32)


def wrap_fix(fn, img: np.ndarray, band: float = 0.25):
    """Make a non-periodic estimator's output tileable (phase 2, not used in v1: the SPAN nets are circular): run fn on the image
    rolled by (0 | h/2, 0 | w/2) (4 runs), un-roll, and blend bilinearly with ramp weights."""
    h, w = img.shape[:2]
    wx, wy = ramp(w, band)[None, :], ramp(h, band)[:, None]
    out = 0
    for sy, ky in ((0, wy), (h // 2, 1 - wy)):
        for sx, kx in ((0, wx), (w // 2, 1 - wx)):
            r = np.roll(fn(np.roll(img, (sy, sx), (0, 1))), (-sy, -sx), (0, 1))
            k = ky * kx
            out = out + r * (k[..., None] if r.ndim == 3 else k)
    return out


# ================================================================== de-lighting, height, normal, AO
def delight(rgb, sigma_frac: float = 0.08, strength: float = 1.0) -> np.ndarray:
    """Flatten large-scale lighting: divide the linear luminance by its periodic low-pass, keep the mean level. strength 0 = unchanged."""
    a = to_float(rgb)[..., :3]
    if strength <= 0:
        return a
    lin = srgb_to_linear(a)
    lum = luminance(lin) + 1e-4
    low = gauss_blur_wrap(lum, sigma_frac * max(lum.shape))
    gain = (lum.mean() / np.maximum(low, 1e-4)) ** strength
    return linear_to_srgb(lin * gain[..., None]).astype(np.float32)


def height_levels(h: np.ndarray, contrast: float = 1.0, min_range: float = 0.15) -> np.ndarray:
    """Levels of an estimated height map: p1..p99 stretched around their midpoint to 0..1. A near-flat map (a metal plate) is NOT
    stretched into noise: the range is at least `min_range`. `contrast` scales around the midpoint (0.5)."""
    h = to_float(h)
    lo, hi = np.percentile(h, [1, 99])
    rng = max(float(hi - lo), min_range)
    mid = float(lo + hi) / 2
    return np.clip(0.5 + (h - mid) / rng * contrast, 0, 1).astype(np.float32)


def normal_from_height(height: np.ndarray, strength: float = 4.0, y_down: bool = False) -> np.ndarray:
    """Tangent-space normal map (RGB in [0, 1]) from a height map: central differences with wrap-around. `strength` = height units
    per texel width (bigger = steeper). OpenGL convention (Y+, green up) unless y_down. Image rows go down, so the slope along
    +Y(up) is -d/drow: n = [-dx, +drow, 1] for OpenGL."""
    hgt = to_float(height)
    dx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) * 0.5 * strength
    dy = (np.roll(hgt, -1, 0) - np.roll(hgt, 1, 0)) * 0.5 * strength
    n = np.stack([-dx, dy, np.ones_like(hgt)], -1)
    if y_down:
        n[..., 1] *= -1
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def flip_green(normal_rgb: np.ndarray) -> np.ndarray:
    """OpenGL (Y+) <-> DirectX (Y-). An involution."""
    n = to_float(normal_rgb).copy()
    n[..., 1] = 1.0 - n[..., 1]
    return n


def renormalize_normal(normal_rgb: np.ndarray) -> np.ndarray:
    n = to_float(normal_rgb)[..., :3] * 2 - 1
    n[..., 2] = np.maximum(n[..., 2], 1e-3)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def normal_scale(normal_rgb: np.ndarray, k: float) -> np.ndarray:
    """Scale the slope of a normal map (x and y by k), then renormalise. k = 1 returns the input."""
    if abs(k - 1.0) < 1e-6:
        return to_float(normal_rgb)[..., :3]
    n = to_float(normal_rgb)[..., :3] * 2 - 1
    n[..., :2] *= k
    n[..., 2] = np.maximum(n[..., 2], 1e-3)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def height_from_normal(normal_rgb: np.ndarray, y_down: bool = False) -> np.ndarray:
    """Frankot-Chellappa integration in the Fourier domain: exact for periodic (tileable) maps. Output [0, 1]. OpenGL input
    unless y_down. Fallback height source (`height_source: from_normal`, hidden in v1)."""
    n = to_float(normal_rgb)[..., :3] * 2 - 1
    nz = np.maximum(n[..., 2], 0.05)
    p = -n[..., 0] / nz                                  # dh/dx
    q = (-n[..., 1] if y_down else n[..., 1]) / nz       # dh/drow (image rows go down)
    h, w = p.shape
    fy = np.fft.fftfreq(h)[:, None] * 2 * np.pi
    fx = np.fft.fftfreq(w)[None, :] * 2 * np.pi
    P, Q = np.fft.fft2(p), np.fft.fft2(q)
    den = fx ** 2 + fy ** 2
    den[0, 0] = 1
    Z = (-1j * fx * P - 1j * fy * Q) / den
    Z[0, 0] = 0
    z = np.real(np.fft.ifft2(Z))
    lo, hi = np.percentile(z, [0.5, 99.5])
    return np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def ao_from_height(height: np.ndarray, radii=(2, 6, 16, 40), strength: float = 1.0) -> np.ndarray:
    """Multi-scale cavity AO: how far each texel sits below its periodic neighbourhood average, summed over radii, normalised at
    p99.5, then 1 - 0.6 * strength * occlusion. 1 = open, 0 = occluded."""
    hgt = to_float(height)
    occ = np.zeros_like(hgt)
    for r in radii:
        occ += np.maximum(gauss_blur_wrap(hgt, r) - hgt, 0) / len(radii)
    occ = occ / max(float(np.percentile(occ, 99.5)), 1e-6)
    return np.clip(1 - strength * 0.6 * occ, 0, 1).astype(np.float32)


def pack_orm(ao: np.ndarray, rough: np.ndarray, metal: np.ndarray) -> np.ndarray:
    """glTF / Unreal ORM: R = occlusion, G = roughness, B = metallic (linear)."""
    return np.stack([to_float(ao), to_float(rough), to_float(metal)], -1)


def roughness_from_rgb(rgb: np.ndarray, bias: float = 0.0) -> np.ndarray:
    """Luminance of a PBRify roughness map (it comes as RGB), plus a user offset, clipped to 0.02..1."""
    return np.clip(luminance(to_float(rgb)[..., :3]) + bias, 0.02, 1.0).astype(np.float32)


# ================================================================== SPAN (PBRify) weights
def reparam_conv3xc(w1, b1, w2, b2, w3, b3, skw, skb):
    """numpy: the single 3x3 conv equal to a SPAN Conv3XC training block  conv1x1(w1,b1) -> conv3x3(w2,b2; no padding) -> conv1x1(w3,b3)
    on the (wrap- or zero-) padded input, plus the 1x1 skip (skw, skb).
    Shapes: w1 (A, I, 1, 1), w2 (B, A, 3, 3), w3 (O, B, 1, 1), skw (O, I, 1, 1). Returns (weight (O, I, 3, 3), bias (O,)).
    The PBRify checkpoints carry STALE `eval_conv` tensors (largest per-layer error 1.1 in NormalV3, 24 in the 4x upscaler; spandrel
    recomputes them in Conv3XC.update_params()), so the loader MUST rebuild them with this. Exact for interior / circular padding."""
    w1, w2, w3, skw = (np.asarray(x, np.float64) for x in (w1, w2, w3, skw))
    b1, b2, b3, skb = (np.asarray(x, np.float64) for x in (b1, b2, b3, skb))
    W = np.einsum("ob,bakl,ai->oikl", w3[:, :, 0, 0], w2, w1[:, :, 0, 0])
    W[:, :, 1, 1] += skw[:, :, 0, 0]
    B = b3 + w3[:, :, 0, 0] @ (b2 + np.einsum("bakl,a->b", w2, b1)) + skb
    return W.astype(np.float32), B.astype(np.float32)


# ================================================================== packing layouts (reuses the engine export's texture packer)
# id -> {label, target (engine_common.pack_textures target), pipeline (unity), normal (default convention), packed (what is packed), files}
LAYOUTS = {
    "gltf": {"label": "glTF / generic (ORM)", "target": "gltf", "normal": "gl", "packed": "orm.png: R occlusion, G roughness, B metallic",
             "summary": "basecolor, OpenGL normal and one ORM image; the glTF metallicRoughness + occlusion convention."},
    "unreal": {"label": "Unreal Engine (ORM, DirectX normal)", "target": "unreal", "normal": "dx",
               "packed": "T_<name>_ORM: R occlusion, G roughness, B metallic",
               "summary": "T_<name>_BC / _N (DirectX, green flipped) / _ORM."},
    "unity_urp": {"label": "Unity URP (metallic-smoothness)", "target": "unity", "pipeline": "urp", "normal": "gl",
                  "packed": "_MetallicSmoothness: RGB metallic, A = 1 - roughness; _Occlusion separate",
                  "summary": "<name>_BaseMap / _Normal / _MetallicSmoothness (alpha = smoothness) / _Occlusion."},
    "unity_hdrp": {"label": "Unity HDRP (mask map)", "target": "unity", "pipeline": "hdrp", "normal": "gl",
                   "packed": "_MaskMap: R metallic, G occlusion, B detail mask, A = 1 - roughness",
                   "summary": "<name>_BaseMap / _Normal / _MaskMap."},
    "godot": {"label": "Godot 4 (ORM)", "target": "godot", "normal": "gl", "packed": "<name>_orm: R occlusion, G roughness, B metallic",
              "summary": "<name>_albedo / _normal / _orm (tick ORM in the material)."},
    "separate": {"label": "Separate maps only", "target": None, "normal": "gl", "packed": "none",
                 "summary": "basecolor, normal, height, roughness, metallic and AO as single images."},
}
DEFAULT_LAYOUT = "gltf"


def layout_info(layout: str) -> dict:
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown material layout '{layout}'. Use one of: {', '.join(LAYOUTS)}.")
    return LAYOUTS[layout]


def normalize_convert(conv: dict | None, stored_size: int = 0) -> dict:
    """The `convert.material` block of an export request -> {layout, normal ('gl'|'dx'), size, height_bits, glb}. Unknown layout /
    normal / size raise ValueError. size 0 = keep; sizes above the stored one are ignored (never upscale)."""
    c = dict(conv or {})
    layout = c.get("layout") or DEFAULT_LAYOUT
    info = layout_info(layout)
    normal = c.get("normal") or info["normal"]
    if normal not in ("gl", "dx"):
        raise ValueError("material.normal must be 'gl' or 'dx'.")
    size = int(c.get("size") or 0)
    if size not in (0, 256, 512, 1024, 2048, 4096):
        raise ValueError("material.size must be 0 (keep), 256, 512, 1024, 2048 or 4096.")
    if stored_size and size >= stored_size:
        size = 0
    bits = int(c.get("height_bits") or 16)
    if bits not in (8, 16):
        raise ValueError("material.height_bits must be 8 or 16.")
    return {"layout": layout, "normal": normal, "size": size, "height_bits": bits, "glb": c.get("glb") is not False}


def pack_layout(maps: dict, layout: str, stem: str, normal: str | None = None, height_bits: int = 16, size: int = 0) -> list[tuple[str, object, dict]]:
    """Files of one exported material -> [(file name, PIL image, meta)].

    maps: float32 arrays {basecolor HxWx3 sRGB, normal HxWx3 OpenGL, height HxW, roughness HxW, metallic HxW, ao HxW} (a missing one is
    skipped). Engine layouts use `engine_common.pack_textures` (the same channel conventions as the game-engine export: Unreal ORM + DirectX
    normal, Unity URP metallic-smoothness + occlusion, HDRP mask map, Godot ORM, glTF ORM) for base colour, normal and the packed map;
    height, roughness, metallic and AO are added as single images (`<stem>_height.png` 16-bit, `<stem>_roughness.png` ...). The
    `separate` layout has no packed map. `size` > 0 downsizes every map first with the exact periodic box filter (never a clamped Lanczos)."""
    from PIL import Image

    info = layout_info(layout)
    nconv = normal or info["normal"]
    if size:
        f = max(1, next(iter(maps.values())).shape[0] // int(size))
        maps = {k: (box_down(v, f) if f > 1 and v.shape[0] % f == 0 else v) for k, v in maps.items()}
    out: list[tuple[str, object, dict]] = []

    def add(name, arr, slot, cs, desc, bits=8):
        img = to_pil(arr, bits)
        out.append((f"{name}.png", img, {"slot": slot, "colorspace": cs, "packing": desc, "size": list(img.size), "bits": bits}))

    bc, nm = maps.get("basecolor"), maps.get("normal")
    ao, ro, me = maps.get("ao"), maps.get("roughness"), maps.get("metallic")
    if info["target"]:
        import engine_common as ec

        eng = {"basecolor": to_pil(bc) if bc is not None else None, "normal": to_pil(nm) if nm is not None else None}
        if ro is not None or me is not None:
            eng["orm"] = to_pil(pack_orm(ao if ao is not None else np.ones_like(ro if ro is not None else me), ro if ro is not None else np.ones_like(me),
                                         me if me is not None else np.zeros_like(ro)))
        eng = {k: v for k, v in eng.items() if v is not None}
        opts = {"unity_pipeline": info.get("pipeline", "both"), "normal": nconv, "prefix": "T_", "tex_max": 0}
        for fname, img, meta in ec.pack_textures(info["target"], eng, stem, opts):
            out.append((fname, img, meta))
    else:
        if bc is not None:
            add(f"{stem}_basecolor", bc, "basecolor", "srgb", "RGB base colour")
        if nm is not None:
            add(f"{stem}_normal", flip_green(nm) if nconv == "dx" else nm, "normal", "linear",
                "normal map, DirectX (green flipped)" if nconv == "dx" else "normal map, OpenGL")
    if maps.get("height") is not None:
        add(f"{stem}_height", maps["height"], "height", "linear", f"grey height, {height_bits}-bit (0 = deepest)", height_bits)
    for key, arr in (("roughness", ro), ("metallic", me), ("ao", ao)):
        if arr is not None:
            add(f"{stem}_{key}", arr, key, "linear", f"grey {key}")
    seen, uniq = set(), []
    for item in out:                      # an engine name and a single-map name can coincide (glTF: _normal): keep the first
        if item[0].lower() not in seen:
            seen.add(item[0].lower())
            uniq.append(item)
    return uniq


# ================================================================== geometry: preview / sample meshes (numpy + stdlib)
def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _tangent_w(N, T, dPdv):
    """glTF tangent w: bitangent = cross(N, T) * w must point in the direction of the normal map's +Y (image up = decreasing v)."""
    up = -dPdv
    return np.where(np.einsum("ij,ij->i", np.cross(N, T), up) >= 0, 1.0, -1.0).astype(np.float32)


def _grid_indices(nu: int, nv: int, flip: bool = False) -> np.ndarray:
    """Triangles of a (nu+1) x (nv+1) vertex grid; vertex (iu, iv) is at iv * (nu + 1) + iu."""
    iv, iu = np.mgrid[0:nv, 0:nu]
    a = (iv * (nu + 1) + iu).ravel()
    b, c, d = a + 1, a + (nu + 1), a + (nu + 1) + 1
    tri = np.stack([a, c, b, b, c, d], -1) if not flip else np.stack([a, b, c, b, d, c], -1)
    return tri.reshape(-1).astype(np.uint32)


def sample_mesh(shape: str = "sphere", tiles: float = 1.0) -> dict:
    """A closed-form sample mesh with POSITION, NORMAL, TEXCOORD_0 (multiplied by `tiles`) and TANGENT (vec4, w = bitangent sign).
    sphere: UV sphere r = 0.5, 64 x 32 segments (U 0..2 so texels are square) · cube: 24 vertices, UV 0..1 per face, 1 m · plane: 32 x 32 grid, 2 m, facing +Z.
    Returns {positions (N,3) f32, normals, uvs (N,2), tangents (N,4), indices (M,) u32}."""
    tiles = float(tiles)
    if shape == "sphere":
        nu, nv, r = 64, 32, 0.5
        iv, iu = np.mgrid[0:nv + 1, 0:nu + 1]
        u = iu / nu
        v = iv / nv
        th, ph = v * math.pi, u * 2 * math.pi
        N = np.stack([np.sin(th) * np.cos(ph), np.cos(th), -np.sin(th) * np.sin(ph)], -1).reshape(-1, 3)   # u grows to the right seen from outside
        P = N * r
        T = np.stack([-np.sin(ph), np.zeros_like(ph), -np.cos(ph)], -1).reshape(-1, 3)     # dP/du, unit also at the poles
        dPdv = np.stack([np.cos(th) * np.cos(ph), -np.sin(th), -np.cos(th) * np.sin(ph)], -1).reshape(-1, 3)
        uv = np.stack([u.reshape(-1) * 2, v.reshape(-1)], -1)        # u runs twice around: the equator is twice as long as pole to pole, so texels stay square
        idx = _grid_indices(nu, nv)
        mesh = {"positions": P, "normals": N, "uvs": uv, "tangents": np.concatenate([T, _tangent_w(N, T, dPdv)[:, None]], -1), "indices": idx}
    elif shape == "plane":
        n, size = 32, 2.0
        iv, iu = np.mgrid[0:n + 1, 0:n + 1]
        u, v = iu / n, iv / n
        P = np.stack([(u - 0.5) * size, (0.5 - v) * size, np.zeros_like(u)], -1).reshape(-1, 3)
        N = np.tile(np.array([0, 0, 1.0]), (P.shape[0], 1))
        T = np.tile(np.array([1.0, 0, 0]), (P.shape[0], 1))
        dPdv = np.tile(np.array([0, -1.0, 0]), (P.shape[0], 1))
        mesh = {"positions": P, "normals": N, "uvs": np.stack([u.reshape(-1), v.reshape(-1)], -1),
                "tangents": np.concatenate([T, _tangent_w(N, T, dPdv)[:, None]], -1), "indices": _grid_indices(n, n)}
    elif shape == "cube":
        faces = [  # (normal, u axis (right in image), v axis (down in image))
            ((0, 0, 1), (1, 0, 0), (0, -1, 0)), ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
            ((1, 0, 0), (0, 0, -1), (0, -1, 0)), ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
            ((0, 1, 0), (1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, -1))]
        Ps, Ns, UVs, Ts, idx = [], [], [], [], []
        for k, (n, ua, va) in enumerate(faces):
            n, ua, va = (np.array(x, np.float64) for x in (n, ua, va))
            for cu, cv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                Ps.append(0.5 * n + (cu - 0.5) * ua + (cv - 0.5) * va)
                Ns.append(n)
                UVs.append((cu, cv))
                Ts.append(np.append(ua, _tangent_w(n[None], ua[None], va[None])[0]))
            b = 4 * k
            idx += [b, b + 1, b + 2, b, b + 2, b + 3] if np.dot(np.cross(Ps[b + 1] - Ps[b], Ps[b + 3] - Ps[b]), n) > 0 else [b, b + 2, b + 1, b, b + 3, b + 2]
        mesh = {"positions": np.array(Ps), "normals": np.array(Ns), "uvs": np.array(UVs, np.float64), "tangents": np.array(Ts),
                "indices": np.array(idx, np.uint32)}
    else:
        raise ValueError(f"Unknown shape '{shape}' (sphere, cube or plane).")
    mesh["uvs"] = (np.asarray(mesh["uvs"], np.float64) * tiles)
    out = {k: np.ascontiguousarray(v, dtype=np.uint32 if k == "indices" else np.float32) for k, v in mesh.items()}
    out["shape"], out["tiles"] = shape, tiles
    return out


def gltf_document(mesh: dict, textures: dict, *, name: str = "material", buffer_uri: str | None = None, images_embedded: dict | None = None,
                  buffer_bytes: bytes | None = None, double_sided: bool = False) -> tuple[dict, bytes]:
    """glTF JSON + the geometry blob. `textures`: {basecolor, normal, orm} -> image URIs (relative file names, or None).
    `images_embedded` {slot: PNG bytes} (GLB) appends the images to the blob and references them by buffer view.
    Material: baseColorTexture = basecolor, metallicRoughnessTexture = orm (G roughness, B metallic), occlusionTexture = orm (R),
    normalTexture = normal (OpenGL). Sampler wrap = REPEAT (glTF default)."""
    parts, views, accessors = [], [], []
    off = 0

    def add_view(data: bytes, target=None):
        nonlocal off
        pad = (-off) % 4
        if pad:
            parts.append(b"\0" * pad)
            off += pad
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), **({"target": target} if target else {})})
        parts.append(data)
        off += len(data)
        return len(views) - 1

    def add_acc(arr, ctype, atype, target, minmax=False):
        bv = add_view(arr.tobytes(), target)
        acc = {"bufferView": bv, "componentType": ctype, "count": int(arr.shape[0]), "type": atype}
        if minmax:
            acc["min"], acc["max"] = [float(x) for x in arr.min(0)], [float(x) for x in arr.max(0)]
        accessors.append(acc)
        return len(accessors) - 1

    pos = add_acc(mesh["positions"], 5126, "VEC3", 34962, True)
    nor = add_acc(mesh["normals"], 5126, "VEC3", 34962)
    uv = add_acc(mesh["uvs"], 5126, "VEC2", 34962)
    tan = add_acc(mesh["tangents"], 5126, "VEC4", 34962)
    ind = add_acc(mesh["indices"], 5125, "SCALAR", 34963)
    images, texs = [], {}
    for slot in ("basecolor", "normal", "orm"):
        if images_embedded and slot in images_embedded:
            images.append({"bufferView": add_view(images_embedded[slot]), "mimeType": "image/png", "name": slot})
        elif textures.get(slot):
            images.append({"uri": textures[slot], "name": slot})
        else:
            continue
        texs[slot] = len(images) - 1
    mat = {"name": name, "pbrMetallicRoughness": {"metallicFactor": 1.0, "roughnessFactor": 1.0}, "doubleSided": bool(double_sided)}
    if "basecolor" in texs:
        mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": texs["basecolor"]}
    else:
        mat["pbrMetallicRoughness"]["baseColorFactor"] = [0.8, 0.8, 0.8, 1.0]
    if "orm" in texs:
        mat["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": texs["orm"]}
        mat["occlusionTexture"] = {"index": texs["orm"], "strength": 1.0}
    if "normal" in texs:
        mat["normalTexture"] = {"index": texs["normal"], "scale": 1.0}
    blob = b"".join(parts)
    if len(blob) % 4:
        blob += b"\0" * (4 - len(blob) % 4)
    buf = {"byteLength": len(blob)}
    if buffer_uri is not None:
        buf["uri"] = buffer_uri
    doc = {"asset": {"version": "2.0", "generator": "Open Asset Studio (materials)"}, "scene": 0,
           "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": mesh.get("shape", "mesh")}],
           "meshes": [{"name": mesh.get("shape", "mesh"), "primitives": [{"attributes": {"POSITION": pos, "NORMAL": nor, "TEXCOORD_0": uv, "TANGENT": tan},
                                                                          "indices": ind, "material": 0}]}],
           "materials": [mat], "buffers": [buf], "bufferViews": views, "accessors": accessors}
    if images:
        doc["images"] = images
        doc["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
        doc["textures"] = [{"source": i, "sampler": 0} for i in range(len(images))]
        # texture index == image index here, in the order basecolor / normal / orm (those present)
    return doc, blob


def write_gltf(path, mesh: dict, textures: dict, name: str = "material", double_sided: bool = False) -> Path:
    """A .gltf with the geometry as a base64 data URI and the maps by relative URI (e.g. 'basecolor.png'). Used by the drawer / studio
    viewer only (the maps are served next to it)."""
    doc, blob = gltf_document(mesh, textures, name=name, double_sided=double_sided)
    doc["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
    p = Path(path)
    p.write_text(json.dumps(doc, separators=(",", ":")))
    return p


def write_glb(path, mesh: dict, images: dict, name: str = "material", double_sided: bool = False) -> Path:
    """A .glb with the textures embedded. `images` {basecolor, normal, orm}: PNG bytes."""
    doc, blob = gltf_document(mesh, {}, name=name, images_embedded=images, double_sided=double_sided)
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    p = Path(path)
    with p.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\0") + blob)
    return p


# ================================================================== preview thumbnail
def render_sphere(alb, nrm_gl, rough, metal, ao, tiles: float = 1.0, size: int = 360, light=(-0.5, 0.6, 0.8)) -> np.ndarray:
    """A tiny Blinn-Phong sphere (the material thumbnail, preview.png). Raised features must look raised with an OpenGL normal map:
    this doubles as the convention test. Returns float RGB (size, size, 3) in sRGB."""
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx + 0.5) / size * 2 - 1
    y = 1 - (yy + 0.5) / size * 2
    r2 = x * x + y * y
    m = r2 < 1
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    N = np.stack([x, y, z], -1)
    u = (np.arctan2(x, z) / (2 * np.pi) + 0.5) * tiles * 2
    v = (np.arccos(np.clip(y, -1, 1)) / np.pi) * tiles
    H, W = alb.shape[:2]
    iu = (np.floor(u * W).astype(int)) % W
    iv = (np.floor(v * H).astype(int)) % H
    A = alb[iv, iu]
    Nt = nrm_gl[iv, iu] * 2 - 1
    R, M, O = rough[iv, iu], metal[iv, iu], ao[iv, iu]
    T = np.stack([np.cos(np.arctan2(x, z)), np.zeros_like(x), -np.sin(np.arctan2(x, z))], -1)   # dP/du direction
    B = np.cross(N, T)                                                                            # image-up (OpenGL +V)
    n = Nt[..., :1] * T + Nt[..., 1:2] * B + Nt[..., 2:3] * N
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    L = np.array(light) / np.linalg.norm(light)
    V = np.array([0, 0, 1.0])
    Hh = (L + V) / np.linalg.norm(L + V)
    ndl = np.clip((n * L).sum(-1), 0, 1)
    ndh = np.clip((n * Hh).sum(-1), 0, 1)
    a = np.clip(R, 0.05, 1) ** 2
    spec_pow = 2 / (a * a) - 2
    spec = (ndh ** spec_pow) * (spec_pow + 2) / 8
    lin = srgb_to_linear(A)
    F0 = 0.04 * (1 - M[..., None]) + lin * M[..., None]
    col = lin * (1 - M[..., None]) * ndl[..., None] + F0 * spec[..., None] * ndl[..., None] + 0.08 * lin * O[..., None]
    img = linear_to_srgb(col * 1.6)
    img[~m] = 0.18
    return img.astype(np.float32)
