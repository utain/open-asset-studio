"""Image helpers of the Image Studio: alpha analysis, checkerboard preview, crop/pad, mask handling, Real-ESRGAN upscale.

Pure PIL / numpy except `Upscaler` (torch + realesrgan, runs in the app process under the shared GPU lock).
"""
from __future__ import annotations

import gc
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

MAX_UPSCALE_SIDE = 8192  # refuse outputs larger than this (a 4x of a 2048px image is already 8192)
MAX_INPUT_PIXELS = 64_000_000  # PIL decompression-bomb style guard for uploads


# ------------------------------------------------------------------------ alpha
def alpha_stats(img: Image.Image) -> tuple[bool, float]:
    """(has real transparency, fraction of pixels that are clearly transparent).

    Qwen returns RGBA even for ordinary images (alpha ~250..255 noise), so mode alone says nothing."""
    if img.mode not in ("RGBA", "LA", "PA") and "transparency" not in img.info:
        return False, 0.0
    a = np.asarray(img.convert("RGBA").getchannel("A"))
    return bool((a < 250).mean() > 0.002), float((a < 128).mean())


def normalize_alpha(img: Image.Image) -> Image.Image:
    """RGBA -> RGBA when it really has transparency, otherwise RGB (drops the near-opaque alpha noise)."""
    if img.mode == "RGBA":
        return img if alpha_stats(img)[0] else img.convert("RGB")
    return img if img.mode in ("RGB", "RGBA") else img.convert("RGBA" if "transparency" in img.info else "RGB")


def clean_alpha(img: Image.Image, thr: int) -> Image.Image:
    """Zero the alpha of pixels at or below `thr` (/255). Qwen leaves a 1-2/255 haze over most of a 'transparent'
    image, which makes any tool that tests alpha > 0 (including Hunyuan3D's recentering) see a full-frame object."""
    if thr <= 0 or img.mode != "RGBA":
        return img
    a = np.asarray(img.getchannel("A"))
    if not (a <= thr).any():
        return img
    out = img.copy()
    out.putalpha(Image.fromarray(np.where(a <= thr, 0, a).astype(np.uint8), "L"))
    return out


def describe(img: Image.Image) -> str:
    has, frac = alpha_stats(img)
    return f"{img.width} x {img.height} px, " + (f"RGBA, {frac * 100:.0f}% transparent" if has else "RGB, no transparency")


def flatten(img: Image.Image, color=(255, 255, 255)) -> Image.Image:
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (*color, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def checker(size: tuple[int, int], cell: int = 12) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((yy // cell + xx // cell) % 2).astype(np.uint8)
    arr = np.where(m[..., None] == 0, 204, 245).astype(np.uint8)
    return Image.fromarray(np.repeat(arr, 3, axis=2), "RGB")


def on_checker(img: Image.Image, cell: int = 12) -> Image.Image:
    img = img.convert("RGBA")
    return Image.alpha_composite(checker(img.size, cell).convert("RGBA"), img).convert("RGB")


def make_thumb(img: Image.Image, side: int = 288) -> Image.Image:
    """Gallery thumbnail; transparent images are shown on a checkerboard so the transparency is visible."""
    t = img.copy()
    t.thumbnail((side, side), Image.LANCZOS)
    return on_checker(t, 10) if alpha_stats(img)[0] else t.convert("RGB")


def open_checked(src, max_pixels: int = MAX_INPUT_PIXELS) -> Image.Image:
    """Open an image from a path / PIL image with a size guard, EXIF orientation applied; returns a loaded copy."""
    from PIL import ImageOps

    if isinstance(src, Image.Image):
        im = src
    else:
        with Image.open(src) as f:
            if f.width * f.height > max_pixels:
                raise ValueError(f"Image too large ({f.width}x{f.height}); the limit is {max_pixels // 10**6} megapixels.")
            f.load()
            im = f.copy()
    if im.width * im.height > max_pixels:
        raise ValueError(f"Image too large ({im.width}x{im.height}); the limit is {max_pixels // 10**6} megapixels.")
    im = ImageOps.exif_transpose(im)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if im.mode in ("LA", "PA") or "transparency" in im.info else "RGB")
    return im


# ------------------------------------------------------------------ crop / pad
def object_bbox(img: Image.Image, thr: int = 16) -> tuple[int, int, int, int] | None:
    a = np.asarray(img.convert("RGBA").getchannel("A"))
    ys, xs = np.nonzero(a > thr)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_pad_square(img: Image.Image, margin: float = 0.15, out_size: int = 0, background: str = "transparent",
                    thr: int = 16) -> Image.Image:
    """Crop to the object (alpha > thr), centre it on a square canvas with `margin` (fraction of the canvas, same meaning
    as Hunyuan's border ratio: the object spans at most 1 - margin of the side) and optionally resize."""
    img = img.convert("RGBA")
    bb = object_bbox(img, thr)
    if bb is None:
        raise ValueError("No object found: the image is empty or fully transparent.")
    obj = img.crop(bb)
    side_obj = max(obj.size)
    margin = min(max(float(margin), 0.0), 0.45)
    side = max(1, int(round(side_obj / (1.0 - margin))))
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    canvas.alpha_composite(obj, ((side - obj.width) // 2, (side - obj.height) // 2))
    if out_size and int(out_size) != side:
        canvas = canvas.resize((int(out_size), int(out_size)), Image.LANCZOS)
    return flatten(canvas) if background == "white" else canvas


# --------------------------------------------------------------------- masks
def strokes_mask(layers, threshold: int = 8) -> Image.Image | None:
    """Union of the painted layers of an ImageEditor value as an 'L' mask (255 where something was drawn), or None."""
    mask = None
    for layer in layers or []:
        if layer is None:
            continue
        a = np.asarray(Image.fromarray(layer) if isinstance(layer, np.ndarray) else layer).astype(np.uint8)
        if a.ndim == 3 and a.shape[2] == 4:
            m = a[..., 3] > threshold
        else:
            continue
        mask = m if mask is None else (mask | m)
    if mask is None or not mask.any():
        return None
    return Image.fromarray((mask * 255).astype(np.uint8), "L")


def paste_back(original: Image.Image, edited: Image.Image, mask: Image.Image, feather: float = 6.0, grow: int = 9) -> Image.Image:
    """Hard guarantee for local edits: keep `original` outside the (slightly grown, feathered) mask."""
    edited = edited.convert("RGBA")
    original = original.convert("RGBA").resize(edited.size, Image.LANCZOS)
    m = mask.resize(edited.size, Image.BILINEAR)
    if grow > 0:
        m = m.filter(ImageFilter.MaxFilter(grow | 1))
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(edited, original, m)


# -------------------------------------------------------------------- upscale
class Upscaler:
    """Real-ESRGAN x4plus (weights shipped with the Hunyuan3D checkout), tiled, with alpha support."""

    def __init__(self, ckpt: Path):
        self.ckpt = Path(ckpt)
        self.net = None
        self.lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.net is not None

    def _load(self):
        if self.net is None:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            if not self.ckpt.exists():
                raise FileNotFoundError(f"Real-ESRGAN weights not found: {self.ckpt}")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.net = RealESRGANer(scale=4, model_path=str(self.ckpt), model=model, tile=0, tile_pad=10, pre_pad=0,
                                    half=torch.cuda.is_available(), gpu_id=None)
        return self.net

    def unload(self):
        self.net = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def __call__(self, img: Image.Image, scale: int = 4, tile: int = 0, progress=None) -> Image.Image:
        """scale 2 or 4. x2 = x4 network output resized back by half (Lanczos), the usual way to get a cleaner x2."""
        if scale not in (2, 4):
            raise ValueError("Scale must be 2 or 4.")
        if max(img.size) * scale > MAX_UPSCALE_SIDE:
            raise ValueError(f"The result would be {img.width * scale} x {img.height * scale} px; the limit is "
                             f"{MAX_UPSCALE_SIDE} px per side. Use x2 or a smaller image.")
        with self.lock:
            net = self._load()
            net.tile_size = int(tile)  # RealESRGANer reads tile_size (set once in its constructor), not `tile`
            has_alpha = alpha_stats(img)[0]
            rgba = img.convert("RGBA")
            rgb = rgba.convert("RGB")
            if has_alpha:
                # colours under fully transparent pixels are garbage (Qwen leaves purple there): bleed the visible edge
                # colour outwards first so it cannot leak into the upscaled edge
                rgb = _bleed(rgba)
            if progress:
                progress(0.1, "Upscaling")
            out, _ = net.enhance(np.asarray(rgb)[..., ::-1].copy(), outscale=4)  # basicsr works in BGR
            out = Image.fromarray(np.asarray(out)[..., ::-1].copy())
            if scale == 2:
                out = out.resize((img.width * 2, img.height * 2), Image.LANCZOS)
            if has_alpha:
                a = rgba.getchannel("A").resize(out.size, Image.BICUBIC)
                out = out.convert("RGBA")
                out.putalpha(a)
            return out


def _bleed(rgba: Image.Image, iterations: int = 24) -> Image.Image:
    """Fill the RGB of (near-)transparent pixels with nearby visible colours by repeated dilation."""
    arr = np.asarray(rgba).astype(np.float32)
    rgb, a = arr[..., :3], arr[..., 3] / 255.0
    known = a > 0.5
    if known.all() or not known.any():
        return Image.fromarray(rgb.astype(np.uint8), "RGB")
    filled = np.where(known[..., None], rgb, 0.0)
    cnt = known.astype(np.float32)
    k = np.ones((3, 3), np.float32)
    from scipy.ndimage import convolve

    for _ in range(iterations):
        if (cnt > 0).all():
            break
        sums = np.stack([convolve(filled[..., c] * (cnt > 0), k, mode="nearest") for c in range(3)], -1)
        n = convolve((cnt > 0).astype(np.float32), k, mode="nearest")
        new = (cnt == 0) & (n > 0)
        filled[new] = sums[new] / n[new][:, None]
        cnt = np.where(new, 1.0, cnt)
    out = np.where(known[..., None], rgb, filled)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
