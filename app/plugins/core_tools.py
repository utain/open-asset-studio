"""Core CPU tools (and the reference for writing a plugin; see docs/CONTRACTS.md section 3.3).

    tool:image.convert   resize / re-encode an image asset (png | webp | jpg), optional square pad, keeps alpha
    tool:asset.export    derive audio / image assets with an export preset (loudness, sample rate, bit depth, format,
                         aspect crop, size) and save them as NEW assets (parents = the originals; originals untouched)
"""
from __future__ import annotations

from PIL import Image, ImageOps

from ..jobs import JobContext, JobError, handler


@handler("tool:image.convert", gpu=False, label="Convert image")
def image_convert(ctx: JobContext):
    p = ctx.params
    inputs = ctx.input_assets()
    if not inputs or inputs[0]["type"] != "image":
        raise JobError("Pick an image asset to convert.", "input")
    fmt = str(p.get("format", "png")).lower()
    if fmt not in ("png", "webp", "jpg"):
        raise JobError("format must be png, webp or jpg", "input")
    size = int(p.get("max_side", 0) or 0)
    if size and not 16 <= size <= 8192:
        raise JobError("max_side must be 16..8192", "input")
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    for i, inp in enumerate(inputs):
        ctx.check_cancel()
        ctx.progress(i / len(inputs), f"Converting {inp['name']}", stage="converting")
        with Image.open(inp["path"]) as im:
            im = ImageOps.exif_transpose(im)
            im.load()
        if size:
            im.thumbnail((size, size), Image.LANCZOS)
        if p.get("square"):
            side = max(im.size)
            canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            canvas.paste(im.convert("RGBA"), ((side - im.width) // 2, (side - im.height) // 2))
            im = canvas
        if fmt == "jpg":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").getchannel("A"))
            im = bg
        out = ctx.scratch / f"converted_{i + 1}.{fmt}"
        im.save(out, {"jpg": "JPEG", "png": "PNG", "webp": "WEBP"}[fmt], **({"quality": 92} if fmt != "png" else {}))
        meta = ctx.asset_meta({"name": f"{inp['name']} ({fmt}{f', {size}px' if size else ''})"}, i)
        meta["parents"] = [inp["asset_id"]]
        ctx.add_output(out, "image", meta)


@handler("tool:asset.export", gpu=False, label="Export preset (derived copy)")
def asset_export(ctx: JobContext):
    """params: {preset?: <id from app/profiles.py>, audio?: {...}, image?: {...}} (same body as POST /api/export `export`)."""
    from .. import profiles

    p = ctx.params
    inputs = ctx.input_assets()
    if not inputs:
        raise JobError("Pick at least one audio or image asset to export.", "input")
    try:
        spec = profiles.resolve_export({k: p[k] for k in ("preset", "audio", "image") if k in p}, ctx.settings)
    except ValueError as e:
        raise JobError(str(e), "input") from None
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    made = 0
    for i, inp in enumerate(inputs):
        ctx.check_cancel()
        ctx.progress(i / len(inputs), f"Exporting {inp['name']}", stage="converting")
        meta = ctx.mgr.assets.get(inp["asset_id"])
        try:
            res = ctx.mgr.assets.convert_main(meta, spec, ctx.scratch / f"x{i}", f"export_{i + 1}")
        except ValueError as e:
            raise JobError(f"{inp['name']}: {e}", "input") from None
        if res is None:
            ctx.log(f"{inp['name']}: nothing to convert for a {inp['type']} asset with this preset; skipped")
            continue
        out, report = res
        label = spec.get("preset") or "custom"
        m = ctx.asset_meta({"name": f"{inp['name']} ({label})"}, i)
        m["parents"] = [inp["asset_id"]]
        m["info"] = {"export": report}
        m["tags"] = sorted({*(m.get("tags") or []), "export", label})
        ctx.add_output(out, "voice" if inp["type"] == "voice" else inp["type"], m)
        made += 1
    if not made:
        raise JobError("None of the picked assets can be converted with this preset (only audio and images are).", "input")
