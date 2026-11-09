"""Seamless PBR materials: handlers, the material export and a small router. docs/research/materials.md has the measurements.

  handler material.make        model `material-zimage`. Text, photo or an existing material in, ONE `material` asset per take out:
                               basecolor.png (main) + normal.png (OpenGL) + height.png (16-bit) + roughness / metallic / ao / orm PNGs +
                               preview.png (sphere) + preview_{sphere,cube,plane}.gltf + material.json. Meta `x_material`.
                               Steps: tileable Z-Image base colour (latent rolling + circular VAE, worker `qwen`) -> project palette (style guide)
                               -> PBRify SPAN maps (worker `material`, beside it in the same venv) -> CPU post (de-light, normal strength, height
                               levels, AO from height, metallic constant, ORM, 4k wrap-safe upscale of the data maps) -> files.
  handler material.tileable    model `tileable-blend` (CPU) -> a NEW image asset "<name> (tileable)" per input image (offset blend).
  handler tool:material.sample_glb   a material asset -> a NEW mesh asset: sphere / cube / plane, UV tiles 1..8, textures embedded.
  router /api/x/material       GET /layouts   export layouts (id, label, normal convention, packing)
                               POST /repack   {id, layout, normal, size, glb?} -> a zip of one material in an engine layout
  exporter assets.EXPORTERS["material"]   zip layout `materials/<stem>/...` (docs/CONTRACTS.md section 13).

Every spatial operation is periodic (wrap padding / FFT / exact box filter), so the result tiles. Stored normal maps are OpenGL (Y+); the
DirectX flip happens only at export. The licence of a result is the most restrictive of the inputs and the models.
"""
from __future__ import annotations

import json
import logging
import random
import re
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from .. import assets as assets_mod
from .. import config
from .. import style_common as sc
from ..assets import AssetError
from ..jobs import JobContext, JobError, handler

log = logging.getLogger("studio")
router = APIRouter()
CORE = None
STYLE_TASKS = {"material": "texture"}      # the project style guide's "texture" variant (app/plugins/style.py picks this up lazily)

MATERIAL_SUFFIX = ", seamless tileable texture, top-down orthographic view, flat even diffuse lighting, no shadows, no perspective, no objects"
GEN_MODEL = "z-image-turbo"
PREVIEW_SHAPES = ("sphere", "cube", "plane")
MAP_FILES = {"basecolor": "basecolor.png", "normal": "normal.png", "height": "height.png", "roughness": "roughness.png",
             "metallic": "metallic.png", "ao": "ao.png", "orm": "orm.png"}
MAT_VERSION = 1


def _mc():
    if str(config.WORKERS) not in sys.path:
        sys.path.append(str(config.WORKERS))
    import material_common
    return material_common


def _lic(parents: list) -> dict | None:
    """The most restrictive licence (non-commercial wins); in a tie the first one, so the source asset comes first (same as retopo)."""
    lics = [x for x in parents if x]
    nc = [x for x in lics if not x.get("commercial", True)]
    return (nc or lics or [None])[0]


def with_suffix(prompt: str) -> str:
    """The prompt plus the texture framing terms it does not already contain (so a re-run of a stored prompt does not repeat them)."""
    low = prompt.lower()
    extra = [t.strip() for t in MATERIAL_SUFFIX.split(",") if t.strip() and t.strip().lower() not in low]
    return (prompt.rstrip().rstrip(",") + ", " + ", ".join(extra)) if extra else prompt


def square_1024(path, mc):
    """Centre-crop to a square and resize to 1024 (PIL, RGB)."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        s = min(im.size)
        l, t = (im.width - s) // 2, (im.height - s) // 2
        im = im.crop((l, t, l + s, t + s))
        return im.resize((1024, 1024), Image.Resampling.LANCZOS) if s != 1024 else im.copy()


def _first_file(res: dict, what: str) -> Path:
    for f in res.get("files") or []:
        if f.get("type", "image") == "image":
            return Path(f["path"])
    raise JobError(f"The {what} step returned no image.", "internal")


# ================================================================== material.make
def build_maps(mc, alb, raw_normal_dx, raw_height, raw_rough, p: dict, map_res: int):
    """Raw worker maps + user parameters -> final float maps at the base-colour size. Pure numpy, no GPU. Returns a dict with
    basecolor, normal (OpenGL), height, roughness, metallic, ao, orm (all float32 [0,1])."""
    import numpy as np

    size = alb.shape[0]
    base = mc.delight(alb, 0.08, float(p.get("delight", 0.5)))
    normal = mc.renormalize_normal(mc.flip_green(raw_normal_dx))                  # PBRify writes DirectX; stored maps are OpenGL
    k = float(p.get("normal_strength", 1.0))
    normal = mc.normal_scale(normal, k) if abs(k - 1.0) > 1e-6 else normal
    height = mc.height_levels(mc.luminance(raw_height), float(p.get("height_contrast", 1.0)))
    rough = np.clip(mc.luminance(raw_rough) + float(p.get("roughness_bias", 0.0)), 0.02, 1.0).astype(np.float32)
    s = map_res / 1024.0
    ao = mc.ao_from_height(height, tuple(r * s for r in (2, 6, 16, 40)), float(p.get("ao_strength", 1.0)))
    if map_res != size:                                                           # 4k: data maps were estimated at 2k, upscale wrap-safely
        f = size // map_res
        normal = mc.renormalize_normal(mc.upscale_wrap(normal, f))
        height, rough, ao = (mc.upscale_wrap(a, f) for a in (height, rough, ao))
    metal = np.full(height.shape, float(p.get("metallic", 0.0)), np.float32)
    return {"basecolor": base, "normal": normal, "height": height, "roughness": rough, "metallic": metal, "ao": ao,
            "orm": mc.pack_orm(ao, rough, metal)}


def sphere_preview(mc, maps: dict, px: int = 360):
    """The 360 px sphere thumbnail, from 512 px copies of the maps (box filter) so it is not aliased."""
    f = max(1, maps["basecolor"].shape[0] // 512)
    s = {k: (mc.box_down(v, f) if f > 1 else v) for k, v in maps.items() if k != "orm"}
    return mc.render_sphere(s["basecolor"], s["normal"], s["roughness"], s["metallic"], s["ao"], tiles=1.0, size=px)


def write_material_files(mc, maps: dict, out: Path, name: str, meta_x: dict) -> dict:
    """Write the PNG maps, preview and glTF previews into `out`. Returns {slot: file name}."""
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for slot, fn in MAP_FILES.items():
        bits = 16 if slot == "height" else 8
        mc.to_pil(maps[slot], bits).save(out / fn)
        files[slot] = fn
    mc.to_pil(sphere_preview(mc, maps)).save(out / "preview.png")
    files["preview"] = "preview.png"
    tex = {"basecolor": "basecolor.png", "normal": "normal.png", "orm": "orm.png"}
    for shape in PREVIEW_SHAPES:
        mc.write_gltf(out / f"preview_{shape}.gltf", mc.sample_mesh(shape, 1.0), tex, name=name[:60] or "material")
        files[f"preview_{shape}"] = f"preview_{shape}.gltf"
    (out / "material.json").write_text(json.dumps(meta_x, indent=1, ensure_ascii=False))
    files["material"] = "material.json"
    return files


def _seams(mc, maps: dict) -> dict:
    return {k: mc.seam_score(maps[k])["score"] for k in ("basecolor", "normal", "height")}


@handler("material.make", gpu=True, label="PBR material")
def make(ctx: JobContext):
    mc = _mc()
    import numpy as np
    from PIL import Image

    p = dict(ctx.params)
    ins = ctx.input_assets()
    base = next((i for i in ins if i["role"] == "base" or (i["role"] == "input" and i["type"] == "material")), None)
    photo = None if base else next((i for i in ins if i["role"] == "image" or (i["role"] == "input" and i["type"] == "image")), None)
    typed = (p.get("prompt") or "").strip()
    mode = "variation" if base else "photo" if photo else "text"
    src_asset = ctx.assets.get((base or photo)["asset_id"]) if (base or photo) else None
    if mode == "text" and not typed:
        raise JobError("Type a description or add a photo.", "input")
    if mode == "variation":
        typed = typed or (src_asset.get("prompt") or src_asset.get("final_prompt") or "").strip()
    size = int(p.get("size") or 1024)
    batch = max(1, min(4, int(p.get("batch") or 1)))
    seed0 = int(p.get("seed", -1))
    if seed0 < 0:
        seed0 = random.randrange(2**31 - 8)
    steps = int(p.get("steps") or 9)
    tiling = p.get("photo_tiling") or "blend"
    post = (ctx.job.get("meta") or {}).get("style_post")
    lic = _lic([(src_asset or {}).get("license"), (ctx.model or {}).get("license"), (ctx.registry.models.get(GEN_MODEL) or {}).get("license")])
    scratch = ctx.scratch
    scratch.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    made = 0
    for k in range(batch):
        lo, hi = 0.97 * k / batch, 0.97 * (k + 1) / batch
        span = hi - lo
        seed = seed0 + k
        ctx.check_cancel()
        t_all = time.time()
        tiling_meta: dict = {"method": "latent_roll+circular_vae", "strength": None}
        timings: dict = {}
        # ---- 1. the 1024 px tileable base colour
        t0 = time.time()
        if mode == "text":
            ctx.progress(lo, "Generating the base colour", stage="base")
            res = ctx.run_worker(GEN_MODEL, task="t2i", frac_range=(lo, lo + span * 0.62), out_dir=scratch / f"gen{k}",
                                 params={"prompt": with_suffix(typed), "size": 1024, "aspect": "1:1", "steps": steps, "seed": seed,
                                         "batch": 1, "tileable": True}, inputs=[])
            alb_img = Image.open(_first_file(res, "base colour")).convert("RGB")
            source, src_seam = "text", None
        else:
            sq = square_1024(Path((base or photo)["path"]), mc)
            arr = mc.to_float(np.asarray(sq))
            src_seam = mc.seam_score(arr)["score"]
            if mode == "photo" and tiling == "blend":
                alb_img = Image.fromarray(mc.to_u8(mc.make_tileable_blend(arr, 0.25, True)))
                tiling_meta = {"method": "offset_blend", "strength": None}
                ctx.progress(lo + span * 0.3, "Blending the seams", stage="base")
            elif mode == "photo" and tiling == "none":
                alb_img = sq
                tiling_meta = {"method": "none", "strength": None}
                if src_seam > mc.SEAM_WARN:
                    warnings.append(f"The photo does not tile (seam score {src_seam:.2f}); the material will show visible seams.")
            else:                                   # regenerate (photo) or variation (base): Z-Image img2img, latent rolling + circular VAE
                strength = float(p.get("variation_strength", 0.5)) if mode == "variation" else 0.35
                sp = scratch / f"src{k}.png"
                sq.save(sp)
                ctx.progress(lo, "Redrawing as a tileable texture", stage="base")
                res = ctx.run_worker(GEN_MODEL, task="tileable", frac_range=(lo, lo + span * 0.62), out_dir=scratch / f"gen{k}",
                                     params={"prompt": with_suffix(typed) if typed else "seamless tileable texture", "size": 1024,
                                             "steps": steps, "seed": seed, "strength": strength},
                                     inputs=[{"asset_id": None, "role": "input", "type": "image", "name": "source", "path": str(sp), "meta": {}}])
                alb_img = Image.open(_first_file(res, "tileable")).convert("RGB")
                tiling_meta = {"method": "regenerate" if mode == "photo" else "variation", "strength": strength}
            source = mode
        if alb_img.size != (1024, 1024):
            alb_img = mc.resize_wrap(alb_img, (1024, 1024))
        timings["base"] = round(time.time() - t0, 2)
        base_seam = mc.seam_score(np.asarray(alb_img))["score"]
        # ---- 2. project palette on the base colour only
        x_style = None
        if post and post.get("mode") not in (None, "off"):
            raw_png = scratch / f"base_raw{k}.png"
            alb_img.save(raw_png)
            alb_img = sc.apply_post(alb_img, post["mode"], post["colors"], float(post.get("strength") or 0.35)).convert("RGB")
            x_style = {"palette": post["mode"], "strength": post.get("strength"), "colors": len(post["colors"]), "map": "basecolor"}
        base_png = scratch / f"base{k}.png"
        alb_img.save(base_png)
        # ---- 3. the maps (PBRify SPAN nets, same venv as Z-Image, a separate worker process)
        ctx.check_cancel()
        ctx.progress(lo + span * 0.64, "Estimating the maps", stage="maps")
        t0 = time.time()
        mres = ctx.run_worker("material-zimage", task="material_maps", frac_range=(lo + span * 0.64, lo + span * 0.82),
                              out_dir=scratch / f"maps{k}", params={"size": size},
                              inputs=[{"asset_id": None, "role": "image", "type": "image", "name": "basecolor", "path": str(base_png), "meta": {}}])
        timings["maps"] = round(time.time() - t0, 2)
        wm = mres.get("meta") or {}
        byname = {Path(f["path"]).stem: Path(f["path"]) for f in mres.get("files") or []}
        for need in ("basecolor_up", "normal_dx", "height_raw", "rough_raw"):
            if need not in byname:
                raise JobError(f"The maps step did not return {need}.", "internal")
        map_res = int(wm.get("map_res") or min(size, 2048))

        def load(stem):
            with Image.open(byname[stem]) as im:
                im.load()
                return mc.to_float(np.asarray(im))
        # ---- 4. CPU post + files
        ctx.progress(lo + span * 0.84, "Finishing the maps", stage="saving")
        t0 = time.time()
        alb = load("basecolor_up")
        maps = build_maps(mc, alb, load("normal_dx"), load("height_raw"), load("rough_raw"), p, map_res)
        seams = _seams(mc, maps)
        timings["post"] = round(time.time() - t0, 2)
        mats_warn = []
        for key, sc_ in seams.items():
            if sc_ > mc.SEAM_WARN:
                mats_warn.append(f"{key} seam score {sc_:.2f}: a seam may show (regular patterns such as brick score high without one); check the 3x3 view")
        name_meta = ctx.asset_meta({"seed": seed, "index": k, "final_prompt": with_suffix(typed) if typed else None}, k)
        title = (ctx.job.get("title") or typed[:60] or "Material").strip()
        if batch > 1:
            title = f"{title} #{k + 1}"
        name_meta["name"] = title
        x_mat = {"version": MAT_VERSION, "size": size, "map_res": map_res, "maps": sorted(MAP_FILES), "normal": "opengl", "height_bits": 16,
                 "orm": "R occlusion, G roughness, B metallic", "color_space": {"basecolor": "srgb", "others": "linear"}, "source": source,
                 "tiling": {**tiling_meta, "seam_source": src_seam, "seam_base": base_seam}, "seam": seams,
                 "generator": {"model": GEN_MODEL, "task": "t2i" if mode == "text" else "tileable", "seed": seed, "steps": steps},
                 "estimators": {"normal": "PBRify_NormalV3", "height": "PBRify_Height", "roughness": "PBRify_RoughnessV2",
                                "upscaler": "PBRify_UpscalerSPANV4" if size > 1024 else None, "license": "CC0-1.0"},
                 "params": {kk: p.get(kk) for kk in ("size", "metallic", "photo_tiling", "variation_strength", "delight", "normal_strength",
                                                      "roughness_bias", "height_contrast", "ao_strength", "steps")},
                 "previews": {"sphere": "preview_sphere.gltf", "cube": "preview_cube.gltf", "plane": "preview_plane.gltf", "thumb": "preview.png"},
                 "physical_size_m": None, "timings": {**timings, "worker": wm.get("timings"), "peak_gb": wm.get("peak_gb")}}
        if x_style:
            x_mat["style_palette"] = x_style
        files = write_material_files(mc, maps, scratch / f"mat{k}", title, x_mat)
        meta = dict(name_meta)
        meta["x_material"] = x_mat
        if x_style:
            meta.setdefault("x_style", {})["post"] = x_style
        meta["license"] = lic
        meta["info"] = {"width": size, "height": size}
        if mats_warn or warnings:
            meta["warnings"] = [*warnings, *mats_warn]
        d = scratch / f"mat{k}"
        ctx.add_output(d / files["basecolor"], "material", meta, attach=[d / v for kk, v in files.items() if kk != "basecolor"])
        made += 1
        timings["total"] = round(time.time() - t_all, 2)
        ctx.log(f"material {k + 1}/{batch}: {mode}, {size} px, {timings}, seams {seams}")
        ctx.progress(hi, "Saved", stage="saving")
    if not made:
        raise JobError("No material was made.", "internal")


# ================================================================== material.tileable
@handler("material.tileable", gpu=None, label="Make tileable")
def tileable(ctx: JobContext):
    mc = _mc()
    import numpy as np
    from PIL import Image

    p = ctx.params
    ins = [i for i in ctx.input_assets() if i["type"] == "image"]
    if not ins:
        raise JobError("Add one or more images to make tileable.", "input")
    band = float(p.get("band", 0.25))
    keep = bool(p.get("keep_contrast", True))
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    for n, inp in enumerate(ins):
        ctx.check_cancel()
        ctx.progress(n / len(ins), f"Blending {n + 1}/{len(ins)}", stage="blend")
        src = ctx.assets.get(inp["asset_id"])
        with Image.open(inp["path"]) as im:
            im.load()
            alpha = im.getchannel("A") if im.mode in ("RGBA", "LA") else None
            a = mc.to_float(np.asarray(im.convert("RGB")))
        before = mc.seam_score(a)["score"]
        out = mc.make_tileable_blend(a, band, keep)
        after = mc.seam_score(out)["score"]
        img = Image.fromarray(mc.to_u8(out))
        if alpha is not None and alpha.getextrema()[0] < 255:
            img = img.convert("RGBA")
            img.putalpha(alpha)
        f = ctx.scratch / f"tileable_{n + 1}.png"
        img.save(f)
        meta = ctx.asset_meta({"index": n}, n)
        meta.update({"name": f"{src.get('name') or 'image'} (tileable)", "parents": [src["id"]], "license": _lic([src.get("license")]) or meta.get("license"),
                     "prompt": src.get("prompt"), "x_tileable": {"method": "offset_blend", "band": band, "keep_contrast": keep,
                                                                  "seam_before": before, "seam_after": after}})
        if after > mc.SEAM_WARN:
            meta["warnings"] = [f"Still a visible seam (score {after:.2f}); try a wider band or Make tileable (regenerate)."]
        ctx.add_output(f, "image", meta)
    ctx.progress(1.0, "Done", stage="saving")


# ================================================================== tool:material.sample_glb
@handler("tool:material.sample_glb", gpu=False, label="Sample GLB (material on a shape)")
def sample_glb(ctx: JobContext):
    mc = _mc()
    p = ctx.params
    shape = p.get("shape") or "sphere"
    if shape not in PREVIEW_SHAPES:
        raise JobError(f"shape must be one of {', '.join(PREVIEW_SHAPES)}.", "input")
    tiles = max(1, min(8, int(p.get("tiles") or 2)))
    tex = int(p.get("size") or 1024)
    if tex not in (512, 1024):
        raise JobError("size must be 512 or 1024.", "input")
    ins = [i for i in ctx.input_assets() if i["type"] == "material"]
    if not ins:
        raise JobError("Pick a material asset.", "input")
    ctx.scratch.mkdir(parents=True, exist_ok=True)
    for n, inp in enumerate(ins):
        ctx.check_cancel()
        src = ctx.assets.get(inp["asset_id"])
        d = ctx.assets.asset_dir(src["project"], src["id"])
        maps = load_maps(mc, d, src, size=tex)
        f = ctx.scratch / f"sample_{n + 1}.glb"
        write_sample_glb(mc, maps, f, shape, tiles, src.get("name") or "material")
        meta = ctx.asset_meta({"index": n}, n)
        meta.update({"name": f"{src.get('name') or 'material'} ({shape} x{tiles})", "parents": [src["id"]], "type": "mesh",
                     "license": src.get("license") or meta.get("license"), "source": "tool"})
        ctx.add_output(f, "mesh", meta)
        ctx.progress((n + 1) / len(ins), "Writing GLB", stage="saving")


def load_maps(mc, asset_dir: Path, meta: dict, size: int = 0) -> dict:
    """Stored PNG maps of a material asset -> float arrays {basecolor, normal(OpenGL), height, roughness, metallic, ao} (box-downsized to
    `size` when it is smaller than the stored size)."""
    import numpy as np
    from PIL import Image

    out = {}
    names = {f["name"] for f in meta.get("files", [])}
    for slot in ("basecolor", "normal", "height", "roughness", "metallic", "ao"):
        fn = MAP_FILES[slot] if slot != "basecolor" else meta.get("main", "basecolor.png")
        if fn not in names or not (asset_dir / fn).is_file():
            continue
        with Image.open(asset_dir / fn) as im:
            im.load()
            a = mc.to_float(np.asarray(im if slot in ("height", "roughness", "metallic", "ao") else im.convert("RGB")))
        if a.ndim == 3 and slot in ("height", "roughness", "metallic", "ao"):
            a = a[..., 0]
        if size and a.shape[0] > size and a.shape[0] % size == 0:
            a = mc.box_down(a, a.shape[0] // size)
        out[slot] = a
    if "basecolor" not in out:
        raise ValueError("The material has no base colour file.")
    return out


def png_bytes(img) -> bytes:
    import io

    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def write_sample_glb(mc, maps: dict, path: Path, shape: str, tiles: int, name: str) -> Path:
    import numpy as np

    ones = np.ones_like(maps["basecolor"][..., 0])
    ao = maps.get("ao", ones)
    ro = maps.get("roughness", ones)
    me = maps.get("metallic", np.zeros_like(ones))
    nm = maps.get("normal")
    imgs = {"basecolor": png_bytes(mc.to_pil(maps["basecolor"])), "orm": png_bytes(mc.to_pil(mc.pack_orm(ao, ro, me)))}
    if nm is not None:
        imgs["normal"] = png_bytes(mc.to_pil(nm))
    return mc.write_glb(path, mc.sample_mesh(shape, tiles), imgs, name=name[:60])


# ================================================================== export
def _stem_of(base: str) -> str:
    return base.rsplit("/", 1)[-1]


def export_material(meta: dict, d: Path, convert: dict, tmp: Path, base: str):
    """assets.EXPORTERS["material"]: files of one material for the zip. `base` is "materials/<stem>". Returns ([(src, arcname)], report)."""
    mc = _mc()
    stored = int((meta.get("x_material") or {}).get("size") or meta.get("info", {}).get("width") or 0)
    conv = mc.normalize_convert(convert, stored)
    maps = load_maps(mc, d, meta)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", _stem_of(base)).strip("_") or "material"
    tmp.mkdir(parents=True, exist_ok=True)
    files = []
    pack = mc.pack_layout(maps, conv["layout"], stem, normal=conv["normal"], height_bits=conv["height_bits"], size=conv["size"])
    listed = []
    for fname, img, info in pack:
        p = tmp / fname
        img.save(p)
        files.append((p, f"{base}/{fname}"))
        listed.append({"file": fname, **{k: info[k] for k in ("slot", "colorspace", "packing", "size", "bits") if k in info}})
    glb_name = None
    if conv["glb"]:
        small = load_maps(mc, d, meta, size=1024)
        glb_name = f"{stem}_sample.glb"
        write_sample_glb(mc, small, tmp / glb_name, "sphere", 2, meta.get("name") or stem)
        files.append((tmp / glb_name, f"{base}/{glb_name}"))
    side = {"version": MAT_VERSION, "name": meta.get("name"), "layout": conv["layout"], "normal": conv["normal"],
            "size": next(iter(listed))["size"][0] if listed and listed[0].get("size") else stored, "files": listed, "sample_glb": glb_name,
            "tiling": (meta.get("x_material") or {}).get("tiling"), "seam": (meta.get("x_material") or {}).get("seam"),
            "license": meta.get("license"), "layout_summary": mc.layout_info(conv["layout"]).get("summary")}
    (tmp / f"{stem}.material.json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=str))
    files.append((tmp / f"{stem}.material.json", f"{base}/{stem}.material.json"))
    return files, {"layout": conv["layout"], "normal": conv["normal"], "size": side["size"]}


# ================================================================== router
class RepackReq(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    layout: str = "gltf"
    normal: str | None = None
    size: int = 0
    glb: bool = True


@router.get("/layouts")
def layouts():
    mc = _mc()
    return {"default": mc.DEFAULT_LAYOUT, "layouts": [{"id": k, "label": v["label"], "normal": v["normal"], "packed": v["packed"], "summary": v["summary"]}
                                                      for k, v in mc.LAYOUTS.items()],
            "sizes": [0, 4096, 2048, 1024, 512, 256], "normals": ["gl", "dx"]}


@router.post("/repack")
def repack(req: RepackReq):
    if CORE is None:
        raise HTTPException(503, "not ready")
    try:
        m = CORE.assets.get(req.id)
        if m["type"] != "material":
            raise AssetError("Not a material asset.")
        conv = {"layout": req.layout, "size": req.size, "glb": req.glb, **({"normal": req.normal} if req.normal else {})}
        _mc().normalize_convert(conv)
        z = CORE.assets.export([req.id], config.DATA / "exports", convert={"material": conv})
    except AssetError as e:
        raise HTTPException(e.status, str(e)) from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return {"url": f"/api/export/{z.name}", "name": z.name, "bytes": z.stat().st_size, "count": 1}


def setup(core):
    global CORE
    CORE = core
    assets_mod.EXPORTERS["material"] = export_material
