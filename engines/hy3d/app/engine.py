"""Model manager for Hunyuan3D-2.1: lazy loading, one-job-at-a-time GPU lock, run folders.

Importing this module prepares the process for the vendor code (cwd, sys.path,
torchvision shim, temp dir). The vendor repo is never edited; every hard-coded
value is overridden from here (see README "How the knobs reach the pipelines").
"""
from __future__ import annotations

import gc
import json
import logging
import os
import secrets
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "Hunyuan3D-2.1"


def _bootstrap():
    """Environment the vendor code expects. Must run before torch / vendor imports."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if not os.environ.get("HY3D_ONLINE"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # all weights are in the local cache
    for bad in ("USE_SAGEATTN", "CA_USE_SAGEATTN"):  # sageattention is not installed
        os.environ.pop(bad, None)
    tmp = ROOT / "tmp" / "hy3d"
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)  # vendor postprocessors leak temp files; keep them inside the project
    cutoff = time.time() - 24 * 3600
    for f in tmp.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass
    os.chdir(VENDOR)  # vendor uses cwd-relative paths (./hy3dshape, ./assets, temp.glb ...)
    for sub in ("hy3dpaint", "hy3dshape", ""):
        p = str(VENDOR / sub) if sub else str(VENDOR)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import bpy  # noqa: F401  (a stub is installed in the venv; build one if it is missing)
    except ImportError:
        sys.modules["bpy"] = types.ModuleType("bpy")


_bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

try:
    from torchvision_fix import apply_fix  # vendor shim for torchvision.transforms.functional_tensor

    apply_fix()
except Exception as e:  # noqa: BLE001
    print(f"[engine] torchvision fix not applied: {e}")

import postprocess as pp  # noqa: E402
from params import APP_VERSION, MAX_SEED, effective_render_size, normalize  # noqa: E402
from runs import RunStore  # noqa: E402

log = logging.getLogger("hy3d.engine")

SHAPE_MODEL = ("tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1")
MAX_INPUT_SIDE = 1536  # larger inputs are downscaled before background removal (CPU cost only)


# ----------------------------------------------------------------- errors
class UserError(Exception):
    """A failure with a message that is safe and useful to show in the UI."""

    def __init__(self, msg, run_id=None):
        super().__init__(msg)
        self.run_id = run_id


class JobCancelled(UserError):
    pass


# --------------------------------------------------------------- progress
class Tracker:
    """Stage/progress bookkeeping shared between the worker thread and the UI poller.

    Stages are given as (lo, hi) spans of the current window so a combined
    shape+texture job can map each half onto its share of the bar.
    """

    def __init__(self, cancel_event: threading.Event | None = None):
        self.cancel_event = cancel_event or threading.Event()
        self.window = (0.0, 1.0)
        self.frac = 0.0
        self.desc = "Queued"
        self.stage_name: str | None = None
        self.stage_started = time.time()
        self.t_start = time.time()
        self.timings: dict[str, float] = {}
        self.peaks: dict[str, float] = {}
        self._span = (0.0, 1.0)

    def check(self):
        if self.cancel_event.is_set():
            raise JobCancelled("Cancelled by user.")

    def set_window(self, lo: float, hi: float):
        self.window = (lo, hi)

    def _emit(self, rel: float, desc: str):
        lo, hi = self.window
        self.frac = lo + (hi - lo) * min(max(rel, 0.0), 1.0)
        self.desc = desc

    def _close(self):
        if self.stage_name is not None:
            self.timings[self.stage_name] = self.timings.get(self.stage_name, 0.0) + time.time() - self.stage_started
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 2**30
                self.peaks[self.stage_name] = max(self.peaks.get(self.stage_name, 0.0), peak)
        self.stage_name = None

    def stage(self, name: str, lo: float = 0.0, hi: float = 1.0):
        self._close()
        self.check()
        self.stage_name, self.stage_started, self._span = name, time.time(), (lo, hi)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._emit(lo, name)

    def sub(self, frac: float, detail: str = ""):
        self.check()
        lo, hi = self._span
        self._emit(lo + (hi - lo) * frac, f"{self.stage_name}  {detail}".strip())

    def finish(self):
        self._close()
        self._emit(1.0, "Done")

    def summary_times(self) -> dict:
        t = {k: round(v, 2) for k, v in self.timings.items()}
        t["total"] = round(time.time() - self.t_start, 2)
        return t


# --------------------------------------------------------- texture context
class TexContext:
    """Per-run settings read by the long-lived wrappers installed on the paint pipeline."""

    def __init__(self, p: dict, seed: int, tracker: Tracker):
        self.tracker = tracker
        self.steps, self.guidance, self.seed = int(p["tex_steps"]), float(p["tex_guidance"]), seed
        self.sr_enabled, self.sr_tile = bool(p["super_res"]), int(p["sr_tile"])
        self.vertex_inpaint, self.uv_padding = bool(p["vertex_inpaint"]), int(p["uv_padding"])
        self.full_res, self.tex_size = bool(p["tex_full_res"]), int(p["tex_size"])
        self.render_size = effective_render_size(p)
        self.n_views = 0
        self.sr_done = 0


# Spans of the texture window (fractions); the stage names are also the keys in the timing table.
TEX_SPANS = {
    "Loading texture models": (0.00, 0.08),
    "Simplify mesh for texturing": (0.08, 0.10),
    "UV unwrap": (0.10, 0.14),
    "Views + control maps": (0.14, 0.20),
    "Multiview diffusion": (0.20, 0.70),
    "Super-resolution": (0.70, 0.82),
    "Baking views": (0.82, 0.92),
    "UV inpaint": (0.92, 0.96),
    "Save textures": (0.96, 0.98),
    "Build GLB": (0.98, 1.00),
}


def _tex_stage(tr: Tracker, name: str):
    tr.stage(name, *TEX_SPANS[name])


class _MVProxy:
    """Stand-in for `multiviewDiffusionNet.pipeline`.

    The vendor hard-codes steps / CFG / seed inside forward_one(); the cleanest
    seam is the single call to `self.pipeline(...)`, so we intercept it here.
    Everything else (unet, scheduler, device, view_size ...) is forwarded.
    """

    def __init__(self, inner, engine):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_engine", engine)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_inner"), name, value)

    def __call__(self, *args, **kw):
        inner = object.__getattribute__(self, "_inner")
        ctx: TexContext = object.__getattribute__(self, "_engine")._tex_ctx
        kw["num_inference_steps"] = ctx.steps
        kw["guidance_scale"] = ctx.guidance
        kw["generator"] = torch.Generator(device=inner.device).manual_seed(ctx.seed)

        def on_step(_pipe, i, _t, cb_kwargs):
            ctx.tracker.sub((i + 1) / ctx.steps, f"step {i + 1}/{ctx.steps}")
            return cb_kwargs

        kw["callback_on_step_end"] = on_step
        _tex_stage(ctx.tracker, "Multiview diffusion")
        out = inner(*args, **kw)
        ctx.n_views = len(out.images) // 2  # albedo views + metallic/roughness views
        ctx.sr_done = 0
        return out


class _SuperRes:
    """Wraps the Real-ESRGAN model: optional bypass (Lanczos x4), tile size, progress."""

    def __init__(self, real, engine):
        self.real, self.engine = real, engine
        self.upsampler = real.upsampler

    def __call__(self, image):
        ctx: TexContext = self.engine._tex_ctx
        total = max(1, 2 * ctx.n_views)
        if ctx.sr_done == 0:
            _tex_stage(ctx.tracker, "Super-resolution")
        self.upsampler.tile_size = ctx.sr_tile  # RealESRGANer reads tile_size (set once in its constructor), not .tile
        if ctx.sr_enabled:
            out = self.real(image)
        else:
            out = image.resize((image.width * 4, image.height * 4), Image.LANCZOS)
        # The vendor only resizes the first two views to render_size (its loop runs over the dict keys),
        # so do it for every view here and the render size means the same for all of them.
        if out.size != (ctx.render_size, ctx.render_size):
            out = out.resize((ctx.render_size, ctx.render_size), Image.LANCZOS)
        ctx.sr_done += 1
        ctx.tracker.sub(ctx.sr_done / total, f"image {ctx.sr_done}/{total}")
        if ctx.sr_done >= total:
            _tex_stage(ctx.tracker, "Baking views")
        return out


def _xatlas_wrap(mesh, padding: int, resolution: int):
    """UV unwrap with atlas padding (vendor uses xatlas defaults: padding 0)."""
    import xatlas

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices, mesh.faces)
    pack = xatlas.PackOptions()
    pack.padding, pack.resolution = int(padding), int(resolution)
    atlas.generate(pack_options=pack)
    vmapping, indices, uvs = atlas.get_mesh(0)
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs
    return mesh


def build_pbr_glb(run_dir: Path, metallic_scale: float, roughness_scale: float) -> Path:
    """Textured OBJ + albedo/metallic/roughness JPEGs -> one PBR GLB (glTF: G=roughness, B=metallic).

    Replaces vendor `create_glb_with_pbr_materials`, which hard-codes the factors, mislabels JPEG
    data as PNG and writes scratch files into the cwd.
    """
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals

    obj = run_dir / "textured_mesh.obj"
    mesh = trimesh.load(str(obj), force="mesh", process=False)
    uv = np.asarray(mesh.visual.uv)
    albedo = Image.open(run_dir / "textured_mesh.jpg").convert("RGB")
    mr = None
    met_p, rough_p = run_dir / "textured_mesh_metallic.jpg", run_dir / "textured_mesh_roughness.jpg"
    if met_p.exists() and rough_p.exists():
        met = Image.open(met_p).convert("L")
        rough = Image.open(rough_p).convert("L").resize(met.size)
        mr = Image.merge("RGB", (Image.new("L", met.size, 255), rough, met))
    mat = PBRMaterial(baseColorTexture=albedo, metallicRoughnessTexture=mr,
                      baseColorFactor=[255, 255, 255, 255],
                      metallicFactor=float(metallic_scale), roughnessFactor=float(roughness_scale))
    out = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    out.visual = TextureVisuals(uv=uv, material=mat)
    _ = out.vertex_normals
    glb = run_dir / "textured_mesh.glb"
    out.export(str(glb), file_type="glb", include_normals=True)
    return glb


def _square_pad(img: Image.Image) -> Image.Image:
    """Pad to a square (transparent for RGBA) so the vendor's 512x512 resize does not distort."""
    if img.width == img.height:
        return img
    side = max(img.size)
    fill = (255, 255, 255, 0) if img.mode == "RGBA" else (255, 255, 255)
    canvas = Image.new(img.mode, (side, side), fill)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


def _gpu_mem_gb() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {"allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


# ----------------------------------------------------------------- engine
class Engine:
    def __init__(self, output_dir: Path, low_vram: bool = False):
        self.store = RunStore(output_dir)
        self.low_vram = low_vram
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.busy = False
        self.active_run: str | None = None  # run folder owned by the running job (must not be deleted)
        self.shape_pipe = None
        self.rembg = None
        self.paint_pipe = None
        self._real_sr = None
        self._flash_on = False
        self._tex_ctx: TexContext | None = None
        self._render_sig: tuple | None = None
        self._paint_mod = None
        self.load_times: dict[str, float] = {}
        # Image Studio hooks (see image_client.ImageService): both are set when the studio is enabled.
        self.external_job: str | None = None     # name of a non-Hunyuan job that currently holds the GPU lock
        self.release_peers = None                # callable, run with the lock held before each 3D job
        self.peer_status = None                  # callable -> dict about the image worker (for status())

    # -- public helpers -----------------------------------------------------
    def cancel(self) -> bool:
        """Ask the running job to stop at its next checkpoint (a diffusion step or a stage boundary)."""
        if self.busy:
            self.cancel_event.set()
        return self.busy

    def status(self) -> dict:
        return {"shape_loaded": self.shape_pipe is not None, "paint_loaded": self.paint_pipe is not None,
                "rembg_loaded": self.rembg is not None, "busy": self.busy, "low_vram": self.low_vram,
                "external_job": self.external_job, "gpu": _gpu_mem_gb()}

    def unload(self, shape: bool = True, paint: bool = True):
        if shape and self.shape_pipe is not None:
            self.shape_pipe, self._flash_on = None, False
        if paint and self.paint_pipe is not None:
            self.paint_pipe, self._real_sr, self._render_sig = None, None, None
        self._free()

    @staticmethod
    def _free():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- model loading ---------------------------------------------------------
    def _ensure_rembg(self):
        if self.rembg is None:
            from hy3dshape.rembg import BackgroundRemover

            self.rembg = BackgroundRemover()
        return self.rembg

    def _ensure_shape(self, tr: Tracker):
        if self.shape_pipe is None:
            tr.stage("Loading shape model")
            t = time.time()
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

            self.shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                SHAPE_MODEL[0], subfolder=SHAPE_MODEL[1], use_safetensors=False, variant="fp16",
                device="cuda", dtype=torch.float16)
            self._flash_on = False
            self.load_times["shape_model_s"] = round(time.time() - t, 1)
        return self.shape_pipe

    def _ensure_paint(self, tr: Tracker):
        if self.paint_pipe is not None:
            return self.paint_pipe
        _tex_stage(tr, "Loading texture models")
        t = time.time()
        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        class LazyPaintPipeline(Hunyuan3DPaintPipeline):
            def load_models(self):  # we load sub-models ourselves, with the wrappers installed
                pass

        conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=512)
        conf.realesrgan_ckpt_path = str(VENDOR / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth")
        conf.multiview_cfg_path = str(VENDOR / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
        conf.custom_pipeline = str(VENDOR / "hy3dpaint" / "hunyuanpaintpbr")
        torch.cuda.empty_cache()
        pipe = LazyPaintPipeline(conf)

        from utils.image_super_utils import imageSuperNet
        from utils.multiview_utils import multiviewDiffusionNet

        self._real_sr = imageSuperNet(conf)
        pipe.models["super_model"] = _SuperRes(self._real_sr, self)
        mv = multiviewDiffusionNet(conf)
        mv.seed_everything = lambda seed: None  # vendor resets python/numpy/torch global RNGs on every run
        mv.pipeline = _MVProxy(mv.pipeline, self)
        pipe.models["multiview_model"] = mv

        # module-level names that the vendor pipeline resolves at call time
        self._paint_mod = sys.modules[Hunyuan3DPaintPipeline.__module__]
        # keep the vendor function across reloads, otherwise every reload would wrap the previous wrapper
        orig_wrap = getattr(self._paint_mod, "_hy3d_orig_mesh_uv_wrap", None) or self._paint_mod.mesh_uv_wrap
        self._paint_mod._hy3d_orig_mesh_uv_wrap = orig_wrap

        def uv_wrap(mesh):
            c = self._tex_ctx
            _tex_stage(c.tracker, "UV unwrap")
            mesh = _xatlas_wrap(mesh, c.uv_padding, c.tex_size) if c.uv_padding > 0 else orig_wrap(mesh)
            _tex_stage(c.tracker, "Views + control maps")
            return mesh

        self._paint_mod.mesh_uv_wrap = uv_wrap
        self.paint_pipe = pipe
        self._render_sig = None
        self.load_times["paint_models_s"] = round(time.time() - t, 1)
        return pipe

    def _configure_paint(self, p: dict, ctx: TexContext):
        """Push this run's settings into the (cached) paint pipeline."""
        from DifferentiableRenderer.MeshRender import MeshRender
        from utils.pipeline_utils import ViewProcessor

        pipe, conf = self.paint_pipe, self.paint_pipe.config
        conf.max_selected_view_num = int(p["tex_views"])
        conf.resolution = int(p["tex_resolution"])
        conf.bake_exp = int(p["bake_exp"])
        render_size, tex_size = effective_render_size(p), int(p["tex_size"])
        conf.render_size, conf.texture_size = render_size, tex_size

        sig = (render_size, tex_size)
        if sig != self._render_sig:  # the renderer is cheap to rebuild; the diffusion weights stay loaded
            pipe.render = MeshRender(default_resolution=render_size, texture_size=tex_size,
                                     bake_mode=conf.bake_mode, raster_mode=conf.raster_mode)
            pipe.view_processor = ViewProcessor(conf, pipe.render)
            render = pipe.render
            cls = type(render)

            def uv_inpaint(texture, mask, **_kw):
                c = self._tex_ctx
                _tex_stage(c.tracker, "UV inpaint")
                return cls.uv_inpaint(render, texture, mask, vertex_inpaint=c.vertex_inpaint)

            def save_mesh(path, downsample=False):
                c = self._tex_ctx
                _tex_stage(c.tracker, "Save textures")
                return cls.save_mesh(render, path, downsample=downsample and not c.full_res)

            render.uv_inpaint, render.save_mesh = uv_inpaint, save_mesh
            self._render_sig = sig
            self._free()
        pipe.render.bake_angle_thres = float(p["bake_angle"])
        self._tex_ctx = ctx

    def preload(self, shape: bool = True, paint: bool = True):
        """Load models up front (the --preload flag); otherwise they load on first use."""
        tr = Tracker()
        with self.lock:
            self._win_shape = self._win_tex = (0.0, 1.0)
            if shape:
                self._ensure_shape(tr)
                self._ensure_rembg()
            if paint:
                self._ensure_paint(tr)
            tr._close()
        self._free()

    # -- job plumbing ------------------------------------------------------------
    def _begin(self, tr: Tracker):
        self.lock.acquire()
        try:
            self.busy = True
            if self.release_peers is not None:
                try:
                    self.release_peers()  # frees the image worker when memory is tight / low-memory mode is on
                except Exception:  # noqa: BLE001
                    log.exception("releasing the image worker failed")
            tr.cancel_event = self.cancel_event
            self.cancel_event.clear()
            tr.t_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except BaseException:
            self.busy = False
            self.lock.release()
            raise

    def _end(self):
        self._tex_ctx = None
        self.busy = False
        self.active_run = None
        self.cancel_event.clear()
        if self.low_vram:
            self.unload()
        else:
            self._free()
        self.lock.release()

    @staticmethod
    def _wrap_error(e: Exception, run_id: str | None):
        if isinstance(e, UserError):
            e.run_id = e.run_id or run_id
            return e
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
            return UserError("GPU ran out of memory. Lower the octree resolution / number of chunks, or for texturing the "
                             "view resolution, number of views or atlas size. Enabling low-memory mode also helps "
                             "(it unloads the other model before each stage).", run_id)
        return UserError(f"{type(e).__name__}: {e}", run_id)

    # -- image prep ----------------------------------------------------------------
    def _prepare_image(self, image: Image.Image, p: dict, tr: Tracker, warnings: list) -> tuple[Image.Image, Image.Image]:
        """Return (cleaned RGBA conditioning image, original RGB(A))."""
        original = image
        img = image if image.mode == "RGBA" else image.convert("RGBA")
        if max(img.size) > MAX_INPUT_SIDE:
            s = MAX_INPUT_SIDE / max(img.size)
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
            warnings.append(f"Input downscaled to fit {MAX_INPUT_SIDE}px before processing.")
        alpha = np.asarray(img.getchannel("A"))
        has_alpha = (alpha < 250).mean() > 0.01
        if p["remove_bg"] and has_alpha:
            warnings.append("Image already has transparency: background removal skipped.")
        elif p["remove_bg"]:
            tr.stage("Remove background", 0.0, 1.0)
            img = self._ensure_rembg()(img.convert("RGB"))
        elif not has_alpha:
            warnings.append("No transparency and background removal is off: the whole frame is treated as the object.")
        # the vendor crashes with a numpy error on an empty mask; say what is actually wrong
        a = np.asarray(img.getchannel("A"))
        ys, xs = np.nonzero(a)
        if len(ys) == 0 or ys.max() - ys.min() < 2 or xs.max() - xs.min() < 2:
            raise UserError("No foreground object was found in the image (it is empty, too small, or everything "
                            "was removed as background). Use a different image, or untick background removal "
                            "if the object already fills the frame.")
        fg = np.asarray(img.convert("RGB"))[a > 127]
        if len(fg) and float(fg.std(axis=0).mean()) < 3.0:
            warnings.append("The object in the image is almost a single flat colour; the result will probably be a featureless blob.")
        return img, original

    # -- shape stage ----------------------------------------------------------------
    def _shape(self, run_dir: Path, image: Image.Image, p: dict, seed: int, tr: Tracker, warnings: list):
        """Returns (mesh, cleaned RGBA image, clean-up report)."""
        tr.set_window(*self._win_shape)
        cleaned, original = self._prepare_image(image, p, tr, warnings)
        cleaned.save(run_dir / "input.png")
        original.save(run_dir / "input_original.png")
        self.store.save_thumb(run_dir, cleaned)

        pipe = self._ensure_shape(tr)
        pipe.image_processor.border_ratio = float(p["border_ratio"])  # vendor config hard-codes 0.15
        pipe.scheduler.register_to_config(shift=float(p["sigma_shift"]))
        if bool(p["flashvdm"]) != self._flash_on:
            pipe.enable_flashvdm(bool(p["flashvdm"]), mc_algo="mc")
            self._flash_on = bool(p["flashvdm"])

        steps = int(p["steps"])
        tr.stage("Shape: diffusion", 0.10, 0.60)

        def on_step(i, _t, _out):
            tr.sub((i + 1) / steps, f"step {i + 1}/{steps}")
            if i + 1 >= steps:
                tr.stage("Shape: volume decode + marching cubes", 0.60, 0.90)

        outputs = pipe(
            image=cleaned, num_inference_steps=steps, guidance_scale=float(p["guidance"]),
            generator=torch.Generator().manual_seed(seed), box_v=float(p["box_v"]),
            octree_resolution=int(p["octree"]), mc_level=float(p["mc_level"]), num_chunks=int(p["num_chunks"]),
            output_type="trimesh", enable_pbar=False, callback=on_step, callback_steps=1)
        mesh = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if mesh is None:
            raise UserError("No surface was extracted. Try a different seed or mc_level, or check the input image.")

        tr.stage("Mesh clean-up", 0.90, 0.95)
        report = {}
        if p["shape_cleanup"]:
            mesh, report = pp.run_pipeline(mesh, degenerate=True, floaters=True, floater_ratio=float(p["floater_ratio"]))
        tr.stage("Export white mesh", 0.95, 1.0)
        pp.export_mesh(mesh, run_dir / "white_mesh.glb", "glb")
        return mesh, cleaned, report

    # -- texture stage ----------------------------------------------------------------
    def _texture(self, run_dir: Path, mesh: trimesh.Trimesh, ref: Image.Image, p: dict, tex_seed: int, tr: Tracker) -> dict:
        tr.set_window(*self._win_tex)
        ctx = TexContext(p, tex_seed, tr)
        pipe = self._ensure_paint(tr)
        self._configure_paint(p, ctx)

        _tex_stage(tr, "Simplify mesh for texturing")
        reduced = pp.reduce_faces(mesh, int(p["tex_faces"]))
        mesh_obj = run_dir / "white_mesh_tex.obj"
        reduced.export(str(mesh_obj), file_type="obj", include_normals=False)

        if ref.mode == "RGBA":
            bg = Image.new("RGB", ref.size, (255, 255, 255))
            bg.paste(ref, mask=ref.getchannel("A"))
            ref = bg
        ref = _square_pad(ref.convert("RGB"))

        pipe(mesh_path=str(mesh_obj), image_path=ref, output_mesh_path=str(run_dir / "textured_mesh.obj"),
             use_remesh=False, save_glb=False)

        _tex_stage(tr, "Build GLB")
        glb = build_pbr_glb(run_dir, p["metallic_scale"], p["roughness_scale"])
        tex_mesh = pp.load_mesh(run_dir / "textured_mesh.obj")
        info = {"mesh": pp.mesh_stats(tex_mesh), "views_used": ctx.n_views or None,
                "texture_files": {}, "glb_mb": round(glb.stat().st_size / 2**20, 2),
                "faces_before_reduction": int(len(mesh.faces)), "faces_after_reduction": int(len(reduced.faces))}
        for name in ("textured_mesh.jpg", "textured_mesh_metallic.jpg", "textured_mesh_roughness.jpg"):
            f = run_dir / name
            if f.exists():
                with Image.open(f) as im:
                    info["texture_files"][name] = f"{im.width}x{im.height}"
        return info

    # -- public jobs ---------------------------------------------------------------------
    def generate(self, image: Image.Image, params: dict, textured: bool, tracker: Tracker) -> dict:
        """Image -> white mesh (and optionally textured mesh). Returns a result dict (see _finish)."""
        if image is None:
            raise UserError("Please provide an image first.")
        p = normalize(params)
        seed = secrets.randbelow(10_000_000) if p["randomize_seed"] else int(p["seed"])
        tex_seed = secrets.randbelow(10_000_000) if p["tex_randomize_seed"] else int(p["tex_seed"])
        kind = "full" if textured else "shape"
        run_id = run_dir = None
        warnings: list[str] = []
        stats: dict = {"time": {}}
        self._begin(tracker)
        try:  # everything after _begin must be inside try so that _end() always releases the GPU lock
            self._win_shape, self._win_tex = ((0.0, 0.4), (0.4, 1.0)) if textured else ((0.0, 1.0), (1.0, 1.0))
            run_id, run_dir = self.store.new_run(kind)
            self.active_run = run_id
            mesh, cleaned, cleanup = self._shape(run_dir, image, p, seed, tracker, warnings)
            shape_stats = pp.mesh_stats(mesh)
            stats.update(white=shape_stats, number_of_faces=shape_stats["faces"],
                         number_of_vertices=shape_stats["vertices"], cleanup=cleanup)
            if textured:
                if self.low_vram:
                    self.unload(shape=True, paint=False)
                stats["textured"] = self._texture(run_dir, mesh, cleaned, p, tex_seed, tracker)
            return self._finish(run_id, run_dir, kind, p, seed, tex_seed, stats, tracker, warnings, source_run=None)
        except Exception as e:  # noqa: BLE001
            kept = run_id if run_dir is not None and (run_dir / "white_mesh.glb").exists() else None
            err = self._wrap_error(e, kept)
            if run_dir is not None:
                self._fail(run_id, run_dir, p, seed, tex_seed, stats, tracker, err, kind)
            if not isinstance(e, UserError):
                log.exception("job failed")
            raise err from e
        finally:
            self._end()

    def retexture(self, source_run: str, params: dict, current_image: Image.Image | None, tracker: Tracker) -> dict:
        """Re-texture the white mesh of an earlier run without regenerating the shape."""
        p = normalize(params)
        if not self.store.exists(source_run) or not (self.store.run_dir(source_run) / "white_mesh.glb").exists():
            raise UserError("Select or generate a shape first: there is no white mesh to texture.")
        tex_seed = secrets.randbelow(10_000_000) if p["tex_randomize_seed"] else int(p["tex_seed"])
        run_dir = None
        warnings: list[str] = []
        stats: dict = {"time": {}}
        self._begin(tracker)
        try:  # everything after _begin must be inside try so that _end() always releases the GPU lock
            self._win_shape, self._win_tex = (0.0, 0.0), (0.0, 1.0)
            src_params = self.store.read_json(source_run, "params.json")
            src_stats = self.store.read_json(source_run, "stats.json")
            if not isinstance(src_params, dict) or not isinstance(src_stats, dict):
                src_params, src_stats = {}, {}
            src_controls = src_params.get("controls") if isinstance(src_params.get("controls"), dict) else {}
            try:
                seed = int(src_controls.get("seed", p["seed"]))
            except (TypeError, ValueError):
                seed = int(p["seed"])
            white_stats = src_stats.get("white") if isinstance(src_stats.get("white"), dict) else {}
            stats.update(white=white_stats, number_of_faces=src_stats.get("number_of_faces"),
                         number_of_vertices=src_stats.get("number_of_vertices"))
            run_id, run_dir = self.store.fork_for_texture(source_run)
            self.active_run = run_id
            mesh = pp.load_mesh(run_dir / "white_mesh.glb")
            if current_image is not None and p["tex_reference"] == "current":
                ref = current_image
            else:
                name = "input_original.png" if p["tex_reference"] == "original" else "input.png"
                f = run_dir / name if (run_dir / name).exists() else run_dir / "input.png"
                if not f.exists():
                    raise UserError("The source run has no saved input image to use as the texture reference.")
                ref = Image.open(f)
                ref.load()
            if self.low_vram:
                self.unload(shape=True, paint=False)
            stats["textured"] = self._texture(run_dir, mesh, ref, p, tex_seed, tracker)
            # keep the shape settings of the source run so the record is accurate
            return self._finish(run_id, run_dir, "retex", p, seed, tex_seed, stats, tracker, warnings,
                                source_run=source_run, shape_controls=src_controls or None)
        except Exception as e:  # noqa: BLE001
            err = self._wrap_error(e, None)
            if run_dir is not None:
                shutil_rm(run_dir)
            if not isinstance(e, UserError):
                log.exception("retexture failed")
            raise err from e
        finally:
            self._end()

    # -- bookkeeping ----------------------------------------------------------------------
    def _controls(self, p: dict, seed: int, tex_seed: int, shape_controls: dict | None = None) -> dict:
        c = dict(p)
        c.update(seed=seed, randomize_seed=False, tex_seed=tex_seed, tex_randomize_seed=False, preset=p["preset"])
        if shape_controls:
            from params import SHAPE_KEYS

            c.update({k: shape_controls[k] for k in SHAPE_KEYS if k in shape_controls})
        return c

    def _meta(self, run_id, kind, controls, source_run, status):
        return {"id": run_id, "kind": kind, "status": status, "source_run": source_run,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "app_version": APP_VERSION, "controls": controls,
                "model": {"shapegen": "/".join(SHAPE_MODEL), "texgen": "tencent/Hunyuan3D-2.1 (paintpbr)"},
                "versions": {"torch": torch.__version__}}

    def _finish(self, run_id, run_dir, kind, p, seed, tex_seed, stats, tr: Tracker, warnings, *, source_run, shape_controls=None):
        tr.finish()
        stats["time"] = tr.summary_times()
        stats["peak_gpu_gb"] = {"overall": round(max(tr.peaks.values(), default=0.0), 2),
                                "per_stage": {k: round(v, 2) for k, v in tr.peaks.items()}}
        stats["models_load_s"] = dict(self.load_times)
        stats["warnings"] = warnings
        controls = self._controls(p, seed, tex_seed, shape_controls)
        self.store.write_json(run_dir / "params.json", self._meta(run_id, kind, controls, source_run, "ok"))
        self.store.write_json(run_dir / "stats.json", stats)
        return {"run_id": run_id, "run_dir": str(run_dir), "stats": stats, "controls": controls, "warnings": warnings,
                "has_textured": (run_dir / "textured_mesh.glb").exists()}

    def _fail(self, run_id, run_dir, p, seed, tex_seed, stats, tr: Tracker, err, kind):
        if not (run_dir / "white_mesh.glb").exists():  # nothing worth keeping
            shutil_rm(run_dir)
            return
        tr._close()
        stats["time"] = tr.summary_times()
        stats["error"] = str(err)
        meta = self._meta(run_id, kind, self._controls(p, seed, tex_seed), None,
                          "cancelled" if isinstance(err, JobCancelled) else "failed")
        self.store.write_json(run_dir / "params.json", meta)
        self.store.write_json(run_dir / "stats.json", stats)


def shutil_rm(path: Path):
    import shutil

    shutil.rmtree(path, ignore_errors=True)
