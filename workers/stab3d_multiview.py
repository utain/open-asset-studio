"""Multi-view / orbit models of the stab3d backend (imported by workers/stab3d_worker.py; same venv envs/stab3d).

  zero123plus  sudo-ai/zero123plus-v1.2 (Apache-2.0 code, open weights): one image -> 6 fixed views (azimuth 30..330 step 60,
               elevation +20 / -10 alternating), one diffusion pass; diffusers custom pipeline from sudo-ai/zero123plus-pipeline
  sv3d-u / sv3d-p  stabilityai/sv3d (gated): Stable Video 3D, 21-frame orbit around the object (576x576), via Stability's
               `sgm` code (vendor/generative-models). _p takes the elevation, _u has a fixed 10 degree orbit.

Task `multiview`, input role `image`. Output assets (type image): first `sheet.png` (all views in a grid; for SV3D the orbit
video `orbit.mp4` is attached to it, the drawer plays it), then one PNG per view when `save_views` is on.
The registry hands the models' parameters in `job.params` (see models.d/3d_more.yaml).
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import m3d_common as C
from studio_worker import InputError

STUDIO = Path(os.environ.get("STUDIO_ROOT") or Path(__file__).resolve().parent.parent)
VENDOR = STUDIO / "vendor"
ZERO123PLUS_VIEWS = [(30, 20), (90, -10), (150, 20), (210, -10), (270, 20), (330, -10)]   # (azimuth, elevation) degrees


def probe(model):
    mid = model["id"]
    if mid == "zero123plus":
        import diffusers  # noqa: F401
        return True, ""
    if mid.startswith("sv3d"):
        p = str(VENDOR / "generative-models")
        if p not in sys.path:
            sys.path.insert(0, p)
        import sgm.models.diffusion  # noqa: F401
        return True, ""
    return False, f"unknown multiview model {mid}"


def _pipeline_dir() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download("sudo-ai/zero123plus-pipeline")


def load(model, ctx):
    import torch
    mid = model["id"]
    if mid == "zero123plus":
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        from huggingface_hub import snapshot_download
        repo = snapshot_download("sudo-ai/zero123plus-v1.2")
        pipe = DiffusionPipeline.from_pretrained(repo, custom_pipeline=_pipeline_dir(), torch_dtype=torch.float16)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.to("cuda")
        return {"kind": "zero123plus", "pipe": pipe}
    if mid.startswith("sv3d"):
        return _load_sv3d(mid)
    raise InputError(f"unknown multiview model {mid}")


# ---------------------------------------------------------------------------- run
def run(backend, job, ctx):
    import torch
    inp = job.input(role="image") or job.input(type="image")
    if inp is None:
        raise InputError("Add the image of the object to view from other sides.")
    p = job.params
    h = backend.handle(job.model_id)
    out = job.out_dir / "asset"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx.progress(0, 1, "image", "Preparing the image")
    rgba = C.prep_object_image(inp["path"], bool(p.get("remove_bg", True)), float(p.get("foreground_ratio", 0.85)))
    seed = C.seed_of(p)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    if h["kind"] == "zero123plus":
        views, meta = _zero123plus(h, rgba, p, seed, ctx)
        video = None
    else:
        views, meta, video = _sv3d(h, job.model_id, rgba, p, seed, ctx)
    ctx.check()
    gen_s = round(time.time() - t1, 2)
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    name = (inp.get("name") or "object")[:40]
    sheet = _sheet(views, out / "sheet.png")
    attach = []
    if video is not None:
        _write_video(views, out / "orbit.mp4", int(p.get("fps", 8)))
        attach = ["orbit.mp4"]
    ctx.output(out / "sheet.png", "image", seed=seed, name=f"{name} · views ({job.model.get('name', job.model_id)})", attach=attach,
               index=0, views=len(views), **meta)
    if bool(p.get("save_views", True)):
        step = 1 if len(views) <= 8 else int(p.get("view_stride", 3))
        for i in range(0, len(views), step):
            f = out / f"view_{i + 1:02d}.png"
            views[i].save(f)
            lab = (meta.get("cameras") or [None] * len(views))[i]
            ctx.output(f, "image", seed=seed, index=i + 1, name=f"{name} · view {i + 1}" + (f" (az {lab[0]:.0f}°, el {lab[1]:.0f}°)" if lab else ""),
                       camera=({"azimuth_deg": lab[0], "elevation_deg": lab[1]} if lab else None))
    ctx.log(f"timings gen {gen_s}s, peak GPU {peak} GB, {len(views)} views")
    return {"meta": {"timings": {"gen_s": gen_s}, "peak_gpu_gb": peak, "views": len(views)}}


def _sheet(views, dst, cols=None):
    from PIL import Image
    n = len(views)
    cols = cols or (3 if n <= 6 else 7 if n == 21 else 4)
    rows = math.ceil(n / cols)
    w, h = views[0].size
    sh = Image.new("RGB", (cols * w, rows * h), "white")
    for i, v in enumerate(views):
        sh.paste(v.convert("RGB"), ((i % cols) * w, (i // cols) * h))
    sh.save(dst)
    return sh


def _write_video(views, dst, fps):
    import numpy as np
    import imageio
    frames = [np.asarray(v.convert("RGB")) for v in views]
    frames = frames + frames[-2:0:-1]            # ping-pong: loop-friendly
    try:
        imageio.mimwrite(str(dst), frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    except Exception:  # noqa: BLE001
        # fallback: ffmpeg from PNG frames
        tmp = dst.parent / "frames"
        tmp.mkdir(exist_ok=True)
        for i, f in enumerate(frames):
            from PIL import Image
            Image.fromarray(f).save(tmp / f"{i:03d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "%03d.png"), "-pix_fmt", "yuv420p", str(dst)], check=True)
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------- Zero123++
def _zero123plus(h, rgba, p, seed, ctx):
    import torch
    from PIL import Image
    pipe = h["pipe"]
    steps = int(p.get("steps", 75))
    cond = C.composite_gray(C.crop_resize_foreground(rgba, float(p.get("foreground_ratio", 0.85)), 320), 1.0)
    with torch.no_grad():
        res = pipe(cond, num_inference_steps=steps, guidance_scale=float(p.get("guidance", 4.0)),
                   generator=torch.Generator("cuda").manual_seed(seed), callback_on_step_end=None,
                   callback=lambda i, t, lat: (ctx.check(), ctx.progress(i + 1, steps, "denoising", f"step {i + 1}/{steps}")),
                   callback_steps=1).images[0]
    grid = res.convert("RGBA") if res.mode != "RGB" else res
    W, H = grid.size              # 640 x 960: 2 columns x 3 rows of 320 x 320
    cw, ch = W // 2, H // 3
    views = [grid.crop(((i % 2) * cw, (i // 2) * ch, (i % 2 + 1) * cw, (i // 2 + 1) * ch)).convert("RGB") for i in range(6)]
    if bool(p.get("upscale_views", False)):
        views = [v.resize((cw * 2, ch * 2), Image.LANCZOS) for v in views]
    return views, {"cameras": ZERO123PLUS_VIEWS, "pose": "azimuth/elevation relative to the input view"}


# --------------------------------------------------------------------------- SV3D
def _load_sv3d(mid):
    import torch
    sys.path.insert(0, str(VENDOR / "generative-models"))
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from sgm.util import instantiate_from_config
    variant = "sv3d_p" if mid.endswith(("-p", "_p")) else "sv3d_u"
    cfg = OmegaConf.load(VENDOR / "generative-models" / "scripts" / "sampling" / "configs" / f"{variant}.yaml")
    ckpt = hf_hub_download("stabilityai/sv3d", f"{variant}.safetensors")
    cfg.model.params.ckpt_path = None
    cfg.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = "cpu"
    # the CLIP image tower is in the checkpoint: do not download another copy to initialise it
    cfg.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.version = ""
    cfg.model.params.sampler_config.params.verbose = False
    model = instantiate_from_config(cfg.model)
    sd = load_file(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "open_clip" in k or k.startswith("model.") or "first_stage" in k]
    if bad:
        raise RuntimeError(f"SV3D checkpoint did not fill {len(bad)} weights (first: {bad[0]})")
    model = model.to("cuda").eval()
    return {"kind": "sv3d", "model": model, "variant": variant, "cfg": cfg}


def _sv3d(h, mid, rgba, p, seed, ctx):
    import numpy as np
    import torch
    from einops import rearrange, repeat
    model, variant = h["model"], h["variant"]
    num_frames = 21
    steps = int(p.get("steps", 50))
    model.sampler.num_steps = steps
    model.sampler.guider.num_frames = num_frames
    fg = C.crop_resize_foreground(rgba, float(p.get("foreground_ratio", 0.85)), 576)
    img = C.composite_gray(fg, 1.0)
    x = torch.from_numpy(np.asarray(img).astype("float32") / 255.0).permute(2, 0, 1)[None].to("cuda") * 2 - 1
    cond_aug = 1e-5
    value = {"cond_frames_without_noise": x, "cond_frames": x + cond_aug * torch.randn_like(x), "cond_aug": cond_aug,
             "fps_id": 6, "motion_bucket_id": 127}
    elev = float(p.get("elevation", 10.0))
    azim_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    if variant == "sv3d_p":
        value["polars_rad"] = [np.deg2rad(90 - elev)] * num_frames
        az = [np.deg2rad((a - azim_deg[-1]) % 360) for a in azim_deg]
        az[:-1] = sorted(az[:-1])
        value["azimuths_rad"] = az
    else:
        elev = 10.0
    torch.manual_seed(seed)
    dev = "cuda"
    with torch.no_grad(), torch.autocast(dev):
        keys = list({e.input_key for e in model.conditioner.embedders})
        batch, batch_uc = _sv3d_batch(keys, value, [1, num_frames], num_frames, dev)
        c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc=batch_uc,
                                                                 force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"])
        for k in ["crossattn", "concat"]:
            uc[k] = rearrange(repeat(uc[k], "b ... -> b t ...", t=num_frames), "b t ... -> (b t) ...", t=num_frames)
            c[k] = rearrange(repeat(c[k], "b ... -> b t ...", t=num_frames), "b t ... -> (b t) ...", t=num_frames)
        randn = torch.randn((num_frames, 4, 72, 72), device=dev)
        extra = {"image_only_indicator": torch.zeros(2, num_frames).to(dev), "num_video_frames": batch["num_video_frames"]}

        def denoiser(inp, sigma, cc):
            ctx.check()
            return model.denoiser(model.model, inp, sigma, cc, **extra)

        zs = model.sampler(denoiser, randn, cond=c, uc=uc)
        model.en_and_decode_n_samples_a_time = int(p.get("decode_t", 7))
        xs = model.decode_first_stage(zs)
        xs[-1:] = value["cond_frames_without_noise"]
        xs = torch.clamp((xs + 1.0) / 2.0, 0.0, 1.0)
    from PIL import Image
    frames = (rearrange(xs, "t c h w -> t h w c") * 255).cpu().numpy().astype("uint8")
    views = [Image.fromarray(f) for f in frames]
    cams = [(float(a), elev) for a in azim_deg]
    return views, {"cameras": cams, "pose": "orbit (azimuth 0 = the input view)"}, True


def _sv3d_batch(keys, value, N, T, device):
    import math as m
    import torch
    from einops import repeat
    batch, batch_uc = {}, {}
    for key in keys:
        if key in ("fps_id", "motion_bucket_id"):
            batch[key] = torch.tensor([value[key]]).to(device).repeat(int(m.prod(N)))
        elif key == "cond_aug":
            batch[key] = repeat(torch.tensor([value["cond_aug"]]).to(device), "1 -> b", b=m.prod(N))
        elif key in ("cond_frames", "cond_frames_without_noise"):
            batch[key] = repeat(value[key], "1 ... -> b ...", b=N[0])
        elif key in ("polars_rad", "azimuths_rad"):
            batch[key] = torch.tensor(value[key]).to(device).repeat(N[0])
        else:
            batch[key] = value[key]
    batch["num_video_frames"] = T
    for key in batch:
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc
