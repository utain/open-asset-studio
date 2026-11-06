"""Video worker: open-weight video generation with diffusers (backend `video`, runs in envs/video).

Families (picked from the registry entry's `family`, else from the id):
  wan21        Wan2.1-T2V-1.3B           t2v     480p / 16 fps, tiny and fast (preview model)
  wan22        Wan2.2-TI2V-5B            t2v + i2v   up to 720p / 24 fps (one model for both)
  ltx2         LTX-2.3 distilled (22B)   t2v + i2v + flf2v   video AND synchronised audio, 8 steps
  kandinsky6   Kandinsky 6.0 Lite distill   t2v + i2v   video AND audio, MIT licence
  hunyuan15    HunyuanVideo-1.5 480p (step-distilled i2v / distilled t2v)
  rife         RIFE v4 frame interpolation (interpolate task, 2x / 4x frame rate)

Every output is an h264 / yuv420p / +faststart mp4 (workers/video_common.write_mp4) so the core can make the poster frame
and the browser can seek. Weights come from the local Hugging Face cache (offline). Components are loaded one by one
straight to the GPU (the proven recipe of qwen_worker.py: safetensors read with device="cuda"), because stock
`.to("cuda")` of memory-mapped weights runs at ~170 MB/s on the GB10.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # the venv is shared read-only: no __pycache__ writes

import glob
import json
import os
import random
import time
from pathlib import Path

from studio_worker import Backend, InputError, log, serve

import video_common as vc

HUB = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")

FAMILIES = {"wan21-t2v-1.3b": "wan21", "wan22-ti2v-5b": "wan22", "ltx23-distilled": "ltx2", "kandinsky6-lite-distill": "kandinsky6",
            "hunyuan15-480p-i2v-distilled": "hunyuan15", "hunyuan15-480p-t2v-distilled": "hunyuan15", "rife-v4": "rife"}
# model grid (width / height multiple) and frame rule (k * step + 1) per family
GRID = {"wan21": 16, "wan22": 32, "ltx2": 32, "kandinsky6": 16, "hunyuan15": 16}
STEP = {"wan21": 4, "wan22": 4, "ltx2": 8, "kandinsky6": 4, "hunyuan15": 4}

WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
                "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
LTX_NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted"


def family_of(model: dict) -> str:
    fam = model.get("family") or FAMILIES.get(model.get("id", ""))
    if not fam:
        raise InputError(f"The video worker does not know the model '{model.get('id')}'.")
    return fam


# =========================================================== loading
def snap(repo: str) -> str:
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"))
    ref = os.path.join(d, "refs", "main")
    if os.path.exists(ref):
        s = os.path.join(d, "snapshots", open(ref).read().strip())
        if os.path.isdir(s):
            return s
    snaps = sorted(glob.glob(d + "/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"{repo} is not in the local Hugging Face cache ({HUB})")
    return snaps[-1]


def weight_repo(model: dict, prefix: str | None = None) -> str:
    """The registry `weights` entry (hf repo) whose files mention `prefix` (e.g. 'transformer'); first one when None."""
    for w in model.get("weights") or []:
        if "hf" not in w:
            continue
        if prefix is None or any(str(f).startswith(prefix) for f in w.get("files") or [prefix]):
            return w["hf"]
    raise InputError(f"{model.get('id')}: no weights entry for '{prefix}'")


def dev_load(cls, path: str, subfolder: str | None = None, dtype=None, **kw):
    """diffusers / transformers component straight to the GPU."""
    import torch
    kw.setdefault("device_map", "cuda")
    dtype = dtype or torch.bfloat16
    if subfolder:
        kw["subfolder"] = subfolder
    if cls.__module__.startswith("diffusers"):
        kw.setdefault("disable_mmap", True)
    return cls.from_pretrained(path, torch_dtype=dtype, **kw)


def fast_te(cls, path: str, **kw):
    """transformers model straight to the GPU: lazy model, then tensors swapped for safetensors device="cuda" reads."""
    import torch
    from safetensors.torch import load_file
    m = cls.from_pretrained(path, dtype=torch.bfloat16, **kw)
    params = dict(m.named_parameters())
    params.update(dict(m.named_buffers()))
    hit = 0
    for f in sorted(glob.glob(path + "/*.safetensors")):
        sd = load_file(f, device="cuda")
        for k, v in sd.items():
            for kk in (k, k.removeprefix("model."), "model." + k):
                t = params.get(kk)
                if t is not None and t.shape == v.shape:
                    t.data = v if v.dtype == t.dtype else v.to(t.dtype)
                    hit += 1
                    break
        del sd
    miss = len(dict(m.named_parameters())) - hit
    if miss > 0:
        log(f"fast_te {os.path.basename(path)}: {miss} tensors not swapped (copied the slow way)")
    return m.to("cuda")


class Handle:
    """What load() returns: the pipelines of one model (several tasks share the same components)."""

    def __init__(self, family, **pipes):
        self.family = family
        self.pipes = pipes          # task -> pipeline (t2v / i2v / flf2v) or {"net": IFNet}
        self.mode = "gpu"           # gpu | model | sequential

    def any(self):
        return next(iter(self.pipes.values()))


class VideoBackend(Backend):
    name = "video"
    max_loaded = 1

    def probe(self, model):
        import diffusers  # noqa: F401
        fam = family_of(model)
        if fam == "rife":
            import video_rife  # noqa: F401
        else:
            import transformers  # noqa: F401
        return True, ""

    # ------------------------------------------------------------------ load
    def load(self, model, ctx):
        import torch
        fam = family_of(model)
        t0 = time.time()
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        if fam == "rife":
            import video_rife
            f = glob.glob(snap(weight_repo(model)) + "/flownet.safetensors")[0]
            return Handle(fam, net=video_rife.load_rife(f))
        fn = getattr(self, f"_load_{fam}")
        h = fn(model, ctx)
        log(f"loaded {model['id']} in {time.time() - t0:.0f}s, gpu {torch.cuda.memory_allocated() / 2**30:.1f} GB")
        return h

    def _load_wan(self, model, ctx, t2v_only: bool):
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel
        from transformers import AutoTokenizer, UMT5EncoderModel
        main = snap(weight_repo(model, "transformer"))
        te_repo = snap(weight_repo(model, "text_encoder"))
        ctx.progress(0, 0, "loading", "Loading the text encoder", interruptible=False)
        te = fast_te(UMT5EncoderModel, te_repo + "/text_encoder")
        tok = AutoTokenizer.from_pretrained(te_repo + "/tokenizer")
        ctx.progress(0, 0, "loading", "Loading the transformer", interruptible=False)
        tr = dev_load(WanTransformer3DModel, main, "transformer")
        vae = dev_load(AutoencoderKLWan, main, "vae", dtype=torch.float32)
        sch = UniPCMultistepScheduler.from_pretrained(main, subfolder="scheduler")
        comps = dict(tokenizer=tok, text_encoder=te, transformer=tr, vae=vae, scheduler=sch)
        pipes = {"t2v": WanPipeline(**comps)}
        if not t2v_only:
            pipes["i2v"] = WanImageToVideoPipeline(**comps, expand_timesteps=True)
        return Handle("wan", **pipes)

    def _load_wan21(self, model, ctx):
        h = self._load_wan(model, ctx, True)
        h.family = "wan21"
        return h

    def _load_wan22(self, model, ctx):
        h = self._load_wan(model, ctx, False)
        h.family = "wan22"
        return h

    def _load_ltx2(self, model, ctx):
        import torch
        from diffusers import LTX2ConditionPipeline
        path = snap(weight_repo(model))
        pipe = LTX2ConditionPipeline.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
        return Handle("ltx2", t2v=pipe, i2v=pipe, flf2v=pipe)

    def _load_kandinsky6(self, model, ctx):
        import torch
        from diffusers import Kandinsky6TI2VAPipeline
        path = snap(weight_repo(model))
        pipe = Kandinsky6TI2VAPipeline.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
        return Handle("kandinsky6", t2v=pipe, i2v=pipe)

    def _load_hunyuan15(self, model, ctx):
        import torch
        path = snap(weight_repo(model))
        if "i2v" in model["id"]:
            from diffusers import HunyuanVideo15ImageToVideoPipeline as P
            task = "i2v"
        else:
            from diffusers import HunyuanVideo15Pipeline as P
            task = "t2v"
        pipe = P.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
        return Handle("hunyuan15", **{task: pipe})

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        if h is not None:
            for p in h.pipes.values():
                try:
                    p.remove_all_hooks()
                except Exception:  # noqa: BLE001
                    pass
        del h

    # ------------------------------------------------------------------ memory options
    @staticmethod
    def _apply_memory(h: Handle, p: dict):
        """offload: none | model (diffusers model cpu offload) | sequential; vae_tiling: bool. Re-applied only on change."""
        want = str(p.get("offload") or "none")
        want = {"none": "gpu"}.get(want, want)
        pipes = list({id(x): x for x in h.pipes.values()}.values())
        if want != h.mode:
            for x in pipes:
                try:
                    x.remove_all_hooks()
                except Exception:  # noqa: BLE001
                    pass
            if want == "gpu":
                for x in pipes:
                    x.to("cuda")
            elif want == "model":
                for x in pipes:
                    x.enable_model_cpu_offload()
            elif want == "sequential":
                for x in pipes:
                    x.enable_sequential_cpu_offload()
            h.mode = want
        tiling = bool(p.get("vae_tiling"))
        for x in pipes:
            vae = getattr(x, "vae", None)
            if vae is None:
                continue
            try:
                (vae.enable_tiling if tiling else vae.disable_tiling)()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ run
    def run(self, job, ctx):
        h = self.handle(job.model_id)
        fam = family_of(job.model)
        if fam == "rife":
            return self._run_rife(job, ctx, h)
        import torch
        p = job.params
        task = job.task
        if task not in h.pipes:
            raise InputError(f"{job.model.get('name', job.model_id)} cannot do '{task}'.")
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Write a prompt first.")
        feat = job.model.get("features") or {}
        first = job.input(role="image")
        last = job.input(role="last_image")
        if task in ("i2v", "flf2v") and not first:
            raise InputError("Add the first-frame image.")
        if task == "flf2v" and not last:
            raise InputError("Add the last-frame image.")
        self._apply_memory(h, p)

        # ---- geometry, frames, fps
        mult = GRID[fam]
        fps = float(p.get("fps") or feat.get("fps") or 24)
        n_frames = vc.frame_count(p.get("seconds", 5), fps, STEP[fam], int(feat.get("max_frames", 121)))
        match = bool(p.get("match_image", True)) and first is not None
        img_size = vc.image_size(first["path"]) if (first and match) else None
        W, H = vc.dims_from_aspect(p.get("aspect", "16:9"), int(p.get("resolution", 480)), mult, img_size)
        first_img = vc.fit_image(first["path"], (W, H)) if first else None
        last_img = vc.fit_image(last["path"], (W, H)) if last else None
        steps = int(p.get("steps") or feat.get("steps", 30))
        guidance = float(p.get("guidance", feat.get("guidance", 5.0)))
        neg = (p.get("negative") or "").strip()
        want_audio = bool(p.get("audio", True))
        pipe = h.pipes[task]
        batch = max(1, int(p.get("batch", 1)))
        seed0 = int(p["seed"]) if p.get("seed", -1) is not None and int(p.get("seed", -1)) >= 0 else random.randrange(2**31)
        ctx.log(f"{fam}/{task}: {W}x{H}, {n_frames} frames @ {fps:g} fps ({n_frames / fps:.1f} s), {steps} steps, guidance {guidance:g}, seed {seed0}")
        results = []
        for i in range(batch):
            ctx.check()
            seed = seed0 + i
            gen = torch.Generator(device="cuda").manual_seed(seed)
            label = f"take {i + 1}/{batch} · " if batch > 1 else ""
            vc.hook_progress(pipe, ctx, label)
            t0 = time.time()
            ctx.progress(0, steps, "encoding", f"{label}Encoding the prompt…")
            frames, audio, a_sr = getattr(self, f"_gen_{fam}")(
                pipe, task, prompt=prompt, neg=neg, W=W, H=H, n=n_frames, fps=fps, steps=steps, guidance=guidance, gen=gen,
                first=first_img, last=last_img, want_audio=want_audio, p=p, ctx=ctx, feat=feat)
            ctx.check()
            ctx.progress(steps, steps, "encoding", f"{label}Writing the mp4…", interruptible=False)
            out = job.out_dir / f"take_{i + 1}.mp4"
            info = vc.write_mp4(out, frames, fps, audio=audio, sample_rate=a_sr)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            ctx.log(f"take {i + 1}: {dt:.0f} s, peak {peak:.1f} GB, {info['bytes'] / 2**20:.1f} MB")
            ctx.output(out, "video", seed=seed, index=i, name=(prompt[:50] + (f" #{i + 1}" if batch > 1 else "")), final_prompt=prompt,
                       fps=fps, frames=info["frames"], width=info["width"], height=info["height"], has_audio=info["has_audio"],
                       steps=steps, gen_s=round(dt, 1), peak_gb=round(peak, 1))
            results.append(info)
            torch.cuda.reset_peak_memory_stats()
        return {"meta": {"family": fam, "frames": n_frames, "fps": fps, "size": [W, H], "seed": seed0}}

    # ------------------------------------------------------------------ per-family generation: -> (uint8 frames, audio, sample_rate)
    def _gen_wan(self, pipe, task, *, prompt, neg, W, H, n, fps, steps, guidance, gen, first, last, want_audio, p, ctx, feat):
        kw = dict(prompt=prompt, negative_prompt=neg or WAN_NEGATIVE, height=H, width=W, num_frames=n, num_inference_steps=steps,
                  guidance_scale=guidance, generator=gen, output_type="np")
        if task == "i2v":
            kw["image"] = first
        out = pipe(**kw)
        return vc.to_uint8(out.frames[0]), None, 0

    _gen_wan21 = _gen_wan22 = _gen_wan

    def _gen_ltx2(self, pipe, task, *, prompt, neg, W, H, n, fps, steps, guidance, gen, first, last, want_audio, p, ctx, feat):
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        conds = []
        if first is not None:
            conds.append(LTX2VideoCondition(frames=first, index=0, strength=1.0))
        if last is not None:
            conds.append(LTX2VideoCondition(frames=last, index=-1, strength=1.0))
        distilled = bool(feat.get("distilled", True))
        kw = dict(conditions=conds or None, prompt=prompt, negative_prompt=neg or LTX_NEGATIVE, height=H, width=W, num_frames=n,
                  frame_rate=fps, guidance_scale=guidance, generator=gen, output_type="np", return_dict=False)
        if distilled:
            kw.update(sigmas=DISTILLED_SIGMA_VALUES, num_inference_steps=len(DISTILLED_SIGMA_VALUES), guidance_scale=1.0)
        else:
            kw.update(num_inference_steps=steps)
        video, audio = pipe(**kw)
        a = audio[0].float().cpu().numpy() if want_audio and audio is not None else None
        sr = int(pipe.vocoder.config.output_sampling_rate)
        return vc.to_uint8(video[0]), a, sr

    def _gen_kandinsky6(self, pipe, task, *, prompt, neg, W, H, n, fps, steps, guidance, gen, first, last, want_audio, p, ctx, feat):
        kw = dict(prompt=prompt, height=H, width=W, num_frames=n, frame_rate=fps, num_inference_steps=steps, guidance_scale=guidance,
                  generator=gen, sample_audio=want_audio, output_type="np")
        if neg and guidance > 1.0:
            kw["negative_prompt"] = neg
        if task == "i2v":
            kw["image"] = first
        out = pipe(**kw)
        a = out.audio[0] if (want_audio and out.audio is not None) else None
        if a is not None and hasattr(a, "float"):
            a = a.float().cpu().numpy()
        return vc.to_uint8(out.frames[0]), a, int(pipe.audio_sample_rate)

    def _gen_hunyuan15(self, pipe, task, *, prompt, neg, W, H, n, fps, steps, guidance, gen, first, last, want_audio, p, ctx, feat):
        kw = dict(prompt=prompt, num_frames=n, num_inference_steps=steps, generator=gen, output_type="np")
        if task == "i2v":
            kw["image"] = first
        else:
            kw.update(height=H, width=W)
        if neg:
            kw["negative_prompt"] = neg
        out = pipe(**kw)
        return vc.to_uint8(out.frames[0]), None, 0

    # ------------------------------------------------------------------ interpolate (RIFE)
    def _run_rife(self, job, ctx, h):
        import torch
        import video_rife
        src = job.input(role="video") or job.input(type="video")
        if not src:
            raise InputError("Add the video to interpolate.")
        p = job.params
        factor = int(p.get("factor", 2))
        ctx.progress(0, 1, "decoding", "Reading the video…")
        frames, fps = vc.read_video_frames(src["path"], max_frames=int(p.get("max_frames") or 0) or None)
        ctx.check()
        passes = {2: 1, 4: 2}.get(factor, 1)
        out = frames
        for k in range(passes):
            out = video_rife.rife_double(h.pipes["net"], out, check=ctx.check, report=lambda a, b, k=k: ctx.progress(
                k * b + a, passes * b, "interpolating", f"Pass {k + 1}/{passes}: pair {a}/{b}"))
        new_fps = fps * factor
        # keep the audio of the source when it has one
        audio = None
        out_f = job.out_dir / "interpolated.mp4"
        info = vc.write_mp4(out_f, out, new_fps)
        if p.get("keep_audio", True):
            try:
                _mux_audio(src["path"], out_f)
            except Exception as e:  # noqa: BLE001
                ctx.log(f"audio not copied: {e}")
        ctx.output(out_f, "video", name=f"{src.get('name', 'video')} {factor}x", fps=new_fps, frames=info["frames"], width=info["width"], height=info["height"],
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        return {"meta": {"family": "rife", "factor": factor, "source_fps": fps, "fps": new_fps}}


def _mux_audio(src: str, video_file: Path):
    """Copy the first audio stream of `src` into `video_file` (in place), if there is one."""
    import subprocess
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "json", src],
                        capture_output=True, text=True)
    if not json.loads(pr.stdout or "{}").get("streams"):
        return
    tmp = video_file.with_name("mux_tmp.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_file), "-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(tmp)], check=True)
    os.replace(tmp, video_file)


if __name__ == "__main__":
    serve(VideoBackend())
