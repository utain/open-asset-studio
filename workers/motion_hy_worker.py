"""Text-to-motion backend (envs/motion_hy): Tencent HY-Motion 1.0-Lite (flow-matching DiT, SMPL-H 22-joint body, 30 fps, <= 12 s).

  task motion   input role `mesh` (a rigged GLB, e.g. a Rig-tab result) + prompt  ->  mesh asset `animated.glb`: the same mesh, skin and
                materials plus ONE animation clip ("motion", first in the file) retargeted onto the character's own skeleton
                (workers/motion_common.py: humanoid detection on any skeleton, global-rotation transfer, root scaling, foot lock);
                attachments `motion.npz` (the raw HY-Motion output: rot6d, transl, keypoints) and `retarget.json` (role map, scale, warnings).

The upstream pipeline (vendor/HY-Motion-1.0, Tencent licence) is used directly, without its T2MRuntime (no prompt rewriter, no FBX
export, no gradio). Text encoders: Qwen/Qwen3-8B (bf16) + openai/clip-vit-large-patch14 from the HF cache (USE_HF_MODELS=1, offline).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_HF_MODELS", "1")          # hymotion's text encoder then resolves "Qwen/Qwen3-8B" / "openai/clip-vit-large-patch14" via the HF cache
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

import motion_common as MC
import rig_anim
from studio_worker import Backend, InputError, serve

STUDIO = Path(os.environ.get("STUDIO_ROOT") or Path(__file__).resolve().parent.parent)
VENDOR = STUDIO / "vendor" / "HY-Motion-1.0"
HY_REPO = "tencent/HY-Motion-1.0"
SUBFOLDER = {"hy-motion-lite": "HY-Motion-1.0-Lite", "hy-motion-full": "HY-Motion-1.0"}
MAX_SECONDS = 12.0


def _snapshot(repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=True))
    except Exception as e:  # noqa: BLE001
        raise InputError(f"{repo} is not in the Hugging Face cache ({type(e).__name__}). Download it on the Models page.") from None


class MotionHYBackend(Backend):
    name = "motion_hy"
    max_loaded = 1

    # ------------------------------------------------------------------ probe / load / unload
    def probe(self, model):
        import einops  # noqa: F401
        import torch  # noqa: F401
        import torchdiffeq  # noqa: F401
        import transformers  # noqa: F401
        for rel in ("hymotion/pipeline/motion_diffusion.py", "stats/Mean.npy", "scripts/gradio/static/assets/dump_wooden/j_template.bin"):
            p = VENDOR / rel
            if not p.is_file():
                return False, f"vendor/HY-Motion-1.0/{rel} is missing (scripts/fetch_vendor.sh HY-Motion-1.0, then `git -C vendor/HY-Motion-1.0 lfs pull`)"
        if (VENDOR / "stats/Mean.npy").stat().st_size < 1000 or (VENDOR / "scripts/gradio/static/assets/dump_wooden/v_template.bin").stat().st_size < 1000:
            return False, "vendor/HY-Motion-1.0 has git-lfs pointer files: run `git -C vendor/HY-Motion-1.0 lfs pull`"
        return True, ""

    def load(self, model, ctx):
        import torch
        import yaml

        sub = SUBFOLDER.get(model["id"])
        if sub is None:
            raise InputError(f"unknown model {model['id']}")
        ctx.progress(0, 0, "loading", "Reading the HY-Motion checkpoint", interruptible=False)
        snap = _snapshot(HY_REPO, [f"{sub}/*"])
        ckpt, cfg_path = snap / sub / "latest.ckpt", snap / sub / "config.yml"
        if not ckpt.is_file() or not cfg_path.is_file():
            raise InputError(f"{HY_REPO}/{sub} is not fully downloaded.")
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from hymotion.utils.loaders import load_object

        with open(cfg_path) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        args = dict(cfg["train_pipeline_args"])
        args["test_cfg"] = dict(args.get("test_cfg") or {}, mean_std_dir=str(VENDOR / "stats"))
        cwd, t0 = os.getcwd(), time.time()
        os.chdir(VENDOR)                                     # WoodenMesh() reads its .bin files relative to the repo root
        try:
            pipe = load_object(cfg["train_pipeline"], args, network_module=cfg["network_module"],
                               network_module_args=cfg["network_module_args"])
            t_net = time.time() - t0
            ctx.progress(0, 0, "loading", "Loading Qwen3-8B + CLIP text encoders", interruptible=False)
            pipe.load_in_demo(str(ckpt), build_text_encoder=True)
        finally:
            os.chdir(cwd)
        t_txt = time.time() - t0 - t_net
        ctx.progress(0, 0, "loading", "Moving to the GPU", interruptible=False)
        pipe.to(torch.device("cuda"))
        pipe.eval()
        ctx.log(f"load: network {t_net:.1f} s, checkpoint + text encoders {t_txt:.1f} s, to GPU {time.time() - t0 - t_net - t_txt:.1f} s")
        # the body model is only used for the grounded keypoints; check it matches the embedded rest skeleton
        rest = MC.load_hy_rest(VENDOR / "scripts/gradio/static/assets/dump_wooden/j_template.bin")
        if float(np.abs(rest - MC.HY_REST).max()) > 2e-4:
            ctx.log("warning: WoodenMesh j_template differs from motion_common.HY_REST")
        return {"pipe": pipe, "sub": sub, "train_frames": int(pipe.train_frames)}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    # ------------------------------------------------------------------ run
    def run(self, job, ctx):
        if job.task != "motion":
            raise InputError(f"{job.model.get('name', job.model_id)} cannot do '{job.task}'.")
        inp = job.input(role="mesh") or job.input(type="mesh")
        if inp is None:
            raise InputError("Pick the rigged character to animate (Auto-rig tab result).")
        p = job.params
        prompt = " ".join(str(p.get("prompt") or "").split())
        if not prompt:
            raise InputError("Describe the motion, e.g. \"a person walks forward and waves\".")
        seconds = min(max(float(p.get("seconds", 5) or 5), 1.0), MAX_SECONDS)
        steps = int(p.get("steps", 50) or 50)
        cfg_scale = float(p.get("guidance", 5.0) or 5.0)
        fps = int(p.get("fps", 30) or 30)
        seed = int(p.get("seed", -1)) if p.get("seed") is not None else -1
        if seed < 0:
            import random
            seed = random.randrange(2**31 - 1)
        src_glb = Path(inp["path"])
        plain = src_glb.with_name("rigged_plain.glb")
        keep = bool(p.get("keep_clips", False))
        if not keep and plain.is_file():
            src_glb = plain                                      # the Rig result without its flex / wiggle test clips
        glb_bytes = src_glb.read_bytes()
        out = job.out_dir / "asset"
        out.mkdir(parents=True, exist_ok=True)
        t = {}

        # 1. refuse early (before any GPU work) when the character cannot be animated
        ctx.progress(0, 4, "prep", "Checking the skeleton")
        try:
            js, binary = rig_anim.read_glb(glb_bytes)
            rig = rig_anim.Rig(js)
            sk, _ = MC.skeleton_from_rig(rig)
            hum = MC.detect_humanoid(sk.X, sk.parents, sk.names, facing=_facing(p))
        except rig_anim.NoSkinError:
            raise InputError("This mesh has no skeleton. Rig it first (Auto-rig tab), then animate the rigged result.") from None
        except rig_anim.RigAnimError as e:
            raise InputError(str(e)) from None
        ctx.log(f"humanoid: {hum['method']}, facing {hum['facing_name']}, {len(rig.joints)} joints, leg length {hum['leg_length']:.3f}")
        ctx.check()

        # 2. generate
        import torch
        h = self.handle(job.model_id)
        pipe = h["pipe"]
        pipe.validation_steps = steps
        calls = {"n": 0}
        net = pipe.motion_transformer
        orig_forward = net.forward

        def counted(*a, **kw):
            ctx.check()
            calls["n"] += 1
            ctx.progress(min(calls["n"], steps), steps, "denoising", f"step {min(calls['n'], steps)}/{steps}")
            return orig_forward(*a, **kw)

        net.forward = counted
        t0 = time.time()
        ctx.progress(0, steps, "encoding", "Encoding the prompt")
        try:
            with torch.no_grad():
                out_d = pipe.generate(prompt, [seed], seconds, cfg_scale=cfg_scale)
        finally:
            net.forward = orig_forward
        t["generate_s"] = round(time.time() - t0, 2)
        rot6d = out_d["rot6d"][0].float().numpy()                # [L, 52, 6]
        transl = out_d["transl"][0].float().numpy()              # [L, 3]
        k3d = out_d["keypoints3d"][0].float().numpy()
        ctx.log(f"generated {len(rot6d)} frames in {t['generate_s']} s (peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)")
        ctx.check()

        # 3. retarget + write the GLB
        ctx.progress(steps, steps, "retarget", "Retargeting onto the skeleton")
        t1 = time.time()
        clip = "motion"
        glb_out, rep = MC.retarget_glb(glb_bytes, rot6d, transl, clip_name=clip, fps=fps, facing=_facing(p),
                                       in_place=(p.get("root_motion") == "in_place"), foot_lock=bool(p.get("foot_lock", True)),
                                       arm_offset_deg=float(p.get("arm_offset", 0) or 0))
        t["retarget_s"] = round(time.time() - t1, 3)
        (out / "animated.glb").write_bytes(glb_out)
        np.savez_compressed(out / "motion.npz", rot6d=rot6d.astype(np.float32), transl=transl.astype(np.float32),
                            keypoints3d=k3d.astype(np.float32), fps=MC.HY_FPS, prompt=prompt, seed=seed,
                            joint_names=np.array(MC.HY_NAMES), parents=np.array(MC.HY_PARENTS))
        (out / "retarget.json").write_text(json.dumps({**rep, "prompt": prompt, "seed": seed, "model": job.model_id}, indent=1, default=float))
        for w in rep["warnings"]:
            ctx.log("warning: " + w)
        ctx.log(f"clip '{clip}': {rep['frames']} frames @ {rep['fps']} fps ({rep['duration']} s), {rep['channels']} channels, "
                f"scale {rep['scale']}, contacts {rep['contact_frames']}, timings {t}")
        ctx.output(out / "animated.glb", "mesh", seed=seed, name=f"{(inp.get('name') or 'mesh')[:40]} - {prompt[:36]}",
                   attach=["motion.npz", "retarget.json"], final_prompt=prompt, rigged=True, joints=len(rig.joints),
                   animations=[clip], motion={"model": job.model_id, "prompt": prompt, "seconds": round(rep["duration"], 2),
                                              "fps": rep["fps"], "frames": rep["frames"], "steps": steps, "guidance": cfg_scale,
                                              "root_motion": p.get("root_motion", "root"), "facing": rep["facing"]})
        return {"meta": {"timings": t, "frames": rep["frames"], "channels": rep["channels"]}}


def _facing(p) -> str:
    f = str(p.get("facing", "auto") or "auto")
    return f if f in ("+Z", "-Z", "+X", "-X") else "auto"


if __name__ == "__main__":
    serve(MotionHYBackend())
