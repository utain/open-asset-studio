"""Foley backend (envs/foley): video-to-audio and text-to-audio.

  mmaudio-large-44k-v2   MMAudio large 44.1 kHz v2 (hkchengrex/MMAudio): video+text -> audio, also text-only (sfx)
  hunyuan-foley          HunyuanVideo-Foley (tencent/HunyuanVideo-Foley): video+text -> 48 kHz audio
  thinksound             ThinkSound (FunAudioLLM/ThinkSound): video + caption + chain-of-thought text -> audio

Task `v2a`: input role `video` (a video file; the core hands the path), params prompt / negative / seconds / ...
Outputs: audio take(s) (wav, 24-bit) + optionally the video with the take muxed in (ffmpeg; attached to the audio
asset as `<take>_video.mp4`, see `mux_video`).  Task `sfx` (text only): MMAudio and (if it supports it) the others.
"""
from __future__ import annotations

import importlib.util
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from studio_worker import Backend, Cancelled, InputError, serve

STUDIO = Path(__file__).resolve().parent.parent
VENDOR = STUDIO / "vendor"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".mpg", ".mpeg", ".ogv", ".flv", ".wmv"}


# ------------------------------------------------------------------ helpers
def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def hf_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        raise InputError(f"Missing weights {repo}/{filename}: run `hf download {repo} {filename}`") from e


def hf_dir(repo: str, patterns: list[str] | None = None) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True, allow_patterns=patterns)
    except Exception as e:  # noqa: BLE001
        raise InputError(f"Missing weights {repo}: run `hf download {repo}`") from e


def save_wav(path: Path, wav, sr: int) -> dict:
    """Float audio (T,) or (C, T) / (T, C) -> 24-bit WAV; overs are brought to -1 dBFS. Returns file meta."""
    import soundfile as sf

    w = np.asarray(wav, dtype=np.float32)
    if w.ndim == 2 and w.shape[0] <= 8 and w.shape[0] < w.shape[1]:
        w = w.T
    if w.ndim == 2 and w.shape[1] == 1:
        w = w[:, 0]
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if w.size == 0 or len(w) < int(0.02 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    peak = float(np.abs(w).max())
    meta = {"sample_rate": int(sr)}
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def video_input(job) -> dict | None:
    """The video of a v2a job: role `video` first, else the first input that looks like a video."""
    v = job.input(role="video") or job.input(type="video")
    if v:
        return v
    return next((i for i in job.inputs if Path(str(i.get("path", ""))).suffix.lower() in VIDEO_EXT), None)


def probe_video(path: str) -> dict:
    """{duration, fps, width, height, has_audio} through ffprobe (None values when unknown)."""
    import json

    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    r = subprocess.run([exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise InputError("This file is not a readable video (ffprobe failed).")
    d = json.loads(r.stdout or "{}")
    vs = [s for s in d.get("streams", []) if s.get("codec_type") == "video"]
    if not vs:
        raise InputError("The file has no video stream.")
    v = vs[0]
    dur = v.get("duration") or d.get("format", {}).get("duration")
    num, _, den = str(v.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else None
    return {"duration": float(dur) if dur not in (None, "N/A") else None, "fps": fps,
            "width": v.get("width"), "height": v.get("height"),
            "has_audio": any(s.get("codec_type") == "audio" for s in d.get("streams", []))}


def mux_video(video: str, audio: Path, out: Path, seconds: float) -> bool:
    """video (original soundtrack dropped) + generated audio -> mp4 (video stream copied when it is h264, else re-encoded)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    base = [exe, "-y", "-loglevel", "error", "-i", video, "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
            "-t", f"{seconds:.3f}", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart"]
    for vcodec in (["-c:v", "copy"], ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]):
        r = subprocess.run(base + vcodec + [str(out)], capture_output=True, text=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return True
    return False


def read_frames_av(path: str, fps_list: list[float], end_sec: float) -> list[np.ndarray]:
    """Decode a video once and resample it to several frame rates (nearest earlier frame). -> [ (T,H,W,3) uint8 ... ]"""
    import av

    out = [[] for _ in fps_list]
    nxt = [0.0 for _ in fps_list]
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for pkt in c.demux(st):
            for fr in pkt.decode():
                t = fr.time
                if t is None or t < 0:
                    continue
                if t > end_sec:
                    break
                arr = None
                for k, fps in enumerate(fps_list):
                    while t >= nxt[k]:
                        if arr is None:
                            arr = fr.to_ndarray(format="rgb24")
                        out[k].append(arr)
                        nxt[k] += 1.0 / fps
    return [np.stack(o) if o else np.zeros((0, 1, 1, 3), np.uint8) for o in out]


def thinksound_frames(path: str, dur: int):
    """ThinkSound inputs: 8 fps frames padded to a square and run through the MetaCLIP processor (224), and 25 fps
    frames resized / centre-cropped to 224 and normalised for Synchformer. Both padded to the expected length."""
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import v2
    from transformers import AutoProcessor

    clip_a, sync_a = read_frames_av(path, [8.0, 25.0], dur + 0.5)
    if len(clip_a) == 0 or len(sync_a) == 0:
        raise InputError("Could not read frames from the video.")

    def fit(x: torch.Tensor, n: int) -> torch.Tensor:
        x = x[:n]
        if x.shape[0] < n:
            x = torch.cat([x, x[-1:].repeat(n - x.shape[0], 1, 1, 1)], 0)
        return x

    clip = fit(torch.from_numpy(clip_a).permute(0, 3, 1, 2), 8 * dur)
    sync = fit(torch.from_numpy(sync_a).permute(0, 3, 1, 2), 25 * dur)
    _, _, h, w = clip.shape
    m = max(h, w)
    clip = F.pad(clip, ((m - w) // 2, m - w - (m - w) // 2, (m - h) // 2, m - h - (m - h) // 2), value=0)
    proc = AutoProcessor.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
    clip = proc(images=clip, return_tensors="pt")["pixel_values"]
    tf = v2.Compose([v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC), v2.CenterCrop(224), v2.ToImage(),
                     v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    return clip, tf(sync)


def free_cuda():
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ backend
class FoleyBackend(Backend):
    name = "foley"
    max_loaded = 1

    NEEDS = {"mmaudio": ["mmaudio", "open_clip", "av", "torchdiffeq"],
             "hunyuan": ["hunyuanvideo_foley", "transformers", "av"],
             "thinksound": ["ThinkSound", "transformers", "av"]}

    @staticmethod
    def family(mid: str) -> str:
        for k, f in (("mmaudio", "mmaudio"), ("hunyuan", "hunyuan"), ("thinksound", "thinksound")):
            if mid.startswith(k):
                return f
        raise InputError(f"The foley backend does not know model '{mid}'.")

    def probe(self, model):
        fam = self.family(model["id"])
        missing = [m for m in self.NEEDS[fam] + ["torch", "soundfile"] if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    # ---------------------------------------------------------------- load
    def load(self, model, ctx):
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        return getattr(self, "_load_" + self.family(model["id"]))(model, ctx)

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        del h
        free_cuda()

    def run(self, job, ctx):
        import torch

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        res = getattr(self, "_run_" + self.family(job.model_id))(job, ctx, self.handle(job.model_id)) or {}
        peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        ctx.log(f"run {time.time() - t0:.1f} s, peak GPU allocated {peak} GB")
        res.setdefault("meta", {})["peak_gpu_gb"] = peak
        return res

    # ================================================================ MMAudio
    def _load_mmaudio(self, model, ctx):
        import torch

        sys.path.insert(0, str(VENDOR / "MMAudio"))
        from mmaudio.model.networks import get_my_mmaudio
        from mmaudio.model.utils.features_utils import FeaturesUtils

        repo = model["weights"][0]["hf"]
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dtype = torch.bfloat16
        net = get_my_mmaudio("large_44k_v2").to("cuda", dtype).eval()
        net.load_weights(torch.load(hf_file(repo, "weights/mmaudio_large_44k_v2.pth"), map_location="cuda", weights_only=True))
        fu = FeaturesUtils(tod_vae_ckpt=hf_file(repo, "ext_weights/v1-44.pth"),
                           synchformer_ckpt=hf_file(repo, "ext_weights/synchformer_state_dict.pth"),
                           enable_conditions=True, mode="44k", bigvgan_vocoder_ckpt=None, need_vae_encoder=False)
        fu = fu.to("cuda", dtype).eval()
        return {"net": net, "fu": fu, "dtype": dtype}

    def _run_mmaudio(self, job, ctx, h):
        import torch
        from mmaudio.eval_utils import generate, load_video
        from mmaudio.model.flow_matching import FlowMatching
        from mmaudio.model.sequence_config import CONFIG_44K

        p = job.params
        net, fu = h["net"], h["fu"]
        prompt = (p.get("prompt") or "").strip()
        negative = (p.get("negative") or "").strip()
        want = float(p.get("seconds") or 8)
        vid = video_input(job) if job.task == "v2a" else None
        if job.task == "v2a" and not vid:
            raise InputError("Add a video first.")
        if job.task == "sfx" and not prompt:
            raise InputError("Describe the sound first.")
        clip_frames = sync_frames = None
        duration = want
        if vid:
            ctx.progress(0, 0, "video", "Reading the video")
            try:
                info = load_video(Path(vid["path"]), want, load_all_frames=False)
            except Exception as e:  # noqa: BLE001
                raise InputError(f"Could not read the video: {e}") from e
            duration = info.duration_sec
            if duration < 1.0:
                raise InputError(f"The video is too short ({duration:.1f} s); MMAudio needs at least 1 s.")
            clip_frames = None if p.get("mask_video") else info.clip_frames.unsqueeze(0)
            sync_frames = info.sync_frames.unsqueeze(0)
            if duration + 0.05 < want:
                ctx.log(f"Video is {duration:.1f} s long: audio will be {duration:.1f} s, not {want:.1f} s.")
        cfg = CONFIG_44K
        cfg.duration = duration
        net.update_seq_lengths(cfg.latent_seq_len, cfg.clip_seq_len, cfg.sync_seq_len)
        steps = int(p.get("steps") or 25)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=steps)
        n = int(p.get("batch") or 1)
        s0 = seed0(p)
        sr = cfg.sampling_rate
        base = prompt[:40] if prompt else (Path(vid["name"]).stem if vid and vid.get("name") else "video")
        for i in range(n):
            ctx.check()
            ctx.progress(i, n, "sampling", f"take {i + 1}/{n} ({steps} steps)")
            rng = torch.Generator(device="cuda")
            rng.manual_seed(s0 + i)
            with torch.inference_mode():
                audios = generate(clip_frames, sync_frames, [prompt], negative_text=[negative] if negative else None,
                                  feature_utils=fu, net=net, fm=fm, rng=rng, cfg_strength=float(p.get("cfg_scale") or 4.5))
            wav = audios.float().cpu()[0].numpy()[:, : int(duration * sr)]
            self._emit(job, ctx, wav, sr, i, n, s0 + i, base, vid, duration, extra={"steps": steps})
        return {"meta": {"model_dtype": "bf16", "video_seconds": duration if vid else None}}

    # ================================================================ HunyuanVideo-Foley
    def _load_hunyuan(self, model, ctx):
        import torch

        sys.path.insert(0, str(VENDOR / "HunyuanVideo-Foley"))
        from hunyuanvideo_foley.utils.model_utils import load_model

        repo = model["weights"][0]["hf"]
        d = hf_dir(repo, ["hunyuanvideo_foley.pth", "vae_128d_48k.pth", "synchformer_state_dict.pth"])
        cfg_path = str(VENDOR / "HunyuanVideo-Foley" / "configs" / "hunyuanvideo-foley-xxl.yaml")
        md, cfg = load_model(d, cfg_path, torch.device("cuda"), enable_offload=False, model_size="xxl")
        return {"md": md, "cfg": cfg}

    def _run_hunyuan(self, job, ctx, h):
        import torch
        from hunyuanvideo_foley.constants import MAX_VIDEO_DURATION_SECONDS
        from hunyuanvideo_foley.utils.feature_utils import feature_process
        from hunyuanvideo_foley.utils.model_utils import denoise_process

        p = job.params
        vid = video_input(job)
        if not vid:
            raise InputError("Add a video first (HunyuanVideo-Foley needs a video).")
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Describe the sound you want (HunyuanVideo-Foley needs a text prompt).")
        info = probe_video(vid["path"])
        want = min(float(p.get("seconds") or 8), float(MAX_VIDEO_DURATION_SECONDS))
        dur = min(want, info["duration"]) if info.get("duration") else want
        if dur < 1.0:
            raise InputError(f"The video is too short ({dur:.1f} s); at least 1 s is needed.")
        job.out_dir.mkdir(parents=True, exist_ok=True)
        clip = job.out_dir / "_input_clip.mp4"                        # trimmed copy: the model uses the whole file
        r = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-i", vid["path"], "-t", f"{dur:.3f}",
                            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(clip)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise InputError(f"Could not read the video: {r.stderr.strip()[-200:]}")
        ctx.progress(0, 0, "features", "Encoding video and text", interruptible=False)
        neg = (p.get("negative") or "").strip() or None
        visual, text, audio_len = feature_process(str(clip), prompt, h["md"], h["cfg"], neg_prompt=neg)
        n = int(p.get("batch") or 1)
        steps = int(p.get("steps") or 50)
        s0 = seed0(p)
        base = prompt[:40]
        for i in range(n):
            ctx.check()
            ctx.progress(i, n, "sampling", f"take {i + 1}/{n} ({steps} steps)", interruptible=False)
            torch.manual_seed(s0 + i)
            torch.cuda.manual_seed_all(s0 + i)
            audio, sr = denoise_process(visual, text, audio_len, h["md"], h["cfg"],
                                        guidance_scale=float(p.get("cfg_scale") or 4.5), num_inference_steps=steps)
            wav = audio[0].numpy() if audio.dim() == 3 else audio.numpy()
            self._emit(job, ctx, wav, int(sr), i, n, s0 + i, base, vid, audio_len, extra={"steps": steps})
        try:
            clip.unlink()
        except OSError:
            pass
        return {"meta": {"video_seconds": audio_len}}

    # ================================================================ ThinkSound
    def _load_thinksound(self, model, ctx):
        import json

        import torch

        sys.path.insert(0, str(VENDOR / "ThinkSound"))
        import torchaudio  # noqa: F401  (ThinkSound imports it)
        from data_utils.v2a_utils.feature_utils_224 import FeaturesUtils
        from ThinkSound.models import create_model_from_config
        from ThinkSound.models.utils import load_ckpt_state_dict

        repo = model["weights"][0]["hf"]
        ck = hf_file(repo, "thinksound_light.ckpt")
        vae = hf_file(repo, "vae.ckpt")
        sync = hf_file(repo, "synchformer_state_dict.pth")
        # feature extractors (MetaCLIP-H, T5-v1.1-XL encoder, Synchformer): fp16 on the GPU
        fu = FeaturesUtils(vae_ckpt=None, vae_config=None, synchformer_ckpt=sync, enable_conditions=True)
        fu.t5_model = fu.t5_model.half().cuda().eval()
        fu.clip_model = fu.clip_model.half().cuda().eval()

        def _text_feats(self, input_ids=None, attention_mask=None, position_ids=None, **_):
            # vendor patch passes `return_dict` to CLIPTextTransformer, which transformers 4.57 no longer accepts
            out = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            return self.text_projection(out[1]), out[0]
        fu.clip_model.get_text_features = _text_feats.__get__(fu.clip_model)
        fu.synchformer = fu.synchformer.half().cuda().eval()
        cfg_json = VENDOR / "ThinkSound" / "ThinkSound" / "configs" / "model_configs" / "thinksound.json"
        cfg = json.loads(cfg_json.read_text())
        return {"fu": fu, "cfg": cfg, "ckpt": ck, "vae": vae, "create": create_model_from_config,
                "load_vae": load_ckpt_state_dict, "dits": {}}     # DiT built per duration (seq lengths are baked in)

    def _thinksound_dit(self, h, duration: int):
        import copy

        import torch

        if duration in h["dits"]:
            return h["dits"][duration]
        h["dits"].clear()
        free_cuda()
        cfg = copy.deepcopy(h["cfg"])
        cfg["sample_size"] = duration * cfg["sample_rate"]
        d = cfg["model"]["diffusion"]["config"]
        d["sync_seq_len"] = 24 * duration
        d["clip_seq_len"] = 8 * duration
        d["latent_seq_len"] = round(44100 / 64 / 32 * duration)
        m = h["create"](cfg)
        m.load_state_dict(torch.load(h["ckpt"], map_location="cpu", weights_only=False))
        m.pretransform.load_state_dict(h["load_vae"](h["vae"], prefix="autoencoder."))
        m = m.to("cuda").eval().requires_grad_(False)
        h["dits"][duration] = (m, cfg)
        return h["dits"][duration]

    def _run_thinksound(self, job, ctx, h):
        import torch
        from ThinkSound.inference.sampling import sample_discrete_euler

        p = job.params
        vid = video_input(job)
        if not vid:
            raise InputError("Add a video first (ThinkSound needs a video).")
        caption = (p.get("prompt") or "").strip()
        if not caption:
            raise InputError("Write a short caption of the sound (for example: 'a door creaks open').")
        cot = (p.get("cot") or "").strip() or caption
        info = probe_video(vid["path"])
        want = int(p.get("seconds") or 9)
        dur = int(min(want, math.floor(info["duration"]))) if info.get("duration") else want
        if dur < 2:
            raise InputError("ThinkSound needs a video of at least 2 s.")
        fu = h["fu"]
        ctx.progress(0, 0, "features", "Encoding video and text", interruptible=False)
        clip_v, sync_v = thinksound_frames(vid["path"], dur)
        with torch.inference_mode():
            cf = fu.encode_video_with_clip(clip_v.unsqueeze(0).cuda().half())
            sf_ = fu.encode_video_with_sync(sync_v.unsqueeze(0).cuda().half())
            glob, txt = fu.encode_text([caption])
            t5 = fu.encode_t5_text([cot])
        meta = {"metaclip_features": cf[0].float(), "sync_features": sf_[0].float(), "metaclip_global_text_features": glob[0].float(),
                "metaclip_text_features": txt[0].float(), "t5_features": t5[0].float(), "video_exist": torch.tensor(True),
                "id": "demo", "caption": caption, "caption_cot": cot}
        ctx.progress(0, 0, "loading", f"Preparing the {dur} s model", interruptible=False)
        net, cfg = self._thinksound_dit(h, dur)
        n = int(p.get("batch") or 1)
        steps = int(p.get("steps") or 24)
        s0 = seed0(p)
        base = caption[:40]
        length = round(44100 / 64 / 32 * dur)
        for i in range(n):
            ctx.check()
            ctx.progress(i, n, "sampling", f"take {i + 1}/{n} ({steps} steps)", interruptible=False)
            torch.manual_seed(s0 + i)
            torch.cuda.manual_seed_all(s0 + i)
            md = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in meta.items()}
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                conditioning = net.conditioner([md], "cuda")
                conditioning["metaclip_features"][~md["video_exist"].reshape(1)] = net.model.model.empty_clip_feat
                conditioning["sync_features"][~md["video_exist"].reshape(1)] = net.model.model.empty_sync_feat
                cond_inputs = net.get_conditioning_inputs(conditioning)
                noise = torch.randn([1, net.io_channels, length], device="cuda")
                fakes = sample_discrete_euler(net.model, noise, steps, **cond_inputs, cfg_scale=float(p.get("cfg_scale") or 5.0), batch_cfg=True)
                fakes = net.pretransform.decode(fakes)
            wav = fakes.float().cpu()[0]
            wav = (wav / wav.abs().max().clamp(min=1e-6)).clamp(-1, 1).numpy()
            wav = wav[:, : int(dur * 44100)]
            self._emit(job, ctx, wav, 44100, i, n, s0 + i, base, vid, dur, extra={"steps": steps, "cot": cot[:200]})
        return {"meta": {"video_seconds": dur}}

    # ---------------------------------------------------------------- shared output
    def _emit(self, job, ctx, wav, sr, i, n, seed, base, vid, duration, extra=None):
        f = job.out_dir / f"take_{i + 1}.wav"
        job.out_dir.mkdir(parents=True, exist_ok=True)
        meta = save_wav(f, wav, sr)
        attach = []
        if vid and job.params.get("mux_video", True):
            mp4 = job.out_dir / f"take_{i + 1}_video.mp4"
            if mux_video(vid["path"], f, mp4, meta["duration"]):
                attach.append(mp4.name)
            else:
                ctx.log("ffmpeg could not mux the video; the audio is still saved.")
        name = f"{base}" + (f" #{i + 1}" if n > 1 else "")
        ctx.output(f, "audio", seed=seed, index=i, name=name, attach=attach, **{**meta, **(extra or {})})


if __name__ == "__main__":
    serve(FoleyBackend())
