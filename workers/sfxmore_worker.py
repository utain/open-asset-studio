"""sfxmore backend (envs/sfxmore): more open-weight text-to-audio models for SFX / ambience / music.

  audioldm-l-full                       AudioLDM 1 (diffusers, deprecated pipeline)   16 kHz mono
  audioldm2 / -large / -music           AudioLDM 2 (diffusers)                         16 kHz mono
  tango2                                declare-lab/tango code (vendor/tango)          16 kHz mono, fixed 10 s window
  tangoflux                             declare-lab/TangoFlux (vendor/TangoFlux)       44.1 kHz stereo, up to 30 s
  stable-audio-open-1.0 / -small        gated on Hugging Face (401 without login): registered as blocked, no inference code
  make-an-audio-2                       ByteDance/Make-An-Audio-2 (vendor/Make-An-Audio-2 + BigVGAN vocoder), 16 kHz, <= 10 s

Every take is a 24-bit WAV; full-scale audio is brought to -1 dBFS. Recipes: docs/research/audio_more/sfx_more.md
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from studio_worker import Backend, Cancelled, InputError, serve

STUDIO = Path(__file__).resolve().parent.parent
VENDOR = STUDIO / "vendor"


# ------------------------------------------------------------------ helpers
def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def seed_all(s: int):
    import torch

    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def save_wav(path: Path, wav, sr: int) -> dict:
    """Write float audio (T,) or (T, C) as 24-bit WAV. Brings overs to -1 dBFS. Returns file meta."""
    import soundfile as sf

    w = np.asarray(wav, dtype=np.float32)
    if w.ndim == 2 and w.shape[0] <= 8 and w.shape[0] < w.shape[1]:
        w = w.T                                     # (C, T) -> (T, C)
    if w.ndim == 2 and w.shape[1] == 1:
        w = w[:, 0]
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if w.size == 0 or len(w) < int(0.02 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    peak = float(np.abs(w).max())
    meta = {"sample_rate": int(sr)}
    if peak < 1e-5:
        raise RuntimeError("the model returned silence")
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def take_name(label: str, i: int, n: int) -> str:
    import re

    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def hf_repo(model: dict, k: int = 0) -> str:
    return model["weights"][k]["hf"]


class _Tqdm:
    """tqdm stand-in for code that only calls .update(): reports progress and raises Cancelled."""

    def __init__(self, ctx, label):
        self.ctx, self.label, self.n, self.total = ctx, label, 0, 0

    def __call__(self, iterable=None, total=None, disable=False, **_kw):
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else 0)
        self.n = 0
        return self

    def update(self, k=1):
        self.ctx.check()
        self.n += k
        self.ctx.progress(self.n, self.total, "sampling", f"{self.label}step {self.n}/{self.total}")

    def __iter__(self):
        return iter(())

    def close(self):
        pass


# ------------------------------------------------------------------ backend
class SfxMoreBackend(Backend):
    name = "sfxmore"
    max_loaded = 1

    NEEDS = {"audioldm": ["diffusers", "transformers"], "audioldm2": ["diffusers", "transformers"],
             "tango2": ["diffusers", "transformers", "librosa", "einops"], "tangoflux": ["diffusers", "transformers", "datasets", "yaml"],
             "sao": ["diffusers", "transformers"], "maa2": ["omegaconf", "pytorch_lightning"]}

    @staticmethod
    def family(mid: str) -> str:
        if mid == "audioldm-l-full":
            return "audioldm"
        if mid.startswith("audioldm2"):
            return "audioldm2"
        if mid == "tango2":
            return "tango2"
        if mid == "tangoflux":
            return "tangoflux"
        if mid.startswith("stable-audio-open"):
            return "sao"
        if mid.startswith("make-an-audio"):
            return "maa2"
        raise InputError(f"The sfxmore backend does not know model '{mid}'.")

    def probe(self, model):
        import importlib.util

        fam = self.family(model["id"])
        missing = [m for m in self.NEEDS[fam] + ["torch", "soundfile"] if importlib.util.find_spec(m) is None]
        if fam == "tango2" and not (VENDOR / "tango" / "models.py").exists():
            missing.append("vendor/tango")
        if fam == "tangoflux" and not (VENDOR / "TangoFlux" / "tangoflux").exists():
            missing.append("vendor/TangoFlux")
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    # ---------------------------------------------------------------- load
    def load(self, model, ctx):
        import torch

        fam = self.family(model["id"])
        repo = hf_repo(model)
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        if fam == "audioldm":
            from diffusers import AudioLDMPipeline

            return AudioLDMPipeline.from_pretrained(repo, torch_dtype=torch.float16, local_files_only=True,
                                                    use_safetensors=False).to("cuda")
        if fam == "audioldm2":
            from diffusers import AudioLDM2Pipeline

            from transformers import GenerationMixin

            pipe = AudioLDM2Pipeline.from_pretrained(repo, torch_dtype=torch.float16, local_files_only=True,
                                                     use_safetensors=True).to("cuda")
            lm = pipe.language_model        # transformers >= 4.50: GPT2Model is no longer a GenerationMixin, but the
            if not hasattr(lm, "_update_model_kwargs_for_generation"):   # pipeline calls its helpers
                lm.__class__ = type("GPT2ModelGen", (lm.__class__, GenerationMixin), {})
            return pipe
        if fam == "tango2":
            return self._load_tango2(repo)
        if fam == "tangoflux":
            return self._load_tangoflux(repo)
        if fam == "sao":
            raise InputError("Stable Audio Open is gated on Hugging Face: accept the licence, log in "
                             "(hf auth login) and download it; no inference code for it is wired yet.")
        if fam == "maa2":
            return self._load_maa2(repo)
        raise InputError(f"{model['name']} cannot run: see docs/research/audio_more/sfx_more.md")

    @staticmethod
    def _load_maa2(repo):
        import types

        import torch
        from huggingface_hub import snapshot_download

        # old APIs the 2023 code imports
        six = types.ModuleType("torch._six")
        six.string_classes = (str,)
        sys.modules["torch._six"] = six
        import pytorch_lightning.utilities.rank_zero as rz

        dist = types.ModuleType("pytorch_lightning.utilities.distributed")
        dist.rank_zero_only = rz.rank_zero_only
        sys.modules["pytorch_lightning.utilities.distributed"] = dist
        root = VENDOR / "Make-An-Audio-2"
        sys.path[:0] = [str(root), str(VENDOR / "BigVGAN-awei6")]
        from bigvgan.models import VocoderBigVGAN
        from ldm.models.diffusion.ddim import DDIMSampler
        from ldm.util import instantiate_from_config
        from omegaconf import OmegaConf

        ckpt = Path(snapshot_download(repo_id=repo, local_files_only=True)) / "maa2.ckpt"
        voc = Path(snapshot_download(repo_id="SC4949/make-an-audio-models", allow_patterns=["bigvgan/*"], local_files_only=True)) / "bigvgan"
        cfg = OmegaConf.load(root / "configs/text2audio-ConcatDiT-ae1dnat_Skl20d2_struct2MLPanylen.yaml")
        # the text encoder constructor reads a CLAP checkpoint it never uses (its weights are inside maa2.ckpt): hand it an empty stub
        stub = STUDIO / "data" / "cache" / "maa2_clap_stub.pth"
        stub.parent.mkdir(parents=True, exist_ok=True)
        if not stub.exists():
            torch.save({"model": {}}, stub)
        cfg.model.params.cond_stage_config.params.weights_path = str(stub)
        model = instantiate_from_config(cfg.model)
        # Lightning checkpoint with pickled callbacks/optimizer state: cannot load with weights_only (ByteDance repo, MIT)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model = model.to("cuda").eval()
        model.cond_stage_model.to("cuda")
        model.cond_stage_model.device = model.device
        # vocoder weights are a plain state dict; torch.load must accept it with weights_only=True
        _orig = torch.load
        torch.load = lambda f, *a, **k: _orig(f, *a, **{**k, "weights_only": True})
        try:
            vocoder = VocoderBigVGAN(str(voc), device="cuda")
        finally:
            torch.load = _orig
        return {"sampler": DDIMSampler(model), "vocoder": vocoder, "missing": len(missing), "unexpected": len(unexpected)}

    @staticmethod
    def _load_tango2(repo):
        import json

        import torch
        from huggingface_hub import snapshot_download

        import diffusers.utils
        from diffusers.utils.torch_utils import randn_tensor

        diffusers.utils.randn_tensor = randn_tensor     # vendor/tango imports it from diffusers.utils (moved in 0.2x)
        sys.path.insert(0, str(VENDOR / "tango"))
        import audioldm.audio.stft as _stft
        import librosa.filters
        import librosa.util

        # librosa >= 0.10 made these keyword-only; vendor/tango calls them positionally
        _stft.pad_center = lambda data, size, **kw: librosa.util.pad_center(data, size=size, **kw)
        _stft.librosa_mel_fn = lambda sr, n_fft, n_mels=128, fmin=0.0, fmax=None, **kw: librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, **kw)
        from audioldm.audio.stft import TacotronSTFT
        from audioldm.variational_autoencoder import AutoencoderKL
        from diffusers import DDPMScheduler
        from models import AudioDiffusion

        path = snapshot_download(repo_id=repo, local_files_only=True)
        vae_cfg = json.load(open(f"{path}/vae_config.json"))
        stft_cfg = json.load(open(f"{path}/stft_config.json"))
        main_cfg = json.load(open(f"{path}/main_config.json"))
        main_cfg["unet_model_config_path"] = str(VENDOR / "tango" / main_cfg["unet_model_config_path"])
        dev = "cuda"
        vae = AutoencoderKL(**vae_cfg)
        stft = TacotronSTFT(**stft_cfg)
        net = AudioDiffusion(**main_cfg)
        vae.load_state_dict(torch.load(f"{path}/pytorch_model_vae.bin", map_location="cpu"))
        stft.load_state_dict(torch.load(f"{path}/pytorch_model_stft.bin", map_location="cpu"))
        net.load_state_dict(torch.load(f"{path}/pytorch_model_main.bin", map_location="cpu"))
        vae, stft, net = vae.to(dev).eval(), stft.to(dev).eval(), net.to(dev).eval()
        sched = DDPMScheduler.from_pretrained(main_cfg["scheduler_name"], subfolder="scheduler", local_files_only=True)
        return {"vae": vae, "stft": stft, "net": net, "sched": sched}

    @staticmethod
    def _load_tangoflux(repo):
        import json

        import torch
        from diffusers import AutoencoderOobleck
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        sys.path.insert(0, str(VENDOR / "TangoFlux"))
        from tangoflux.model import TangoFlux

        path = snapshot_download(repo_id=repo, local_files_only=True)
        vae = AutoencoderOobleck()
        vae.load_state_dict(load_file(f"{path}/vae.safetensors"))
        model = TangoFlux(json.load(open(f"{path}/config.json")))
        # missing key text_encoder.encoder.embed_tokens.weight is expected (tied to shared.weight)
        model.load_state_dict(load_file(f"{path}/tangoflux.safetensors"), strict=False)
        return {"vae": vae.to("cuda").eval(), "model": model.to("cuda").eval()}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    # ----------------------------------------------------------------- run
    def run(self, job, ctx):
        fam = self.family(job.model_id)
        prompt = (job.params.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Write a prompt first: describe the sound.")
        import torch

        torch.cuda.reset_peak_memory_stats()
        res = getattr(self, "_run_" + fam)(job, ctx, self.handle(job.model_id), prompt) or {}
        res.setdefault("meta", {})["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        return res

    def _emit(self, job, ctx, wav, sr, i, n, prompt, seed, **extra):
        f = job.out_dir / f"take_{i + 1}.wav"
        fm = save_wav(f, wav, sr)
        ctx.output(f, "audio", seed=seed, index=i, name=take_name(prompt, i, n), final_prompt=prompt, **fm, **extra)

    # --- AudioLDM 1 / 2 (diffusers pipelines; waveform 16 kHz)
    def _run_diffusers(self, job, ctx, pipe, prompt, **extra_call):
        import torch

        p = job.params
        seconds = float(p.get("seconds", 5))
        steps = int(p.get("steps", 100))
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        neg = (p.get("negative") or "").strip()
        sr = 16000
        for i in range(n):
            ctx.check()
            seed = s0 + i
            gen = torch.Generator("cuda").manual_seed(seed)

            def cb(step, _t, _lat, i=i):
                ctx.check()
                ctx.progress(i * steps + step + 1, n * steps, "sampling", f"take {i + 1}/{n}: step {step + 1}/{steps}")
            kw = dict(prompt=prompt, audio_length_in_s=seconds + 0.2, num_inference_steps=steps,
                      guidance_scale=float(p.get("guidance", 2.5)), generator=gen, callback=cb, callback_steps=1, **extra_call)
            if neg:
                kw["negative_prompt"] = neg
            with torch.inference_mode():
                audio = pipe(**kw).audios[0]
            audio = np.asarray(audio[: int(seconds * sr)], dtype=np.float32)
            pk = float(np.abs(audio).max())
            if pk < 1e-5:
                raise RuntimeError("the model returned silence")
            audio = audio / pk * 0.5          # the AudioLDM reference code normalises every take to 0.5 peak
            self._emit(job, ctx, audio, sr, i, n, prompt, seed, normalized_from_db=round(20 * math.log10(pk), 1))
        return {"meta": {"seed": s0}}

    def _run_audioldm(self, job, ctx, pipe, prompt):
        return self._run_diffusers(job, ctx, pipe, prompt)

    def _run_audioldm2(self, job, ctx, pipe, prompt):
        return self._run_diffusers(job, ctx, pipe, prompt)

    # --- Tango 2 (fixed 10.24 s window, trimmed to `seconds`)
    def _run_tango2(self, job, ctx, h, prompt):
        import torch
        import models as tango_models  # vendor/tango/models.py

        p = job.params
        seconds = min(float(p.get("seconds", 10)), 10.0)
        steps = int(p.get("steps", 100))
        guidance = float(p.get("guidance", 3.0))
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        sr = 16000
        for i in range(n):
            ctx.check()
            seed = s0 + i
            seed_all(seed)
            tango_models.tqdm = _Tqdm(ctx, f"take {i + 1}/{n}: ")
            with torch.inference_mode():
                lat = h["net"].inference([prompt], h["sched"], steps, guidance, 1, disable_progress=False)
                mel = h["vae"].decode_first_stage(lat)
                wave = h["vae"].decode_to_waveform(mel)[0]
            wav = wave.astype(np.float32) / 32768.0
            self._emit(job, ctx, wav[: int(seconds * sr)], sr, i, n, prompt, seed)
        return {"meta": {"seed": s0}}

    # --- TangoFlux (flow matching, 44.1 kHz stereo, up to 30 s)
    def _run_tangoflux(self, job, ctx, h, prompt):
        import torch

        p = job.params
        seconds = float(p.get("seconds", 10))
        steps = int(p.get("steps", 25))
        guidance = float(p.get("guidance", 4.5))
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        m, vae = h["model"], h["vae"]
        sr = int(vae.config.sampling_rate)
        for i in range(n):
            ctx.check()
            seed = s0 + i
            for attempt in range(4):
                done = [0]

                def step(i=i):
                    done[0] += 1
                    ctx.check()
                    ctx.progress(i * steps + done[0], n * steps, "sampling", f"take {i + 1}/{n}: step {done[0]}/{steps}")
                with torch.inference_mode():
                    lat = m.inference_flow(prompt, duration=seconds, num_inference_steps=steps, guidance_scale=guidance,
                                           seed=seed, disable_progress=True, callback_on_step_end=step)
                    wave = vae.decode(lat.transpose(2, 1)).sample.float().cpu()[0]
                wave = wave[:, : int(seconds * sr)].numpy()
                if float(np.abs(wave).max()) > 3e-3:
                    break
                ctx.log(f"seed {seed} gave a silent clip (a known TangoFlux failure mode): retrying with another seed")
                seed += 7919
            self._emit(job, ctx, wave, sr, i, n, prompt, seed)
        return {"meta": {"seed": s0}}

    def _run_sao(self, job, ctx, h, prompt):
        raise InputError("Stable Audio Open is gated on Hugging Face and not wired.")

    # --- Make-An-Audio 2 (structured caption "<event& start>@<event& end>", DDIM, <= 10 s)
    def _run_maa2(self, job, ctx, h, prompt):
        import torch

        p = job.params
        seconds = min(float(p.get("seconds", 10)), 10.0)
        steps = int(p.get("steps", 100))
        scale = float(p.get("guidance", 4.0))
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        struct = (p.get("struct_prompt") or "").strip() or f"<{prompt}& all>"
        sampler, vocoder = h["sampler"], h["vocoder"]
        model = sampler.model
        sr = 16000
        width = int(seconds * 31.5)
        width += width % 2
        for i in range(n):
            ctx.check()
            seed = s0 + i
            seed_all(seed)
            with torch.inference_mode():
                x_T = torch.randn(1, model.first_stage_model.embed_dim, width, device="cuda")
                uc = model.get_learned_conditioning({"ori_caption": [""], "struct_caption": [""]}) if scale != 1.0 else None
                c = model.get_learned_conditioning({"ori_caption": [prompt], "struct_caption": [struct]})

                def cb(step, i=i):
                    ctx.check()
                    ctx.progress(i * steps + step + 1, n * steps, "sampling", f"take {i + 1}/{n}: step {step + 1}/{steps}")
                z, _ = sampler.sample(S=steps, conditioning=c, batch_size=1, shape=[model.first_stage_model.embed_dim, width],
                                      verbose=False, unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=x_T, callback=cb)
                spec = model.decode_first_stage(z)
                wav = vocoder.vocode(spec[0].unsqueeze(0))
            wav = np.asarray(wav, dtype=np.float32)[: int(seconds * sr)]
            pk = float(np.abs(wav).max())
            extra = {}
            if 1e-5 < pk < 0.1:               # quiet beds (ambience) come out at -40 dBFS or lower: bring them up to 0.3 peak
                wav = wav / pk * 0.3
                extra["normalized_from_db"] = round(20 * math.log10(pk), 1)
            self._emit(job, ctx, wav, sr, i, n, prompt, seed, struct_caption=struct, **extra)
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(SfxMoreBackend())
