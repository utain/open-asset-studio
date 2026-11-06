"""Music backend `music_heartmula` (envs/music_heartmula): HeartMuLa (Jan-Feb 2026, Apache-2.0; vendor/heartlib).

Songs with vocals from tags + lyrics, multilingual, 48 kHz stereo (12.5 Hz HeartCodec). Uses the recommended pair:
HeartMuLa-oss-3B-happy-new-year (LM, 3B) + HeartCodec-oss-20260123 (codec) + HeartMuLaGen (tokenizer / gen_config).
heartlib wants one directory `ckpt/{tokenizer.json,gen_config.json,HeartMuLa-oss-3B/,HeartCodec-oss/}`; it is built as a
symlink farm at vendor/heartmula_ckpt from the HF cache. torch.compile is turned off (no nvcc / triton tuning).
The generation loop of `HeartMuLaGenPipeline._forward` is re-implemented here only to add progress + cancel.
"""
from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

STUDIO = Path(__file__).resolve().parent.parent
CKPT = STUDIO / "vendor" / "heartmula_ckpt"
SR = 48000


def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def take_name(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def _link(dst: Path, src: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and os.readlink(dst) != str(src):
        dst.unlink()
    if not dst.exists() and not dst.is_symlink():
        dst.symlink_to(src)


def build_ckpt_dir() -> Path:
    from huggingface_hub import snapshot_download

    gen = Path(snapshot_download("HeartMuLa/HeartMuLaGen", local_files_only=True))
    mula = Path(snapshot_download("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", local_files_only=True))
    codec = Path(snapshot_download("HeartMuLa/HeartCodec-oss-20260123", local_files_only=True))
    _link(CKPT / "tokenizer.json", gen / "tokenizer.json")
    _link(CKPT / "gen_config.json", gen / "gen_config.json")
    for sub, snap in (("HeartMuLa-oss-3B", mula), ("HeartCodec-oss", codec)):
        for f in snap.iterdir():
            if f.name != "README.md" and not f.name.startswith("."):
                _link(CKPT / sub / f.name, f)
    return CKPT


class MusicHeartmula(Backend):
    name = "music_heartmula"
    max_loaded = 1

    def probe(self, model):
        import heartlib  # noqa: F401
        from heartlib import HeartMuLaGenPipeline  # noqa: F401

        return True, ""

    def load(self, model, ctx):
        import torch
        from heartlib import HeartMuLaGenPipeline

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        HeartMuLaGenPipeline._apply_compile = lambda self, m: None      # no torch.compile on this box
        pipe = HeartMuLaGenPipeline.from_pretrained(
            str(build_ckpt_dir()), device=torch.device("cuda"), dtype=torch.bfloat16, version="3B", lazy_load=False)
        torch.cuda.synchronize()
        return pipe

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import soundfile as sf
        import torch

        pipe = self.handle(job.model_id)
        p = job.params
        tags = (p.get("prompt") or "").strip()
        lyrics = (p.get("lyrics") or "").strip()
        if not tags:
            raise InputError("Give the style tags first (comma separated: genre, mood, instruments, voice).")
        if not lyrics:
            lyrics = "[instrumental]"
            ctx.log("No lyrics given: asking for an instrumental ([instrumental]); vocals may still appear.")
        n = int(p.get("batch", 1))
        s0 = seed0(p)
        max_ms = int(float(p.get("seconds", 60)) * 1000)
        temperature = float(p.get("temperature", 1.0))
        topk = int(p.get("top_k", 50))
        cfg = float(p.get("cfg_scale", 1.5))
        cfg_ids = {"cfg_scale": cfg}
        for i in range(n):
            ctx.check()
            seed = s0 + i
            random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            inputs = pipe.preprocess({"tags": tags, "lyrics": lyrics}, **cfg_ids)
            wav = self._forward(pipe, inputs, ctx, max_ms, temperature, topk, cfg, f"take {i + 1}/{n}: ")
            wav = wav.to(torch.float32).cpu().numpy()          # (C, T)
            if wav.ndim == 1:
                wav = wav[None]
            wav = np.nan_to_num(wav)
            peak = float(np.abs(wav).max())
            if peak < 1e-5:
                raise RuntimeError("the model produced silence")
            if peak > 0.89:
                wav = wav * (0.89 / peak)
            f = job.out_dir / f"take_{i + 1}.wav"
            sf.write(str(f), wav.T, SR, subtype="PCM_24")
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(tags, i, n), final_prompt=tags, sample_rate=SR,
                       duration=round(wav.shape[1] / SR, 3))
        return {"meta": {"lyrics_chars": len(lyrics)}}

    # heartlib's HeartMuLaGenPipeline._forward + postprocess, with progress and cancel
    @staticmethod
    def _forward(pipe, model_inputs, ctx, max_audio_length_ms, temperature, topk, cfg_scale, label):
        import torch

        dev = pipe.mula_device
        prompt_tokens = model_inputs["tokens"].to(dev)
        prompt_tokens_mask = model_inputs["tokens_mask"].to(dev)
        continuous_segment = model_inputs["muq_embed"].to(dev)
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"].to(dev)
        bs = 2 if cfg_scale != 1.0 else 1
        mula = pipe.mula
        mula.setup_caches(bs)
        with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=pipe.mula_dtype):
            cur = mula.generate_frame(tokens=prompt_tokens, tokens_mask=prompt_tokens_mask, input_pos=prompt_pos,
                                      temperature=temperature, topk=topk, cfg_scale=cfg_scale,
                                      continuous_segments=continuous_segment, starts=starts)
            frames = [cur[0:1]]
            par = pipe._parallel_number
            max_frames = max_audio_length_ms // 80
            for i in range(max_frames):
                ctx.check()
                if i % 10 == 0:
                    ctx.progress(i, max_frames, "sampling", f"{label}{i * 0.08:.0f}/{max_frames * 0.08:.0f} s of audio")
                tok = torch.ones((cur.shape[0], par), device=cur.device, dtype=torch.long) * pipe.config.empty_id
                tok[:, :-1] = cur
                tok = tok.unsqueeze(1)
                mask = torch.ones_like(tok, dtype=torch.bool)
                mask[..., -1] = False
                cur = mula.generate_frame(tokens=tok, tokens_mask=mask, input_pos=prompt_pos[..., -1:] + i + 1,
                                          temperature=temperature, topk=topk, cfg_scale=cfg_scale,
                                          continuous_segments=None, starts=None)
                if torch.any(cur[0:1, :] >= pipe.config.audio_eos_id):
                    break
                frames.append(cur[0:1])
            frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
            ctx.progress(1, 1, "decoding", "HeartCodec decode", interruptible=False)
            return pipe.codec.detokenize(frames.to(pipe.codec_device))


if __name__ == "__main__":
    serve(MusicHeartmula())
