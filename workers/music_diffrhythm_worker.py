"""Music backend `music_diffrhythm` (envs/music_diffrhythm): DiffRhythm 1.2 (ASLP-lab, vendor/DiffRhythm).

Latent-diffusion songs with vocals, 44.1 kHz stereo: `ASLP-lab/DiffRhythm-1_2` (95 s) and `DiffRhythm-1_2-full`
(96-285 s). Inputs: a style prompt (text, encoded by MuQ-MuLan) and lyrics. DiffRhythm wants LRC lyrics
(`[mm:ss.xx] line`); plain lyrics are timed automatically (lines spread evenly over the song, [section] tags dropped).
Empty lyrics = the model hums / plays without words (used for the `music` task).

The upstream code uses paths relative to its repo and downloads from the Hub; here everything is read from the local
HF cache (offline) and the process chdir()s into the repo. espeak-ng (system lib) is used by the g2p front end.
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
REPO = STUDIO / "vendor" / "DiffRhythm"
ESPEAK_LIB = "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1"
ESPEAK_DATA = "/usr/lib/aarch64-linux-gnu/espeak-ng-data"
if os.path.exists(ESPEAK_LIB):
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", ESPEAK_LIB)
    os.environ.setdefault("ESPEAK_DATA_PATH", ESPEAK_DATA)

CFG = {  # model id -> (hf repo, max_frames)
    "diffrhythm-1.2": ("ASLP-lab/DiffRhythm-1_2", 2048),
    "diffrhythm-1.2-full": ("ASLP-lab/DiffRhythm-1_2-full", 6144),
}
SR = 44100


def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def take_name(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


LRC_LINE = re.compile(r"^\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]")
TAG_LINE = re.compile(r"^\[[^\]\d][^\]]*\]\s*$")


def to_lrc(lyrics: str, seconds: float) -> str:
    """Plain lyrics -> LRC with evenly spread timestamps. Text that already has [mm:ss.xx] stamps is kept."""
    lines = [l.strip() for l in (lyrics or "").splitlines() if l.strip()]
    if not lines:
        return ""
    if sum(1 for l in lines if LRC_LINE.match(l)) >= max(1, len(lines) // 2):
        return "\n".join(l for l in lines if LRC_LINE.match(l))
    body = [l for l in lines if not TAG_LINE.match(l)]
    if not body:
        return ""
    t0, t1 = 4.0, max(8.0, seconds - 14.0)
    step = (t1 - t0) / max(len(body), 1)
    out = []
    for i, l in enumerate(body):
        t = t0 + i * step
        out.append(f"[{int(t // 60):02d}:{t % 60:05.2f}] {l}")
    return "\n".join(out)


def _map_legacy_names():
    """MuQ-MuLan asks for the legacy id `xlm-roberta-base`, which cannot be resolved offline."""
    import transformers

    for cls in (transformers.XLMRobertaModel, transformers.AutoTokenizer):
        if getattr(cls.from_pretrained, "_studio_patched", False):
            continue
        orig = cls.from_pretrained.__func__

        def _fp(klass, name, *a, _orig=orig, **k):
            if name == "xlm-roberta-base":
                name = "FacebookAI/xlm-roberta-base"
            return _orig(klass, name, *a, **k)

        _fp._studio_patched = True
        cls.from_pretrained = classmethod(_fp)


class MusicDiffrhythm(Backend):
    name = "music_diffrhythm"
    max_loaded = 1

    def _prep_path(self):
        os.chdir(REPO)
        for p in (str(REPO), str(REPO / "infer")):
            if p not in sys.path:
                sys.path.insert(0, p)

    def probe(self, model):
        self._prep_path()
        import infer_utils  # noqa: F401

        return True, ""

    def load(self, model, ctx):
        import json

        import torch
        from huggingface_hub import snapshot_download

        self._prep_path()
        from infer_utils import CNENTokenizer, load_checkpoint
        from model import CFM, DiT
        from muq import MuQMuLan

        repo, max_frames = CFG[model["id"]]
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        dit_path = Path(snapshot_download(repo, local_files_only=True)) / "cfm_model.pt"
        vae_path = Path(snapshot_download("ASLP-lab/DiffRhythm-vae", local_files_only=True)) / "vae_model.pt"
        cfg = json.load(open(REPO / "config" / "diffrhythm-1b.json"))
        cfm = CFM(transformer=DiT(**cfg["model"], max_frames=max_frames), num_channels=cfg["model"]["mel_dim"],
                  max_frames=max_frames)
        cfm = load_checkpoint(cfm.to("cuda"), str(dit_path), device="cuda", use_ema=False)
        tok = CNENTokenizer()
        _map_legacy_names()
        muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to("cuda").eval()
        vae = torch.jit.load(str(vae_path), map_location="cpu").to("cuda")
        return {"cfm": cfm, "tok": tok, "muq": muq, "vae": vae, "max_frames": max_frames}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import soundfile as sf
        import torch
        from einops import rearrange

        torch.cuda.reset_peak_memory_stats()
        self._prep_path()
        from infer_utils import decode_audio, get_lrc_token, get_negative_style_prompt, get_style_prompt

        h = self.handle(job.model_id)
        p = job.params
        style = (p.get("prompt") or "").strip()
        ref = job.input(role="ref") or job.input(type="audio")
        if not style and not ref:
            raise InputError("Describe the style (genre, mood, instruments, voice) or give a reference clip.")
        max_frames = h["max_frames"]
        secs = 95 if max_frames == 2048 else int(max(96, min(285, round(float(p.get("seconds", 120))))))
        instrumental = job.task == "music"
        n = int(p.get("batch", 1))
        s0 = seed0(p)
        steps = int(p.get("steps", 32))
        cfg = float(p.get("cfg_scale", 4.0))
        lrc = "" if instrumental else to_lrc(p.get("lyrics", ""), secs)
        if not instrumental and not lrc:
            ctx.log("No lyrics: the model will sing / hum without words.")
        device = "cuda"
        lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(max_frames, lrc, h["tok"], secs, device)
        style_prompt = get_style_prompt(h["muq"], ref["path"]) if (ref and not style) else get_style_prompt(h["muq"], prompt=style)
        neg = get_negative_style_prompt(device)
        prompt = torch.zeros(1, max_frames, 64).to(device)
        for i in range(n):
            ctx.check()
            random.seed(s0 + i)
            np.random.seed((s0 + i) % 2**32)
            torch.manual_seed(s0 + i)
            torch.cuda.manual_seed_all(s0 + i)
            ctx.progress(i, n, "sampling", f"take {i + 1}/{n}: {steps} diffusion steps", interruptible=False)
            with torch.inference_mode():
                latents, _ = h["cfm"].sample(
                    cond=prompt, text=lrc_prompt, duration=end_frame, style_prompt=style_prompt,
                    max_duration=end_frame, song_duration=song_duration, negative_style_prompt=neg, steps=steps,
                    cfg_strength=cfg, start_time=start_time, latent_pred_segments=[(0, max_frames)], batch_infer_num=1)
                latent = latents[0].to(torch.float32).transpose(1, 2)
                ctx.progress(i, n, "decoding", "VAE decode", interruptible=False)
                out = decode_audio(latent, h["vae"], chunked=True)
            out = rearrange(out, "b d n -> d (b n)").to(torch.float32).cpu()
            peak = float(out.abs().max())
            if not np.isfinite(peak) or peak < 1e-5:
                raise RuntimeError("the model produced silence")
            out = (out / peak * 10 ** (-1 / 20)).numpy().T
            f = job.out_dir / f"take_{i + 1}.wav"
            sf.write(str(f), out, SR, subtype="PCM_24")
            ctx.output(f, "audio", seed=s0 + i, index=i, name=take_name(style or "song", i, n),
                       final_prompt=style, sample_rate=SR, duration=round(out.shape[0] / SR, 3))
        return {"meta": {"lrc": lrc[:2000], "steps": steps, "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}}


if __name__ == "__main__":
    serve(MusicDiffrhythm())
