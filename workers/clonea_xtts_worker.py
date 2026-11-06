"""clonea_xtts backend (envs/clonea_xtts): Coqui XTTS-v2 through the maintained `coqui-tts` fork.

tasks: tts (one of the 58 built-in speakers) and voice_clone (3-30 s reference clip, no transcript needed).
17 languages, 24 kHz. License CPML = non-commercial. Notes in docs/research/audio_more/tts_clone_a.md.
"""
from __future__ import annotations

import importlib.util
import os
import re

os.environ.setdefault("COQUI_TOS_AGREED", "1")

import numpy as np

from studio_worker import Backend, InputError, serve
from clonea_common import (patch_torchaudio_load, read_mono, ref_input, save_wav, seed0, seed_all, take_name,
                           text_of, tidy_voice)

LANGS = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]
CHAR_LIMIT = {"en": 250, "de": 253, "fr": 273, "es": 239, "it": 213, "pt": 203, "pl": 224, "zh-cn": 82, "ar": 166,
              "cs": 186, "ru": 182, "nl": 251, "tr": 226, "ja": 71, "hu": 224, "ko": 95, "hi": 150}


def chunks(text: str, limit: int) -> list[str]:
    """Split on sentence ends, then on commas / spaces, so every piece is under the model's per-language limit."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?。！？…])\s*", text) if s.strip()]
    out: list[str] = []
    for s in parts:
        while len(s) > limit:
            cut = max(s.rfind(c, 0, limit) for c in (",", "，", ";", " ", "、"))
            cut = cut if cut > limit // 3 else limit
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    # merge short neighbours so the model gets full sentences where possible
    merged: list[str] = []
    for s in out:
        if merged and len(merged[-1]) + 1 + len(s) <= limit:
            merged[-1] += " " + s
        else:
            merged.append(s)
    return merged or [text]


class XttsBackend(Backend):
    name = "clonea_xtts"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("torch", "TTS", "soundfile") if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        patch_torchaudio_load()
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        d = snapshot_download(model["weights"][0]["hf"], local_files_only=True)
        cfg = XttsConfig()
        cfg.load_json(os.path.join(d, "config.json"))
        m = Xtts.init_from_config(cfg)
        m.load_checkpoint(cfg, checkpoint_dir=d, use_deepspeed=False, eval=True)
        m.to("cuda").eval()
        speakers = sorted(m.speaker_manager.speakers.keys()) if getattr(m, "speaker_manager", None) else []
        return {"model": m, "cfg": cfg, "speakers": speakers, "cache": {}}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import torch

        h = self.handle(job.model_id)
        m = h["model"]
        p = job.params
        text = text_of(job)
        lang = (p.get("language") or "en").lower()
        if lang == "zh":
            lang = "zh-cn"
        if lang not in LANGS:
            raise InputError(f"XTTS-v2 does not speak '{lang}'. Languages: {', '.join(LANGS)}.")
        extra = {}
        if job.task == "voice_clone":
            ref = ref_input(job)
            if not ref:
                raise InputError("Pick a reference clip (3-30 s of clean speech from the library).")
            key = (ref["path"], os.path.getmtime(ref["path"]))
            if key not in h["cache"]:
                ctx.progress(0, 0, "conditioning", "Analysing the reference voice", interruptible=False)
                with torch.inference_mode():
                    h["cache"][key] = m.get_conditioning_latents(audio_path=[ref["path"]], load_sr=22050)
            gpt, spk = h["cache"][key]
            extra["ref_asset"] = ref.get("asset_id")
            voice = None
        else:
            voice = p.get("speaker") or "Claribel Dervla"
            if voice not in h["speakers"]:
                raise InputError(f"Unknown built-in speaker '{voice}'.")
            sp = m.speaker_manager.speakers[voice]
            gpt, spk = sp["gpt_cond_latent"], sp["speaker_embedding"]
            extra["speaker"] = voice
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        pieces = chunks(text, CHAR_LIMIT.get(lang, 200))
        total = n * len(pieces)
        sr = 24000
        for i in range(n):
            ctx.check()
            seed = s0 + i
            seed_all(seed)
            wavs = []
            for k, piece in enumerate(pieces):
                ctx.check()
                ctx.progress(i * len(pieces) + k, total, "speaking", f"take {i + 1}/{n}: part {k + 1}/{len(pieces)}")
                with torch.inference_mode():
                    out = m.inference(piece, lang, gpt, spk, temperature=float(p.get("temperature", 0.75)),
                                      length_penalty=float(p.get("length_penalty", 1.0)),
                                      repetition_penalty=float(p.get("repetition_penalty", 10.0)),
                                      top_k=int(p.get("top_k", 50)), top_p=float(p.get("top_p", 0.85)),
                                      speed=float(p.get("speed", 1.0)), enable_text_splitting=False)
                w = np.asarray(out["wav"], dtype=np.float32).reshape(-1)
                wavs.append(w)
                if k < len(pieces) - 1:
                    wavs.append(np.zeros(int(0.15 * sr), dtype=np.float32))
            wav = tidy_voice(np.concatenate(wavs), sr)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, sr)
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(text, i, n), final_prompt=text, language=lang,
                       **extra, **fm)
            ctx.progress((i + 1) * len(pieces), total, "speaking", f"take {i + 1}/{n} done")
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(XttsBackend())
