"""clonea_chatterbox backend (envs/clonea_chatterbox): Resemble AI Chatterbox (English, 0.5B) and
Chatterbox Multilingual (23 languages). MIT. Zero-shot cloning from a 5-15 s clip (no transcript), plus the built-in
default voice for plain TTS. `exaggeration` (emotion intensity) and `cfg_weight` (pace / text adherence) are the
signature controls. Output is 24 kHz and carries the Resemble "Perth" inaudible watermark (part of the upstream code).
"""
from __future__ import annotations

import importlib.util
import os
import re
import tempfile

import numpy as np

from studio_worker import Backend, InputError, serve
from clonea_common import (read_mono, ref_input, ref_wav_to_file, save_wav, seed0, seed_all, take_name, text_of,
                           tidy_voice)

EN_FILES = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
MTL_FILES = ["ve.pt", "t3_23lang.safetensors", "s3gen.pt", "mtl_tokenizer.json", "conds.pt", "Cangjie5_TC.json",
             "grapheme_mtl_merged_expanded_v1.json"]


def pieces_of(text: str, limit: int = 280) -> list[str]:
    """Chatterbox T3 stops at 1000 speech tokens (~40 s): feed it sentence-sized pieces."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?。！？…])\s+", text) if s.strip()]
    out: list[str] = []
    for s in parts:
        while len(s) > limit:
            cut = max(s.rfind(c, 0, limit) for c in (",", "，", ";", " ", "、"))
            cut = cut if cut > limit // 3 else limit
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    merged: list[str] = []
    for s in out:
        if merged and len(merged[-1]) + 1 + len(s) <= limit:
            merged[-1] += " " + s
        else:
            merged.append(s)
    return merged or [text]


class ChatterboxBackend(Backend):
    name = "clonea_chatterbox"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("torch", "chatterbox", "soundfile", "librosa") if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        multilingual = model["id"].endswith("multilingual")
        d = snapshot_download(model["weights"][0]["hf"], allow_patterns=MTL_FILES if multilingual else EN_FILES,
                              local_files_only=True)
        if multilingual:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            m = ChatterboxMultilingualTTS.from_local(d, "cuda")
        else:
            from chatterbox.tts import ChatterboxTTS

            m = ChatterboxTTS.from_local(d, "cuda")
        # keep the shipped voice: a clone job replaces m.conds, and a later plain tts job must not inherit it
        return {"model": m, "multilingual": multilingual, "ref": None,
                "default_conds": getattr(m, "conds", None)}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        h = self.handle(job.model_id)
        m = h["model"]
        p = job.params
        text = text_of(job)
        lang = (p.get("language") or "en").lower()
        if h["multilingual"]:
            from chatterbox.mtl_tts import SUPPORTED_LANGUAGES

            if lang not in SUPPORTED_LANGUAGES:
                raise InputError(f"Chatterbox Multilingual does not speak '{lang}'.")
        extra = {}
        ref_path = None
        if job.task == "voice_clone":
            ref = ref_input(job)
            if not ref:
                raise InputError("Pick a reference clip (5-15 s of clean speech from the library).")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            ref_path = ref_wav_to_file(ref["path"], tmp.name, seconds=30, sr=24000)
            extra["ref_asset"] = ref.get("asset_id")
        exag = float(p.get("exaggeration", 0.5))
        cfg = float(p.get("cfg_weight", 0.5))
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        pieces = pieces_of(text, 250 if h["multilingual"] else 300)
        total = n * len(pieces)
        sr = int(m.sr)
        if ref_path:
            m.prepare_conditionals(ref_path, exaggeration=exag)       # once per job: later pieces reuse the voice
        else:
            if h.get("default_conds") is not None:
                m.conds = h["default_conds"]
        if not ref_path and m.conds is None:
            raise InputError("This install has no built-in voice (conds.pt missing); use voice clone with a reference clip.")
        for i in range(n):
            ctx.check()
            seed = s0 + i
            seed_all(seed)
            wavs = []
            for k, piece in enumerate(pieces):
                ctx.check()
                ctx.progress(i * len(pieces) + k, total, "speaking", f"take {i + 1}/{n}: part {k + 1}/{len(pieces)}")
                kw = dict(exaggeration=exag, cfg_weight=cfg, temperature=float(p.get("temperature", 0.8)),
                          repetition_penalty=float(p.get("repetition_penalty", 2.0 if h["multilingual"] else 1.2)),
                          min_p=float(p.get("min_p", 0.05)), top_p=float(p.get("top_p", 1.0)))
                if h["multilingual"]:
                    wav = m.generate(piece, lang, **kw)
                else:
                    wav = m.generate(piece, **kw)
                wavs.append(wav.detach().float().cpu().numpy().reshape(-1))
                if k < len(pieces) - 1:
                    wavs.append(np.zeros(int(0.12 * sr), dtype=np.float32))
            wav = tidy_voice(np.concatenate(wavs), sr)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, sr)
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(text, i, n), final_prompt=text,
                       language=lang if h["multilingual"] else "en", exaggeration=exag, cfg_weight=cfg, **extra, **fm)
            ctx.progress((i + 1) * len(pieces), total, "speaking", f"take {i + 1}/{n} done")
        if ref_path:
            try:
                os.unlink(ref_path)
            except OSError:
                pass
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(ChatterboxBackend())
