"""clonea_zonos backend (envs/clonea_zonos): Zyphra Zonos-v0.1 (transformer). Apache-2.0, 44.1 kHz.
tasks: tts (random speaker) and voice_clone (speaker embedding from a 5-30 s clip; no transcript needed).
Emotion control = 8-way emotion vector + pitch variation + speaking rate + fmax. The hybrid variant needs mamba-ssm
CUDA kernels (no nvcc here), so only the transformer checkpoint is wired.
"""
from __future__ import annotations

import importlib.util
import os
import re

import numpy as np

from studio_worker import Backend, InputError, serve
from clonea_common import (patch_torchaudio_load, ref_input, save_wav, seed0, seed_all, take_name, text_of, tidy_voice)

EMOTIONS = ["happiness", "sadness", "disgust", "fear", "surprise", "anger", "other", "neutral"]
DEFAULT_EMO = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]
DAC_REPO = "descript/dac_44khz"


def setup_espeak():
    """phonemizer needs libespeak-ng; no system package here, so use the one bundled in `espeakng-loader`."""
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.data_path = espeakng_loader.get_data_path()
    os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())


def pieces_of(text: str, limit: int = 220) -> list[str]:
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


class ZonosBackend(Backend):
    name = "clonea_zonos"
    max_loaded = 1

    HYBRID_WHY = "the hybrid checkpoint needs mamba-ssm / causal-conv1d CUDA kernels, which cannot be built here (no nvcc)"

    def probe(self, model):
        if "hybrid" in model["id"] and importlib.util.find_spec("mamba_ssm") is None:
            return False, self.HYBRID_WHY
        missing = [m for m in ("torch", "zonos", "phonemizer", "espeakng_loader", "soundfile") if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download

        if "hybrid" in model["id"] and importlib.util.find_spec("mamba_ssm") is None:
            raise InputError("Zonos hybrid is blocked: " + self.HYBRID_WHY + ". Use Zonos v0.1 (transformer).")
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        patch_torchaudio_load()
        setup_espeak()
        from zonos.model import Zonos

        d = snapshot_download(model["weights"][0]["hf"], local_files_only=True)
        m = Zonos.from_local(os.path.join(d, "config.json"), os.path.join(d, "model.safetensors"), "cuda", backbone="torch")
        m.eval()
        return {"model": m, "spk": {}}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import torch
        import torchaudio
        from zonos.conditioning import make_cond_dict, supported_language_codes

        h = self.handle(job.model_id)
        m = h["model"]
        p = job.params
        text = text_of(job)
        lang = (p.get("language") or "en-us").lower()
        if lang not in supported_language_codes:
            raise InputError(f"Zonos does not know the language code '{lang}'.")
        speaker, extra = None, {}
        if job.task == "voice_clone":
            ref = ref_input(job)
            if not ref:
                raise InputError("Pick a reference clip (5-30 s of clean speech from the library).")
            key = (ref["path"], os.path.getmtime(ref["path"]))
            if key not in h["spk"]:
                ctx.progress(0, 0, "conditioning", "Analysing the reference voice", interruptible=False)
                wav, sr = torchaudio.load(ref["path"])
                with torch.no_grad():
                    h["spk"][key] = m.make_speaker_embedding(wav.mean(0, keepdim=True), sr)
            speaker = h["spk"][key]
            extra["ref_asset"] = ref.get("asset_id")
        emo = [max(0.0, min(1.0, float(p.get("emo_" + e, d)))) for e, d in zip(EMOTIONS, DEFAULT_EMO)]
        if not any(k.startswith("emo_") for k in p):
            emo = DEFAULT_EMO
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        pieces = pieces_of(text)
        sr = int(m.autoencoder.sampling_rate)
        total = n * len(pieces)
        sampling = dict(min_p=float(p.get("min_p", 0.1)), top_p=float(p.get("top_p", 0)), top_k=int(p.get("top_k", 0)),
                        linear=float(p.get("linear", 0)), conf=float(p.get("conf", 0)), quad=float(p.get("quad", 0)))
        for i in range(n):
            ctx.check()
            seed = s0 + i
            seed_all(seed)
            wavs = []
            for k, piece in enumerate(pieces):
                ctx.check()
                base = i * len(pieces) + k
                cond = make_cond_dict(text=piece, language=lang, speaker=speaker, emotion=emo,
                                      fmax=float(p.get("fmax", 22050)), pitch_std=float(p.get("pitch_std", 20)),
                                      speaking_rate=float(p.get("speaking_rate", 15)), device="cuda")
                with torch.no_grad():
                    conditioning = m.prepare_conditioning(cond)
                max_new = int(86 * min(30, max(4, len(piece) / 8 + 6)))

                def cb(frame, step, max_steps, base=base):
                    ctx.check()
                    if step % 20 == 0:
                        ctx.progress(base + step / max_steps, total, "speaking", f"take {i + 1}/{n}: part {k + 1}/{len(pieces)} ({step} tokens)")
                    return True
                with torch.no_grad():
                    codes = m.generate(conditioning, max_new_tokens=max_new, cfg_scale=float(p.get("cfg_scale", 2.0)),
                                       batch_size=1, sampling_params=sampling, progress_bar=False,
                                       disable_torch_compile=True, callback=cb)
                    w = m.autoencoder.decode(codes).float().cpu().numpy()
                wavs.append(w.reshape(-1))
                if k < len(pieces) - 1:
                    wavs.append(np.zeros(int(0.12 * sr), dtype=np.float32))
            wav = tidy_voice(np.concatenate(wavs), sr)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, sr)
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(text, i, n), final_prompt=text, language=lang,
                       emotion=dict(zip(EMOTIONS, [round(x, 3) for x in emo])), **extra, **fm)
            ctx.progress((i + 1) * len(pieces), total, "speaking", f"take {i + 1}/{n} done")
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(ZonosBackend())
