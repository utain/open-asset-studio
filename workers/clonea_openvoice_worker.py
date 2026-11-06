"""clonea_openvoice backend (envs/clonea_openvoice): OpenVoice V2 = MeloTTS base voice + tone-colour converter.
MIT. tasks: tts (plain MeloTTS English accent, no conversion) and voice_clone (MeloTTS line -> tone-colour converter
-> the timbre of a 3-30 s reference clip; no transcript needed). English only here (the other MeloTTS languages are
separate 200 MB checkpoints plus their own BERTs and were not fetched). The converter's WavMark watermark is off.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np

from studio_worker import Backend, InputError, serve
from clonea_common import (read_mono, ref_input, save_wav, seed0, seed_all, take_name, text_of, tidy_voice)

ACCENTS = {"EN-US": "en-us", "EN-BR": "en-br", "EN_INDIA": "en-india", "EN-AU": "en-au", "EN-Default": "en-default"}


class OpenVoiceBackend(Backend):
    name = "clonea_openvoice"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("torch", "melo", "openvoice", "soundfile", "librosa") if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        from melo.api import TTS
        from openvoice.api import OpenVoiceBaseClass, ToneColorConverter

        class _Converter(ToneColorConverter):
            """Upstream forwards `enable_watermark` to the base class (TypeError) and wants `wavmark` otherwise:
            skip both, no watermark."""
            def __init__(self, config_path, device="cuda:0"):
                OpenVoiceBaseClass.__init__(self, config_path, device=device)
                self.watermark_model = None
                self.version = getattr(self.hps, "_version_", "v1")

        base = snapshot_download(model["weights"][0]["hf"], local_files_only=True)           # myshell-ai/OpenVoiceV2
        melo_repo = model["weights"][1]["hf"]                                                   # myshell-ai/MeloTTS-English
        conv = _Converter(os.path.join(base, "converter", "config.json"), device="cuda")
        conv.load_ckpt(os.path.join(base, "converter", "checkpoint.pth"))
        melo_dir = snapshot_download(melo_repo, local_files_only=True)
        tts = TTS(language="EN", device="cuda", config_path=os.path.join(melo_dir, "config.json"),
                  ckpt_path=os.path.join(melo_dir, "checkpoint.pth"))
        ses = {k: torch.load(os.path.join(base, "base_speakers", "ses", v + ".pth"), map_location="cuda")
               for k, v in ACCENTS.items()}
        return {"tts": tts, "conv": conv, "ses": ses, "tgt": {}}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        h = self.handle(job.model_id)
        tts, conv = h["tts"], h["conv"]
        p = job.params
        text = text_of(job)
        accent = p.get("accent") or "EN-US"
        spk = tts.hps.data.spk2id
        if accent not in spk:
            raise InputError(f"Unknown accent '{accent}'. Choices: {', '.join(spk)}.")
        clone = job.task == "voice_clone"
        extra = {}
        tgt = None
        if clone:
            ref = ref_input(job)
            if not ref:
                raise InputError("Pick a reference clip (3-30 s of clean speech from the library).")
            key = (ref["path"], os.path.getmtime(ref["path"]))
            if key not in h["tgt"]:
                ctx.progress(0, 0, "conditioning", "Extracting the reference tone colour", interruptible=False)
                h["tgt"][key] = conv.extract_se(ref["path"])
            tgt = h["tgt"][key]
            extra["ref_asset"] = ref.get("asset_id")
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        sr = int(conv.hps.data.sampling_rate) if clone else int(tts.hps.data.sampling_rate)
        for i in range(n):
            ctx.check()
            seed = s0 + i
            seed_all(seed)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}: base voice")
            audio = tts.tts_to_file(text, spk[accent], None, speed=float(p.get("speed", 1.0)), quiet=True,
                                    sdp_ratio=float(p.get("sdp_ratio", 0.2)), noise_scale=float(p.get("noise_scale", 0.6)),
                                    noise_scale_w=float(p.get("noise_scale_w", 0.8)))
            audio = np.asarray(audio, dtype=np.float32)
            ctx.check()
            if clone:
                ctx.progress(i + 0.5, n, "converting", f"take {i + 1}/{n}: tone-colour conversion")
                import soundfile as sf

                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                try:
                    sf.write(tmp.name, audio, int(tts.hps.data.sampling_rate))
                    audio = conv.convert(audio_src_path=tmp.name, src_se=h["ses"][accent], tgt_se=tgt, output_path=None,
                                         tau=float(p.get("tau", 0.3)), message="default")
                finally:
                    os.unlink(tmp.name)
                audio = np.asarray(audio, dtype=np.float32)
            wav = tidy_voice(audio, sr)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, sr)
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(text, i, n), final_prompt=text, accent=accent,
                       **extra, **fm)
            ctx.progress(i + 1, n, "speaking", f"take {i + 1}/{n} done")
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(OpenVoiceBackend())
