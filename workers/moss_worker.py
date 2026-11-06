"""MOSS backend (envs/moss, read-only: transformers 5.0.0 exactly, trust_remote_code).

  moss-voicegen    task voice_design  a new voice from a free-prose description; outputs are `voice` assets
                                      (the clip is the clone reference, its line the transcript)
  moss-tts-v1.5    task voice_clone   long-form clone from a reference clip in 31 languages incl. Thai
                   task tts           without a reference the model picks a voice
Codec: OpenMOSS-Team/MOSS-Audio-Tokenizer. Recipe: audio_snippets/moss_tts.py and the MOSS-TTS examples.
generate() has no step hook, so a take cannot be interrupted midway (cancel waits for the current take).
"""
from __future__ import annotations

import importlib.util
import math
import random
import re

import numpy as np

from studio_worker import Backend, InputError, serve

TOKENS_PER_S = 12.5
THAI = re.compile(r"[฀-๿]")


def seed0(p):
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def tidy_voice(w, sr, pad_s=0.12, floor_db=-50.0):
    idx = np.flatnonzero(np.abs(w) > 10 ** (floor_db / 20))
    if idx.size == 0:
        return w
    pad = int(pad_s * sr)
    return w[max(0, idx[0] - pad): min(len(w), idx[-1] + pad + 1)]


def save_wav(path, wav, sr):
    import soundfile as sf

    w = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1))
    if len(w) < int(0.05 * sr):
        raise RuntimeError("the model returned (almost) no audio; try another seed")
    meta = {"sample_rate": int(sr)}
    peak = float(np.abs(w).max())
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def name_of(text, i, n):
    t = re.sub(r"\s+", " ", text).strip()
    return f"{t[:48]}{'…' if len(t) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


class MossBackend(Backend):
    name = "moss"
    max_loaded = 1

    def probe(self, model):
        if model["id"] == "moss-tts-nano":
            return False, "MOSS-TTS-Nano needs its 88 MB codec and the upstream MOSS-TTS-Nano package (not installed)"
        missing = [m for m in ("transformers", "torch", "soundfile") if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch
        from transformers import AutoModel, AutoProcessor

        if model["id"] == "moss-tts-nano":
            raise InputError("MOSS-TTS-Nano is not set up on this machine.")
        torch.backends.cuda.enable_cudnn_sdp(False)     # model card: the cuDNN SDPA backend is broken
        repo = model["weights"][0]["hf"]
        design = model["id"] == "moss-voicegen"
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True, **({"normalize_inputs": True} if design else {}))
        proc.audio_tokenizer = proc.audio_tokenizer.to("cuda")
        m = AutoModel.from_pretrained(repo, trust_remote_code=True, attn_implementation=attn, torch_dtype=torch.bfloat16).to("cuda").eval()
        return {"proc": proc, "model": m, "sr": int(proc.model_config.sampling_rate), "refs": {}}

    def _ref_codes(self, h, path):
        import soundfile as sf
        import torch

        if path not in h["refs"]:
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = wav[: sr * 30]
            h["refs"][path] = h["proc"].encode_audios_from_wav([torch.from_numpy(wav.T.copy())], sr)[0]  # no torchaudio.load
            if len(h["refs"]) > 16:
                h["refs"].pop(next(iter(h["refs"])))
        return h["refs"][path]

    def run(self, job, ctx):
        import torch

        h = self.handle(job.model_id)
        proc, model, sr = h["proc"], h["model"], h["sr"]
        p = job.params
        text = (p.get("text") or "").strip()
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        design = job.model_id == "moss-voicegen"
        extra, as_voice = {}, False
        if design:
            ins = (p.get("instruction") or p.get("prompt") or "").strip()
            if not ins:
                raise InputError("Describe the voice first (age, gender, timbre, pace, accent, emotion).")
            if not text:
                raise InputError("Write a short neutral line for the voice to say (it becomes the clone reference).")
            msg = proc.build_user_message(text=text, instruction=ins)
            max_new = 1000
            sampling = dict(audio_temperature=float(p.get("temperature", 1.5)), audio_top_p=float(p.get("top_p", 0.6)),
                            audio_top_k=int(p.get("top_k", 50)), audio_repetition_penalty=1.1)
            as_voice = True
            extra["design_prompt"] = ins
        else:
            if not text:
                text = (p.get("prompt") or "").strip()
            if not text:
                raise InputError("Write the line to speak first.")
            lang = p.get("language") or None
            if lang in (None, "", "auto") and THAI.search(text):
                lang = "Thai"
            ref = job.input(role="ref") or next((i for i in job.inputs if i.get("type") in ("voice", "audio")), None)
            if job.task == "voice_clone" and not ref:
                raise InputError("Pick a reference clip to clone.")
            kw = {"text": text, "language": lang}
            if ref:
                ctx.progress(0, n, "speaking", "Encoding the reference voice", interruptible=False)
                kw["reference"] = [self._ref_codes(h, ref["path"])]
                extra["ref_asset"] = ref.get("asset_id")
            msg = proc.build_user_message(**kw)
            max_new = int(TOKENS_PER_S * float(p.get("max_seconds", 20)))
            sampling = dict(audio_temperature=float(p.get("temperature", 1.7)), audio_top_p=float(p.get("top_p", 0.8)),
                            audio_top_k=int(p.get("top_k", 25)), audio_repetition_penalty=1.0)
            extra["language"] = lang
        batch = proc([[msg]], mode="generation")
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n} (cannot be interrupted midway)", interruptible=False)
            with torch.no_grad():
                o = model.generate(input_ids=batch["input_ids"].to("cuda"), attention_mask=batch["attention_mask"].to("cuda"),
                                   max_new_tokens=max_new, **sampling)
            audio = proc.decode(o)[0].audio_codes_list[0].float().cpu().numpy().reshape(-1)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, tidy_voice(audio, sr), sr)
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(("Voice: " + extra["design_prompt"]) if design else text, i, n), final_prompt=text, as_voice=as_voice, **extra, **fm)
            ctx.progress(i + 1, n, "speaking", f"take {i + 1}/{n} done")
        return {"meta": {"seed": s0}}

    def unload(self, model_id):
        self.models.pop(model_id, None)


if __name__ == "__main__":
    serve(MossBackend())
