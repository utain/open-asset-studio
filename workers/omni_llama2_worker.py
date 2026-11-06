"""LLaMA-Omni2 backend (envs/omni_llama2: transformers 4.43.4, torch 2.9 cu130; upstream code in vendor/LLaMA-Omni2).

  llama-omni2-0_5b   task s2s   Whisper-large-v3 encoder -> Qwen2.5 LM that writes the answer text and, in parallel, speech
                                 units (autoregressive speech decoder) -> CosyVoice2 flow + HiFT at 24 kHz.
English only for the 0.5B/1.5B/3B "-Bilingual" variants are Chinese+English. Academic-only license.
"""
from __future__ import annotations

import gc
import importlib.util
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "LLaMA-Omni2"
REPO = "ICTNLP/LLaMA-Omni2-0.5B"
DECODER_REPO = "ICTNLP/cosy2_decoder"
WHISPER_PT = Path(os.path.expanduser("~/.cache/whisper/large-v3.pt"))


def seed0(p):
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def save_wav(path, wav, sr):
    import soundfile as sf

    w = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1))
    if len(w) < int(0.05 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    meta = {"sample_rate": int(sr)}
    peak = float(np.abs(w).max())
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_16")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def name_of(text, i, n):
    t = re.sub(r"\s+", " ", text).strip()
    return f"{t[:48]}{'…' if len(t) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def patch_torchaudio():
    """torchaudio 2.9 load/save need torchcodec (no aarch64 wheel): route them through soundfile."""
    import soundfile as sf
    import torch
    import torchaudio

    def load(path, *a, **k):
        y, sr = sf.read(str(path) if not hasattr(path, "read") else path, dtype="float32", always_2d=True)
        return torch.from_numpy(y.T.copy()), sr

    def save(path, wav, sr=None, *a, **k):
        sr = sr or k.get("sample_rate")
        sf.write(str(path), wav.detach().cpu().numpy().T, sr)

    torchaudio.load, torchaudio.save = load, save
    # removed in torchaudio 2.9 but still called at import time by cosyvoice.dataset.processor
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
        torchaudio.get_audio_backend = lambda: "soundfile"


def stub_text_normalizers():
    """cosyvoice's frontend imports WeTextProcessing (`tn`, needs pynini: no aarch64 wheel). We only use its speaker-embedding
    extractor (speech units are decoded, no text goes through the frontend), so give it inert stand-ins."""
    import types

    if "tn" in sys.modules:
        return

    class Normalizer:
        def __init__(self, *a, **k):
            pass

        def normalize(self, text):
            return text

    for name in ("tn", "tn.chinese", "tn.english", "tn.chinese.normalizer", "tn.english.normalizer"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["tn.chinese.normalizer"].Normalizer = Normalizer
    sys.modules["tn.english.normalizer"].Normalizer = Normalizer


class Llama2Backend(Backend):
    name = "omni_llama2"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("transformers", "whisper", "hyperpyyaml", "torch") if importlib.util.find_spec(m) is None]
        if not VENDOR.exists():
            missing.append(f"vendor/LLaMA-Omni2 ({VENDOR})")
        if not WHISPER_PT.exists():
            missing.append(f"{WHISPER_PT} (openai-whisper large-v3 checkpoint, wget from the URL in whisper._MODELS)")
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        patch_torchaudio()
        stub_text_normalizers()
        for p in (VENDOR, VENDOR.parent / "Matcha-TTS"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from llama_omni2.model import Omni2Speech2SQwen2ForCausalLM
        from llama_omni2.inference.run_cosy2_decoder import SpeechDecoder
        from cosyvoice.utils.file_utils import load_wav

        local = snapshot_download(REPO, local_files_only=True)
        cfg = AutoConfig.from_pretrained(local)
        cfg.tts_tokenizer = os.path.join(local, "tts_tokenizer")
        cfg.speech_encoder = str(WHISPER_PT)
        tok = AutoTokenizer.from_pretrained(local, use_fast=False)
        lm = Omni2Speech2SQwen2ForCausalLM.from_pretrained(local, config=cfg, torch_dtype=torch.bfloat16).cuda().eval()
        dec = SpeechDecoder(model_dir=snapshot_download(DECODER_REPO, local_files_only=True), device="cuda")
        prompt = load_wav(str(VENDOR / "llama_omni2" / "inference" / "prompt_en.wav"), 16000)
        return {"tok": tok, "lm": lm, "dec": dec, "prompt": prompt}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import soundfile as sf
        import torch
        import whisper
        from llama_omni2.constants import DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX

        h = self.handle(job.model_id)
        tok, lm, dec = h["tok"], h["lm"], h["dec"]
        p = job.params
        src = job.input(role="audio") or next((i for i in job.inputs if i.get("type") in ("audio", "voice")), None)
        if not src:
            raise InputError("Pick an audio clip to answer (this model only takes spoken questions).")
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        temp = float(p.get("temperature") or 0.0)
        system = (p.get("instruction") or "").strip()
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": DEFAULT_SPEECH_TOKEN}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")[0]
        ids[ids == tok.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
        ids = ids.unsqueeze(0).cuda()
        a = whisper.load_audio(src["path"])
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(a[: 30 * 16000]), n_mels=128).permute(1, 0)
        speech = mel.unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
        slen = torch.LongTensor([mel.shape[0]]).cuda()
        max_new = max(64, int(float(p.get("seconds") or 20) * 4))  # ~4 text tokens/s of speech is plenty
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ctx.progress(i, n, "thinking", f"take {i + 1}/{n}: writing the reply", interruptible=False)
            with torch.inference_mode():
                out_ids, units = lm.generate(ids, speech=speech, speech_lengths=slen, do_sample=temp > 0,
                                             temperature=temp if temp > 0 else None, top_p=0.9 if temp > 0 else None,
                                             top_k=None, num_beams=1, max_new_tokens=max_new, use_cache=True,
                                             pad_token_id=tok.pad_token_id)
                reply = tok.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                u = [int(x) for x in re.findall(r"\d+", units)] if isinstance(units, str) else [int(x) for x in units]
                if not u:
                    raise RuntimeError("the model produced no speech units (try again or a different seed)")
                ctx.progress(i, n, "decoding", "flow-matching + vocoder", interruptible=False)
                wav = dec.entry(torch.LongTensor(u).cuda(), h["prompt"], stream=False)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav.numpy(), 24000)
            (job.out_dir / f"take_{i + 1}.txt").write_text(reply, encoding="utf-8")
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(reply or "Reply", i, n), final_prompt=reply,
                       reply_text=reply, input_asset=src.get("asset_id"), attach=[f"take_{i + 1}.txt"], **fm)
            gc.collect()
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(Llama2Backend())
