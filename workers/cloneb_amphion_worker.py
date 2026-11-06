"""Amphion backend (envs/cloneb_amphion, vendor/Amphion): MaskGCT, Vevo, Vevo1.5.

  maskgct   tasks tts, voice_clone       non-autoregressive zero-shot TTS (en zh ja ko fr de), 24 kHz
  vevo      tasks tts, voice_clone, voice_convert
              tts / voice_clone: AR (PhoneToVq8192) + flow matching, style ref + optional timbre ref
              voice_convert: mode timbre (keep style, swap timbre; FM only) | style (keep timbre, imitate
              the reference's accent/emotion) | voice (imitate both)
  vevo15    tasks tts, voice_clone, voice_convert   speech AND singing (Vevo1.5 = amphion/Vevo1.5 weights,
              code models/svc/vevosing). voice_convert modes: timbre | style (keeps the source timbre, takes
              the reference's speaking/singing style; needs transcripts, made with Whisper when not given).
              Optional `melody` input on tts: sings the text to the pitch contour of that clip.

Inputs (job.inputs roles): `ref` (reference voice), `source` (clip to convert), `timbre` (optional second
reference for tts: style from `ref`, timbre from `timbre`), `melody` (Vevo1.5 tts, optional).
Needs espeak-ng: provided by the espeakng-loader wheel (no apt); paths are exported before Amphion imports.
"""
from __future__ import annotations

import math
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

STUDIO = Path(__file__).resolve().parent.parent
AMPHION = STUDIO / "vendor" / "Amphion"
LANGS = {"en", "zh", "ja", "ko", "fr", "de"}
MAX_REF_S = 15.0
MAX_SRC_S = 40.0


def _setup_env():
    """espeak-ng from the espeakng-loader wheel + Amphion on sys.path (its configs use paths relative to its root)."""
    import espeakng_loader

    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
    os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
    if str(AMPHION) not in sys.path:
        sys.path.insert(0, str(AMPHION))
    os.chdir(AMPHION)


def hf_dir(repo, *sub):
    from huggingface_hub import snapshot_download

    p = Path(snapshot_download(repo, local_files_only=True))
    return str(p.joinpath(*sub)) if sub else str(p)


def seed0(p):
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def set_seed(s):
    import torch

    random.seed(s)
    np.random.seed(s % (2**32))
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def prep_wav(path, tmp, name, max_s):
    """Mono wav at 24 kHz, trimmed to max_s seconds, in the temp dir (Amphion loads from paths)."""
    import librosa
    import soundfile as sf

    y = librosa.load(path, sr=24000, mono=True)[0]
    if len(y) < 24000 * 0.5:
        raise InputError("The audio clip is too short (under 0.5 s).")
    y = y[: int(max_s * 24000)]
    out = Path(tmp) / f"{name}.wav"
    sf.write(str(out), y, 24000)
    return str(out), len(y) / 24000


def save_wav(path, wav, sr=24000, target_db=-22.0):
    import soundfile as sf

    w = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1))
    if len(w) < int(0.1 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    rms = float(np.sqrt(np.mean(w**2)) + 1e-9)
    w = w * (10 ** (target_db / 20) / rms)
    peak = float(np.abs(w).max())
    meta = {"sample_rate": int(sr)}
    if peak > 0.97:
        w = w * (0.97 / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def name_of(text, i, n):
    t = re.sub(r"\s+", " ", text or "").strip()
    return f"{t[:48]}{'…' if len(t) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


# Voices bundled in the Amphion repo (so `tts` works without a reference clip): file, transcript
BUNDLED = {
    "maskgct_prompt": ("models/tts/maskgct/wav/prompt.wav", " We do not break. We never give in. We never back down."),
    "arabic_male": ("models/vc/vevo/wav/arabic_male.wav", "Flip stood undecided, his ears strained to catch the slightest sound."),
    "mandarin_female": ("models/vc/vevo/wav/mandarin_female.wav", None),
}


class AmphionBackend(Backend):
    name = "cloneb_amphion"
    max_loaded = 1

    def probe(self, model):
        import importlib.util

        missing = [m for m in ("torch", "librosa", "phonemizer", "whisper", "espeakng_loader", "transformers", "soundfile")
                   if importlib.util.find_spec(m) is None]
        if not (AMPHION / "models").is_dir():
            missing.append("vendor/Amphion")
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    # ------------------------------------------------------------------ load
    def load(self, model, ctx):
        _setup_env()
        import torch

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        mid = model["id"]
        h = {"id": mid, "asr": None, "pipes": {}}
        if mid == "maskgct":
            h["pipe"] = self._load_maskgct()
        elif mid == "vevo":
            h["pipes"]["vc"] = self._load_vevo("vc")      # timbre / style / voice conversion
        elif mid == "vevo15":
            h["pipe"] = self._load_vevo15()
        else:
            raise InputError(f"unknown model {mid}")
        torch.cuda.synchronize()
        return h

    def _load_maskgct(self):
        import safetensors
        import torch
        from models.tts.maskgct.maskgct_utils import (MaskGCT_Inference_Pipeline, build_acoustic_codec, build_s2a_model,
                                                      build_semantic_codec, build_semantic_model, build_t2s_model)
        from utils.util import load_config

        dev = torch.device("cuda:0")
        cfg = load_config("./models/tts/maskgct/config/maskgct.json")
        w2v = hf_dir("facebook/w2v-bert-2.0")
        # build_semantic_model / the pipeline call from_pretrained("facebook/w2v-bert-2.0"): offline-safe (HF_HUB_OFFLINE=1)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        sem, mean, std = build_semantic_model(dev)
        sem_codec = build_semantic_codec(cfg.model.semantic_codec, dev)
        enc, dec = build_acoustic_codec(cfg.model.acoustic_codec, dev)
        t2s = build_t2s_model(cfg.model.t2s_model, dev)
        s2a1 = build_s2a_model(cfg.model.s2a_model.s2a_1layer, dev)
        s2af = build_s2a_model(cfg.model.s2a_model.s2a_full, dev)
        d = hf_dir("amphion/MaskGCT")
        for m, f in ((sem_codec, "semantic_codec/model.safetensors"), (enc, "acoustic_codec/model.safetensors"),
                     (dec, "acoustic_codec/model_1.safetensors"), (t2s, "t2s_model/model.safetensors"),
                     (s2a1, "s2a_model/s2a_model_1layer/model.safetensors"), (s2af, "s2a_model/s2a_model_full/model.safetensors")):
            safetensors.torch.load_model(m, os.path.join(d, f))
        del w2v
        return MaskGCT_Inference_Pipeline(sem, sem_codec, enc, dec, t2s, s2a1, s2af, mean, std, dev)

    def _load_vevo(self, kind):
        import torch
        from models.vc.vevo.vevo_utils import VevoInferencePipeline

        d = hf_dir("amphion/Vevo")
        cfgd = "./models/vc/vevo/config/"
        kw = dict(
            content_style_tokenizer_ckpt_path=os.path.join(d, "tokenizer/vq8192"),
            fmt_cfg_path=cfgd + "Vq8192ToMels.json",
            fmt_ckpt_path=os.path.join(d, "acoustic_modeling/Vq8192ToMels"),
            vocoder_cfg_path=cfgd + "Vocoder.json",
            vocoder_ckpt_path=os.path.join(d, "acoustic_modeling/Vocoder"),
            device=torch.device("cuda"),
        )
        if kind == "vc":
            kw.update(content_tokenizer_ckpt_path=os.path.join(d, "tokenizer/vq32/hubert_large_l18_c32.pkl"),
                      ar_cfg_path=cfgd + "Vq32ToVq8192.json",
                      ar_ckpt_path=os.path.join(d, "contentstyle_modeling/Vq32ToVq8192"))
        else:
            kw.update(ar_cfg_path=cfgd + "PhoneToVq8192.json",
                      ar_ckpt_path=os.path.join(d, "contentstyle_modeling/PhoneToVq8192"))
        return VevoInferencePipeline(**kw)

    def _load_vevo15(self):
        import torch
        from models.svc.vevosing.vevosing_utils import VevosingInferencePipeline

        d = hf_dir("amphion/Vevo1.5")
        cfgd = "./models/svc/vevosing/config/"
        return VevosingInferencePipeline(
            prosody_tokenizer_ckpt_path=os.path.join(d, "tokenizer/prosody_fvq512_6.25hz"),
            content_style_tokenizer_ckpt_path=os.path.join(d, "tokenizer/contentstyle_fvq16384_12.5hz"),
            ar_cfg_path=cfgd + "ar_emilia101k_singnet7k.json",
            ar_ckpt_path=os.path.join(d, "contentstyle_modeling/ar_emilia101k_singnet7k"),
            fmt_cfg_path=cfgd + "fm_emilia101k_singnet7k.json",
            fmt_ckpt_path=os.path.join(d, "acoustic_modeling/fm_emilia101k_singnet7k"),
            vocoder_cfg_path=cfgd + "vocoder.json",
            vocoder_ckpt_path=os.path.join(d, "acoustic_modeling/Vocoder"),
            device=torch.device("cuda"),
        )

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        if isinstance(h, dict):
            h.clear()

    # ------------------------------------------------------------------ helpers
    def _transcribe(self, h, path, lang):
        """Reference transcript when the user gave none (openai-whisper medium, the model Vevo1.5 loads anyway)."""
        wm = None
        if h.get("pipe") is not None and hasattr(h["pipe"], "whisper_model"):
            wm = h["pipe"].whisper_model
        else:
            if h.get("asr") is None:
                import torch
                import whisper

                h["asr"] = whisper.load_model("medium", "cuda")
            wm = h["asr"]
        kw = {"language": lang} if lang in LANGS else {}
        return wm.transcribe(path, fp16=True, **kw)["text"].strip()

    def _ref(self, job, p, tmp, ctx, role="ref", required=True):
        """-> (wav path in tmp, transcript or None)"""
        ref = job.input(role=role) or (None if role != "ref" else next((i for i in job.inputs if i.get("type") in ("voice", "audio")), None))
        if ref is None:
            return None, None
        path, _ = prep_wav(ref["path"], tmp, role, MAX_REF_S)
        text = (p.get("ref_text") or "").strip() if role == "ref" else ""
        if not text:
            text = (((ref.get("meta") or {}).get("x_voice") or {}).get("ref_text") or "").strip() if role == "ref" else ""
        return path, text or None

    # ------------------------------------------------------------------ run
    def run(self, job, ctx):
        _setup_env()
        import torch

        h = self.handle(job.model_id)
        p = job.params
        mid = job.model_id
        lang = p.get("language") or "en"
        if lang not in LANGS:
            lang = "en"
        s0, n = seed0(p), max(1, int(p.get("batch", 1) or 1))
        text = (p.get("text") or p.get("prompt") or "").strip()
        convert = job.task == "voice_convert"
        if not convert and not text and not (mid == "vevo15" and job.input(role="melody")):
            raise InputError("Write the line to speak first.")
        if mid == "maskgct" and convert:
            raise InputError("MaskGCT does not convert voices. Use Vevo or Vevo1.5.")
        with tempfile.TemporaryDirectory(prefix="amphion_") as tmp:
            src_path = None
            if convert:
                src = job.input(role="source")
                if src is None:
                    raise InputError("Pick the source clip (the audio to convert).")
                src_path, src_s = prep_wav(src["path"], tmp, "source", MAX_SRC_S)
                if src_s >= MAX_SRC_S - 0.1:
                    ctx.log(f"source clip trimmed to {MAX_SRC_S:.0f} s")
            ref_path, ref_text = self._ref(job, p, tmp, ctx)
            if ref_path is None:
                if convert:
                    raise InputError("Pick a reference voice clip.")
                v = p.get("voice") or "maskgct_prompt"
                f, t = BUNDLED.get(v, BUNDLED["maskgct_prompt"])
                ref_path, _ = prep_wav(str(AMPHION / f), tmp, "ref", MAX_REF_S)
                ref_text = ref_text or t
                ctx.log(f"no reference clip: using the bundled voice '{v}'")
            timbre_path, _ = self._ref(job, p, tmp, ctx, role="timbre")
            melody = job.input(role="melody")
            melody_path = prep_wav(melody["path"], tmp, "melody", MAX_SRC_S)[0] if melody else None
            ref_lang = p.get("ref_language") or lang
            need_ref_text = not (convert and p.get("mode", "timbre") in ("timbre", "voice") and mid == "vevo") and not (convert and mid == "vevo15" and p.get("mode", "timbre") == "timbre")
            if need_ref_text and not ref_text and not (mid == "vevo" and convert):
                ctx.progress(0, n, "transcribing", "Transcribing the reference clip (Whisper)")
                ref_text = self._transcribe(h, ref_path, ref_lang)
                ctx.log(f"reference transcript (auto): {ref_text}")
            steps = int(p.get("steps") or 0)
            files = []
            for i in range(n):
                ctx.check()
                seed = s0 + i
                set_seed(seed)
                ctx.progress(i, n, "synthesizing", f"take {i + 1}/{n}")
                with torch.inference_mode():
                    if mid == "maskgct":
                        wav, sr = self._run_maskgct(h, p, text, lang, ref_lang, ref_path, ref_text, steps), 24000
                    elif mid == "vevo":
                        wav, sr = self._run_vevo(h, p, job, text, lang, ref_lang, ref_path, ref_text, timbre_path, src_path, steps), 24000
                    else:
                        wav, sr = self._run_vevo15(h, p, job, text, lang, ref_lang, ref_path, ref_text, timbre_path, src_path, melody_path, steps), 24000
                f = job.out_dir / f"take_{i + 1}.wav"
                fm = save_wav(f, wav, sr)
                label = text or (f"{p.get('mode', 'timbre')} conversion")
                extra = {"mode": p.get("mode")} if convert else {}
                ctx.output(f, "audio", seed=seed, index=i, name=name_of(label, i, n), final_prompt=text or None, **extra, **fm)
                files.append(str(f))
        return {"meta": {"seed": s0}}

    def _run_maskgct(self, h, p, text, lang, ref_lang, ref_path, ref_text, steps):
        secs = float(p.get("target_seconds") or 0)
        return h["pipe"].maskgct_inference(
            ref_path, ref_text, text, language=ref_lang, target_language=lang,
            target_len=secs if secs > 0 else None, n_timesteps=steps or 25,
            cfg=float(p.get("cfg_scale") or 2.5), n_timesteps_s2a=[int(p.get("s2a_steps") or 25), 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def _vevo_pipe(self, h, kind):
        """Vevo has two AR models (phone->tokens for tts, tokens->tokens for conversion): keep one resident."""
        if kind not in h["pipes"]:
            import gc

            import torch

            h["pipes"].clear()   # one AR at a time keeps memory down; the other pipeline is rebuilt on demand
            gc.collect()
            torch.cuda.empty_cache()
            h["pipes"][kind] = self._load_vevo(kind)
        return h["pipes"][kind]

    def _run_vevo(self, h, p, job, text, lang, ref_lang, ref_path, ref_text, timbre_path, src_path, steps):
        steps = steps or 32
        if job.task == "voice_convert":
            pipe = self._vevo_pipe(h, "vc")
            mode = p.get("mode") or "timbre"
            if mode == "timbre":
                a = pipe.inference_fm(src_wav_path=src_path, timbre_ref_wav_path=ref_path, flow_matching_steps=steps)
            elif mode == "style":
                a = pipe.inference_ar_and_fm(src_wav_path=src_path, src_text=None, style_ref_wav_path=ref_path,
                                             timbre_ref_wav_path=src_path, flow_matching_steps=steps)
            else:
                a = pipe.inference_ar_and_fm(src_wav_path=src_path, src_text=None, style_ref_wav_path=ref_path,
                                             timbre_ref_wav_path=ref_path, flow_matching_steps=steps)
            return a.numpy()[0]
        pipe = self._vevo_pipe(h, "tts")
        a = pipe.inference_ar_and_fm(src_wav_path=None, src_text=text, style_ref_wav_path=ref_path,
                                     timbre_ref_wav_path=timbre_path or ref_path, style_ref_wav_text=ref_text,
                                     src_text_language=lang, style_ref_wav_text_language=ref_lang, flow_matching_steps=steps)
        return a.numpy()[0]

    def _run_vevo15(self, h, p, job, text, lang, ref_lang, ref_path, ref_text, timbre_path, src_path, melody_path, steps):
        pipe = h["pipe"]
        steps = steps or 32
        if job.task == "voice_convert":
            mode = p.get("mode") or "timbre"
            if mode == "timbre":
                a = pipe.inference_fm(src_wav_path=src_path, timbre_ref_wav_path=ref_path, flow_matching_steps=steps)
            else:   # style: keep the source timbre, take the reference's style (singing style conversion)
                src_text = (p.get("source_text") or "").strip() or self._transcribe(h, src_path, lang)
                a = pipe.inference_ar_and_fm(
                    task="recognition-synthesis", src_wav_path=src_path, src_text=src_text, style_ref_wav_path=ref_path,
                    style_ref_wav_text=ref_text, src_text_language=lang, style_ref_wav_text_language=ref_lang,
                    timbre_ref_wav_path=src_path, use_style_tokens_as_ar_input=True, flow_matching_steps=steps)
            return a.numpy()[0]
        if melody_path:
            a = pipe.inference_ar_and_fm(
                task="recognition-synthesis", src_wav_path=melody_path, src_text=text, style_ref_wav_path=ref_path,
                style_ref_wav_text=ref_text, src_text_language=lang, style_ref_wav_text_language=ref_lang,
                timbre_ref_wav_path=timbre_path or ref_path, use_style_tokens_as_ar_input=True, flow_matching_steps=steps)
        else:
            a = pipe.inference_ar_and_fm(
                task="synthesis", src_wav_path=None, src_text=text, style_ref_wav_path=ref_path,
                timbre_ref_wav_path=timbre_path or ref_path, style_ref_wav_text=ref_text,
                src_text_language=lang, style_ref_wav_text_language=ref_lang, flow_matching_steps=steps)
        return a.numpy()[0]


if __name__ == "__main__":
    serve(AmphionBackend())
