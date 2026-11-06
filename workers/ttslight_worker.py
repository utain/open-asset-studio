"""Light TTS backends (envs/ttslight and envs/ttslight_parler).

  envs/ttslight          kokoro-82m (task tts), piper-en / piper-thai (task tts, CPU), styletts2-libritts (tts + voice_clone)
  envs/ttslight_parler   parler-tts-mini-v1_1 / parler-tts-large-v1 (tts + voice_design), melo-english (tts)

One module, lazy imports per family: workers/ttslight_parler_worker.py re-uses it with `name = "ttslight_parler"`.
Every take is a 24-bit WAV (speech is trimmed of leading / trailing silence, overs are brought to -1 dBFS).
"""
from __future__ import annotations

import importlib.util
import math
import os
import random
import re
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np

from studio_worker import Backend, InputError, serve

STUDIO = Path(__file__).resolve().parent.parent
THAI = re.compile(r"[฀-๿]")


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


def tidy_voice(w: np.ndarray, sr: int, pad_s: float = 0.12, floor_db: float = -50.0) -> np.ndarray:
    idx = np.flatnonzero(np.abs(w) > 10 ** (floor_db / 20))
    if idx.size == 0:
        return w
    pad = int(pad_s * sr)
    return w[max(0, idx[0] - pad): min(len(w), idx[-1] + pad + 1)]


def save_wav(path: Path, wav, sr: int) -> dict:
    import soundfile as sf

    w = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if len(w) < int(0.05 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    meta = {"sample_rate": int(sr)}
    peak = float(np.abs(w).max())
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def name_of(text: str, i: int, n: int) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return f"{t[:48]}{'…' if len(t) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def line_of(job) -> str:
    t = (job.params.get("text") or job.params.get("prompt") or "").strip()
    if not t:
        raise InputError("Write the line to speak first.")
    return t


def hf_snapshot(repo: str) -> str:
    """Local snapshot dir of a cached HF repo. (snapshot_download(local_files_only=True) insists that EVERY file
    of the repo is cached, which is false for repos we fetched partially, e.g. piper-voices.)"""
    from huggingface_hub import constants

    root = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    ref = root / "refs" / "main"
    snaps = root / "snapshots"
    if ref.exists() and (snaps / ref.read_text().strip()).is_dir():
        return str(snaps / ref.read_text().strip())
    cand = sorted(snaps.glob("*"), key=lambda q: q.stat().st_mtime) if snaps.is_dir() else []
    if not cand:
        raise InputError(f"weights of {repo} are not in the HF cache ({root}).")
    return str(cand[-1])


def ref_clip(job):
    for role in ("ref", "voice", "reference"):
        r = job.input(role=role)
        if r:
            return r
    return next((i for i in job.inputs if i.get("type") in ("voice", "audio")), None)


FAMILIES = ("kokoro", "piper", "styletts2", "parler", "melo")
NEEDS = {"kokoro": ["kokoro", "misaki"], "piper": ["piper", "onnxruntime"], "styletts2": ["munch", "nltk", "librosa"],
         "parler": ["parler_tts", "transformers"], "melo": ["g2p_en", "cached_path", "librosa"]}

# Piper voices: id -> (path inside rhasspy/piper-voices, speakers)
PIPER_VOICES = {
    "en_US-lessac-high": "en/en_US/lessac/high/en_US-lessac-high",
    "en_US-amy-medium": "en/en_US/amy/medium/en_US-amy-medium",
    "en_US-libritts_r-medium": "en/en_US/libritts_r/medium/en_US-libritts_r-medium",
    "en_GB-alba-medium": "en/en_GB/alba/medium/en_GB-alba-medium",
    "en_GB-northern_english_male-medium": "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium",
    "th_TH-tsync2-medium": "th/th_TH/tsync2/medium/th_TH-tsync2-medium",
}

MELO_SPEAKERS = ["EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-Default"]


def stub_melo_languages():
    """MeloTTS imports every language module at import time and each one downloads a tokenizer / needs
    packages (mecab, jieba, gruut ...). We only use English: replace the others with empty stand-ins."""
    def distribute_phone(n_phone, n_word):  # the only thing english.py takes from the Japanese module
        phones_per_word = [0] * n_word
        for _ in range(n_phone):
            phones_per_word[phones_per_word.index(min(phones_per_word))] += 1
        return phones_per_word

    def unavailable(*_a, **_k):
        raise RuntimeError("only MeloTTS-English is installed")

    for name in ("chinese", "japanese", "korean", "french", "spanish", "chinese_mix",
                 "chinese_bert", "japanese_bert", "french_bert", "spanish_bert"):
        full = f"melo.text.{name}"
        if full not in sys.modules:
            m = types.ModuleType(full)
            m.__getattr__ = lambda attr, _u=unavailable: _u        # any other symbol the package asks for
            if name == "japanese":
                m.distribute_phone = distribute_phone
            sys.modules[full] = m


# ------------------------------------------------------------------ backend
class TTSLightBackend(Backend):
    name = "ttslight"
    max_loaded = 3

    @staticmethod
    def family(mid: str) -> str:
        for k, pre in (("kokoro", "kokoro"), ("piper", "piper"), ("styletts2", "styletts2"), ("parler", "parler"), ("melo", "melo")):
            if mid.startswith(pre):
                return k
        raise InputError(f"The {TTSLightBackend.name} backend does not know model '{mid}'.")

    def probe(self, model):
        fam = self.family(model["id"])
        need = NEEDS[fam] + ["torch", "soundfile"] + (["tltk", "gensim"] if model["id"] == "piper-thai" else [])
        missing = [m for m in need if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    # ---------------------------------------------------------------- load
    def load(self, model, ctx):
        import torch

        fam = self.family(model["id"])
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if fam == "kokoro":
            from kokoro import KModel

            return {"kmodel": KModel(repo_id=model["weights"][0]["hf"]).to(dev).eval(), "pipes": {}, "dev": dev}
        if fam == "piper":
            return {"voices": {}, "snap": hf_snapshot(model["weights"][0]["hf"])}
        if fam == "styletts2":
            from ttslight_styletts2 import StyleTTS2Infer

            return StyleTTS2Infer(hf_snapshot(model["weights"][0]["hf"]), dev)
        if fam == "parler":
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            repo = hf_snapshot(model["weights"][0]["hf"])
            m = ParlerTTSForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.float32).to(dev).eval()
            tok = AutoTokenizer.from_pretrained(repo)
            # v1.1 uses two tokenizers (description: the T5 text encoder's; prompt: the repo's); v1 uses one
            dtok = AutoTokenizer.from_pretrained(hf_snapshot(m.config.text_encoder._name_or_path)) \
                if (model.get("features") or {}).get("two_tokenizers") else tok
            return {"model": m, "tok": tok, "dtok": dtok, "dev": dev}
        if fam == "melo":
            vend = str(STUDIO / "vendor" / "MeloTTS-ttslight")
            if vend not in sys.path:
                sys.path.insert(0, vend)
            stub_melo_languages()
            from melo.api import TTS

            return TTS(language="EN", device=dev)
        raise InputError(f"unknown model {model['id']}")

    # ----------------------------------------------------------------- run
    def run(self, job, ctx):
        fam = self.family(job.model_id)
        res = getattr(self, "_run_" + fam)(job, ctx, self.handle(job.model_id)) or {}
        try:
            import torch

            if torch.cuda.is_available():
                res.setdefault("meta", {})["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        except Exception:  # noqa: BLE001
            pass
        return res

    # --- Kokoro
    def _run_kokoro(self, job, ctx, h):
        from kokoro import KPipeline

        p = job.params
        text = line_of(job)
        voice = p.get("voice") or "af_heart"
        lang = voice[0]                                   # a = American, b = British
        if THAI.search(text):
            raise InputError("Kokoro has no Thai. Use Piper Thai, OmniVoice or Higgs for Thai lines.")
        pipe = h["pipes"].get(lang)
        if pipe is None:
            pipe = h["pipes"][lang] = KPipeline(lang_code=lang, repo_id=job.model["weights"][0]["hf"], model=h["kmodel"])
        speed = float(p.get("speed", 1.0))
        parts = []
        gap = np.zeros(int(0.18 * 24000), dtype=np.float32)
        ctx.progress(0, 1, "speaking", "Kokoro: synthesizing")
        for k, (_gs, _ps, audio) in enumerate(pipe(text, voice=voice, speed=speed, split_pattern=r"\n+")):
            ctx.check()
            if audio is None:
                continue
            a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            parts += [a.astype(np.float32), gap]
            ctx.progress(k + 1, 0, "speaking", f"Kokoro: chunk {k + 1}")
        if not parts:
            raise RuntimeError("Kokoro produced no audio")
        wav = tidy_voice(np.concatenate(parts[:-1]), 24000)
        f = job.out_dir / "take_1.wav"
        fm = save_wav(f, wav, 24000)
        ctx.output(f, "audio", seed=-1, index=0, name=name_of(text, 0, 1), final_prompt=text, voice=voice, **fm)
        return {"meta": {"voice": voice, "deterministic": True}}

    # --- Piper (CPU, onnx)
    def _piper_voice(self, h, vid):
        from piper import PiperVoice

        v = h["voices"].get(vid)
        if v is None:
            rel = PIPER_VOICES.get(vid)
            if not rel:
                raise InputError(f"unknown Piper voice {vid}")
            onnx = Path(h["snap"]) / (rel + ".onnx")
            if not onnx.exists():
                raise InputError(f"Piper voice {vid} is not downloaded ({onnx.name}).")
            v = h["voices"][vid] = PiperVoice.load(str(onnx), config_path=str(onnx) + ".json", use_cuda=False)
        return v

    def _run_piper(self, job, ctx, h):
        from piper import SynthesisConfig

        p = job.params
        text = line_of(job)
        vid = p.get("voice") or job.model.get("features", {}).get("default_voice") or next(iter(PIPER_VOICES))
        if THAI.search(text) and not vid.startswith("th_"):
            raise InputError("This Piper voice is English. Pick th_TH-tsync2-medium (model 'Piper Thai') for Thai text.")
        v = self._piper_voice(h, vid)
        sr = v.config.sample_rate
        cfg = SynthesisConfig(speaker_id=int(p["speaker"]) if v.config.num_speakers > 1 and p.get("speaker") is not None else None,
                              length_scale=1.0 / max(0.25, float(p.get("speed", 1.0))),
                              noise_scale=float(p.get("noise_scale", 0.667)), noise_w_scale=float(p.get("noise_w", 0.8)))
        n = max(1, int(p.get("batch", 1)))
        for i in range(n):                      # piper has no seed: the noise comes from onnxruntime (noise_scale 0 = repeatable)
            ctx.check()
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}")
            chunks = []
            for ch in v.synthesize(text, syn_config=cfg):
                ctx.check()
                chunks.append(np.asarray(ch.audio_float_array, dtype=np.float32))
                chunks.append(np.zeros(int(0.15 * sr), dtype=np.float32))
            wav = tidy_voice(np.concatenate(chunks[:-1]), sr)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, sr)
            ctx.output(f, "audio", seed=-1, index=i, name=name_of(text, i, n), final_prompt=text, voice=vid, **fm)
        return {"meta": {"voice": vid}}

    # --- MeloTTS
    def _run_melo(self, job, ctx, tts):
        p = job.params
        text = line_of(job)
        if THAI.search(text):
            raise InputError("MeloTTS-English has no Thai.")
        spk = p.get("speaker") or "EN-US"
        spk_ids = tts.hps.data.spk2id
        if spk not in spk_ids:
            raise InputError(f"unknown MeloTTS speaker {spk}; choose one of {list(spk_ids.keys())}")
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        for i in range(n):
            ctx.check()
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}")
            seed_all(s0 + i)
            wav = tts.tts_to_file(text, spk_ids[spk], None, speed=float(p.get("speed", 1.0)), quiet=True,
                                  sdp_ratio=float(p.get("sdp_ratio", 0.2)), noise_scale=float(p.get("noise_scale", 0.6)),
                                  noise_scale_w=float(p.get("noise_w", 0.8)))
            sr = int(tts.hps.data.sampling_rate)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, tidy_voice(np.asarray(wav, dtype=np.float32), sr), sr)
            ctx.output(f, "audio", seed=s0 + i, index=i, name=name_of(text, i, n), final_prompt=text, voice=spk, **fm)
        return {"meta": {"speaker": spk, "seed": s0}}

    # --- Parler-TTS (tts with a named speaker, voice_design with a free description)
    def _run_parler(self, job, ctx, h):
        import torch

        p = job.params
        text = line_of(job)
        if THAI.search(text):
            raise InputError("Parler-TTS is English only.")
        desc = (p.get("instruction") or p.get("instruct") or "").strip()
        spk = (p.get("speaker") or "").strip()
        if job.task == "voice_design" and not desc:
            raise InputError("Describe the voice first (age, gender, pitch, pace, accent, recording quality).")
        named = job.task == "tts" and spk and spk.lower() != "none"
        if named:
            if not desc:
                desc = f"{spk}'s voice is clear and expressive with a moderate speed and pitch, in a very close recording that has almost no background noise."
            elif not desc.lower().startswith(spk.lower()):
                desc = f"{spk}'s voice. {desc}"
        elif not desc:
            desc = "A clear, natural voice with a moderate speed and pitch, in a very close recording that has almost no background noise."
        m, tok, dev = h["model"], h["tok"], h["dev"]
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        sr = int(m.config.sampling_rate)
        gen_kw = dict(do_sample=True, temperature=float(p.get("temperature", 1.0)))
        for i in range(n):
            ctx.check()
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}")
            seed_all(s0 + i)
            ids = h["dtok"](desc, return_tensors="pt").to(dev)
            pids = tok(text, return_tensors="pt").to(dev)
            with torch.inference_mode():
                out = m.generate(input_ids=ids.input_ids, attention_mask=ids.attention_mask,
                                 prompt_input_ids=pids.input_ids, prompt_attention_mask=pids.attention_mask, **gen_kw)
            wav = out.float().cpu().numpy().reshape(-1)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, tidy_voice(wav, sr), sr)
            ctx.output(f, "audio", seed=s0 + i, index=i, name=name_of(text, i, n), final_prompt=f"{desc} || {text}", **fm)
        return {"meta": {"description": desc, "seed": s0}}

    # --- StyleTTS 2 (tts: diffusion-sampled style; voice_clone: style from a reference clip)
    def _run_styletts2(self, job, ctx, st):
        import soundfile as sf

        p = job.params
        text = line_of(job)
        if THAI.search(text):
            raise InputError("StyleTTS 2 (LibriTTS) is English only.")
        ref, ref_wav, ref_sr = None, None, None
        if job.task == "voice_clone":
            ref = ref_clip(job)
            if not ref:
                raise InputError("Pick a reference clip to clone.")
            y, ref_sr = sf.read(ref["path"], dtype="float32", always_2d=True)
            ref_wav = y.mean(axis=1)[: ref_sr * 30]
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        for i in range(n):
            ctx.check()
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}")
            seed_all(s0 + i)
            wav = st.synth(text, ref_wav=ref_wav, ref_sr=ref_sr, voice=p.get("voice"), alpha=float(p.get("alpha", 0.3)), beta=float(p.get("beta", 0.7)),
                           diffusion_steps=int(p.get("steps", 10)), embedding_scale=float(p.get("embedding_scale", 1.0)),
                           cancel=ctx.check)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, tidy_voice(wav, 24000), 24000)
            extra = {"ref_asset": ref.get("asset_id")} if ref else {"voice": p.get("voice")}
            ctx.output(f, "audio", seed=s0 + i, index=i, name=name_of(text, i, n), final_prompt=text, **extra, **fm)
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    serve(TTSLightBackend())
