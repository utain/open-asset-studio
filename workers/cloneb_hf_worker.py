"""Dialogue / expressive TTS backend (envs/cloneb_hf: torch 2.9.1+cu130, transformers 5.22.1, diffusers, dia2).

  dia-1.6b     task tts / voice_clone   Nari Dia 1.6B (0626), transformers-native. [S1]/[S2] tags, (laughs) etc.
  dia2-2b      task tts / voice_clone   Nari Dia2 2B, streaming dialogue model (`dia2` package, vendor/dia2).
                                        Voice clone = audio prefix for speaker 1 (+ transcript via Whisper).
  bark         task tts                 suno/bark (transformers BarkModel): [laughs] [sighs] ♪ music ♪, 100+ voice presets
  vibevoice-*  task tts / voice_clone   Microsoft VibeVoice 1.5B / 7B (transformers-native `-hf` conversion):
                                        long-form, up to 4 speakers ("Speaker 1: ..." lines), optional voice per speaker
  csm-1b       task tts / voice_clone   Sesame CSM-1B. GATED repo: blocked (no HF login), code path untested.
"""
from __future__ import annotations

import importlib.util
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

from studio_worker import Backend, Cancelled, InputError, serve

ROOT = Path(__file__).resolve().parents[1]
DIA2_SRC = ROOT / "vendor" / "dia2"


def seed0(p):
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def set_seeds(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def snap(repo):
    """Local snapshot dir of a cached repo. (snapshot_download(local_files_only=True) refuses partial snapshots, and we
    deliberately download only the files a model needs.)"""
    from huggingface_hub.constants import HF_HUB_CACHE

    root = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    ref = root / "refs" / "main"
    if not ref.exists():
        raise InputError(f"{repo} is not downloaded (hf download {repo})")
    return str(root / "snapshots" / ref.read_text().strip())


def name_of(text, i, n):
    t = re.sub(r"\s+", " ", text).strip()
    return f"{t[:48]}{'…' if len(t) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def save_wav(path, wav, sr):
    """float mono/stereo -> 24-bit wav. Rejects silent / NaN output. Returns meta."""
    import soundfile as sf

    w = np.nan_to_num(np.asarray(wav, dtype=np.float32))
    w = w.reshape(-1) if w.ndim > 1 and 1 in w.shape else w
    if w.size < int(0.2 * sr) or float(np.abs(w).max()) < 1e-4:
        raise RuntimeError("the model returned silent or (almost) empty audio")
    meta = {"sample_rate": int(sr)}
    peak = float(np.abs(w).max())
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(w.shape[0] / sr, 3)
    return meta


def load_ref(path, sr, seconds=None):
    """mono float32 at `sr` from any audio file."""
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True, duration=seconds)
    if y.size < sr // 2:
        raise InputError("The reference clip is too short (need at least a second of speech).")
    return y


def cancel_criteria(ctx, total=None, stage="generating"):
    """A transformers StoppingCriteria that reports progress and raises Cancelled."""
    from transformers import StoppingCriteria

    class _C(StoppingCriteria):
        def __init__(self):
            self.n = 0

        def __call__(self, input_ids, scores, **kw):
            self.n += 1
            ctx.check()
            if total and self.n % 25 == 0:
                ctx.progress(min(self.n, total), total, stage, f"{self.n}/{total} tokens")
            return False

    return _C()


def ref_input(job, role="ref"):
    return job.input(role=role) or (next((i for i in job.inputs if i.get("type") in ("voice", "audio") and i.get("role") in (None, "", role)), None)
                                    if role == "ref" else None)


def clean_text(job):
    t = (job.params.get("text") or job.params.get("prompt") or "").strip()
    if not t:
        raise InputError("Write the line to speak first.")
    return t


class HFBackend(Backend):
    name = "cloneb_hf"
    max_loaded = 1

    def probe(self, model):
        need = ["torch", "transformers", "soundfile", "librosa"]
        if model["id"].startswith("vibevoice"):
            need.append("diffusers")
        if model["id"].startswith("dia2"):
            need += ["dia2", "sphn"]
        missing = [m for m in need if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    # ------------------------------------------------------------------ load
    def load(self, model, ctx):
        import torch

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        mid = model["id"]
        repo = model["weights"][0]["hf"]
        if mid.startswith("dia2"):
            return self.load_dia2(model, repo)
        local = snap(repo)
        if mid == "dia-1.6b":
            from transformers import AutoProcessor, DiaForConditionalGeneration

            proc = AutoProcessor.from_pretrained(local)
            m = DiaForConditionalGeneration.from_pretrained(local, dtype=torch.bfloat16, device_map="cuda:0").eval()
            return {"kind": "dia", "proc": proc, "model": m}
        if mid.startswith("bark"):
            from transformers import AutoProcessor, BarkModel

            proc = AutoProcessor.from_pretrained(local)
            m = BarkModel.from_pretrained(local, dtype=torch.float16).to("cuda:0").eval()
            return {"kind": "bark", "proc": proc, "model": m}
        if mid.startswith("vibevoice"):
            from transformers import AutoModelForTextToWaveform, AutoProcessor

            proc = AutoProcessor.from_pretrained(local)
            m = AutoModelForTextToWaveform.from_pretrained(local, dtype=torch.bfloat16, device_map="cuda:0").eval()
            return {"kind": "vibevoice", "proc": proc, "model": m}
        if mid == "csm-1b":
            from transformers import AutoProcessor, CsmForConditionalGeneration

            proc = AutoProcessor.from_pretrained(local)
            m = CsmForConditionalGeneration.from_pretrained(local, dtype=torch.bfloat16, device_map="cuda:0").eval()
            return {"kind": "csm", "proc": proc, "model": m}
        raise InputError(f"unknown model {mid}")

    def load_dia2(self, model, repo):
        import torch

        if str(DIA2_SRC) not in sys.path:
            sys.path.insert(0, str(DIA2_SRC))
        from dia2 import Dia2
        from dia2.runtime import voice_clone as vc

        def transcribe_words(audio_path, device, language=None):
            """Upstream hard-codes openai/whisper-large-v3 (3 GB, not cached): use the cached large-v3-turbo instead."""
            import gc

            import whisper_timestamped as wts

            wm = wts.load_model("openai/whisper-large-v3-turbo", device=str(device))
            try:
                res = wts.transcribe(wm, audio_path, language=language)
            finally:
                del wm
                gc.collect()
                torch.cuda.empty_cache()
            words = []
            for seg in res.get("segments", []):
                for w in seg.get("words", []):
                    t = (w.get("text") or w.get("word") or "").strip()
                    if t:
                        words.append(vc.WhisperWord(text=t, start=float(w.get("start", 0.0)), end=float(w.get("end", 0.0))))
            return words

        vc.transcribe_words = transcribe_words

        from dia2.audio import codec as _codec

        def mimi_decode(self, codes):
            """transformers 5: MimiModel.decode(return_dict=False) returns a 1-tuple (upstream expects 2)."""
            codes = codes.to(self.device)
            with torch.inference_mode():
                out = self.model.decode(codes, return_dict=True)
                return torch.clamp(out.audio_values, -1.0, 1.0)

        _codec.MimiCodec.decode = mimi_decode
        dia = Dia2.from_repo(repo, device="cuda", dtype="bfloat16")
        dia._ensure_runtime()
        return {"kind": "dia2", "dia": dia}

    def unload(self, model_id):
        h = self.models.pop(model_id, None)
        if h and h.get("kind") == "dia2":
            try:
                h["dia"].close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ run
    def run(self, job, ctx):
        h = self.handle(job.model_id)
        fn = {"dia": self.run_dia, "dia2": self.run_dia2, "bark": self.run_bark,
              "vibevoice": self.run_vibevoice, "csm": self.run_csm}[h["kind"]]
        return fn(job, ctx, h)

    def emit(self, job, ctx, wav, sr, text, i, n, seed, **extra):
        f = job.out_dir / f"take_{i + 1}.wav"
        meta = save_wav(f, wav, sr)
        ctx.output(f, "audio", seed=seed, index=i, name=name_of(text, i, n), final_prompt=text, **extra, **meta)

    # ---- Dia 1.6B
    def run_dia(self, job, ctx, h):
        import torch

        p, proc, m = job.params, h["proc"], h["model"]
        text = clean_text(job)
        if "[S1]" not in text and "[S2]" not in text:
            text = "[S1] " + text
        sr_codec = proc.feature_extractor.sampling_rate
        audio, extra = None, {}
        if job.task == "voice_clone":
            ref = ref_input(job)
            if not ref:
                raise InputError("Pick a reference clip (5-20 s of clean speech) to clone.")
            rt = (p.get("ref_text") or "").strip() or (((ref.get("meta") or {}).get("x_voice") or {}).get("ref_text") or "").strip()
            if not rt:
                raise InputError("Dia clones from a transcript of the reference clip: fill 'Reference transcript' (use [S1]/[S2] tags).")
            if "[S1]" not in rt and "[S2]" not in rt:
                rt = "[S1] " + rt
            audio = load_ref(ref["path"], sr_codec, 20)
            text = rt.rstrip() + "\n" + text
            extra = {"ref_asset": ref.get("asset_id")}
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        max_tokens = int(float(p.get("seconds", 30)) * 86)
        for i in range(n):
            ctx.check()
            set_seeds(s0 + i)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}")
            kw = dict(text=[text], padding=True, return_tensors="pt")
            if audio is not None:
                kw["audio"] = audio
            inputs = proc(**kw).to(m.device)
            plen = proc.get_audio_prompt_len(inputs["decoder_attention_mask"]) if audio is not None else None
            with torch.inference_mode():
                out = m.generate(**inputs, max_new_tokens=max_tokens, guidance_scale=float(p.get("guidance_scale", 3.0)),
                                 temperature=float(p.get("temperature", 1.8)), top_p=float(p.get("top_p", 0.9)),
                                 top_k=int(p.get("top_k", 45)), do_sample=True,
                                 stopping_criteria=[cancel_criteria(ctx, max_tokens, "speaking")])
            dec = proc.batch_decode(out, audio_prompt_len=plen) if plen is not None else proc.batch_decode(out)
            wav = dec[0] if isinstance(dec, (list, tuple)) else dec
            wav = wav.detach().float().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
            self.emit(job, ctx, wav, sr_codec, text if audio is None else p.get("text", text), i, n, s0 + i, **extra)
        return {"meta": {"seed": s0}}

    # ---- Dia2
    def run_dia2(self, job, ctx, h):
        from dia2 import GenerationConfig, SamplingConfig

        p, dia = job.params, h["dia"]
        text = clean_text(job)
        if "[S1]" not in text and "[S2]" not in text:
            text = "[S1] " + text
        refs = {}
        if job.task == "voice_clone":
            r1, r2 = ref_input(job), job.input(role="ref2")
            if not r1:
                raise InputError("Pick a reference clip (role ref = speaker 1) to clone. Dia2 conditions on a spoken prefix.")
            refs["prefix_speaker_1"] = r1["path"]
            if r2:
                refs["prefix_speaker_2"] = r2["path"]
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        for i in range(n):
            ctx.check()
            set_seeds(s0 + i)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}", interruptible=False)
            cfg = GenerationConfig(cfg_scale=float(p.get("cfg_scale", 2.0)),
                                   audio=SamplingConfig(temperature=float(p.get("temperature", 0.8)), top_k=int(p.get("top_k", 50))),
                                   use_cuda_graph=False)
            res = dia.generate(text, config=cfg, verbose=False, **refs)
            wav = res.waveform.detach().float().cpu().numpy()
            self.emit(job, ctx, wav, res.sample_rate, p.get("text", text), i, n, s0 + i, **({"ref_asset": job.input(role="ref")["asset_id"]} if refs else {}))
        return {"meta": {"seed": s0}}

    # ---- Bark
    def run_bark(self, job, ctx, h):
        import torch

        p, proc, m = job.params, h["proc"], h["model"]
        text = clean_text(job)
        voice = p.get("voice") or "v2/en_speaker_6"
        voice = None if voice == "random" else voice
        sr = m.generation_config.sample_rate
        # Bark keeps ~13 s of audio per call: split long lines at sentence ends, keep tags with their sentence
        parts = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
        chunks, cur = [], ""
        for s in parts:
            if cur and len(cur) + len(s) > int(p.get("chunk_chars", 190)):
                chunks.append(cur)
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            chunks.append(cur)
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        gap = np.zeros(int(sr * float(p.get("pause_s", 0.25))), dtype=np.float32)
        for i in range(n):
            set_seeds(s0 + i)
            wavs = []
            for k, c in enumerate(chunks):
                ctx.check()
                ctx.progress(i * len(chunks) + k, n * len(chunks), "speaking", f"take {i + 1}/{n}, part {k + 1}/{len(chunks)}", interruptible=False)
                inputs = proc(c, voice_preset=voice, return_tensors="pt").to(m.device)
                with torch.inference_mode():
                    out = m.generate(**inputs, do_sample=True, semantic_temperature=float(p.get("text_temp", 0.7)),
                                     coarse_temperature=float(p.get("waveform_temp", 0.7)), fine_temperature=0.5,
                                     pad_token_id=proc.tokenizer.pad_token_id)
                wavs += [out[0].detach().float().cpu().numpy(), gap]
            self.emit(job, ctx, np.concatenate(wavs[:-1]), sr, text, i, n, s0 + i, voice=voice or "random", chunks=len(chunks))
        return {"meta": {"seed": s0}}

    # ---- VibeVoice
    SPK = re.compile(r"^\s*(?:speaker\s*)?\[?([1-4])\]?\s*[:：]\s*(.*)$", re.I)

    def parse_script(self, text):
        """'Speaker 2: hi' / '2: hi' / '[2]: hi' lines -> [(role "0".."3", text)]; plain lines continue the previous speaker."""
        turns = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            mt = self.SPK.match(line)
            if mt:
                turns.append([str(int(mt.group(1)) - 1), mt.group(2).strip()])
            elif turns:
                turns[-1][1] += " " + line
            else:
                turns.append(["0", line])
        return [(r, t) for r, t in turns if t]

    def run_vibevoice(self, job, ctx, h):
        import torch

        p, proc, m = job.params, h["proc"], h["model"]
        text = clean_text(job)
        turns = self.parse_script(text)
        sr_in = proc.feature_extractor.sampling_rate
        refs = {}
        for k in range(4):
            a = ref_input(job) if k == 0 else job.input(role=f"ref{k + 1}")
            if a:
                refs[str(k)] = a
        if job.task == "voice_clone" and "0" not in refs:
            raise InputError("Pick a reference clip (role ref = speaker 1) to clone, or use the plain text-to-speech task.")
        conv, seen = [], set()
        for role, t in turns:
            content = [{"type": "text", "text": t}]
            if role in refs and role not in seen:
                content.append({"type": "audio", "audio": load_ref(refs[role]["path"], sr_in, 30)})
                seen.add(role)
            conv.append({"role": role, "content": content})
        for role in refs:   # a reference for a speaker who never speaks would shift the voice mapping
            if role not in seen:
                raise InputError(f"Speaker {int(role) + 1} has a reference clip but no lines in the script.")
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        gen = {"cfg_scale": None}
        for i in range(n):
            ctx.check()
            set_seeds(s0 + i)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}")
            inputs = proc.apply_chat_template(conv, return_dict=True, tokenize=True, add_generation_prompt=True).to(m.device, m.dtype)
            kw = dict(guidance_scale=float(p.get("cfg_scale", 1.3)), num_diffusion_steps=int(p.get("steps", 10)),
                      stopping_criteria=[cancel_criteria(ctx, None, "speaking")])
            if p.get("max_tokens"):
                kw["max_new_tokens"] = int(p["max_tokens"])
            with torch.inference_mode():
                audio = m.generate(**inputs, **kw)
            wav = audio[0] if isinstance(audio, (list, tuple)) else audio
            wav = wav.detach().float().cpu().numpy().squeeze()
            self.emit(job, ctx, wav, sr_in, text, i, n, s0 + i, speakers=len({r for r, _ in turns}),
                      voices=sorted(int(k) + 1 for k in refs))
        return {"meta": {"seed": s0}}

    # ---- Sesame CSM (gated: untested)
    def run_csm(self, job, ctx, h):
        p, proc, m = job.params, h["proc"], h["model"]
        text = clean_text(job)
        spk = str(int(p.get("speaker", 0)))
        conv = []
        if job.task == "voice_clone":
            ref = ref_input(job)
            rt = (p.get("ref_text") or "").strip()
            if not ref or not rt:
                raise InputError("CSM clones from a context: pick a reference clip and fill its transcript.")
            conv.append({"role": spk, "content": [{"type": "text", "text": rt}, {"type": "audio", "audio": load_ref(ref["path"], 24000, 20)}]})
        conv.append({"role": spk, "content": [{"type": "text", "text": text}]})
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        for i in range(n):
            ctx.check()
            set_seeds(s0 + i)
            inputs = proc.apply_chat_template(conv, tokenize=True, return_dict=True).to(m.device)
            audio = m.generate(**inputs, output_audio=True, max_new_tokens=int(float(p.get("seconds", 20)) * 12.5))
            wav = audio[0].detach().float().cpu().numpy().squeeze()
            self.emit(job, ctx, wav, 24000, text, i, n, s0 + i)
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(HFBackend())
