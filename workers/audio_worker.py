"""Audio backend (envs/audio): Stable Audio 3 (small-sfx, medium + extend), MOSS-SoundEffect v2,
MiniMax-Music3, OmniVoice (+ Thai fine-tune), F5-TTS, Orpheus 3B, and the QA helpers CLAP + Whisper.

Loaders follow docs/research/audio_snippets/*.py (the versions measured on this DGX Spark). Every take is written
as a 24-bit WAV; anything that reaches full scale is brought to -1 dBFS (SA3 music, F5) and noted in the asset.
QA models (clap-htsat, whisper-v3-turbo) write no files: they return scores in `meta.scores` for the sound plugin.
"""
from __future__ import annotations

import importlib.util
import math
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # MOSS-SFX: no torch.compile on this box

import numpy as np

from studio_worker import Backend, Cancelled, InputError, serve

THAI = re.compile(r"[฀-๿]")
CJK = re.compile(r"[぀-ヿ一-鿿]")


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
    """Trim leading / trailing silence of a speech take (keeps `pad_s` around the voice)."""
    mono = np.abs(w).max(axis=1) if w.ndim == 2 else np.abs(w)
    idx = np.flatnonzero(mono > 10 ** (floor_db / 20))
    if idx.size == 0:
        return w
    pad = int(pad_s * sr)
    a, b = max(0, idx[0] - pad), min(len(mono), idx[-1] + pad + 1)
    return w[a:b]


def save_wav(path: Path, wav, sr: int) -> dict:
    """Write float audio (T,) or (T, C) as 24-bit WAV. Brings overs to -1 dBFS. Returns file meta."""
    import soundfile as sf

    w = np.asarray(wav, dtype=np.float32)
    if w.ndim == 2 and w.shape[0] <= 8 and w.shape[0] < w.shape[1]:
        w = w.T                                     # (C, T) -> (T, C)
    if w.ndim == 2 and w.shape[1] == 1:
        w = w[:, 0]
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if w.size == 0 or len(w) < int(0.02 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    peak = float(np.abs(w).max())
    meta = {"sample_rate": int(sr)}
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def read_mono(path: str, sr: int | None = None) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, s = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr and s != sr:
        import librosa

        y = librosa.resample(y, orig_sr=s, target_sr=sr)
        s = sr
    return y, s


def ref_input(job) -> dict | None:
    for role in ("ref", "voice", "reference"):
        r = job.input(role=role)
        if r:
            return r
    return next((i for i in job.inputs if i.get("type") in ("voice", "audio")), None)


def ref_text_of(job, ref: dict | None) -> str:
    t = (job.params.get("ref_text") or "").strip()
    if not t and ref:
        t = (((ref.get("meta") or {}).get("x_voice") or {}).get("ref_text") or "").strip()
    return t


def text_of(job) -> str:
    t = (job.params.get("text") or job.params.get("prompt") or "").strip()
    if not t:
        raise InputError("Write the line to speak first.")
    return t


def take_name(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def words(s: str) -> list[str]:
    return re.findall(r"\w+", (s or "").lower())


def recall(ref: str, hyp: str, lang: str | None) -> float:
    """Share of the script's words (characters for Thai / CJK) found in the transcript, in order-insensitive form."""
    if lang in ("th", "zh", "ja", "ko") or THAI.search(ref or "") or CJK.search(ref or ""):
        r = [c for c in "".join(words(ref))]
        h = list("".join(words(hyp)))
    else:
        r, h = words(ref), words(hyp)
    if not r:
        return 0.0
    pool: dict[str, int] = {}
    for x in h:
        pool[x] = pool.get(x, 0) + 1
    hit = 0
    for x in r:
        if pool.get(x):
            pool[x] -= 1
            hit += 1
    return round(hit / len(r), 3)


def tqdm_factory(ctx, stage: str, label: str):
    """A tqdm stand-in: reports progress to the core and raises Cancelled between steps."""
    class _T:
        def __init__(self, iterable=None, total=None, **_kw):
            self.it = iterable
            self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else 0)
            self.n = 0

        def __iter__(self):
            for x in self.it:
                ctx.check()
                yield x
                self.update()

        def update(self, k=1):
            self.n += k
            ctx.check()
            if self.total:
                ctx.progress(self.n, self.total, stage, f"{label}{self.n}/{self.total}")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def close(self):
            pass

        def set_description(self, *a, **k):
            pass

        def set_postfix(self, *a, **k):
            pass

        def refresh(self, *a, **k):
            pass
    return _T


def hf_repo(model: dict, k: int = 0) -> str:
    return model["weights"][k]["hf"]


# ------------------------------------------------------------------ backend
class AudioBackend(Backend):
    name = "audio"
    max_loaded = 2

    NEEDS = {"sa3": ["stable_audio_tools"], "moss-sfx": ["moss_soundeffect_v2"], "minimax": ["diffusers"],
             "omnivoice": ["omnivoice"], "f5": ["f5_tts", "vocos"], "orpheus": ["snac", "transformers"],
             "clap": ["transformers", "librosa"], "whisper": ["transformers", "librosa"]}

    @staticmethod
    def family(mid: str) -> str:
        for k in ("sa3", "moss-sfx", "minimax", "omnivoice", "f5", "orpheus", "clap", "whisper"):
            if mid.startswith(k):
                return k
        raise InputError(f"The audio backend does not know model '{mid}'.")

    def probe(self, model):
        fam = self.family(model["id"])
        missing = [m for m in self.NEEDS[fam] + ["torch", "soundfile"] if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    # ---------------------------------------------------------------- load
    def load(self, model, ctx):
        import torch

        fam = self.family(model["id"])
        repo = hf_repo(model)
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        if fam == "sa3":
            from stable_audio_tools import get_pretrained_model

            m, cfg = get_pretrained_model(repo)
            m = m.to("cuda").eval().requires_grad_(False).to(torch.float16)
            return {"model": m, "cfg": cfg}
        if fam == "moss-sfx":
            from moss_soundeffect_v2 import MossSoundEffectPipeline

            return MossSoundEffectPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16, device="cuda")
        if fam == "minimax":
            from diffusers import ModularPipeline

            pipe = ModularPipeline.from_pretrained(repo)
            # offline: local_files_only (sharded components call model_info otherwise); device_map: no CPU copy
            pipe.load_components(dtype=torch.bfloat16, local_files_only=True, device_map="cuda")
            pipe.to("cuda")
            return pipe
        if fam == "omnivoice":
            from omnivoice import OmniVoice

            return OmniVoice.from_pretrained(repo, device_map="cuda:0", dtype=torch.float16)
        if fam == "f5":
            import soundfile as sf
            import torchaudio

            def _sf_load(path, *a, **k):  # torchaudio 2.9 load() needs torchcodec (no aarch64 wheel)
                w, sr = sf.read(path, dtype="float32", always_2d=True)
                return torch.from_numpy(w.T.copy()), sr
            torchaudio.load = _sf_load
            from f5_tts.api import F5TTS

            return F5TTS(model="F5TTS_v1_Base")
        if fam == "orpheus":
            from snac import SNAC
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(repo)
            lm = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda").eval()
            snac = SNAC.from_pretrained(hf_repo(model, 1)).to("cuda").eval()
            return {"tok": tok, "lm": lm, "snac": snac}
        if fam == "clap":
            from transformers import ClapModel, ClapProcessor

            return {"model": ClapModel.from_pretrained(repo).to("cuda").eval(), "proc": ClapProcessor.from_pretrained(repo)}
        if fam == "whisper":
            from transformers import pipeline

            return pipeline("automatic-speech-recognition", repo, dtype=torch.float16, device="cuda")
        raise InputError(f"unknown model {model['id']}")

    # ----------------------------------------------------------------- run
    def run(self, job, ctx):
        fam = self.family(job.model_id)
        return getattr(self, "_run_" + fam.replace("-", "_"))(job, ctx, self.handle(job.model_id))

    # --- Stable Audio 3 (small-sfx / medium; medium can extend a clip by inpainting its continuation)
    def _run_sa3(self, job, ctx, h):
        import torch
        from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

        p = job.params
        m, cfg = h["model"], h["cfg"]
        sr, max_samples = int(cfg["sample_rate"]), int(cfg["sample_size"])
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Write a prompt first (✨ Improve turns a short idea into the Stable Audio format).")
        seconds = float(p.get("seconds", 5))
        steps = int(p.get("steps", 8))
        cfg_scale = float(p.get("cfg_scale", 1.0))
        neg = (p.get("negative") or "").strip()
        n = max(1, int(p.get("batch", 1)))
        s0 = seed0(p)
        src = job.input(role="source")
        inpaint = {}
        head_s = 0.0
        if src or p.get("mode") == "extend":
            if not (job.model.get("features") or {}).get("inpaint"):
                raise InputError(f"{job.model['name']} cannot extend audio; use Stable Audio 3 Medium.")
            if not src:
                raise InputError("Pick the clip to extend.")
            import soundfile as sf

            y, ysr = sf.read(src["path"], dtype="float32", always_2d=True)
            if ysr != sr:
                import librosa

                y = librosa.resample(y.T, orig_sr=ysr, target_sr=sr).T
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            y = y[:, :2]
            head_s = len(y) / sr
            ext = float(p.get("extend_seconds") or seconds)
            seconds = head_s + ext
            if seconds > max_samples / sr - 7:
                raise InputError(f"Clip + extension is too long for this model ({max_samples / sr - 7:.0f} s max).")
            inpaint = {"inpaint_audio": (sr, torch.from_numpy(y.T.copy()).to("cuda", torch.float16)),
                       "inpaint_mask_start_seconds": head_s, "inpaint_mask_end_seconds": seconds}
            ctx.log(f"extending {src['name']} ({head_s:.1f} s) by {ext:.1f} s")
        if neg and cfg_scale <= 1.0:
            ctx.log("negative prompt is only used with CFG > 1 (ARC-distilled models run at CFG 1)")
        label = prompt if not src else f"{src['name']} (extended)"
        for i in range(n):
            ctx.check()
            seed = s0 + i
            cond = [{"prompt": prompt, "seconds_total": seconds}]
            kw = dict(steps=steps, cfg_scale=cfg_scale, conditioning=cond, sample_size=max_samples,
                      sampler_type="pingpong", seed=seed, device="cuda", **inpaint)
            if neg and cfg_scale > 1.0:
                kw["negative_conditioning"] = [{"prompt": neg, "seconds_total": seconds}]

            def cb(d, i=i):
                ctx.check()
                ctx.progress(i * steps + d.get("i", 0) + 1, n * steps, "sampling", f"take {i + 1}/{n}: step {d.get('i', 0) + 1}/{steps}")
            with torch.inference_mode():
                try:
                    audio = generate_diffusion_cond_inpaint(m, callback=cb, **kw)
                except TypeError as e:
                    if "callback" not in str(e):
                        raise
                    audio = generate_diffusion_cond_inpaint(m, **kw)
            wav = audio[0, :, : int(seconds * sr)].float().clamp(-1, 1).cpu().numpy().T
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, sr)
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(label, i, n), final_prompt=prompt, **fm,
                       **({"extended_from_s": round(head_s, 3)} if src else {}))
        return {"meta": {"sampler": "pingpong", "seed": s0}}

    # --- MOSS-SoundEffect v2
    def _run_moss_sfx(self, job, ctx, pipe):
        p = job.params
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Write a caption of the sound first.")
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        steps = int(p.get("steps", 100))
        for i in range(n):
            ctx.check()
            seed = s0 + i
            audio = pipe(prompt=prompt, seconds=float(p.get("seconds", 5)), num_inference_steps=steps,
                         cfg_scale=float(p.get("cfg_scale", 4.0)), negative_prompt=(p.get("negative") or ""), seed=seed,
                         progress_bar_cmd=tqdm_factory(ctx, "sampling", f"take {i + 1}/{n}: step "))
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, audio[0].detach().float().cpu().numpy().T, pipe.sample_rate)
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(prompt, i, n), final_prompt=prompt, **fm)
        return {"meta": {"seed": s0}}

    # --- MiniMax-Music3 (songs with vocals; instrumental with [instrumental] sections)
    def _run_minimax(self, job, ctx, pipe):
        import torch
        import diffusers.modular_pipelines.modular_pipeline as mp
        from diffusers.guiders import ClassifierFreeGuidance

        p = job.params
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Describe the song (genre, BPM, key, vocals, arrangement) first.")
        lyrics = (p.get("lyrics") or "").strip()
        if not lyrics:
            if job.task == "song":
                raise InputError("Write the lyrics with section tags ([verse], [chorus] ...), or let ✨ Improve draft them.")
            lyrics = "[intro]\n[instrumental]\n[outro]"
        # section tags must stand alone on their line (text after a tag on the same line is dropped by the model)
        lyrics = re.sub(r"^\s*(\[[a-z\- ]+\])\s*(\S.*)$", r"\1\n\2", lyrics, flags=re.I | re.M)
        seconds = float(p.get("seconds", 60))
        steps = int(p.get("steps", 30))
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=float(p.get("cfg_scale", 1.7))))
        frames_total = int(25 * seconds)
        state = {"frames": 0, "take": 0}

        def on_frame(_mod, _inp, _out):
            state["frames"] += 1
            if state["frames"] % 25 == 0:
                ctx.check()
                ctx.progress(state["frames"], frames_total, "composing",
                             f"take {state['take'] + 1}/{n}: composing {state['frames'] / 25:.0f}/{seconds:.0f} s (stage 1 of 2)")
        lm = getattr(pipe, "language_model", None)
        hook = lm.lm_head.register_forward_hook(on_frame) if lm is not None and hasattr(lm, "lm_head") else None
        old_tqdm = mp.tqdm
        try:
            for i in range(n):
                ctx.check()
                state.update(frames=0, take=i)
                seed = s0 + i
                mp.tqdm = tqdm_factory(ctx, "rendering", f"take {i + 1}/{n}: rendering audio (stage 2 of 2) ")
                audio = pipe(prompt=prompt, lyrics=lyrics, audio_duration=seconds, num_inference_steps=steps,
                             generator=torch.Generator("cuda").manual_seed(seed), output="audios")[0]
                wav = audio.float().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)
                f = job.out_dir / f"take_{i + 1}.wav"
                fm = save_wav(f, wav, pipe.sampling_rate)
                ctx.output(f, "audio", seed=seed, index=i, name=take_name(prompt.split(".")[0], i, n), final_prompt=prompt,
                           lyrics=lyrics, **fm)
        finally:
            mp.tqdm = old_tqdm
            if hook is not None:
                hook.remove()
        return {"meta": {"seed": s0}}

    # --- OmniVoice / omnivoice-thai: clone (ref + transcript), design (attribute list), auto voice
    def _run_omnivoice(self, job, ctx, model):
        p = job.params
        text = text_of(job)
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        lang = p.get("language") or "auto"
        lang = None if lang == "auto" else lang
        if lang is None and THAI.search(text):
            lang = "th"
        kw = dict(num_step=int(p.get("steps", 32)), guidance_scale=float(p.get("guidance", 2.0)))
        if p.get("speed") not in (None, 1, 1.0):
            kw["speed"] = float(p["speed"])
        extra, as_voice = {}, False
        if job.task == "voice_clone":
            ref = ref_input(job)
            if not ref:
                raise InputError("Pick a reference clip (a voice from the library or any speech take).")
            rt = ref_text_of(job, ref)
            if not rt:
                rt = self.transcribe(ref["path"], ctx)
            kw.update(ref_audio=ref["path"], ref_text=rt)
            extra["ref_asset"] = ref.get("asset_id")
        elif job.task == "voice_design":
            ins = (p.get("instruct") or p.get("prompt") or "").strip()
            if ins:
                kw["instruct"] = ins
            as_voice = True
            extra["design_prompt"] = ins
        for i in range(n):
            ctx.check()
            seed = s0 + i
            seed_all(seed)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}")
            audio = model.generate(text=text, language=lang, **kw)
            sr = int(model.sampling_rate)
            wav = tidy_voice(np.asarray(audio[0], dtype=np.float32), sr)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, sr)
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(("Voice: " + extra["design_prompt"]) if as_voice and extra.get("design_prompt") else text, i, n), final_prompt=text, as_voice=as_voice,
                       language=lang, **extra, **fm)
            ctx.progress(i + 1, n, "speaking", f"take {i + 1}/{n} done")
        return {"meta": {"seed": s0}}

    # --- F5-TTS (clone, EN/ZH)
    def _run_f5(self, job, ctx, tts):
        p = job.params
        text = text_of(job)
        ref = ref_input(job)
        if not ref:
            raise InputError("F5-TTS clones a voice: pick a reference clip (<= 12 s).")
        rt = ref_text_of(job, ref) or self.transcribe(ref["path"], ctx)
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        for i in range(n):
            ctx.check()
            seed = s0 + i
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}")
            wav, sr, _ = tts.infer(ref_file=ref["path"], ref_text=rt, gen_text=text, nfe_step=int(p.get("steps", 32)),
                                   cfg_strength=float(p.get("cfg_strength", 2.0)), sway_sampling_coef=-1,
                                   speed=float(p.get("speed", 1.0)), seed=seed)
            wav = tidy_voice(np.asarray(wav, dtype=np.float32), int(sr))
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, int(sr))
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(text, i, n), final_prompt=text,
                       ref_asset=ref.get("asset_id"), **fm)
        return {"meta": {"seed": s0}}

    # --- Orpheus 3B (English presets + <laugh>-style tags)
    def _run_orpheus(self, job, ctx, h):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        p = job.params
        text = text_of(job)
        voice = p.get("voice") or "leo"
        tok, lm, snac = h["tok"], h["lm"], h["snac"]
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        max_new = 1200

        class Watch(StoppingCriteria):
            def __init__(self, i):
                self.i, self.k = i, 0

            def __call__(self, input_ids, scores, **kw):
                self.k += 1
                if self.k % 40 == 0:
                    ctx.check()
                    ctx.progress(self.i * max_new + self.k, n * max_new, "speaking", f"take {self.i + 1}/{n}: {self.k} tokens")
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for i in range(n):
            ctx.check()
            seed = s0 + i
            seed_all(seed)
            ids = tok(f"{voice}: {text}", return_tensors="pt").input_ids
            ids = torch.cat([torch.tensor([[128259]]), ids, torch.tensor([[128009, 128260]])], dim=1).to("cuda")
            with torch.no_grad():
                gen = lm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=True,
                                  temperature=float(p.get("temperature", 0.6)), top_p=float(p.get("top_p", 0.95)),
                                  repetition_penalty=1.1, eos_token_id=128258, stopping_criteria=StoppingCriteriaList([Watch(i)]))
            g = gen[0, ids.shape[1]:].tolist()
            if 128257 in g:
                g = g[len(g) - g[::-1].index(128257):]
            codes = [t - 128266 for t in g if t != 128258 and t >= 128266]
            codes = codes[: len(codes) // 7 * 7]
            if not codes:
                raise RuntimeError("Orpheus produced no audio codes; try another seed")
            l1, l2, l3 = [], [], []
            for k in range(len(codes) // 7):
                c = codes[7 * k: 7 * k + 7]
                l1.append(c[0])
                l2 += [c[1] - 4096, c[4] - 4 * 4096]
                l3 += [c[2] - 2 * 4096, c[3] - 3 * 4096, c[5] - 5 * 4096, c[6] - 6 * 4096]
            with torch.no_grad():
                audio = snac.decode([torch.tensor(x, device="cuda").clamp(0, 4095).unsqueeze(0) for x in (l1, l2, l3)])
            wav = tidy_voice(audio[0, 0].float().cpu().numpy(), 24000)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, 24000)
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(text, i, n), final_prompt=f"{voice}: {text}", **fm)
        return {"meta": {"seed": s0}}

    # --- QA: CLAP prompt match (no files; meta.scores = {asset_id: {...}})
    def _run_clap(self, job, ctx, h):
        import torch

        p = job.params
        prompts = p.get("prompts") or {}
        default = (p.get("prompt") or "").strip()
        model, proc = h["model"], h["proc"]
        scores = {}
        items = [i for i in job.inputs if i.get("type") in ("audio", "voice")]
        if not items:
            raise InputError("Pick the takes to score.")
        for k, inp in enumerate(items):
            ctx.check()
            ctx.progress(k, len(items), "scoring", f"CLAP {k + 1}/{len(items)}")
            text = (prompts.get(inp["asset_id"]) or default or (inp.get("meta") or {}).get("final_prompt")
                    or (inp.get("meta") or {}).get("prompt") or "").strip()
            if not text:
                scores[inp["asset_id"]] = {"error": "no prompt to compare with"}
                continue
            y, _ = read_mono(inp["path"], 48000)
            y = y[: 48000 * 30]
            batch = proc(text=[text[:500], "silence", "speech"], audio=[y], sampling_rate=48000, return_tensors="pt", padding=True)
            batch = {k2: v.to("cuda") for k2, v in batch.items()}
            with torch.no_grad():
                o = model(**batch)
            sims = torch.nn.functional.cosine_similarity(o.audio_embeds, o.text_embeds).tolist()
            scores[inp["asset_id"]] = {"clap": round(sims[0], 4), "clap_silence": round(sims[1], 4),
                                       "clap_speech": round(sims[2], 4), "prompt": text[:300]}
        ctx.progress(len(items), len(items), "scoring", "done")
        return {"meta": {"scores": scores, "kind": "clap"}}

    # --- QA: Whisper transcript + recall against the script
    def _run_whisper(self, job, ctx, asr):
        p = job.params
        texts = p.get("texts") or {}
        lang_default = p.get("language") or "auto"
        items = [i for i in job.inputs if i.get("type") in ("audio", "voice")]
        if not items:
            raise InputError("Pick the voice takes to check.")
        scores = {}
        for k, inp in enumerate(items):
            ctx.check()
            ctx.progress(k, len(items), "transcribing", f"Whisper {k + 1}/{len(items)}")
            meta = inp.get("meta") or {}
            spec = texts.get(inp["asset_id"]) or {}
            script = (spec.get("text") if isinstance(spec, dict) else spec) or meta.get("final_prompt") or ""
            lang = (spec.get("language") if isinstance(spec, dict) else None) or lang_default
            if lang in (None, "auto") and THAI.search(script):
                lang = "th"
            y, _ = read_mono(inp["path"], 16000)
            kw = {"language": lang} if lang and lang != "auto" else {}
            text = asr({"raw": y, "sampling_rate": 16000}, generate_kwargs=kw, return_timestamps=len(y) > 16000 * 30)["text"].strip()
            r = {"transcript": text, "language": lang or "auto"}
            if script:
                r["recall"] = recall(script, text, lang)
                r["script"] = script[:500]
            scores[inp["asset_id"]] = r
        ctx.progress(len(items), len(items), "transcribing", "done")
        return {"meta": {"scores": scores, "kind": "asr"}}

    def transcribe(self, path: str, ctx) -> str:
        """Reference transcript for cloners when the voice has none (Whisper, kept while models are loaded)."""
        import torch
        from transformers import pipeline

        ctx.progress(0, 0, "transcribing", "No reference transcript: transcribing the clip with Whisper", interruptible=False)
        if getattr(self, "_asr", None) is None:
            self._asr = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo", dtype=torch.float16, device="cuda")
        y, _ = read_mono(path, 16000)
        text = self._asr({"raw": y[: 16000 * 30], "sampling_rate": 16000})["text"].strip()
        ctx.log(f"reference transcript (Whisper): {text}")
        return text

    def unload(self, model_id):
        self.models.pop(model_id, None)
        if not self.models:
            self._asr = None


if __name__ == "__main__":
    serve(AudioBackend())
