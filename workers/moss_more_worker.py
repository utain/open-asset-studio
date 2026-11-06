"""MOSS "more" TTS backend (runs in envs/moss, read-only: transformers 5.0.0 exactly, trust_remote_code).

  moss-ttsd-v1      tasks tts / voice_clone   multi-speaker dialogue (up to 5 speakers, [S1] [S2] ... tags), 8 B.
                    Without references the model invents the voices ("generation" mode); with a reference clip +
                    transcript per speaker it continues in those voices ("continuation" mode, the card's recipe).
  moss-tts-realtime tasks tts / voice_clone   1.7 B, fast, optional single reference clip (no transcript needed).
                    Non-streaming use of the upstream MossTTSRealtimeInference (vendor/MOSS-TTS/moss_tts_realtime).
  moss-tts-local-v1.5 tasks tts / voice_clone 1.7 B local transformer, 48 kHz STEREO, language tag, duration control.

This is a copy-and-extend of workers/moss_worker.py (not edited). Codecs: MOSS-Audio-Tokenizer (TTSD, Realtime) and
MOSS-Audio-Tokenizer-v2 (Local v1.5), both loaded by the model processors / AutoModel through remote code.
generate() has no step hook, so a take cannot be interrupted midway (cancel waits for the current take).
"""
from __future__ import annotations

import importlib.util
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

TOKENS_PER_S = 12.5
THAI = re.compile(r"[฀-๿]")
SPEAKER_ROLES = ["ref", "ref2", "ref3", "ref4"]          # -> [S1] .. [S4]
REALTIME_SR = 24000
VENDOR_RT = Path(__file__).resolve().parent.parent / "vendor" / "MOSS-TTS" / "moss_tts_realtime"


# ---------------------------------------------------------------- helpers shared with the other MOSS workers
def seed0(p):
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def tidy(w, sr, pad_s=0.12, floor_db=-50.0):
    """Trim leading / trailing near-silence (works on [samples] or [samples, channels])."""
    mono = np.abs(w).max(axis=1) if w.ndim == 2 else np.abs(w)
    idx = np.flatnonzero(mono > 10 ** (floor_db / 20))
    if idx.size == 0:
        return w
    pad = int(pad_s * sr)
    return w[max(0, idx[0] - pad): min(len(w), idx[-1] + pad + 1)]


def save_wav(path, wav, sr):
    """wav: float array [samples] or [samples, channels]. 24-bit PCM, peak-limited to -1 dBFS if it clips."""
    import soundfile as sf

    w = np.nan_to_num(np.asarray(wav, dtype=np.float32))
    if len(w) < int(0.05 * sr):
        raise RuntimeError("the model returned (almost) no audio; try another seed")
    meta = {"sample_rate": int(sr), "channels": 1 if w.ndim == 1 else int(w.shape[1])}
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


def read_ref(path, sr=None, max_s=20.0):
    """Reference clip -> mono float tensor [1, T] (resampled to `sr`). soundfile, so no torchaudio.load / torchcodec."""
    import soundfile as sf
    import torch
    import torchaudio

    wav, fsr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)[: int(fsr * max_s)]
    t = torch.from_numpy(wav.copy()).unsqueeze(0)
    if sr and fsr != sr:
        t = torchaudio.functional.resample(t, fsr, sr)
    return t, (sr or fsr)


# ---------------------------------------------------------------- MOSS-TTSD script handling
def _glue(a: str, b: str) -> str:
    """Join two pieces of one speaker's text: the card's app concatenates directly (fine for Chinese); add a space between
    Latin-script pieces so 'Hello.' + 'World' does not become 'Hello.World'."""
    if a and b and a[-1].isascii() and b[0].isascii() and not a[-1].isspace():
        return a + " " + b
    return a + b


def normalize_script(text: str) -> str:
    """The card app's normalisation: [1] -> [S1], strip decorative punctuation, merge consecutive tags of a speaker."""
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)
    remove_chars = "【】《》（）『』「」" '"-_“”～~‘’'
    segs = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
    parts = []
    for seg in segs:
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"^(\[S\d+\])\s*(.*)", seg)
        tag, content = m.groups() if m else ("", seg)
        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
        content = re.sub(r"哈{2,}", "[笑]", content)
        content = re.sub(r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE)
        for a in ("——", "……", "...", "⸺", "―", "—", "…"):
            content = content.replace(a, "，")
        content = content.translate(str.maketrans({"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"})).strip()
        content = re.sub(r"([，。？！,.?!])[，。？！,.?!]+", r"\1", content)
        if len(content) > 1:
            last = "。" if content[-1] == "，" else ("." if content[-1] == "," else content[-1])
            content = content[:-1].replace("。", "，") + last
        parts.append((tag, content))
    if not parts:
        return ""
    lines, cur_tag, cur = [], parts[0][0], [parts[0][1]]
    for tag, content in parts[1:]:
        if tag == cur_tag and cur_tag:
            cur.append(content)
        else:
            lines.append(f"{cur_tag}{_join(cur)}".strip())
            cur_tag, cur = tag, [content]
    lines.append(f"{cur_tag}{_join(cur)}".strip())
    return "".join(lines).replace("‘", "'").replace("’", "'")


def _join(parts: list[str]) -> str:
    out = ""
    for x in parts:
        out = _glue(out, x)
    return out


def merge_tags(text: str) -> str:
    out, cur = [], None
    for seg in re.split(r"(?=\[S\d+\])", text):
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"^(\[S\d+\])\s*(.*)", seg, re.DOTALL)
        if not m:
            out.append(seg)
            continue
        tag, content = m.groups()
        if tag == cur:
            out[-1] = _glue(out[-1], content)
        else:
            cur = tag
            out.append(f"{tag}{content}")
    return "".join(out)


def script_speakers(text: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\[S(\d+)\]", text)})


def ref_transcript(job, role, p, key):
    """Transcript of a speaker's reference: the param, else the voice asset's own transcript."""
    t = (p.get(key) or "").strip()
    if t:
        return t
    inp = job.input(role=role) or {}
    m = inp.get("meta") or {}
    return ((m.get("x_voice") or {}).get("ref_text") or m.get("final_prompt") or "").strip()


class MossMoreBackend(Backend):
    name = "moss_more"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("transformers", "torch", "soundfile", "torchaudio") if importlib.util.find_spec(m) is None]
        if model["id"] == "moss-tts-realtime" and not (VENDOR_RT / "mossttsrealtime").is_dir():
            missing.append("vendor/MOSS-TTS/moss_tts_realtime")
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    # ------------------------------------------------------------ load
    def load(self, model, ctx):
        import torch

        torch.backends.cuda.enable_cudnn_sdp(False)     # model card: the cuDNN SDPA backend is broken
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        mid = model["id"]
        if mid == "moss-tts-realtime":
            return self._load_realtime(model, attn, torch)
        from transformers import AutoModel, AutoProcessor

        repo = model["weights"][0]["hf"]
        proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        proc.audio_tokenizer = proc.audio_tokenizer.to("cuda")
        m = AutoModel.from_pretrained(repo, trust_remote_code=True, attn_implementation=attn, torch_dtype=torch.bfloat16).to("cuda").eval()
        return {"kind": "delay", "proc": proc, "model": m, "sr": int(proc.model_config.sampling_rate), "refs": {}}

    def _load_realtime(self, model, attn, torch):
        from transformers import AutoModel, AutoTokenizer

        sys.dont_write_bytecode = True                      # vendor/MOSS-TTS is a shared, read-only checkout
        if str(VENDOR_RT) not in sys.path:
            sys.path.insert(0, str(VENDOR_RT))
        from inferencer import MossTTSRealtimeInference
        from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime

        repo, codec_repo = model["weights"][0]["hf"], model["weights"][1]["hf"]
        m = MossTTSRealtime.from_pretrained(repo, attn_implementation=attn, torch_dtype=torch.bfloat16).to("cuda").eval()
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        codec = AutoModel.from_pretrained(codec_repo, trust_remote_code=True).eval().to("cuda")
        inf = MossTTSRealtimeInference(m, tok, max_length=5000, codec=codec, codec_sample_rate=REALTIME_SR,
                                       codec_encode_kwargs={"chunk_duration": 8})

        def _load_audio(path, target_sample_rate):          # upstream uses torchaudio.load (needs torchcodec)
            return read_ref(path, target_sample_rate, 30.0)[0]
        inf._load_audio = _load_audio
        return {"kind": "realtime", "inf": inf, "model": m, "codec": codec, "sr": REALTIME_SR}

    # ------------------------------------------------------------ run
    def run(self, job, ctx):
        h = self.handle(job.model_id)
        if h["kind"] == "realtime":
            return self._run_realtime(job, ctx, h)
        if job.model_id == "moss-ttsd-v1":
            return self._run_ttsd(job, ctx, h)
        return self._run_local(job, ctx, h)

    def _ref_codes(self, h, path):
        import torch

        if path not in h["refs"]:
            t, sr = read_ref(path)
            h["refs"][path] = h["proc"].encode_audios_from_wav([t], sr)[0]
            if len(h["refs"]) > 16:
                h["refs"].pop(next(iter(h["refs"])))
        return h["refs"][path]

    def _generate(self, job, ctx, h, msgs, mode, max_new, sampling, label, n, s0, extra, stereo=False):
        import torch

        proc, model, sr = h["proc"], h["model"], h["sr"]
        batch = proc(msgs, mode=mode)
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n} (cannot be interrupted midway)", interruptible=False)
            with torch.no_grad():
                o = model.generate(input_ids=batch["input_ids"].to("cuda"), attention_mask=batch["attention_mask"].to("cuda"),
                                   max_new_tokens=max_new, **sampling)
            msg = proc.decode(o)[0]
            audio = msg.audio_codes_list[0].float().cpu().numpy()
            if audio.ndim == 2:                             # [channels, samples] -> [samples, channels]
                audio = audio.T if audio.shape[0] <= 8 else audio.reshape(-1)
                if audio.ndim == 2 and audio.shape[1] == 1:
                    audio = audio[:, 0]
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, tidy(audio, sr), sr)
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(label, i, n), **extra, **fm)
            ctx.progress(i + 1, n, "speaking", f"take {i + 1}/{n} done")
        return {"meta": {"seed": s0}}

    def _run_ttsd(self, job, ctx, h):
        import torch

        proc = h["proc"]
        p = job.params
        text = (p.get("text") or p.get("prompt") or "").strip()
        if not text:
            raise InputError("Write the dialogue first: one turn per line, each starting with [S1], [S2] ...")
        if p.get("normalize", True):
            text = normalize_script(text)
        text = merge_tags(text)
        spk = script_speakers(text)
        if not spk:
            raise InputError("The script needs speaker tags: start every turn with [S1], [S2] ... (up to [S5]).")
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        sr = h["sr"]
        # references (role ref -> S1, ref2 -> S2 ...) with their transcripts
        cloned, wavs, prompts = [], [], []
        for k, role in enumerate(SPEAKER_ROLES, start=1):
            inp = job.input(role=role)
            if not inp:
                continue
            tr = ref_transcript(job, role, p, "ref_text" if k == 1 else f"ref_text{k}")
            if not tr:
                raise InputError(f"Speaker {k} has a reference clip but no transcript. MOSS-TTSD continues the dialogue after the "
                                 f"reference, so it needs what the clip says (Reference transcript {k}).")
            tag = f"[S{k}]"
            cloned.append(k)
            wavs.append(read_ref(inp["path"], sr, 20.0)[0])
            prompts.append(tr if tr.lstrip().startswith(tag) else f"{tag} {tr}")
        too_high = [s for s in spk if s > 5]
        if too_high:
            raise InputError("MOSS-TTSD supports speakers [S1] to [S5].")
        max_new = int(TOKENS_PER_S * float(p.get("max_seconds", 90)))
        sampling = dict(audio_temperature=float(p.get("temperature", 1.1)), audio_top_p=float(p.get("top_p", 0.9)),
                        audio_top_k=int(p.get("top_k", 50)), audio_repetition_penalty=float(p.get("repetition_penalty", 1.1)))
        extra = {"speakers": spk, "cloned_speakers": cloned}
        if not cloned:
            msgs = [[proc.build_user_message(text=text)]]
            mode = "generation"
        else:
            ctx.progress(0, n, "speaking", "Encoding the reference voices", interruptible=False)
            n_spk = max(spk + cloned)
            codes = proc.encode_audios_from_wav(wavs, sampling_rate=sr)
            reference = [None] * n_spk
            for k, c in zip(cloned, codes):
                reference[k - 1] = c
            prompt_audio = proc.encode_audios_from_wav([torch.cat(wavs, dim=-1)], sampling_rate=sr)[0]
            full = merge_tags("".join(prompts) + text)
            msgs = [[proc.build_user_message(text=full, reference=reference), proc.build_assistant_message(audio_codes_list=[prompt_audio])]]
            mode = "continuation"
            extra["ref_assets"] = [(job.input(role=SPEAKER_ROLES[k - 1]) or {}).get("asset_id") for k in cloned]
        extra["mode"] = mode
        extra["final_prompt"] = text
        return self._generate(job, ctx, h, msgs, mode, max_new, sampling, text, n, s0, extra)

    def _run_local(self, job, ctx, h):
        p = job.params
        text = (p.get("text") or p.get("prompt") or "").strip()
        if not text:
            raise InputError("Write the line to speak first.")
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        lang = p.get("language") or None
        if lang in (None, "", "auto") and THAI.search(text):
            lang = "Thai"
        if lang in ("", "auto"):
            lang = None
        ref = job.input(role="ref") or next((i for i in job.inputs if i.get("type") in ("voice", "audio")), None)
        if job.task == "voice_clone" and not ref:
            raise InputError("Pick a reference clip to clone.")
        kw = {"text": text, "language": lang}
        extra = {"language": lang, "final_prompt": text}
        if ref:
            ctx.progress(0, n, "speaking", "Encoding the reference voice", interruptible=False)
            kw["reference"] = [self._ref_codes(h, ref["path"])]
            extra["ref_asset"] = ref.get("asset_id")
        dur = float(p.get("seconds") or 0)
        if dur > 0:
            kw["tokens"] = int(round(dur * TOKENS_PER_S))
        msgs = [[h["proc"].build_user_message(**kw)]]
        max_new = int(TOKENS_PER_S * float(p.get("max_seconds", 30)))
        sampling = dict(do_sample=True, audio_temperature=float(p.get("temperature", 1.7)), audio_top_p=float(p.get("top_p", 0.8)),
                        audio_top_k=int(p.get("top_k", 25)), audio_repetition_penalty=float(p.get("repetition_penalty", 1.0)))
        return self._generate(job, ctx, h, msgs, "generation", max_new, sampling, text, n, s0, extra)

    def _run_realtime(self, job, ctx, h):
        import torch

        inf, codec, sr = h["inf"], h["codec"], h["sr"]
        p = job.params
        text = (p.get("text") or p.get("prompt") or "").strip()
        if not text:
            raise InputError("Write the line to speak first.")
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        ref = job.input(role="ref") or next((i for i in job.inputs if i.get("type") in ("voice", "audio")), None)
        if job.task == "voice_clone" and not ref:
            raise InputError("Pick a reference clip to clone.")
        extra = {"final_prompt": text}
        if ref:
            extra["ref_asset"] = ref.get("asset_id")
        max_len = int(TOKENS_PER_S * float(p.get("max_seconds", 30)))
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n} (cannot be interrupted midway)", interruptible=False)
            res = inf.generate(text=[text], reference_audio_path=[ref["path"]] if ref else None, max_length=max_len,
                               temperature=float(p.get("temperature", 0.8)), top_p=float(p.get("top_p", 0.6)),
                               top_k=int(p.get("top_k", 30)), repetition_penalty=float(p.get("repetition_penalty", 1.1)),
                               repetition_window=50, device=torch.device("cuda"))
            codes = torch.tensor(res[0]).to("cuda")
            dec = codec.decode(codes.permute(1, 0), chunk_duration=8)
            audio = dec["audio"][0].float().cpu().numpy().reshape(-1)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, tidy(audio, sr), sr)
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(text, i, n), **extra, **fm)
            ctx.progress(i + 1, n, "speaking", f"take {i + 1}/{n} done")
        return {"meta": {"seed": s0}}

    def unload(self, model_id):
        self.models.pop(model_id, None)


if __name__ == "__main__":
    serve(MossMoreBackend())
