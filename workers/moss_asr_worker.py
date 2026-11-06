"""MOSS audio-understanding backend (envs/moss_asr: torch 2.9.1 cu130, transformers 5.22.1).

  moss-transcribe-diarize  task transcribe    long-form transcript with speaker labels ([S01], [S02] ...) and timestamps,
                                              50+ languages, audio or video up to ~90 min. 0.9 B.
                                              Output: a `text` asset (one line per segment) + .srt + .json attachments.
  moss-audio-4b-instruct   task audio_ask     ask anything about a clip: describe it, caption the speech, name the
                                              instruments / events / mood, transcribe with timestamps, summarise.
                                              Output: a `text` asset with the answer.

Code: vendor/MOSS-Transcribe-Diarize (pip -e, --no-deps) and vendor/MOSS-Audio (added to sys.path; `src` package).
Audio is decoded with soundfile / ffmpeg (no torchaudio.load, which needs torchcodec).
"""
from __future__ import annotations

import importlib.util
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

ROOT = Path(__file__).resolve().parent.parent
VENDOR_AUDIO = ROOT / "vendor" / "MOSS-Audio"
AUDIO_ROLES = ("audio", "source", "ref", "video")

MTD_PROMPTS = {
    ("diarized", "en"): "Transcribe the audio. For each segment, start with the timestamp and speaker ID ([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp.",
    ("diarized", "zh"): "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。",
    ("speakers", "en"): "Transcribe the audio as text using speaker labels such as [S01], [S02], and [S03].",
    ("speakers", "zh"): "转录为文本，使用 [S01] [S02] [S03]等说话人标签。",
}
HOTWORD_SUFFIX = {"en": " Hotwords: ", "zh": "热词提示："}


def load_wave(path, sr):
    """Any audio / video file -> mono float32 [T] at `sr` (soundfile first, ffmpeg for everything else)."""
    try:
        import soundfile as sf

        wav, fsr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if fsr != sr:
            import soxr

            wav = soxr.resample(wav, fsr, sr)
        return np.ascontiguousarray(wav, dtype=np.float32)
    except Exception:
        pass
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise InputError("Could not decode the input as audio: " + r.stderr.decode("utf-8", "replace")[-200:])
    return np.frombuffer(r.stdout, dtype="<f4").copy()


def clock(t, ms=True):
    t = max(0.0, float(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}" if ms else f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def srt_time(t):
    return clock(t).replace(".", ",")


def pick_input(job):
    for r in AUDIO_ROLES:
        i = job.input(role=r)
        if i:
            return i
    return next((i for i in job.inputs if i.get("type") in ("audio", "voice", "video")), None)


def seed0(p):
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


class MossAsrBackend(Backend):
    name = "moss_asr"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("transformers", "torch", "soundfile", "soxr") if importlib.util.find_spec(m) is None]
        if model["id"] == "moss-transcribe-diarize" and importlib.util.find_spec("moss_transcribe_diarize") is None:
            missing.append("moss_transcribe_diarize (pip install --no-deps -e vendor/MOSS-Transcribe-Diarize)")
        if model["id"].startswith("moss-audio") and not (VENDOR_AUDIO / "src").is_dir():
            missing.append("vendor/MOSS-Audio")
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    # ------------------------------------------------------------ load
    def load(self, model, ctx):
        import torch

        repo = model["weights"][0]["hf"]
        if model["id"] == "moss-transcribe-diarize":
            from transformers import AutoModelForCausalLM, AutoProcessor

            m = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa")
            m = m.to("cuda").eval()
            proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
            return {"kind": "mtd", "model": m, "proc": proc}
        sys.dont_write_bytecode = True
        if str(VENDOR_AUDIO) not in sys.path:
            sys.path.insert(0, str(VENDOR_AUDIO))
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor

        m = MossAudioModel.from_pretrained(repo, trust_remote_code=True, dtype="auto", device_map="cuda").eval()
        proc = MossAudioProcessor.from_pretrained(repo, trust_remote_code=True, enable_time_marker=True)
        return {"kind": "audio", "model": m, "proc": proc}

    # ------------------------------------------------------------ run
    def run(self, job, ctx):
        h = self.handle(job.model_id)
        inp = pick_input(job)
        if not inp:
            raise InputError("Add the audio (or video) clip first.")
        return self._run_mtd(job, ctx, h, inp) if h["kind"] == "mtd" else self._run_audio(job, ctx, h, inp)

    def _run_mtd(self, job, ctx, h, inp):
        import torch
        from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription
        from moss_transcribe_diarize.transcript_parser import parse_transcript

        p = job.params
        lang = "zh" if str(p.get("prompt_language", "en")).lower().startswith("zh") else "en"
        mode = p.get("mode") or "diarized"
        prompt = MTD_PROMPTS[(mode, lang)]
        hot = re.sub(r"\s+", " ", (p.get("hotwords") or "").strip())
        if hot:
            prompt += HOTWORD_SUFFIX[lang] + hot
        ctx.progress(0, 0, "transcribing", "Decoding the audio", interruptible=False)
        sr = int(h["proc"].feature_extractor.sampling_rate)
        wav = load_wave(inp["path"], sr)
        dur = len(wav) / sr
        if dur < 0.3:
            raise InputError("The clip is too short.")
        if dur > 100 * 60:
            raise InputError("The model handles recordings up to about 90 minutes; trim the clip first.")
        import soundfile as sf

        tmp = job.out_dir / "_input16k.wav"                   # the helper loads from a path; hand it our decoded copy
        sf.write(str(tmp), wav, sr, subtype="PCM_16")
        messages = build_transcription_messages(tmp, prompt)
        max_new = int(p.get("max_new_tokens", 8192))
        ctx.progress(0, max_new, "transcribing", f"Transcribing {clock(dur, False)} of audio (cannot be interrupted midway)", interruptible=False)
        try:
            out = generate_transcription(h["model"], h["proc"], messages, max_new_tokens=max_new, do_sample=False,
                                         device=torch.device("cuda"), dtype=torch.bfloat16)
        finally:
            tmp.unlink(missing_ok=True)
        raw = out["text"]
        try:
            segs = list(parse_transcript(raw))
        except Exception as e:                              # keep the raw text if the model drifted from the format
            ctx.log(f"parse failed ({e}); keeping the raw transcript")
            segs = []
        lines, js, srt = [], [], []
        for k, s in enumerate(segs, start=1):
            lines.append(f"[{clock(s.start)} - {clock(s.end)}] {s.speaker}: {s.text}")
            srt.append(f"{k}\n{srt_time(s.start)} --> {srt_time(s.end)}\n{s.speaker}: {s.text}\n")
            js.append({"start": round(float(s.start), 3), "end": round(float(s.end), 3), "speaker": s.speaker, "text": s.text})
        body = "\n".join(lines) if lines else raw
        d = job.out_dir
        (d / "take_1.txt").write_text(body + "\n", encoding="utf-8")
        attach = []
        if segs:
            import json

            (d / "take_1.srt").write_text("\n".join(srt), encoding="utf-8")
            (d / "take_1.json").write_text(json.dumps({"segments": js, "raw": raw, "duration": round(dur, 2)}, ensure_ascii=False, indent=1), encoding="utf-8")
            attach = ["take_1.srt", "take_1.json"]
        speakers = sorted({s.speaker for s in segs})
        ctx.output(d / "take_1.txt", "text", index=0, name=f"Transcript: {inp.get('name') or 'clip'}", attach=attach,
                   final_prompt=prompt, input_asset=inp.get("asset_id"), segments=len(segs), speakers=speakers,
                   audio_seconds=round(dur, 2), generated_tokens=out["generated_tokens"], parsed=bool(segs))
        ctx.progress(1, 1, "transcribing", "done")
        return {"meta": {"segments": len(segs), "speakers": speakers}}

    def _run_audio(self, job, ctx, h, inp):
        import torch

        p = job.params
        q = (p.get("question") or p.get("prompt") or "").strip() or "Describe this audio."
        model, proc = h["model"], h["proc"]
        ctx.progress(0, 0, "listening", "Decoding the audio", interruptible=False)
        wav = load_wave(inp["path"], int(proc.config.mel_sr))
        dur = len(wav) / proc.config.mel_sr
        if dur < 0.3:
            raise InputError("The clip is too short.")
        if dur > 20 * 60:
            raise InputError("MOSS-Audio is meant for clips up to a few minutes; trim the clip first (20 min max).")
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        temp = float(p.get("temperature", 0.0))
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            ctx.progress(i, n, "listening", f"answering {i + 1}/{n} (cannot be interrupted midway)", interruptible=False)
            inputs = proc(text=q, audios=[wav], return_tensors="pt").to(model.device)
            if inputs.get("audio_data") is not None:
                inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
            inputs["audio_input_mask"] = inputs["input_ids"] == proc.audio_token_id
            gen = dict(max_new_tokens=int(p.get("max_new_tokens", 1024)), num_beams=1, use_cache=True)
            if temp > 0:
                gen.update(do_sample=True, temperature=temp, top_p=float(p.get("top_p", 0.95)), top_k=int(p.get("top_k", 50)))
            else:
                gen.update(do_sample=False)
            with torch.no_grad():
                ids = model.generate(**inputs, **gen)
            ans = proc.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            f = job.out_dir / f"take_{i + 1}.txt"
            f.write_text(ans + "\n", encoding="utf-8")
            ctx.output(f, "text", seed=seed, index=i, name=f"{q[:40]}" + (f" #{i + 1}" if n > 1 else ""), final_prompt=q,
                       input_asset=inp.get("asset_id"), audio_seconds=round(dur, 2), reply_text=ans)
            ctx.progress(i + 1, n, "listening", f"answer {i + 1}/{n} done")
        return {"meta": {"seed": s0}}

    def unload(self, model_id):
        self.models.pop(model_id, None)


if __name__ == "__main__":
    serve(MossAsrBackend())
