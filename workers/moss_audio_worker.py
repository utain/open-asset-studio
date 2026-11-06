"""MOSS-Audio backend (envs/moss_audio: torch 2.9.1 cu130, transformers 4.57.1 EXACTLY).

  moss-audio-4b-instruct   task audio_ask   ask anything about a clip: describe it, caption the speech, name the
                                            instruments / events / mood, transcribe with timestamps, summarise.
                                            Output: a `text` asset with the answer.

Why its own venv: MOSS-Audio's Whisper-style encoder uses the transformers 4.x WhisperEncoderLayer signature
(it fails with a tensor-size error on 5.22). Code: vendor/MOSS-Audio (put on sys.path; its package is the generic `src`).
Helpers are shared with workers/moss_asr_worker.py (imported; that module imports no heavy libraries at top level).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from moss_asr_worker import load_wave, pick_input, seed0
from studio_worker import Backend, InputError, serve

VENDOR_AUDIO = Path(__file__).resolve().parent.parent / "vendor" / "MOSS-Audio"


class MossAudioBackend(Backend):
    name = "moss_audio"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("transformers", "torch", "soundfile", "soxr", "tiktoken") if importlib.util.find_spec(m) is None]
        if not (VENDOR_AUDIO / "src").is_dir():
            missing.append("vendor/MOSS-Audio")
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        repo = model["weights"][0]["hf"]
        sys.dont_write_bytecode = True
        if str(VENDOR_AUDIO) not in sys.path:
            sys.path.insert(0, str(VENDOR_AUDIO))
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor
        from transformers import AutoTokenizer

        m = MossAudioModel.from_pretrained(repo, trust_remote_code=True, dtype="auto", device_map="cuda").eval()
        # MossAudioProcessor.from_pretrained drops trust_remote_code, and the tokenizer then asks on stdin: build it ourselves
        tok = AutoTokenizer.from_pretrained(repo, use_fast=False, trust_remote_code=True)
        return {"model": m, "proc": MossAudioProcessor(tok, enable_time_marker=True)}

    def run(self, job, ctx):
        h = self.handle(job.model_id)
        inp = pick_input(job)
        if not inp:
            raise InputError("Add the audio (or video) clip first.")
        return self._run_audio(job, ctx, h, inp)

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
    serve(MossAudioBackend())
