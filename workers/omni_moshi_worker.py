"""Kyutai backend (envs/omni_moshi: moshi 0.2.13, torch 2.9 cu130).

  kyutai-tts   task tts   Kyutai TTS 1.6B (en/fr), streaming-style model run offline, voice = pre-computed embedding
                          from kyutai/tts-voices (CC0 voice bank; no free voice cloning by design)
  moshi        task s2s   Moshiko, the full-duplex speech-text model, run as ONE offline turn: the user's clip is fed
                          frame by frame, then silence so it can finish answering. Output = everything Moshi said
                          (it may start answering before the clip ends) + its inner-monologue text.
Recipes: vendor/delayed-streams-modeling/scripts/tts_pytorch.py and moshi/run_inference.py.
"""
from __future__ import annotations

import importlib.util
import math
import os
import random
import re

import numpy as np

from studio_worker import Backend, InputError, serve

# torch.compile / inductor hangs on sm_121 here (no triton ptxas for GB10): run eager
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("NO_CUDA_GRAPH", "1")

TTS_REPO = "kyutai/tts-1.6b-en_fr"
MOSHI_REPO = "kyutai/moshiko-pytorch-bf16"
DEFAULT_VOICE = "unmute-prod-website/default_voice.wav"


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


def read_mono(path, sr):
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    y, s = sf.read(path, dtype="float32", always_2d=True)
    y = torch.from_numpy(y.mean(axis=1))
    if s != sr:
        y = AF.resample(y, s, sr)
    return y


class MoshiBackend(Backend):
    name = "omni_moshi"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("moshi", "torch", "soundfile", "sentencepiece") if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    # ------------------------------------------------------------------ load
    def load(self, model, ctx):
        import torch
        from moshi.models import LMGen
        from moshi.models.loaders import CheckpointInfo

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        if model["id"] == "kyutai-tts":
            from moshi.models.tts import TTSModel

            info = CheckpointInfo.from_hf_repo(TTS_REPO)
            tts = TTSModel.from_checkpoint_info(info, n_q=32, temp=0.6, device="cuda")
            return {"kind": "tts", "tts": tts}
        if model["id"] == "moshi":
            info = CheckpointInfo.from_hf_repo(MOSHI_REPO)
            mimi = info.get_mimi(device="cuda")
            tok = info.get_text_tokenizer()
            lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)
            gen = LMGen(lm, **info.lm_gen_config)
            mimi.streaming_forever(1)
            gen.streaming_forever(1)
            return {"kind": "moshi", "mimi": mimi, "tok": tok, "gen": gen, "lm": lm}
        raise InputError(f"unknown model {model['id']}")

    def unload(self, model_id):
        self.models.pop(model_id, None)

    # ------------------------------------------------------------------- run
    def run(self, job, ctx):
        h = self.handle(job.model_id)
        return self.run_tts(job, ctx, h) if h["kind"] == "tts" else self.run_moshi(job, ctx, h)

    def run_tts(self, job, ctx, h):
        import torch

        tts, p = h["tts"], job.params
        text = (p.get("text") or p.get("prompt") or "").strip()
        if not text:
            raise InputError("Write the line to speak first.")
        voice = p.get("voice") or DEFAULT_VOICE
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        tts.temp = float(p.get("temperature") or 0.6)
        try:
            voice_path = tts.get_voice_path(voice)
        except Exception as e:  # noqa: BLE001
            raise InputError(f"voice {voice!r} not found in the local kyutai/tts-voices cache ({e})")
        entries = tts.prepare_script([text], padding_between=1)
        cond = tts.make_condition_attributes([voice_path], cfg_coef=float(p.get("cfg_coef") or 2.0))
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            cnt = [0]

            def on_frame(frame):
                if (frame != -1).all():
                    cnt[0] += 1
                    if cnt[0] % 10 == 0:
                        ctx.progress(i, n, "speaking", f"take {i + 1}/{n}: {cnt[0] / 12.5:.1f} s generated")
                ctx.check()

            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}")
            res = tts.generate([entries], [cond], on_frame=on_frame)
            pcms = []
            with tts.mimi.streaming(1), torch.no_grad():
                for frame in res.frames[tts.delay_steps:]:
                    pcm = tts.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                    pcms.append(np.clip(pcm[0, 0], -1, 1))
            wav = np.concatenate(pcms, axis=-1)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, int(tts.mimi.sample_rate))
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(text, i, n), final_prompt=text, voice=voice, **fm)
        return {"meta": {"seed": s0}}

    def run_moshi(self, job, ctx, h):
        import torch

        mimi, tok, gen = h["mimi"], h["tok"], h["gen"]
        p = job.params
        src = job.input(role="audio") or job.input(role="ref") or next(
            (i for i in job.inputs if i.get("type") in ("audio", "voice")), None)
        if not src:
            raise InputError("Pick the audio clip Moshi should answer.")
        max_in = 60.0
        y = read_mono(src["path"], int(mimi.sample_rate))[: int(max_in * mimi.sample_rate)]
        reply_s = float(p.get("seconds") or 10)
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        gen.temp = float(p.get("temperature") or 0.8)
        gen.temp_text = float(p.get("temperature_text") or 0.7)
        frame = int(mimi.sample_rate / mimi.frame_rate)
        # the clip, then silence for the reply (Moshi answers while it "listens")
        pad = torch.zeros(int(reply_s * mimi.sample_rate) + frame)
        x = torch.cat([y, pad])
        x = x[: (len(x) // frame) * frame].to("cuda")[None, None]
        chunks = list(x.split(frame, dim=2))
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            mimi.reset_streaming()
            gen.reset_streaming()
            pcms, pieces, first = [], [], True
            with torch.no_grad():
                for k, chunk in enumerate(chunks):
                    ctx.check()
                    codes = mimi.encode(chunk)
                    if first:
                        gen.step(codes)  # first slice is replaced by the initial tokens otherwise
                        first = False
                    out = gen.step(codes)
                    if out is None:
                        continue
                    pcms.append(mimi.decode(out[:, 1:]).cpu()[0, 0])
                    t = out[0, 0, 0].item()
                    if t not in (0, 3):
                        pieces.append(tok.id_to_piece(t).replace("▁", " "))
                    if k % 10 == 0:
                        ctx.progress(k, len(chunks), "speaking", f"take {i + 1}/{n}: {k * 0.08:.0f} s")
            if not pcms:
                raise RuntimeError("Moshi returned no audio")
            wav = torch.cat(pcms).numpy()
            reply = "".join(pieces).strip()
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, int(mimi.sample_rate))
            (job.out_dir / f"take_{i + 1}.txt").write_text(reply, encoding="utf-8")
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(reply or "Moshi reply", i, n), final_prompt=reply,
                       reply_text=reply, input_asset=src.get("asset_id"), attach=[f"take_{i + 1}.txt"], **fm)
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(MoshiBackend())
