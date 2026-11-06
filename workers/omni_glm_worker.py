"""GLM-4-Voice backend (envs/omni_glm: transformers 4.44.2, torch 2.9 cu130; upstream code in vendor/GLM-4-Voice).

  glm-4-voice-9b   task s2s   speech tokenizer (Whisper-VQ, 12.5 Hz) -> GLM-4-9B LM interleaving 13 text : 26 audio
                              tokens -> CosyVoice-style flow-matching + HiFT decoder (22.05 kHz).
One turn per job, offline (upstream streams in 25/50/100/150/200-token blocks; we replay the same schedule after
generation, which gives the same audio). Chinese and English.
"""
from __future__ import annotations

import gc
import importlib.util
import math
import os
import random
import re
import sys
import uuid
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "GLM-4-Voice"
REPO_LM, REPO_TOK, REPO_DEC = "zai-org/glm-4-voice-9b", "zai-org/glm-4-voice-tokenizer", "zai-org/glm-4-voice-decoder"
SYS_AUDIO = ("User will provide you with a speech instruction. Do it step by step. First, think about the instruction and "
             "respond in a interleaved manner, with 13 text token followed by 26 audio tokens. ")
SYS_TEXT = ("User will provide you with a text instruction. Do it step by step. First, think about the instruction and "
            "respond in a interleaved manner, with 13 text token followed by 26 audio tokens.")
SR_OUT = 22050


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


class GLMBackend(Backend):
    name = "omni_glm"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("transformers", "hyperpyyaml", "torch", "soundfile", "conformer") if importlib.util.find_spec(m) is None]
        if not VENDOR.exists():
            missing.append(f"vendor/GLM-4-Voice ({VENDOR})")
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        for p in (VENDOR, VENDOR / "cosyvoice", VENDOR.parent / "Matcha-TTS"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from flow_inference import AudioDecoder
        from speech_tokenizer.modeling_whisper import WhisperVQEncoder

        lm_dir = snapshot_download(REPO_LM, local_files_only=True)
        tok_dir = snapshot_download(REPO_TOK, local_files_only=True)
        dec_dir = snapshot_download(REPO_DEC, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(lm_dir, trust_remote_code=True)
        lm = AutoModel.from_pretrained(lm_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
        dec = AudioDecoder(config_path=os.path.join(dec_dir, "config.yaml"), flow_ckpt_path=os.path.join(dec_dir, "flow.pt"),
                           hift_ckpt_path=os.path.join(dec_dir, "hift.pt"), device="cuda")
        wh = WhisperVQEncoder.from_pretrained(tok_dir).eval().to("cuda")
        fe = WhisperFeatureExtractor.from_pretrained(tok_dir)
        return {"lm": lm, "tok": tok, "dec": dec, "wh": wh, "fe": fe}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import soundfile as sf
        import torch
        from speech_tokenizer.utils import extract_speech_token
        from transformers import StoppingCriteria, StoppingCriteriaList

        h = self.handle(job.model_id)
        lm, tok, dec = h["lm"], h["tok"], h["dec"]
        p = job.params
        src = job.input(role="audio") or job.input(role="ref") or next(
            (i for i in job.inputs if i.get("type") in ("audio", "voice")), None)
        typed = (p.get("text") or "").strip()
        if not src and not typed:
            raise InputError("Pick an audio clip to answer (or type a message).")
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        max_new = max(100, int(float(p.get("seconds") or 20) * 12.5 * 3.0))  # 12.5 audio tokens/s + text tokens
        if src:
            y, sr = sf.read(src["path"], dtype="float32", always_2d=True)
            y = torch.from_numpy(y.mean(axis=1))[None][:, : sr * 60]
            toks = extract_speech_token(h["wh"], h["fe"], [(y, sr)])[0]
            if not toks:
                raise InputError("No speech found in the clip.")
            user_input = "<|begin_of_audio|>" + "".join(f"<|audio_{x}|>" for x in toks) + "<|end_of_audio|>"
            system = SYS_AUDIO
        else:
            user_input, system = typed, SYS_TEXT
        if (p.get("instruction") or "").strip():
            system = system.rstrip() + " " + p["instruction"].strip()
        prompt = f"<|system|>\n{system}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        enc = tok([prompt], return_tensors="pt").to("cuda")
        audio_offset = tok.convert_tokens_to_ids("<|audio_0|>")
        end_id = tok.convert_tokens_to_ids("<|user|>")

        class Crit(StoppingCriteria):
            def __init__(self):
                self.k = 0

            def __call__(self, input_ids, scores, **kw):
                ctx.check()
                self.k += 1
                if self.k % 25 == 0:
                    ctx.progress(self.k, max_new, "speaking", f"generating tokens {self.k}/{max_new}")
                return False

        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                out = lm.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=float(p.get("temperature") or 0.2),
                                  top_p=float(p.get("top_p") or 0.8), stopping_criteria=StoppingCriteriaList([Crit()]))
            gen = out[0, enc["input_ids"].shape[1]:].tolist()
            if end_id in gen:
                gen = gen[: gen.index(end_id)]
            text_ids = [t for t in gen if t < audio_offset]
            audio_ids = [t - audio_offset for t in gen if t >= audio_offset]
            reply = tok.decode(text_ids, spaces_between_special_tokens=False).strip()
            reply = re.sub(r"<\|[^|]*\|>", "", reply).strip()
            if not audio_ids:
                raise RuntimeError("the model produced no audio tokens (try a higher max duration or another seed)")
            ctx.progress(i, n, "decoding", "flow-matching + vocoder", interruptible=False)
            wav = self.decode(dec, audio_ids)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, SR_OUT)
            (job.out_dir / f"take_{i + 1}.txt").write_text(reply, encoding="utf-8")
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(reply or "Reply", i, n), final_prompt=reply,
                       reply_text=reply, input_asset=(src or {}).get("asset_id"), attach=[f"take_{i + 1}.txt"], **fm)
            del out
            gc.collect()
        return {"meta": {"seed": s0}}

    @staticmethod
    def decode(dec, audio_ids):
        import torch

        dev = "cuda"
        sizes, k = [25, 50, 100, 150, 200], 0
        this_uuid = str(uuid.uuid4())
        prompt_feat = torch.zeros(1, 0, 80, device=dev)
        prompt_tok = torch.zeros(1, 0, dtype=torch.int64, device=dev)
        speechs, mels, pos = [], [], 0
        with torch.inference_mode():
            while pos < len(audio_ids):
                bs = sizes[min(k, len(sizes) - 1)]
                chunk = audio_ids[pos: pos + bs]
                last = pos + bs >= len(audio_ids)
                if not last and len(audio_ids) - (pos + bs) < 5:  # avoid a tiny tail block
                    chunk, last = audio_ids[pos:], True
                t = torch.tensor(chunk, device=dev).unsqueeze(0)
                if mels:
                    prompt_feat = torch.cat(mels, dim=-1).transpose(1, 2)
                sp, mel = dec.token2wav(t, uuid=this_uuid, prompt_token=prompt_tok, prompt_feat=prompt_feat, finalize=last)
                speechs.append(sp.squeeze().float().cpu())
                mels.append(mel)
                prompt_tok = torch.cat((prompt_tok, t), dim=-1)
                pos += len(chunk)
                k += 1
        return torch.cat(speechs, dim=-1).numpy()


if __name__ == "__main__":
    serve(GLMBackend())
