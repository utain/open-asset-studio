"""Qwen Omni backend (envs/omni_qwen: transformers 4.57.3, qwen-omni-utils, torch 2.9 cu130).

  qwen25-omni-3b / qwen25-omni-7b    task s2s   thinker (text) + talker (speech tokens) + DiT token2wav, 24 kHz
  qwen3-omni-30b-a3b                 task s2s   MoE thinker + multi-codebook talker + code2wav, 24 kHz
One turn per job: the user's clip (and/or a typed message, optional instruction text) in, spoken reply + its text out.
Qwen2.5-Omni only produces speech with its default system prompt, so the user's instruction goes in the user turn.
"""
from __future__ import annotations

import gc
import importlib.util
import math
import random
import re

import numpy as np

from studio_worker import Backend, InputError, serve

SYS_25 = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
          "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
REPOS = {"qwen25-omni-3b": "Qwen/Qwen2.5-Omni-3B", "qwen25-omni-7b": "Qwen/Qwen2.5-Omni-7B",
         "qwen3-omni-30b-a3b": "Qwen/Qwen3-Omni-30B-A3B-Instruct"}


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


class QwenOmniBackend(Backend):
    name = "omni_qwen"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("transformers", "qwen_omni_utils", "torch", "soundfile") if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        repo = REPOS.get(model["id"])
        if not repo:
            raise InputError(f"unknown model {model['id']}")
        local = snapshot_download(repo, local_files_only=True)
        if model["id"].startswith("qwen25"):
            from transformers import Qwen2_5OmniForConditionalGeneration as M, Qwen2_5OmniProcessor as P
        else:
            from transformers import Qwen3OmniMoeForConditionalGeneration as M, Qwen3OmniMoeProcessor as P
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        mdl = M.from_pretrained(local, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation=attn).eval()
        proc = P.from_pretrained(local)
        return {"model": mdl, "proc": proc, "v3": model["id"].startswith("qwen3")}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import soundfile as sf
        import torch
        from qwen_omni_utils import process_mm_info

        h = self.handle(job.model_id)
        mdl, proc, v3 = h["model"], h["proc"], h["v3"]
        p = job.params
        src = job.input(role="audio") or job.input(role="ref") or next(
            (i for i in job.inputs if i.get("type") in ("audio", "voice")), None)
        typed = (p.get("text") or "").strip()
        instruction = (p.get("instruction") or "").strip()
        if not src and not (typed or instruction):
            raise InputError("Pick an audio clip to answer (or type a message).")
        user = []
        if src:
            user.append({"type": "audio", "audio": src["path"]})
        if typed:
            user.append({"type": "text", "text": typed})
        if instruction:
            user.append({"type": "text", "text": instruction})
        conv = [{"role": "user", "content": user}]
        if not v3:  # speech output needs exactly this system prompt
            conv.insert(0, {"role": "system", "content": [{"type": "text", "text": SYS_25}]})
        elif p.get("system"):
            conv.insert(0, {"role": "system", "content": [{"type": "text", "text": p["system"]}]})
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                      use_audio_in_video=False)
        inputs = inputs.to(mdl.device).to(mdl.dtype)
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        speaker = p.get("voice") or ("Ethan" if v3 else "Chelsie")
        max_tokens = max(32, int(float(p.get("seconds") or 20) * 6))  # ~4-5 text tokens per spoken second, upper bound
        kw = dict(speaker=speaker, use_audio_in_video=False, thinker_max_new_tokens=max_tokens)
        if p.get("temperature"):
            kw["talker_temperature"] = float(p["temperature"])
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}: thinking + speaking", interruptible=False)
            with torch.inference_mode():
                out = mdl.generate(**inputs, **kw)
            if v3:
                text_ids, wav = out
                seqs = text_ids.sequences
            else:
                seqs, wav = out
            if wav is None:
                raise RuntimeError("the model returned no audio (talker disabled?)")
            reply = proc.batch_decode(seqs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav.reshape(-1).float().cpu().numpy(), 24000)
            (job.out_dir / f"take_{i + 1}.txt").write_text(reply, encoding="utf-8")
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(reply or "Reply", i, n), final_prompt=reply,
                       reply_text=reply, voice=speaker, input_asset=(src or {}).get("asset_id"),
                       attach=[f"take_{i + 1}.txt"], **fm)
            del out, wav
            gc.collect()
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(QwenOmniBackend())
