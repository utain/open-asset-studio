"""clonea_codec backend (envs/clonea_codec): codec-token LLM TTS.

  spark-tts   SparkAudio/Spark-TTS-0.5B (Qwen2.5-0.5B LM + BiCodec, 16 kHz). tasks: tts / voice_design (gender, pitch and
              speed levels -> a made-up voice) and voice_clone (reference clip, transcript optional but better).
  llasa-1b    HKUSTAudio/Llasa-1B (Llama-3.2-1B LM) + HKUST-Audio/xcodec2 (16 kHz). tasks: tts (random voice) and
              voice_clone (reference clip + its transcript; Whisper writes the transcript when none is given).

Both have non-commercial weights (CC-BY-NC). Long text is cut into sentence pieces; a made-up Spark voice is pinned
to the global tokens of the first piece so the voice stays the same across pieces.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve
from clonea_common import (Asr, read_mono, ref_input, ref_text_of, save_wav, seed0, seed_all, take_name, text_of,
                           tidy_voice)

STUDIO = Path(__file__).resolve().parent.parent
SPARK_SRC = STUDIO / "vendor" / "Spark-TTS"
LEVELS = ["very_low", "low", "moderate", "high", "very_high"]


def pieces_of(text: str, limit: int) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?。！？…])\s+", text) if s.strip()]
    out: list[str] = []
    for s in parts:
        while len(s) > limit:
            cut = max(s.rfind(c, 0, limit) for c in (",", "，", ";", " ", "、"))
            cut = cut if cut > limit // 3 else limit
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    merged: list[str] = []
    for s in out:
        if merged and len(merged[-1]) + 1 + len(s) <= limit:
            merged[-1] += " " + s
        else:
            merged.append(s)
    return merged or [text]


def generate(lm, input_ids, ctx, label: str, max_new: int, eos, **kw):
    """HF generate with cancel + progress (tokens / budget)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Watch(StoppingCriteria):
        k = 0

        def __call__(self, ids, scores, **_kw):
            self.k += 1
            if self.k % 25 == 0:
                ctx.check()
                ctx.progress(self.k, max_new, "speaking", f"{label}: {self.k} tokens")
            return torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    return lm.generate(input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=max_new, do_sample=True,
                       eos_token_id=eos, stopping_criteria=StoppingCriteriaList([Watch()]), **kw)


class CodecBackend(Backend):
    name = "clonea_codec"
    max_loaded = 1

    NEEDS = {"spark-tts": ["transformers", "omegaconf", "einx", "einops", "soxr"], "llasa": ["transformers", "xcodec2"]}

    @staticmethod
    def family(mid: str) -> str:
        for k in ("spark-tts", "llasa"):
            if mid.startswith(k):
                return k
        raise InputError(f"The clonea_codec backend does not know model '{mid}'.")

    def probe(self, model):
        fam = self.family(model["id"])
        missing = [m for m in self.NEEDS[fam] + ["torch", "soundfile"] if importlib.util.find_spec(m) is None]
        return (not missing, f"missing python packages: {', '.join(missing)}" if missing else "")

    # ------------------------------------------------------------------ load
    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download

        fam = self.family(model["id"])
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        if fam == "spark-tts":
            if str(SPARK_SRC) not in sys.path:
                sys.path.insert(0, str(SPARK_SRC))
            from cli.SparkTTS import SparkTTS
            from sparktts.utils.token_parser import TASK_TOKEN_MAP

            d = snapshot_download(model["weights"][0]["hf"], local_files_only=True)
            s = SparkTTS(Path(d), torch.device("cuda:0"))
            s.model.to(torch.bfloat16).eval()
            return {"spark": s, "task_tts": TASK_TOKEN_MAP["tts"]}
        # llasa
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Wav2Vec2BertModel
        import xcodec2.modeling_xcodec2 as xm
        from xcodec2.modeling_xcodec2 import XCodec2Model

        class _W2VBert(Wav2Vec2BertModel):
            """XCodec2Model.__init__ loads the 2.3 GB facebook/w2v-bert-2.0 weights, which its own checkpoint holds
            again: build the module from the config only (saves the download and the double load)."""
            @classmethod
            def from_pretrained(cls, name, **kw):
                cfg = AutoConfig.from_pretrained(name)
                cfg.output_hidden_states = bool(kw.get("output_hidden_states"))
                return cls(cfg)
        xm.Wav2Vec2BertModel = _W2VBert

        repo = snapshot_download(model["weights"][0]["hf"], local_files_only=True)
        tok = AutoTokenizer.from_pretrained(repo)
        lm = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
        x2dir = snapshot_download(model["weights"][1]["hf"], local_files_only=True)
        codec = XCodec2Model.from_pretrained(x2dir).eval()
        # transformers renames SnakeBeta's `beta` parameter to `bias` while loading, so the 36 encoder activations
        # silently stay at their init values (only the reference-voice encoder is affected): copy them by hand.
        from safetensors import safe_open

        fixed = 0
        with safe_open(os.path.join(x2dir, "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                if k.endswith(".act.beta"):
                    codec.get_submodule(k[: -len(".beta")]).bias.data.copy_(f.get_tensor(k))
                    fixed += 1
        codec = codec.to("cuda")
        return {"tok": tok, "lm": lm, "codec": codec}

    def unload(self, model_id):
        self.models.pop(model_id, None)
        if not self.models:
            Asr.free()

    def run(self, job, ctx):
        fam = self.family(job.model_id)
        return getattr(self, "_run_" + fam.replace("-", "_"))(job, ctx, self.handle(job.model_id))

    # ------------------------------------------------------------------ Spark-TTS
    def _run_spark_tts(self, job, ctx, h):
        import torch

        s = h["spark"]
        p = job.params
        text = text_of(job)
        clone = job.task == "voice_clone"
        extra = {}
        prompt_text = None
        if clone:
            ref = ref_input(job)
            if not ref:
                raise InputError("Pick a reference clip (3-20 s of clean speech from the library).")
            prompt_text = ref_text_of(job, ref) or None
            extra["ref_asset"] = ref.get("asset_id")
            extra["ref_text_used"] = bool(prompt_text)
        else:
            gender, pitch, speed = p.get("gender") or "female", p.get("pitch") or "moderate", p.get("speed_level") or "moderate"
            if gender not in ("female", "male") or pitch not in LEVELS or speed not in LEVELS:
                raise InputError("gender is female|male; pitch and speed_level are very_low|low|moderate|high|very_high.")
            extra.update(gender=gender, pitch=pitch, speed_level=speed)
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        pieces = pieces_of(text, 220)
        sr = int(s.sample_rate)
        total = n * len(pieces)
        for i in range(n):
            ctx.check()
            seed = s0 + i
            seed_all(seed)
            wavs, pinned = [], None                      # pinned: (global token ids) fixed after the first piece (designed voice)
            for k, piece in enumerate(pieces):
                ctx.check()
                label = f"take {i + 1}/{n}: part {k + 1}/{len(pieces)}"
                if clone:
                    prompt, glob = s.process_prompt(piece, Path(ref["path"]), prompt_text)
                elif pinned is None:
                    prompt, glob = s.process_prompt_control(gender, pitch, speed, piece), None
                else:
                    gtok = "".join(f"<|bicodec_global_{t}|>" for t in pinned.squeeze().tolist())
                    prompt = "".join([h["task_tts"], "<|start_content|>", piece, "<|end_content|>",
                                      "<|start_global_token|>", gtok, "<|end_global_token|>"])
                    glob = pinned
                ids = s.tokenizer([prompt], return_tensors="pt").to(s.device)
                with torch.inference_mode():
                    out = generate(s.model, ids.input_ids, ctx, label, 3000, s.tokenizer.eos_token_id,
                                   temperature=float(p.get("temperature", 0.8)), top_k=int(p.get("top_k", 50)),
                                   top_p=float(p.get("top_p", 0.95)))
                    txt = s.tokenizer.batch_decode(out[:, ids.input_ids.shape[1]:], skip_special_tokens=True)[0]
                    sem = torch.tensor([int(t) for t in re.findall(r"bicodec_semantic_(\d+)", txt)]).long().unsqueeze(0)
                    if sem.numel() == 0:
                        raise RuntimeError("Spark-TTS produced no audio tokens; try another seed")
                    if glob is None:
                        glob = torch.tensor([int(t) for t in re.findall(r"bicodec_global_(\d+)", txt)]).long().unsqueeze(0).unsqueeze(0)
                        pinned = glob
                    wav = s.audio_tokenizer.detokenize(glob.to(s.device).squeeze(0), sem.to(s.device))
                wavs.append(np.asarray(wav, dtype=np.float32).reshape(-1))
                if k < len(pieces) - 1:
                    wavs.append(np.zeros(int(0.12 * sr), dtype=np.float32))
                ctx.progress(i * len(pieces) + k + 1, total, "speaking", label + " done")
            wav = tidy_voice(np.concatenate(wavs), sr)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, sr)
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(text, i, n), final_prompt=text,
                       as_voice=job.task == "voice_design", **extra, **fm)
        return {"meta": {"seed": s0}}

    # ------------------------------------------------------------------ Llasa
    def _run_llasa(self, job, ctx, h):
        import torch

        tok, lm, codec = h["tok"], h["lm"], h["codec"]
        p = job.params
        text = text_of(job)
        clone = job.task == "voice_clone"
        extra, prefix, prefix_n = {}, "", 0
        if clone:
            ref = ref_input(job)
            if not ref:
                raise InputError("Pick a reference clip (3-12 s of clean speech from the library).")
            y, _ = read_mono(ref["path"], 16000)
            cut = len(y) > 16000 * 12
            y = y[: 16000 * 12]
            rt = "" if cut else ref_text_of(job, ref)
            if not rt:
                import soundfile as sf
                import tempfile

                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                try:
                    sf.write(tmp.name, y, 16000)
                    rt = Asr.transcribe(tmp.name, ctx)
                finally:
                    os.unlink(tmp.name)
            ctx.progress(0, 0, "conditioning", "Encoding the reference voice", interruptible=False)
            with torch.inference_mode():
                codes = codec.encode_code(input_waveform=torch.from_numpy(y).float().unsqueeze(0))[0, 0, :]
            ids = codes.tolist()
            prefix = "".join(f"<|s_{c}|>" for c in ids)
            prefix_n = len(ids)
            extra.update(ref_asset=ref.get("asset_id"), ref_text=rt)
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        pieces = pieces_of(text, 200 if not clone else 160)
        total = n * len(pieces)
        end_id = tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        for i in range(n):
            ctx.check()
            seed = s0 + i
            seed_all(seed)
            wavs = []
            for k, piece in enumerate(pieces):
                ctx.check()
                label = f"take {i + 1}/{n}: part {k + 1}/{len(pieces)}"
                full = (extra["ref_text"] + " " + piece) if clone else piece
                chat = [{"role": "user", "content": "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>" + full + "<|TEXT_UNDERSTANDING_END|>"},
                        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + prefix}]
                input_ids = tok.apply_chat_template(chat, tokenize=True, return_tensors="pt", continue_final_message=True)
                if not torch.is_tensor(input_ids):
                    input_ids = input_ids["input_ids"]
                input_ids = input_ids.to("cuda")
                budget = max(64, 2048 - input_ids.shape[1])
                with torch.inference_mode():
                    out = generate(lm, input_ids, ctx, label, budget, end_id, top_p=float(p.get("top_p", 1.0)),
                                   temperature=float(p.get("temperature", 0.8)), top_k=int(p.get("top_k", 0)) or None)
                    gen = out[0, input_ids.shape[1] - prefix_n:-1] if clone else out[0, input_ids.shape[1]:-1]
                    toks = tok.batch_decode(gen, skip_special_tokens=True)
                    sp = [int(m) for t in toks for m in re.findall(r"<\|s_(\d+)\|>", t)]
                    if len(sp) <= prefix_n:
                        raise RuntimeError("Llasa produced no speech tokens; try another seed")
                    wav = codec.decode_code(torch.tensor(sp, device="cuda").unsqueeze(0).unsqueeze(0))[0, 0, :]
                w = wav.float().cpu().numpy()[prefix_n * 320:]              # drop the re-synthesised reference
                wavs.append(w)
                if k < len(pieces) - 1:
                    wavs.append(np.zeros(int(0.12 * 16000), dtype=np.float32))
                ctx.progress(i * len(pieces) + k + 1, total, "speaking", label + " done")
            wav = tidy_voice(np.concatenate(wavs), 16000)
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav, 16000)
            ctx.output(f, "audio", seed=seed, index=i, name=take_name(text, i, n), final_prompt=text, **extra, **fm)
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(CodecBackend())
