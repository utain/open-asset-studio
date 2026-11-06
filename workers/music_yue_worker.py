"""Music backend `music_yue` (envs/music_yue): YuE v1 (m-a-p, Apache-2.0), lyrics-to-song with vocals.

Two-stage LM pipeline from vendor/YuE-v1/inference/infer.py, ported in-process (load once, many jobs):
  stage 1  `m-a-p/YuE-s1-7B-anneal-en-cot` (7B LLaMA, bf16, SDPA): text -> interleaved vocal / instrumental xcodec tokens
  stage 2  `m-a-p/YuE-s2-1B-general` (1B): upsamples the 1-codebook tokens to 8 codebooks, 6 s windows
  codec    `m-a-p/xcodec_mini_infer` (xcodec decoder 16 kHz + Vocos vocal / instrumental decoders 44.1 kHz)
The result is the vocoder mix with the <5.5 kHz band replaced by the 16 kHz xcodec mix (energy matched), as upstream.
Differences from upstream: no torch.compile (no nvcc here), no mp3 round trips (everything stays in memory), one
segment of lyrics by default (`segments`, each ~`seconds` long), cancel / progress through a StoppingCriteria.
The repo's `inference/xcodec_mini_infer` dir is a symlink to the HF snapshot (created on first use).
"""
from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

STUDIO = Path(__file__).resolve().parent.parent
REPO = STUDIO / "vendor" / "YuE-v1" / "inference"
S1, S2, XC = "m-a-p/YuE-s1-7B-anneal-en-cot", "m-a-p/YuE-s2-1B-general", "m-a-p/xcodec_mini_infer"
TPS = 50            # xcodec frames per second


def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def take_name(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def link_xcodec() -> Path:
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(XC, local_files_only=True))
    link = REPO / "xcodec_mini_infer"
    if link.is_symlink():
        if os.readlink(link) != str(snap):
            link.unlink()
            link.symlink_to(snap)
    else:
        if link.exists():
            if any(link.iterdir()):
                raise RuntimeError(f"{link} is a non-empty directory; expected a symlink to {snap}")
            link.rmdir()
        link.symlink_to(snap)
    return link


def split_lyrics(lyrics: str) -> list[str]:
    segs = re.findall(r"\[(\w+)\](.*?)(?=\[|\Z)", lyrics, re.DOTALL)
    return [f"[{a}]\n{b.strip()}\n\n" for a, b in segs]


class MusicYue(Backend):
    name = "music_yue"
    max_loaded = 1

    def _prep_path(self):
        link_xcodec()
        os.chdir(REPO)                       # xcodec configs use paths relative to inference/
        for p in (str(REPO), str(REPO / "xcodec_mini_infer"), str(REPO / "xcodec_mini_infer" / "descriptaudiocodec")):
            if p not in sys.path:
                sys.path.insert(0, p)

    def probe(self, model):
        self._prep_path()
        import codecmanipulator  # noqa: F401
        import mmtokenizer  # noqa: F401
        from models.soundstream_hubert_new import SoundStream  # noqa: F401
        from vocoder import build_codec_model  # noqa: F401

        return True, ""

    def load(self, model, ctx):
        import torch
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf
        from transformers import AutoModelForCausalLM

        self._prep_path()
        from codecmanipulator import CodecManipulator
        from mmtokenizer import _MMSentencePieceTokenizer
        from models.soundstream_hubert_new import SoundStream
        from vocoder import build_codec_model

        xc = REPO / "xcodec_mini_infer"
        ctx.progress(0, 4, "loading", "tokenizer + xcodec", interruptible=False)
        tok = _MMSentencePieceTokenizer(str(REPO / "mm_tokenizer_v0.2_hf" / "tokenizer.model"))
        cfg = OmegaConf.load(xc / "final_ckpt" / "config.yaml")
        codec = eval(cfg.generator.name)(**cfg.generator.config)
        sd = torch.load(xc / "final_ckpt" / "ckpt_00360000.pth", map_location="cpu", weights_only=False)
        codec.load_state_dict(sd["codec_model"])
        codec.to("cuda").eval()
        ctx.progress(1, 4, "loading", "vocoder decoders", interruptible=False)
        vdec, idec = build_codec_model(str(xc / "decoders" / "config.yaml"), str(xc / "decoders" / "decoder_131000.pth"),
                                       str(xc / "decoders" / "decoder_151000.pth"))
        vdec.to("cuda").eval()
        idec.to("cuda").eval()
        ctx.progress(2, 4, "loading", "stage 1 (7B)", interruptible=False)
        s1 = AutoModelForCausalLM.from_pretrained(snapshot_download(S1, local_files_only=True), torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa").to("cuda").eval()
        ctx.progress(3, 4, "loading", "stage 2 (1B)", interruptible=False)
        s2 = AutoModelForCausalLM.from_pretrained(snapshot_download(S2, local_files_only=True), torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa").to("cuda").eval()
        torch.cuda.synchronize()
        return {"tok": tok, "codec": codec, "vdec": vdec, "idec": idec, "s1": s1, "s2": s2,
                "ct1": CodecManipulator("xcodec", 0, 1), "ct2": CodecManipulator("xcodec", 0, 8)}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    # ------------------------------------------------------------------ run
    def run(self, job, ctx):
        import copy
        from collections import Counter

        import soundfile as sf
        import torch
        from einops import rearrange
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        self._prep_path()
        h = self.handle(job.model_id)
        tok, codec, ct1, ct2 = h["tok"], h["codec"], h["ct1"], h["ct2"]
        p = job.params
        genres = (p.get("prompt") or "").strip()
        lyr = split_lyrics((p.get("lyrics") or "").strip())
        if not genres:
            raise InputError("Give the genre tags first (genre, mood, instrument, vocal timbre and gender, space separated).")
        if not lyrics_ok(lyr):
            raise InputError("YuE needs lyrics with section tags, e.g. [verse] ... [chorus] ... (one tag per section).")
        n = int(p.get("batch", 1))
        s0 = seed0(p)
        nseg = max(1, min(int(p.get("segments", 1)), len(lyr)))
        seg_s = float(p.get("seconds", 30))
        max_new = int(max(600, min(3000, seg_s * 100)))          # 2 interleaved tokens per frame, 50 frames / s
        rep = float(p.get("repetition_penalty", 1.1))
        top_p, temp = float(p.get("top_p", 0.93)), float(p.get("temperature", 1.0))
        cfg1 = float(p.get("cfg_scale", 1.5))
        ref = job.input(role="ref") or job.input(type="audio")

        class Block(LogitsProcessor):
            def __init__(self, a, b):
                self.ids = list(range(a, b))

            def __call__(self, input_ids, scores):
                scores[:, self.ids] = -float("inf")
                return scores

        class Watch(StoppingCriteria):
            def __init__(self, base, label, total):
                self.base, self.label, self.total, self.k = base, label, total, 0

            def __call__(self, input_ids, scores, **kw):
                self.k += 1
                ctx.check()
                if self.k % 20 == 0:
                    ctx.progress(self.k, self.total, "sampling", f"{self.label}: {self.k}/{self.total} tokens")
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        def read_mono16(path):
            y, sr = sf.read(path, dtype="float32", always_2d=True)
            y = torch.from_numpy(y.mean(axis=1))[None]
            if sr != 16000:
                import torchaudio

                y = torchaudio.functional.resample(y, sr, 16000)
            return y

        torch.cuda.reset_peak_memory_stats()
        audio_prompt_codec = None
        if ref:
            ctx.log("Encoding the reference clip (first 30 s) as an audio prompt.")
            y = read_mono16(ref["path"])[:, : 16000 * 30]
            with torch.no_grad():
                raw = codec.encode(y[None].to("cuda"), target_bw=0.5).transpose(0, 1).cpu().numpy().astype(np.int16)
            audio_prompt_codec = ct1.npy2ids(raw[0])[: 30 * TPS]

        full = "\n".join(lyr)
        head_text = f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full}"
        soseg, eoseg = tok.tokenize("[start_of_segment]"), tok.tokenize("[end_of_segment]")
        results = []
        for t in range(n):
            ctx.check()
            seed = s0 + t
            random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            raw_out = None
            for si in range(nseg):
                section = lyr[si].replace("[start_of_segment]", "").replace("[end_of_segment]", "")
                gs = cfg1 if si == 0 else min(cfg1, 1.2)
                if si == 0:
                    head = tok.tokenize(head_text)
                    if audio_prompt_codec is not None:
                        ap = [tok.soa] + ct1.sep_ids + audio_prompt_codec + [tok.eoa]
                        head = head + tok.tokenize("[start_of_reference]") + ap + tok.tokenize("[end_of_reference]")
                    prompt_ids = head + soseg + tok.tokenize(section) + [tok.soa] + ct1.sep_ids
                else:
                    prompt_ids = eoseg + soseg + tok.tokenize(section) + [tok.soa] + ct1.sep_ids
                prompt = torch.as_tensor(prompt_ids).unsqueeze(0).to("cuda")
                inp = torch.cat([raw_out, prompt], dim=1) if raw_out is not None else prompt
                max_ctx = 16384 - max_new - 1
                if inp.shape[-1] > max_ctx:
                    inp = inp[:, -max_ctx:]
                label = f"take {t + 1}/{n} stage 1 segment {si + 1}/{nseg}"
                with torch.no_grad():
                    out = h["s1"].generate(
                        input_ids=inp, max_new_tokens=max_new, min_new_tokens=100, do_sample=True, top_p=top_p,
                        temperature=temp, repetition_penalty=rep, eos_token_id=tok.eoa, pad_token_id=tok.eoa,
                        logits_processor=LogitsProcessorList([Block(0, 32002)]),
                        stopping_criteria=StoppingCriteriaList([Watch(0, label, max_new)]), guidance_scale=gs)
                if out[0][-1].item() != tok.eoa:
                    out = torch.cat((out, torch.as_tensor([[tok.eoa]]).to("cuda")), dim=1)
                raw_out = out if si == 0 else torch.cat([raw_out, prompt, out[:, inp.shape[-1]:]], dim=1)

            ids = raw_out[0].cpu().numpy()
            soa, eoa = np.where(ids == tok.soa)[0].tolist(), np.where(ids == tok.eoa)[0].tolist()
            if len(soa) != len(eoa) or not soa:
                raise RuntimeError(f"stage 1 produced an invalid audio token layout (soa {len(soa)} / eoa {len(eoa)})")
            voc, ins = [], []
            for a, b in zip(soa, eoa):
                c = ids[a + 1:b]
                if len(c) and c[0] == 32016:
                    c = c[1:]
                c = c[: 2 * (c.shape[0] // 2)]
                if not len(c):
                    continue
                pair = rearrange(c, "(n b) -> b n", b=2)
                voc.append(ct1.ids2npy(pair[0]))
                ins.append(ct1.ids2npy(pair[1]))
            if not voc:
                raise RuntimeError("stage 1 produced no audio tokens")
            tracks1 = [np.concatenate(voc, axis=1), np.concatenate(ins, axis=1)]
            if tracks1[0].shape[-1] < 6 * TPS:
                raise RuntimeError("stage 1 output is shorter than 6 s; raise 'seconds'")

            # ---- stage 2 (6 s windows, teacher forced over the first codebook)
            tracks2 = []
            for ti, prm in enumerate(tracks1):
                prm = prm.astype(np.int32)
                dur = prm.shape[-1] // TPS // 6 * 6
                nb = dur // 6
                bs = int(p.get("stage2_batch", 4))
                outs = []
                for b0 in range(0, nb, bs):
                    cb = min(bs, nb - b0)
                    outs.append(self._stage2(h, prm[:, b0 * 300:(b0 + cb) * 300], cb, f"take {t + 1}/{n} stage 2 {'vocal' if ti == 0 else 'instr.'} {b0 + 1}/{nb}", ctx, Block, LogitsProcessorList))
                if dur * TPS != prm.shape[-1]:
                    outs.append(self._stage2(h, prm[:, dur * TPS:], 1, f"take {t + 1}/{n} stage 2 tail", ctx, Block, LogitsProcessorList))
                o = ct2.ids2npy(np.concatenate(outs, axis=0))
                fixed = copy.deepcopy(o)
                for r, line in enumerate(o):
                    bad = (line < 0) | (line > 1023)
                    if bad.any():
                        fixed[r, bad] = Counter(line.tolist()).most_common(1)[0][0]
                tracks2.append(fixed)

            # ---- decode: 16 kHz xcodec mix + 44.1 kHz vocos mix
            ctx.progress(1, 1, "decoding", "codec + vocoder", interruptible=False)
            wav = self._decode(h, tracks2)
            peak = float(np.abs(wav).max())
            if not np.isfinite(peak) or peak < 1e-5:
                raise RuntimeError("the model produced silence")
            wav = wav / peak * 10 ** (-1 / 20)
            f = job.out_dir / f"take_{t + 1}.wav"
            sf.write(str(f), wav, 44100, subtype="PCM_24")
            ctx.output(f, "audio", seed=seed, index=t, name=take_name(genres, t, n), final_prompt=genres,
                       sample_rate=44100, duration=round(len(wav) / 44100, 3))
            results.append(len(wav) / 44100)
        return {"meta": {"segments": nseg, "durations": results,
                         "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}}

    def _stage2(self, h, prompt, bs, label, ctx, Block, LPL):
        import torch

        tok, ct1 = h["tok"], h["ct1"]
        codec_ids = ct1.unflatten(prompt, n_quantizer=1)
        codec_ids = ct1.offset_tok_ids(codec_ids, global_offset=ct1.global_offset, codebook_size=ct1.codebook_size,
                                       num_codebooks=ct1.num_codebooks).astype(np.int32)
        if bs > 1:
            codec_ids = np.concatenate([codec_ids[:, i * 300:(i + 1) * 300] for i in range(bs)], axis=0)
            pid = np.concatenate([np.tile([tok.soa, tok.stage_1], (bs, 1)), codec_ids, np.tile([tok.stage_2], (bs, 1))], axis=1)
        else:
            pid = np.concatenate([np.array([tok.soa, tok.stage_1]), codec_ids.flatten(), np.array([tok.stage_2])]).astype(np.int32)[None]
        codec_ids = torch.as_tensor(codec_ids).to("cuda")
        pid = torch.as_tensor(pid).to("cuda")
        n0 = pid.shape[-1]
        block = LPL([Block(0, 46358), Block(53526, tok.vocab_size)])
        total = codec_ids.shape[1]
        for fi in range(total):
            ctx.check()
            if fi % 15 == 0:
                ctx.progress(fi, total, "sampling", f"{label}: frame {fi}/{total}")
            pid = torch.cat([pid, codec_ids[:, fi:fi + 1]], dim=1)
            with torch.no_grad():
                pid = h["s2"].generate(input_ids=pid, min_new_tokens=7, max_new_tokens=7, eos_token_id=tok.eoa,
                                       pad_token_id=tok.eoa, logits_processor=block)
        out = pid.cpu().numpy()[:, n0:]
        return np.concatenate([out[i] for i in range(out.shape[0])], axis=0) if bs > 1 else out[0]

    def _decode(self, h, tracks2):
        import torch
        import torchaudio.functional as F

        codec = h["codec"]
        n_min = min(t.shape[-1] for t in tracks2)
        lo, hi = [], []
        for t, dec in zip(tracks2, (h["vdec"], h["idec"])):
            codes = torch.as_tensor(t[:, :n_min].astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to("cuda")
            with torch.no_grad():
                lo.append(codec.decode(codes).cpu().squeeze(0))                       # (1, T16)
                emb = codec.get_embed(codes)
                hi.append(dec(torch.as_tensor(emb).to("cuda")).detach().cpu())       # (1, T44)
        a = lo[0] + lo[1]
        b = hi[0] + hi[1]
        a = F.resample(a, 16000, 44100)
        a_low = F.lowpass_biquad(a, 44100, 5500.0)
        b_low = F.lowpass_biquad(b, 44100, 5500.0)
        a_low = a_low * ((b_low.pow(2).mean().sqrt().item() + 1e-10) / (a_low.pow(2).mean().sqrt().item() + 1e-10))
        b_high = F.highpass_biquad(b, 44100, 5500.0)
        m = min(a_low.shape[1], b_high.shape[1])
        return (a_low[:, :m] + b_high[:, :m])[0].numpy().astype(np.float32)


def lyrics_ok(segs) -> bool:
    return bool(segs) and any(s.split("\n", 1)[1].strip() for s in segs if "\n" in s)


if __name__ == "__main__":
    serve(MusicYue())
