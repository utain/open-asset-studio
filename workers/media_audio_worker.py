"""Backend `media_audio` (envs/media_audio): production audio tools for video / podcast work.

  subtitles   Whisper (large-v3-turbo: transcribe; large-v3: transcribe + translate to English) -> SRT + VTT + JSON
              (cues with word timestamps). Audio or video input (audio is decoded with ffmpeg).
  separate    Stem separation: Demucs (htdemucs, htdemucs_ft, htdemucs_6s) or Mel-Band RoFormer (vocals).
              Output: one audio asset per stem + the instrumental (everything but the vocals).

The dubbing pipeline is a server-side handler (app/plugins/media_audio.py) that calls this worker for `separate`
and `subtitles`, then a voice-clone TTS worker per line. Pure helpers (SRT/VTT, cue splitting, time-fit) live in
`media_audio_common.py` so the tests can run without torch.

Weights come from the HF cache (HF_HUB_OFFLINE=1):
  openai/whisper-large-v3-turbo, openai/whisper-large-v3        MIT
  adefossez/HTDemucs, HTDemucs-ft, HTDemucs-6s                  MIT (official mirrors of the Demucs checkpoints)
  KimberleyJSN/melbandroformer                                  MIT
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import media_audio_common as C  # noqa: E402
from studio_worker import Backend, InputError, serve  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

# Mel-Band RoFormer "vocals" (Kimberley Jensen): the config of the checkpoint (ZFTurbo's Music-Source-Separation-Training)
MELROFORMER_CFG = dict(dim=384, depth=6, stereo=True, num_stems=1, time_transformer_depth=1, freq_transformer_depth=1,
                       num_bands=60, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, flash_attn=True,
                       dim_freqs_in=1025, sample_rate=44100, stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048,
                       stft_normalized=False, mask_estimator_depth=2, multi_stft_resolution_loss_weight=1.0,
                       multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256), multi_stft_hop_size=147,
                       multi_stft_normalized=False)
MELROFORMER_CHUNK = 352800


# ------------------------------------------------------------------ audio io (ffmpeg)
def decode(path: str, sr: int, channels: int, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Any audio / video file -> float32 array [channels, samples] through ffmpeg (first audio stream)."""
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-protocol_whitelist", "file", "-i", str(path)]
    if start:
        cmd[6:6] = ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-map", "0:a:0", "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        err = p.stderr.decode("utf8", "replace").strip().splitlines()
        if err and "does not contain any stream" in err[-1] or err and "matches no streams" in err[-1]:
            raise InputError("The file has no audio track.")
        raise InputError("Could not read the audio of this file" + (f": {err[-1][:200]}" if err else "."))
    x = np.frombuffer(p.stdout, dtype=np.float32)
    n = (len(x) // channels) * channels
    return x[:n].reshape(-1, channels).T.copy()


def write_audio(path: Path, x: np.ndarray, sr: int, fmt: str = "wav"):
    import soundfile as sf

    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), x.T, sr, subtype="PCM_24" if fmt in ("wav", "flac") else None, format=fmt.upper())


def audio_input(job):
    for role in ("media", "audio", "source", "video"):
        r = job.input(role=role)
        if r:
            return r
    r = next((i for i in job.inputs if i.get("type") in ("audio", "voice", "video")), None)
    if not r:
        raise InputError("Pick an audio or video file.")
    return r


def _sentence_end(t: str) -> bool:
    return (t or "").rstrip().endswith((".", "?", "!", "\u3002", "\uff01", "\uff1f", "\u2026"))


def refine_edges(words: list[dict], y16: np.ndarray, hop: float = 0.02) -> list[dict]:
    """Whisper's attention often starts a phrase early and ends it late; trim leading / trailing silence of each run of
    words (a run = words closer than 0.3 s) using the short-time energy of the audio itself. Runs also end at sentence punctuation."""
    if not words or len(y16) < 1600:
        return words
    n = int(hop * 16000)
    m = len(y16) // n
    e = 20 * np.log10(np.sqrt((y16[: m * n].reshape(m, n) ** 2).mean(axis=1)) + 1e-7)
    floor, top = np.percentile(e, 10), np.percentile(e, 95)
    thr = max(floor + 10.0, top - 32.0)
    act = e > thr
    out = [dict(w) for w in words]
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1]["start"] - out[j]["end"] < 0.3 and not _sentence_end(out[j]["text"]):
            j += 1       # a run = one sentence / breath group (so a sentence parked early inside a pause is still trimmed)
        a, b = int(out[i]["start"] / hop), min(int(out[j]["end"] / hop) + 1, m)
        sustained = act[a:b].copy()                 # a click or the reverb tail of the previous line is not speech
        if len(sustained) > 8:
            sustained = np.convolve(sustained.astype(float), np.ones(8) / 8, mode="same") >= 0.75
        idx = np.nonzero(act[a:b] & sustained)[0]
        if idx.size:
            s0 = max((a + int(idx[0])) * hop - 0.03, 0.0)
            e0 = (a + int(idx[-1]) + 1) * hop + 0.05
            if s0 - out[i]["start"] > 0.12:
                out[i]["start"] = round(min(s0, out[i]["end"] - 0.02), 3)
                for k in range(i + 1, j + 1):          # keep order if later words started before the new start
                    out[k]["start"] = max(out[k]["start"], out[i]["start"])
            if out[j]["end"] - e0 > 0.15:
                out[j]["end"] = round(max(e0, out[j]["start"] + 0.02), 3)
        i = j + 1
    return out


# ------------------------------------------------------------------ backend
class MediaAudioBackend(Backend):
    name = "media_audio"
    max_loaded = 2

    def probe(self, model):
        eng = (model.get("features") or {}).get("engine")
        if eng == "whisper":
            import transformers  # noqa: F401
        elif eng == "demucs":
            import demucs.apply  # noqa: F401
        elif eng == "roformer":
            import bs_roformer  # noqa: F401
            import librosa  # noqa: F401
        import soundfile  # noqa: F401
        return True, ""

    # ---- load
    def load(self, model, ctx):
        import torch

        eng = (model.get("features") or {}).get("engine")
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        if eng == "whisper":
            from transformers import pipeline

            repo = model["features"]["repo"]
            return {"engine": "whisper", "asr": pipeline("automatic-speech-recognition", repo, dtype=torch.float16, device="cuda")}
        if eng == "demucs":
            from demucs.hf import get_hf_model

            bag = get_hf_model(model["features"]["demucs_name"])
            bag.eval().to("cuda")
            return {"engine": "demucs", "model": bag}
        if eng == "roformer":
            from bs_roformer import MelBandRoformer
            from huggingface_hub import hf_hub_download

            # the checkpoint predates the (optional) linear-attention slot at index 0 of every layer: shift its keys by one
            net = MelBandRoformer(**MELROFORMER_CFG, linear_transformer_depth=0)
            sd = torch.load(hf_hub_download("KimberleyJSN/melbandroformer", "MelBandRoformer.ckpt"), map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {re.sub(r"^layers\.(\d+)\.(\d+)\.", lambda m: f"layers.{m.group(1)}.{int(m.group(2)) + 1}.", k): v for k, v in sd.items()}
            net.load_state_dict(sd, strict=True)
            return {"engine": "roformer", "model": net.eval().to("cuda")}
        raise InputError(f"unknown model {model['id']}")

    def unload(self, model_id):
        self.models.pop(model_id, None)

    # ---- run
    def run(self, job, ctx):
        h = self.handle(job.model_id)
        if job.task == "subtitles":
            return self.run_subtitles(job, ctx, h)
        if job.task == "separate":
            return self.run_separate(job, ctx, h)
        raise InputError(f"{job.model_id} cannot do '{job.task}'")

    # ================================================================ subtitles
    def run_subtitles(self, job, ctx, h):
        import torch

        p = job.params
        asr = h["asr"]
        model = asr.model
        translate = bool(p.get("translate"))
        if translate and not (job.model.get("features") or {}).get("translate"):
            raise InputError("This Whisper model only transcribes (large-v3-turbo was not trained to translate). "
                             "Pick Whisper large-v3 to translate to English.")
        items = [i for i in job.inputs if i.get("type") in ("audio", "voice", "video")]
        if not items:
            raise InputError("Pick an audio or video file.")
        lang_req = C.norm_lang(p.get("language"))
        max_chars = int(p.get("max_chars") or 42)
        max_lines = int(p.get("max_lines") or 2)
        max_dur = float(p.get("max_cue_s") or 7.0)
        n = len(items)
        out_meta = {}
        for k, inp in enumerate(items):
            ctx.check()
            name = inp.get("name") or Path(inp["path"]).stem
            ctx.progress(k, n, "decoding", f"Reading {name}")
            y = decode(inp["path"], 16000, 1)[0]
            dur = len(y) / 16000
            if dur < 0.3:
                raise InputError(f"{name}: the audio is empty.")
            # --- language
            lang = lang_req
            if lang is None:
                lang = self.detect_language(asr, y[: 16000 * 30])
                ctx.log(f"{name}: detected language {lang or '?'}")
            gk: dict = {"task": "translate" if translate else "transcribe"}
            if lang:
                gk["language"] = C.LANG_NAMES.get(lang, lang).lower()
            if lang in (None, "") and translate:
                raise InputError("The language could not be detected; choose it in the language field.")
            ctx.progress(k, n, "transcribing", f"Whisper: {name} ({dur:.0f} s)")
            t0 = time.time()
            # Whisper segments (fast: 0.7 s per minute), then word times by cross-attention DTW on our own forward pass
            # (the pipeline's return_timestamps="word" copies the attention caches to the CPU per token: ~1 s per second of audio here)
            res = asr({"raw": y, "sampling_rate": 16000}, return_timestamps=True, generate_kwargs=gk)
            ctx.check()
            text = (res.get("text") or "").strip()
            segs = [{"text": c["text"], "start": (c.get("timestamp") or (0.0, None))[0] or 0.0,
                     "end": (c.get("timestamp") or (0.0, None))[1]} for c in res.get("chunks") or [] if str(c.get("text", "")).strip()]
            ctx.progress(k, n, "aligning", f"Word timestamps: {name}")
            try:
                words = self.align_words(asr, y, segs, gk["task"], lang or "en", dur)
                aligned = "dtw"
            except Exception as e:  # noqa: BLE001 - keep going with interpolated word times
                ctx.log(f"word alignment failed ({type(e).__name__}: {e}); interpolating inside segments")
                words, aligned = C.interpolate_words(segs, dur), "interpolated"
            words = C.clean_words(words, dur)
            words = refine_edges(words, y)
            if not words:
                raise InputError(f"{name}: no speech found.")
            out_lang = "en" if translate else lang
            cues = C.words_to_segments(words, max_chars=max_chars, max_lines=max_lines, max_dur=max_dur,
                                       min_dur=float(p.get("min_cue_s") or 0.9), pause=float(p.get("pause_s") or 0.7),
                                       sentences=bool(p.get("split_sentences", True)))
            stem = f"subtitles_{k + 1}"
            fmts = p.get("main_format") or "srt"
            (job.out_dir / f"{stem}.srt").write_text(C.to_srt(cues), encoding="utf-8")
            (job.out_dir / f"{stem}.vtt").write_text(C.to_vtt(cues), encoding="utf-8")
            doc = {"language": out_lang, "source_language": lang, "task": gk["task"], "duration": round(dur, 3), "text": text,
                   "segments": [{"start": c["start"], "end": c["end"], "text": c["text"]} for c in cues],
                   "words": [{"text": w["text"], "start": w["start"], "end": w["end"]} for w in words], "word_timing": aligned,
                   "model": job.model_id}
            (job.out_dir / f"{stem}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            main = f"{stem}.vtt" if fmts == "vtt" else f"{stem}.srt"
            attach = [f"{stem}.vtt" if main.endswith(".srt") else f"{stem}.srt", f"{stem}.json"]
            ctx.output(job.out_dir / main, "text", name=f"{name} ({out_lang or 'subtitles'} subtitles)", attach=attach,
                       index=k, language=out_lang, source_language=lang, cues=len(cues), word_timing=aligned, duration=round(dur, 2),
                       seconds=round(time.time() - t0, 2), preview=" ".join(c["text"].replace("\n", " ") for c in cues[:3])[:300])
            out_meta[inp["asset_id"]] = {"language": out_lang, "cues": len(cues)}
            ctx.progress(k + 1, n, "transcribing", f"{name}: {len(cues)} cues")
        return {"meta": {"subtitles": out_meta}}


    # ---- word timestamps (Whisper's own method: cross-attention heads + DTW)
    def align_words(self, asr, y, segs, task, lang, dur):
        import torch
        from scipy.ndimage import median_filter

        model, tok, fe = asr.model, asr.tokenizer, asr.feature_extractor
        heads = [tuple(h) for h in model.generation_config.alignment_heads]
        prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>", "<|notimestamps|>"])
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        # windows of whole segments, at most 30 s of audio each
        groups, cur = [], []
        for sg in segs:
            e = sg["end"] if sg["end"] is not None else min(dur, sg["start"] + 10)
            lead = (groups[-1][-1]["end"] - 0.3) if groups else cur[0]["start"] - 1.5 if cur else 0.0
            if cur and e + 1.0 - max(lead, 0) > 28.0:
                groups.append(cur)
                cur = []
            cur.append({**sg, "end": e})
        if cur:
            groups.append(cur)
        words: list[dict] = []
        model.set_attn_implementation("eager")      # output_attentions needs the eager attention
        try:
            for g in groups:
                # Whisper's own segment times can be ~1 s off, so the window starts early (not before the previous window's last word)
                # (a segment start can also be too LATE, e.g. after a long digital silence: so start right after the previous words)
                w0 = max(g[0]["start"] - 1.5, 0.0)
                if words:
                    w0 = max(words[-1]["end"] - 0.3, 0.0)
                w1 = min(max(g[-1]["end"] + 1.0, w0 + 1.0), dur, w0 + 30.0)
                yw = y[int(w0 * 16000):int(w1 * 16000)]
                text = "".join(sg["text"] for sg in g)
                toks = tok.encode(text, add_special_tokens=False)
                if not toks or len(yw) < 800:
                    continue
                feats = fe(yw, sampling_rate=16000, return_tensors="pt").input_features.to(model.device, model.dtype)
                ids = torch.tensor([prefix + toks + [eot]], device=model.device)
                with torch.no_grad():
                    out = model(input_features=feats, decoder_input_ids=ids, output_attentions=True)
                w = torch.stack([out.cross_attentions[l][0, h] for l, h in heads]).float()
                nf = max(2, int(len(yw) / 16000 * 50))
                w = w[:, :, :nf]
                w = (w - w.mean(dim=-2, keepdim=True)) / (w.std(dim=-2, keepdim=True, unbiased=False) + 1e-6)
                wn = median_filter(w.cpu().numpy(), size=(1, 1, 7))
                mat = wn.mean(axis=0)[3:-1]                  # rows: <|notimestamps|> + the text tokens
                ti, tj = C.dtw(-mat)
                jump = np.pad(np.diff(ti), (1, 0), constant_values=1).astype(bool)
                times = tj[jump] / 50.0
                units = C.split_token_units(tok, toks, spaces=lang not in C.NO_SPACE_LANGS)
                bounds = np.pad(np.cumsum([len(u[1]) for u in units[:-1]]), (1, 0)).astype(int) if units else []
                for ui, (txt, ut) in enumerate(units):
                    a = bounds[ui]
                    b = bounds[ui + 1] if ui + 1 < len(bounds) else len(toks)
                    if a >= len(times):
                        break
                    st = float(times[a])
                    en = float(times[min(b, len(times) - 1)])
                    words.append({"text": txt, "start": w0 + st, "end": w0 + max(en, st + 0.02)})
        finally:
            model.set_attn_implementation("sdpa")
        if not words:
            raise RuntimeError("no words aligned")
        return words

    @staticmethod
    def detect_language(asr, y16) -> str | None:
        """Whisper language id on the first 30 s -> ISO code (None when unsure)."""
        import torch

        fe, tok, model = asr.feature_extractor, asr.tokenizer, asr.model
        feats = fe(y16, sampling_rate=16000, return_tensors="pt").input_features.to(model.device, model.dtype)
        with torch.no_grad():
            ids = model.detect_language(input_features=feats)
        tid = int(ids.reshape(-1)[0])
        s = tok.convert_ids_to_tokens(tid)
        code = str(s).strip("<|>")
        return C.norm_lang(code) or (code if code in C.LANG_NAMES else None)

    # ================================================================ separate
    def run_separate(self, job, ctx, h):
        p = job.params
        inp = audio_input(job)
        fmt = p.get("format") or "wav"
        eng = h["engine"]
        name = inp.get("name") or Path(inp["path"]).stem
        ctx.progress(0, 1, "decoding", f"Reading {name}")
        mix = decode(inp["path"], 44100, 2)
        dur = mix.shape[1] / 44100
        if dur < 0.5:
            raise InputError("The audio is too short.")
        t0 = time.time()
        if eng == "demucs":
            stems = self.demucs(h["model"], mix, p, ctx)
        else:
            stems = self.roformer(h["model"], mix, p, ctx)
        ctx.check()
        want = p.get("stems") or "vocals_instrumental"
        vocals = stems["vocals"]
        others = [v for k, v in stems.items() if k != "vocals"]
        instr = np.sum(others, axis=0) if len(others) > 1 else (others[0] if others else mix - vocals)
        outs: list[tuple[str, np.ndarray]] = [("vocals", vocals), ("instrumental", instr)]
        if want == "all" and eng == "demucs":
            outs = [("vocals", vocals)] + [(k, v) for k, v in stems.items() if k != "vocals"] + [("instrumental", instr)]
        # one gain for all files so the stems still add up to the mix (and nothing clips)
        peak = max(float(np.abs(v).max()) for _, v in outs)
        g = 0.98 / peak if peak > 0.98 else 1.0
        if g != 1.0:
            ctx.log(f"peak {peak:.2f}: stems scaled by {g:.3f} to avoid clipping")
        for i, (label, x) in enumerate(outs):
            f = job.out_dir / f"{label}.{fmt}"
            write_audio(f, x * g, 44100, fmt)
            ctx.output(f, "audio", name=f"{name} - {label}", index=i, stem=label, source=inp.get("asset_id"),
                       source_seconds=round(dur, 2), separation_seconds=round(time.time() - t0, 1))
        return {"meta": {"stems": [l for l, _ in outs], "gain": round(g, 4), "seconds": round(time.time() - t0, 1)}}

    def demucs(self, bag, mix: np.ndarray, p: dict, ctx) -> dict[str, np.ndarray]:
        import torch
        from demucs.apply import apply_model

        wav = torch.from_numpy(mix)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        shifts = int(p.get("shifts") or 1)
        total_models = len(getattr(bag, "models", [bag]))

        length = wav.shape[-1]

        def cb(d):
            ctx.check()
            if d.get("state") == "end":
                done = d.get("model_idx_in_bag", 0) * shifts + d.get("shift_idx", 0) + min(d.get("segment_offset", 0) / max(length, 1), 1.0)
                ctx.progress(int(100 * done / (total_models * shifts)), 100, "separating",
                             f"model {d.get('model_idx_in_bag', 0) + 1}/{total_models}, pass {d.get('shift_idx', 0) + 1}/{shifts}")

        with torch.no_grad():
            out = apply_model(bag, wav[None], device="cuda", shifts=shifts, split=True, overlap=float(p.get("overlap") or 0.25),
                              progress=False, callback=cb)[0]
        out = out * ref.std() + ref.mean()
        out = out.cpu().numpy().astype(np.float32)
        return {s: out[i] for i, s in enumerate(bag.sources)}

    def roformer(self, net, mix: np.ndarray, p: dict, ctx) -> dict[str, np.ndarray]:
        """Overlap-add inference of the Mel-Band RoFormer (8 s chunks, 4x overlap, linear fades)."""
        import torch

        chunk = MELROFORMER_CHUNK
        overlap = max(2, int(p.get("overlap_factor") or 4))
        step = chunk // overlap
        border = chunk - step
        x = torch.from_numpy(mix)
        total = x.shape[1]
        pad = border
        if total > 2 * border and border > 0:
            x = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
        else:
            x = torch.nn.functional.pad(x, (pad, pad))
        fade = chunk // 10
        win = torch.ones(chunk)
        win[:fade] = torch.linspace(0, 1, fade)
        win[-fade:] = torch.linspace(1, 0, fade)
        win = win.to("cuda")
        starts = list(range(0, max(1, x.shape[1] - chunk + step), step))
        acc = torch.zeros(2, x.shape[1], device="cuda")
        cnt = torch.zeros(1, x.shape[1], device="cuda")
        bs = 4
        with torch.no_grad():
            for b in range(0, len(starts), bs):
                ctx.check()
                ctx.progress(b, len(starts), "separating", f"chunk {b + 1}/{len(starts)}")
                batch, lens = [], []
                for s in starts[b:b + bs]:
                    part = x[:, s:s + chunk]
                    lens.append(part.shape[1])
                    if part.shape[1] < chunk:
                        part = torch.nn.functional.pad(part, (0, chunk - part.shape[1]))
                    batch.append(part)
                y = net(torch.stack(batch).to("cuda")).float()
                for s, ln, yy in zip(starts[b:b + bs], lens, y):
                    w = win[:ln].clone()
                    if s == 0:
                        w[:fade] = 1.0                      # no fade-in on the very first chunk
                    acc[:, s:s + ln] += yy[:, :ln] * w
                    cnt[:, s:s + ln] += w
        res = (acc / cnt.clamp_min(1e-6))[:, pad:pad + total].cpu().numpy().astype(np.float32)
        return {"vocals": res}


if __name__ == "__main__":
    serve(MediaAudioBackend())
