"""DiffSinger backend (envs/vc_diffsinger).

  diffsinger-opencpop   task sing   score (Mandarin lyrics + notes + durations) -> singing, 24 kHz, one female voice.

Upstream is the ORIGINAL MoonInTheRiver/DiffSinger (vendor/diffsinger-legacy, MIT code), end-to-end Opencpop
model "0228_opencpop_ds100_rel" (lyrics + MIDI -> mel, shallow diffusion K=100) + the HiFi-GAN "0109_hifigan_bigpopcs_hop128"
+ pitch extractor "0102_xiaoma_pe", all from HF PillowTa1k/DiffSinger. The openvpi/DiffSinger fork (vendor/diffsinger)
uses another checkpoint format and has no open voicebank on the Hub, so it is vendored but not wired.

Score format (the `score` param), one word per line:
    <word> <note> <seconds>
    <word> <note>,<note2> <sec>,<sec2>      # a slur: the vowel is held over several notes
  word  a Chinese character, a pinyin syllable (xiao), SP (short pause) or AP (breath)
  note  C4, C#4, Db4 ... or rest
Lines starting with # are comments. A JSON object (upstream's {"text","notes","notes_duration"} or the phoneme-level
{"ph_seq","note_seq","note_dur_seq","is_slur_seq"}) is accepted as well.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

STUDIO = Path(__file__).resolve().parent.parent
LEGACY = STUDIO / "vendor" / "diffsinger-legacy"
REPO = "PillowTa1k/DiffSinger"
EXP = "0228_opencpop_ds100_rel"
CONFIG = "usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml"
NOTE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)(/.*)?$")


def seed_all(s: int):
    import torch

    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def save_wav(path: Path, wav, sr: int) -> dict:
    import soundfile as sf

    w = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if len(w) < int(0.05 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    peak = float(np.abs(w).max())
    meta = {"sample_rate": int(sr)}
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def transpose(note: str, key: int) -> str:
    if note == "rest" or not key:
        return note
    m = NOTE.match(note)
    if not m:
        return note
    import librosa

    midi = librosa.note_to_midi(m.group(1).upper() + m.group(2) + m.group(3)) + key
    return librosa.midi_to_note(midi, unicode=False)


def parse_score(text: str, pinyin2ph: dict, key: int = 0, speed: float = 1.0) -> dict:
    """The line format described in the module doc -> upstream's phoneme-level input dict."""
    from pypinyin import lazy_pinyin

    text = (text or "").strip()
    if not text:
        raise InputError("Write the score first: one word per line as '<word> <note> <seconds>' (e.g. 你 C4 0.5).")
    if text.startswith("{"):
        try:
            d = json.loads(text)
        except ValueError as e:
            raise InputError(f"The score looks like JSON but is not valid: {e}")
        if "notes_duration" in d and (key or speed != 1.0):
            d["notes_duration"] = " | ".join(" ".join(f"{float(x) / speed:.4f}" for x in w.split())
                                             for w in d["notes_duration"].split("|"))
        return d
    ph, notes, durs, slurs, words = [], [], [], [], []
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip() if line.lstrip().startswith("#") else line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise InputError(f"Line {ln} ('{line}'): expected '<word> <note> <seconds>'.")
        word, nts, ds = parts
        nts, ds = nts.split(","), ds.split(",")
        if len(nts) != len(ds):
            raise InputError(f"Line {ln}: {len(nts)} notes but {len(ds)} durations.")
        if word in pinyin2ph:
            py = word
        else:
            cand = [p for p in lazy_pinyin(word, strict=False) if p in pinyin2ph]
            if len(cand) != 1:
                raise InputError(f"Line {ln}: '{word}' is not one Mandarin syllable (use a single character or pinyin like 'xiao').")
            py = cand[0]
        phs = pinyin2ph[py].split()
        for n in nts:
            if n != "rest" and not NOTE.match(n):
                raise InputError(f"Line {ln}: '{n}' is not a note name (C4, C#4, Db4 ... or rest).")
        try:
            dv = [float(d) / speed for d in ds]
        except ValueError:
            raise InputError(f"Line {ln}: durations must be numbers in seconds.")
        if any(d <= 0 for d in dv):
            raise InputError(f"Line {ln}: durations must be positive.")
        words.append(py)
        for p in phs:
            ph.append(p); notes.append(transpose(nts[0], key)); durs.append(dv[0]); slurs.append(0)
        for n, d in zip(nts[1:], dv[1:]):
            ph.append(phs[-1]); notes.append(transpose(n, key)); durs.append(d); slurs.append(1)
    return {"text": " ".join(words), "ph_seq": " ".join(ph), "note_seq": " ".join(notes),
            "note_dur_seq": " ".join(f"{d:.6f}" for d in durs), "is_slur_seq": " ".join(map(str, slurs)),
            "input_type": "phoneme"}


class Handle:
    def __init__(self, infer, sr):
        self.infer = infer
        self.sr = sr


class DSBackend(Backend):
    name = "vc_diffsinger"
    max_loaded = 1

    def probe(self, model):
        if not (LEGACY / "inference" / "svs" / "ds_e2e.py").exists():
            return False, "vendor/diffsinger-legacy is missing: git clone https://github.com/MoonInTheRiver/DiffSinger vendor/diffsinger-legacy"
        import torch  # noqa: F401
        import pypinyin  # noqa: F401
        import pycwt  # noqa: F401
        return True, ""

    def load(self, model, ctx):
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        import scipy.signal
        import scipy.signal.windows as sw
        import torch
        from huggingface_hub import snapshot_download

        if not hasattr(scipy.signal, "kaiser"):          # removed in scipy 1.13, upstream pqmf.py still imports it
            scipy.signal.kaiser = sw.kaiser
        snap = Path(snapshot_download(REPO, local_files_only=True, allow_patterns=[
            f"{EXP}/*", "0109_hifigan_bigpopcs_hop128/*", "0102_xiaoma_pe/*"]))
        ck = LEGACY / "checkpoints"
        ck.mkdir(exist_ok=True)
        for d in (EXP, "0109_hifigan_bigpopcs_hop128", "0102_xiaoma_pe"):
            link = ck / d
            if link.is_symlink() or link.exists():
                if link.resolve() != (snap / d).resolve():
                    link.unlink()
                else:
                    continue
            link.symlink_to(snap / d)
        os.chdir(LEGACY)                                  # upstream reads configs and the dictionary by relative path
        sys.path.insert(0, str(LEGACY))
        from utils.hparams import hparams, set_hparams

        set_hparams(config=CONFIG, exp_name=EXP, print_hparams=False)
        from inference.svs.ds_e2e import DiffSingerE2EInfer

        infer = DiffSingerE2EInfer(hparams, device="cuda" if torch.cuda.is_available() else "cpu")
        return Handle(infer, int(hparams["audio_sample_rate"]))

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import torch

        h: Handle = self.handle(job.model_id)
        p = job.params
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        inp = parse_score(p.get("score") or p.get("text") or p.get("prompt") or "", h.infer.pinyin2phs,
                          key=int(p.get("key", 0)), speed=float(p.get("speed", 1.0)))
        n = max(1, int(p.get("batch", 1)))
        s0 = int(p.get("seed", -1)) if p.get("seed") is not None else -1
        s0 = s0 if s0 >= 0 else random.randrange(2**31)
        for i in range(n):
            ctx.check()
            seed_all(s0 + i)
            t0 = time.time()
            ctx.progress(i, n, "singing", f"take {i + 1}/{n}", interruptible=False)
            try:
                wav = h.infer.infer_once(dict(inp))
            except (KeyError, AssertionError, ValueError) as e:
                raise InputError(f"The score could not be used: {e}")
            f = job.out_dir / f"take_{i + 1}.wav"
            meta = save_wav(f, wav, h.sr)
            label = (inp.get("text") or "score")[:40]
            ctx.output(f, "audio", seed=s0 + i, index=i, name=label + (f" #{i + 1}" if n > 1 else ""),
                       run_s=round(time.time() - t0, 2), **meta)
        return {"meta": {"model": job.model_id, "phonemes": len(inp["ph_seq"].split()) if "ph_seq" in inp else None,
                         "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None}}


if __name__ == "__main__":
    serve(DSBackend())
