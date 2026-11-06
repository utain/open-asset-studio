"""Voice-conversion backend (envs/vc).

  seed-vc   task voice_convert   zero-shot speech + singing voice conversion (Plachtaa/seed-vc, DiT + BigVGAN).
                                 Inputs: role `source` (the audio to convert) + role `ref` (a 3-25 s clip of the
                                 target voice). `singing: true` switches to the F0-conditioned 44.1 kHz model.

  rvc       task voice_convert   Retrieval-based Voice Conversion (RVC-Project, v1/v2 .pth voices). A voice is a trained
                                 model the user drops into data/rvc_voices/ (<name>.pth + optional <name>*.index,
                                 flat or one folder per voice). Input: role `source` only (no reference clip).

Upstream code lives in vendor/seed-vc and vendor/rvc (GPL-3.0, not on PyPI). Its loader downloads into ./checkpoints with
hf_hub_download; here that function is replaced with a resolver for the normal HF cache (offline safe).
"""
from __future__ import annotations

import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from studio_worker import Backend, Cancelled, InputError, serve

STUDIO = Path(__file__).resolve().parent.parent
VENDOR = STUDIO / "vendor"


# ------------------------------------------------------------------ helpers
def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def seed_all(s: int):
    import torch

    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def save_wav(path: Path, wav, sr: int) -> dict:
    """Write float audio (T,) as 24-bit WAV; overs are brought to -1 dBFS. Returns file meta."""
    import soundfile as sf

    w = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if len(w) < int(0.05 * sr):
        raise RuntimeError("the model returned (almost) no audio")
    meta = {"sample_rate": int(sr)}
    peak = float(np.abs(w).max())
    if peak > 0.999:
        w = w * (10 ** (-1 / 20) / peak)
        meta["peak_limited"] = round(20 * math.log10(peak), 2)
    sf.write(str(path), w, int(sr), subtype="PCM_24")
    meta["duration"] = round(len(w) / sr, 3)
    return meta


def audio_inputs(job) -> tuple[dict, dict | None]:
    """(source, reference) inputs by role, falling back to the order of the audio inputs."""
    def pick(*roles):
        for r in roles:
            i = job.input(role=r)
            if i:
                return i
        return None

    src = pick("source", "src", "input", "audio")
    ref = pick("ref", "reference", "target", "voice")
    rest = [i for i in job.inputs if i.get("type") in ("audio", "voice") and i is not src and i is not ref]
    if src is None and rest:
        src = rest.pop(0)
    if ref is None and rest:
        ref = rest.pop(0)
    if src is None:
        raise InputError("Add the source audio (the recording whose voice should change).")
    return src, ref


def peak_gb() -> float | None:
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None


def name_of(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}" + (f" #{i + 1}" if n > 1 else "")


def hf_snapshot_file(repo: str, filename: str) -> str:
    """Resolve a file of a cached HF repo without touching the network."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, local_files_only=True)


# ------------------------------------------------------------------ Seed-VC
# the F0 checkpoint named in seed_vc_wrapper.py has been superseded by the _v2 file (what inference.py uses)
SEEDVC_FILES = {
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth":
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
}


def _seedvc_loader(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    model_filename = SEEDVC_FILES.get(model_filename, model_filename)
    path = hf_snapshot_file(repo_id, model_filename)
    if config_filename is None:
        return path
    return path, hf_snapshot_file(repo_id, config_filename)


def load_seed_vc():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(VENDOR / "seed-vc"))
    import hf_utils

    hf_utils.load_custom_model_from_hf = _seedvc_loader
    import seed_vc_wrapper

    seed_vc_wrapper.load_custom_model_from_hf = _seedvc_loader
    cwd = os.getcwd()
    try:
        os.chdir(VENDOR / "seed-vc")        # a few upstream modules open files by relative path
        import torch

        w = seed_vc_wrapper.SeedVCWrapper(device=torch.device("cuda"))
    finally:
        os.chdir(cwd)
    w.bitrate = "64k"                       # streamed mp3 chunks are discarded; keep the encode cheap
    return w


# ------------------------------------------------------------------ RVC
VOICES = STUDIO / "data" / "rvc_voices"


def list_rvc_voices() -> dict[str, dict]:
    """name -> {pth, index}. A voice is any *.pth under data/rvc_voices (name = file stem). Its index is the
    non-"trained" *.index next to it (same folder) whose name starts with the stem, else the only index there."""
    out: dict[str, dict] = {}
    if not VOICES.is_dir():
        return out
    for pth in sorted(VOICES.rglob("*.pth")):
        idx = sorted(i for i in pth.parent.glob("*.index") if "trained" not in i.name.lower())
        best = next((i for i in idx if i.name.lower().startswith(pth.stem.lower())), idx[0] if len(idx) == 1 else None)
        out[pth.stem] = {"pth": pth, "index": best}
    return out


class RVCHandle:
    def __init__(self, hubert, rmvpe_dir: str):
        self.hubert = hubert
        self.rmvpe_dir = rmvpe_dir
        self.voices: dict[str, dict] = {}       # name -> {net_g, cpt-info, pipeline}


def load_rvc():
    import torch
    from huggingface_hub import snapshot_download

    os.environ["RVC_CUDA_GRAPH"] = "0"           # CUDA graphs are an upstream speed-up we do not need (and cannot test)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = str(VENDOR / "rvc")
    if root not in sys.path:
        sys.path.insert(0, root)
    snap = Path(snapshot_download("lj1995/VoiceConversionWebUI", local_files_only=True,
                                  allow_patterns=["hubert_base/*", "rmvpe.pt"]))
    import infer.hubert as H

    H.HUBERT_MODEL_PATH = (snap / "hubert_base").resolve()
    hubert = H.load_hubert_model(torch.device("cuda"), is_half=False)
    return RVCHandle(hubert, str(snap))


def get_rvc_voice(h: RVCHandle, name: str, ctx):
    import torch

    root = str(VENDOR / "rvc")
    if root not in sys.path:
        sys.path.insert(0, root)
    if name in h.voices:
        return h.voices[name]
    avail = list_rvc_voices()
    if name not in avail:
        # also accept a folder name, or a case-insensitive stem
        low = {k.lower(): k for k in avail}
        key = low.get(name.lower())
        if key is None:
            folder = [k for k, v in avail.items() if v["pth"].parent.name.lower() == name.lower()]
            key = folder[0] if len(folder) == 1 else None
        if key is None:
            have = ", ".join(sorted(avail)) or "none"
            raise InputError(f"No RVC voice called '{name}' in data/rvc_voices/ (found: {have}). "
                             "Put <name>.pth (+ <name>.index) there.")
        name = key
        if name in h.voices:
            return h.voices[name]
    v = avail[name]
    from infer.module.models import (SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono,
                                     SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono)
    from infer.vc.pipeline import Pipeline

    # RVC voices are pickles of plain dicts and tensors; weights_only=False is what upstream does too
    cpt = torch.load(str(v["pth"]), map_location="cpu", weights_only=False)
    if "config" not in cpt or "weight" not in cpt:
        raise InputError(f"{v['pth'].name} is not an RVC voice model (no config/weight in the checkpoint).")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    cls = {("v1", 1): SynthesizerTrnMs256NSFsid, ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
           ("v2", 1): SynthesizerTrnMs768NSFsid, ("v2", 0): SynthesizerTrnMs768NSFsid_nono}[(version, if_f0)]
    net_g = cls(*cpt["config"], is_half=False)
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to("cuda").float()

    class Cfg:
        x_pad, x_query, x_center, x_max = 3, 10, 60, 65
        is_half = False
        device = torch.device("cuda")

    entry = {"net_g": net_g, "tgt_sr": tgt_sr, "if_f0": if_f0, "version": version,
             "n_spk": cpt["config"][-3], "pipeline": Pipeline(tgt_sr, Cfg), "index": v["index"], "pth": v["pth"]}
    while len(h.voices) >= 2:                     # keep at most two voices resident
        h.voices.pop(next(iter(h.voices)))
    h.voices[name] = entry
    ctx.log(f"RVC voice {name}: {version}, {tgt_sr} Hz, f0={if_f0}, speakers={entry['n_spk']}, "
            f"index={'yes' if entry['index'] else 'none'}")
    return entry


class VCBackend(Backend):
    name = "vc"
    max_loaded = 1

    def probe(self, model):
        if model.get("id") == "rvc":
            if not (VENDOR / "rvc" / "infer" / "vc" / "pipeline.py").exists():
                return False, "vendor/rvc is missing: git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI vendor/rvc"
            import faiss  # noqa: F401
            import parselmouth  # noqa: F401
            import transformers  # noqa: F401
            return True, ""
        sys.path.insert(0, str(VENDOR / "seed-vc"))
        if not (VENDOR / "seed-vc" / "seed_vc_wrapper.py").exists():
            return False, "vendor/seed-vc is missing: git clone https://github.com/Plachtaa/seed-vc vendor/seed-vc"
        import torch  # noqa: F401
        import librosa  # noqa: F401
        import munch  # noqa: F401
        import dac  # noqa: F401
        return True, ""

    def load(self, model, ctx):
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        mid = model["id"]
        if mid == "seed-vc":
            return load_seed_vc()
        if mid == "rvc":
            return load_rvc()
        raise InputError(f"unknown model {mid}")

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        if job.model_id == "seed-vc":
            return self.run_seed_vc(job, ctx)
        if job.model_id == "rvc":
            return self.run_rvc(job, ctx)
        raise InputError(f"unknown model {job.model_id}")

    # --------------------------------------------------------------
    def run_seed_vc(self, job, ctx):
        import torch

        w = self.handle("seed-vc")
        torch.cuda.reset_peak_memory_stats()
        p = job.params
        src, ref = audio_inputs(job)
        if ref is None:
            raise InputError("Add the target voice: a clean 3-25 s clip of the voice to convert into.")
        singing = bool(p.get("singing", False))
        n = max(1, int(p.get("batch", 1)))
        s0 = seed0(p)
        label = src.get("name") or Path(src["path"]).stem
        for i in range(n):
            ctx.check()
            seed_all(s0 + i)
            t0 = time.time()
            gen = w.convert_voice(
                src["path"], ref["path"],
                diffusion_steps=int(p.get("steps", 30)),
                length_adjust=float(p.get("length_adjust", 1.0)),
                inference_cfg_rate=float(p.get("cfg_rate", 0.7)),
                f0_condition=singing,
                auto_f0_adjust=bool(p.get("auto_f0_adjust", True)),
                pitch_shift=int(p.get("pitch_shift", 0)),
                stream_output=True,
            )
            full = None
            chunks = 0
            for _mp3, audio in gen:
                chunks += 1
                ctx.check()
                ctx.progress(chunks, 0, "converting", f"take {i + 1}/{n}: chunk {chunks}")
                if audio is not None:
                    full = audio
            if full is None:
                raise RuntimeError("Seed-VC returned no audio")
            sr, wav = full
            f = job.out_dir / f"take_{i + 1}.wav"
            meta = save_wav(f, wav, sr)
            ctx.output(f, "audio", seed=s0 + i, index=i, name=name_of(f"VC {label}", i, n),
                       mode="singing" if singing else "speech", run_s=round(time.time() - t0, 2), **meta)
        return {"meta": {"model": "seed-vc", "singing": singing, "gpu_gb": ctx.gpu_gb(), "peak_gb": peak_gb()}}

    def run_rvc(self, job, ctx):
        import librosa
        import soundfile as sf

        import torch

        torch.cuda.reset_peak_memory_stats()
        h: RVCHandle = self.handle("rvc")
        p = job.params
        src, _ref = audio_inputs(job)
        name = str(p.get("voice") or "").strip()
        if not name:
            have = ", ".join(sorted(list_rvc_voices())) or "none yet"
            raise InputError(f"Choose the RVC voice (a .pth in data/rvc_voices/; found: {have}).")
        os.environ["rmvpe_root"] = h.rmvpe_dir
        os.environ["RVC_CUDA_GRAPH"] = "0"
        v = get_rvc_voice(h, name, ctx)
        y, sr = sf.read(src["path"], dtype="float32", always_2d=True)
        y = librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=16000) if sr != 16000 else y.mean(axis=1)
        peak = np.abs(y).max() / 0.95
        if peak > 1:
            y = y / peak
        n = max(1, int(p.get("batch", 1)))
        s0 = seed0(p)
        index_path = str(v["index"]) if (v["index"] and float(p.get("index_rate", 0.6)) > 0) else ""
        label = src.get("name") or Path(src["path"]).stem
        for i in range(n):
            ctx.check()
            seed_all(s0 + i)
            t0 = time.time()
            ctx.progress(i, n, "converting", f"take {i + 1}/{n}")
            out = v["pipeline"].pipeline(
                h.hubert, v["net_g"], int(p.get("speaker_id", 0)), y.copy(), [0, 0, 0],
                int(p.get("pitch_shift", 0)), str(p.get("f0_method", "rmvpe")), index_path,
                float(p.get("index_rate", 0.6)), v["if_f0"], v["tgt_sr"], 0,
                float(p.get("rms_mix_rate", 0.25)), v["version"], float(p.get("protect", 0.33)),
            )
            f = job.out_dir / f"take_{i + 1}.wav"
            meta = save_wav(f, out.astype(np.float32) / 32768.0, v["tgt_sr"])
            ctx.output(f, "audio", seed=s0 + i, index=i, name=name_of(f"{name} <- {label}", i, n),
                       voice=name, run_s=round(time.time() - t0, 2), **meta)
        return {"meta": {"model": "rvc", "voice": name, "version": v["version"], "index": bool(index_path),
                         "gpu_gb": ctx.gpu_gb(), "peak_gb": peak_gb()}}


if __name__ == "__main__":
    serve(VCBackend())
