"""Music backend `music_acestep` (envs/music_acestep): ACE-Step 1.5 (vendor/ACE-Step-1.5, MIT).

Song with vocals + lyrics, or instrumental music, 48 kHz stereo. `acestep-v15-turbo` (8-step DiT, ~2B) plus the optional
5 Hz language model `acestep-5Hz-lm-1.7B` ("thinking": it writes the metadata / caption and the semantic codes first,
which improves structure and lyric adherence; run through plain transformers, `backend="pt"`).

ACE-Step expects `<project_root>/checkpoints/<name>/...` on disk. The HF snapshot (read-only cache) is mirrored into
`vendor/ace_project/checkpoints` as real directories with symlinked files (the code may write helper modules
next to the weights, which it cannot do inside the HF cache).
"""
from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from studio_worker import Backend, InputError, serve

STUDIO = Path(__file__).resolve().parent.parent
VENDOR = STUDIO / "vendor" / "ACE-Step-1.5"
PROJECT = STUDIO / "vendor" / "ace_project"
HF_REPO = "ACE-Step/Ace-Step1.5"
LM_NAME = "acestep-5Hz-lm-1.7B"
DIT_NAME = {"acestep-1.5-turbo": "acestep-v15-turbo"}


def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def take_name(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def mirror_checkpoints() -> Path:
    """Symlink-farm of the cached snapshot -> PROJECT/checkpoints (idempotent)."""
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(HF_REPO, local_files_only=True))
    dst = PROJECT / "checkpoints"
    for src in snap.rglob("*"):
        rel = src.relative_to(snap)
        if rel.parts[0].startswith(".") or rel.name == "README.md":
            continue
        t = dst / rel
        if src.is_dir():
            t.mkdir(parents=True, exist_ok=True)
        else:
            t.parent.mkdir(parents=True, exist_ok=True)
            if t.is_symlink() and os.readlink(t) != str(src):
                t.unlink()
            if not t.exists() and not t.is_symlink():
                t.symlink_to(src)
    return dst


class MusicAcestep(Backend):
    name = "music_acestep"
    max_loaded = 1

    def probe(self, model):
        sys.path.insert(0, str(VENDOR))
        import acestep.inference  # noqa: F401

        return True, ""

    def load(self, model, ctx):
        import torch

        sys.path.insert(0, str(VENDOR))
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        ck = mirror_checkpoints()
        dit = AceStepHandler()
        msg, ok = dit.initialize_service(project_root=str(PROJECT), config_path=DIT_NAME[model["id"]], device="cuda",
                                         offload_to_cpu=False)
        if not ok:
            raise RuntimeError(f"ACE-Step DiT failed to load: {msg}")
        llm = LLMHandler()
        msg, ok = llm.initialize(checkpoint_dir=str(ck), lm_model_path=LM_NAME, backend="pt", device="cuda",
                                 offload_to_cpu=False, dtype=None)
        if not ok:
            ctx.log(f"The 5 Hz language model did not load ({msg}); 'thinking' is disabled.")
            llm = None
        torch.cuda.synchronize()
        return {"dit": dit, "llm": llm}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        sys.path.insert(0, str(VENDOR))
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        h = self.handle(job.model_id)
        p = job.params
        caption = (p.get("prompt") or "").strip()
        if not caption:
            raise InputError("Describe the style of the music first (genre, mood, instruments, vocals).")
        instrumental = job.task == "music" or not (p.get("lyrics") or "").strip()
        lyrics = "[Instrumental]" if instrumental else p["lyrics"].strip()
        n = int(p.get("batch", 1))
        s0 = seed0(p)
        think = bool(p.get("thinking", True)) and h["llm"] is not None
        params = GenerationParams(
            task_type="text2music", caption=caption, lyrics=lyrics, instrumental=instrumental,
            vocal_language=p.get("language", "unknown") or "unknown",
            bpm=(int(p["bpm"]) if p.get("bpm") else None), keyscale=(p.get("keyscale") or "").strip(),
            timesignature=str(p.get("timesignature") or "").strip(),
            duration=float(p.get("seconds", 60)), inference_steps=int(p.get("steps", 8)), seed=s0,
            thinking=think, use_cot_metas=think, use_cot_caption=think, use_cot_language=think,
            lm_temperature=float(p.get("lm_temperature", 0.85)),
            enable_normalization=True, normalization_db=-1.0,
        )
        config = GenerationConfig(batch_size=n, use_random_seed=False, seeds=[s0 + i for i in range(n)],
                                  audio_format="wav")
        ctx.check()
        ctx.progress(0, 0, "generating", "Writing the song" + (" (language model + diffusion)" if think else " (diffusion)"),
                     interruptible=False)
        res = generate_music(h["dit"], h["llm"], params, config, save_dir=str(job.out_dir))
        if not res.success or not res.audios:
            raise RuntimeError(f"ACE-Step failed: {res.error or res.status_message}")
        import soundfile as sf

        meta_out = {}
        for i, a in enumerate(res.audios):
            src = Path(a["path"])
            if not src.exists():
                raise RuntimeError(f"ACE-Step reported {src} but the file is missing")
            f = job.out_dir / f"take_{i + 1}{src.suffix}"
            src.replace(f)
            info = sf.info(str(f))
            seed_used = a.get("params", {}).get("seed", s0 + i) if isinstance(a.get("params"), dict) else s0 + i
            ctx.output(f, "audio", seed=int(seed_used) if str(seed_used).lstrip("-").isdigit() else s0 + i, index=i,
                       name=take_name(caption, i, n), final_prompt=caption, sample_rate=info.samplerate,
                       duration=round(info.duration, 3))
        lm_meta = (res.extra_outputs or {}).get("lm_metadata") if isinstance(res.extra_outputs, dict) else None
        if lm_meta:
            meta_out["lm_metadata"] = {k: v for k, v in lm_meta.items() if isinstance(v, (str, int, float))}
        return {"meta": meta_out}


if __name__ == "__main__":
    serve(MusicAcestep())
