"""Music backend `music_magenta` (envs/music_magenta): Magenta RealTime 2 (google/magenta-realtime-2, JAX).

Instrumental music from a text style (MusicCoCa embedding), 48 kHz stereo, generated as a stream of 25 frames/s
(40 ms steps). Offline use of the JAX backend on the GB10 GPU (jax[cuda13]); `mrt2_small` (230 M) and `mrt2_base`
(2.4 B). The code (vendor/magenta-realtime) is Apache-2.0, the weights CC-BY-4.0.

Paths: magenta_rt resolves its assets under $MAGENTA_HOME/magenta-rt-v2/{resources,checkpoints}. That layout is the HF
repo layout, so a symlink `vendor/magenta_home/magenta-rt-v2 -> <HF snapshot>` is enough.
The sampler's random seed is fixed inside the library (PRNGKey(0) at compile time); here a take gets its own seed by
building the initial state with a seeded rng (the compiled step is reused).
"""
from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")     # unified memory: do not grab 75 % of it
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from studio_worker import Backend, InputError, serve

STUDIO = Path(__file__).resolve().parent.parent
HOME = STUDIO / "vendor" / "magenta_home"
HF_REPO = "google/magenta-realtime-2"
SIZES = {"magenta-rt2-small": "mrt2_small", "magenta-rt2-base": "mrt2_base"}


def seed0(p: dict) -> int:
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def take_name(label: str, i: int, n: int) -> str:
    label = re.sub(r"\s+", " ", label).strip()
    return f"{label[:48]}{'…' if len(label) > 48 else ''}" + (f" #{i + 1}" if n > 1 else "")


def link_home():
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(HF_REPO, local_files_only=True))
    HOME.mkdir(parents=True, exist_ok=True)
    link = HOME / "magenta-rt-v2"
    if link.is_symlink() and os.readlink(link) != str(snap):
        link.unlink()
    if not link.exists() and not link.is_symlink():
        link.symlink_to(snap)
    os.environ["MAGENTA_HOME"] = str(HOME)


class MusicMagenta(Backend):
    name = "music_magenta"
    max_loaded = 1

    def probe(self, model):
        link_home()
        import jax

        if not any(d.platform in ("gpu", "cuda") for d in jax.devices()):
            return False, "JAX does not see the GPU"
        import magenta_rt  # noqa: F401

        return True, ""

    def load(self, model, ctx):
        link_home()
        import jax
        from magenta_rt import MagentaRT2Jax

        ctx.progress(0, 0, "loading", f"Loading {model['name']} (JAX compile)", interruptible=False)
        mrt = MagentaRT2Jax(size=SIZES[model["id"]])
        ctx.log(f"JAX devices: {jax.devices()}")
        return mrt

    def unload(self, model_id):
        self.models.pop(model_id, None)
        import gc

        import jax

        gc.collect()
        try:
            jax.clear_caches()
        except Exception:  # noqa: BLE001
            pass

    def _init_state(self, mrt, seed: int):
        """Fresh streaming state whose sampler rng is derived from `seed` (the library always uses PRNGKey(0))."""
        import jax

        if not hasattr(mrt, "_studio_init"):
            spec = jax.ShapeDtypeStruct([mrt._num_channels], jax.numpy.int32)

            @jax.jit
            def _init(params, constants, seed_):
                rngs = {"params": jax.random.PRNGKey(42), "random": jax.random.PRNGKey(seed_)}
                return mrt._sampler.apply(params, 1, spec, constants=constants, training=False, rngs=rngs,
                                          method=mrt._sampler.get_initial_state)

            mrt._studio_init = _init
        return mrt._studio_init(mrt._params, {}, jax.numpy.int32(seed % (2**31 - 1)))

    def run(self, job, ctx):
        import soundfile as sf
        from magenta_rt.config import MUSICCOCA

        mrt = self.handle(job.model_id)
        p = job.params
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise InputError("Describe the music first (genre, mood, instruments).")
        n = int(p.get("batch", 1))
        secs = float(p.get("seconds", 20))
        frames_total = max(25, int(secs * 25))
        chunk = 125                                                  # 5 s per call: progress + cancel points
        s0 = seed0(p)
        cfg = {"musiccoca": float(p.get("cfg_style", 3.0)), "notes": 1.0, "drums": 1.0}
        emb = mrt.embed_style(prompt, use_mapper=True)
        for i in range(n):
            ctx.check()
            state = self._init_state(mrt, s0 + i)
            parts, done = [], 0
            while done < frames_total:
                ctx.check()
                k = min(chunk, frames_total - done)
                wav, state = mrt.generate(conditioning={MUSICCOCA.key: emb}, cfg_scales=cfg,
                                          temperature=float(p.get("temperature", 1.3)),
                                          top_k=int(p.get("top_k", 40)), frames=k, state=state)
                parts.append(wav.samples)
                done += k
                ctx.progress(done, frames_total, "sampling", f"take {i + 1}/{n}: {done / 25:.0f}/{frames_total / 25:.0f} s")
            x = np.concatenate(parts, axis=0)
            x = np.nan_to_num(x.astype(np.float32))
            peak = float(np.abs(x).max())
            if peak < 1e-5:
                raise RuntimeError("the model produced silence")
            if peak > 0.89:
                x = x * (0.89 / peak)
            f = job.out_dir / f"take_{i + 1}.wav"
            sf.write(str(f), x, 48000, subtype="PCM_24")
            ctx.output(f, "audio", seed=s0 + i, index=i, name=take_name(prompt, i, n), final_prompt=prompt,
                       sample_rate=48000, duration=round(len(x) / 48000, 3))
        return {"meta": {"steps_per_s": 25}}


if __name__ == "__main__":
    serve(MusicMagenta())
