"""vLLM-Omni backend: Higgs TTS 3, Fish Audio S2 Pro, Voxtral 4B TTS, CosyVoice 3 (tasks tts / voice_clone).

Runs in the read-only venv envs/vllm_omni with PYTHONPATH = workers:envs/vllm_omni_overlay
(the overlay adds fish_speech DAC, s3tokenizer, soxr). One long-lived `vllm_omni.Omni` engine per loaded model
(max_loaded = 1): its stages are child processes, so unloading closes the engine, waits for every child to exit
and kills stragglers. The recipes mirror vllm-omni/examples/offline_inference/text_to_speech/*/end2end.py and
docs/research/audio_snippets/vllm_omni_tts.sh, with the low-memory deploy configs of
docs/research/audio_snippets/vllm_omni_deploy/*_studio.yaml (the stock ones reserve 40-90 % of the GPU).

Gotchas handled here (docs/research/audio_models.md section 7):
  * Voxtral: the LOCAL snapshot dir is passed as model (repo id + HF_HUB_OFFLINE cannot detect model_type).
    Cloning is impossible with the open weights (no audio encoder): presets only.
  * CosyVoice3: the cached config.json is "{}", so a symlinked model dir with a real config.json is built in
    data/cache/cosyvoice3_model; ~/.cache/s3tokenizer/speech_tokenizer_v3_25hz.onnx is symlinked to the cached
    (byte-identical) speech_tokenizer_v3.onnx so s3tokenizer never downloads 925 MB from modelscope; the prompt
    text is "You are a helpful assistant.<|endofprompt|>" + the reference transcript; output is 24 kHz.
  * Omni.generate() closes the engine when a request fails: the handle is then dropped (and reloaded next job).
"""
from __future__ import annotations

import os
import threading
import time


# ---------------------------------------------------------------- child-process safety net
# Engine stages are multiprocessing children (spawn re-imports this file as __mp_main__, fork inherits it).
# If this worker dies hard (SIGKILL, os._exit on stdin EOF) they would be orphaned holding GPU memory, so every
# descendant watches its parent pid and exits as soon as it changes.
def _start_parent_watch():
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=watch, name="studio-parent-watch", daemon=True).start()


if __name__ == "__mp_main__":
    _start_parent_watch()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_start_parent_watch)

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")   # never fork a process that runs threads

import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import signal  # noqa: E402
from pathlib import Path  # noqa: E402

from studio_worker import Backend, InputError, log, send, serve  # noqa: E402

STUDIO = Path(__file__).resolve().parent.parent
DEPLOY = STUDIO / "docs" / "research" / "audio_snippets" / "vllm_omni_deploy"
COSY_DIR = STUDIO / "data" / "cache" / "cosyvoice3_model"
COSY_PROMPT_HEAD = "You are a helpful assistant."
COSY_EOP = "<|endofprompt|>"
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".aif"}

# kind -> recipe. Registry entries are matched by id first, then by their first weights repo.
SPECS = {
    "higgs": {"repo": "bosonai/higgs-tts-3-4b", "deploy": "higgs_multimodal_qwen3_low_latency_studio.yaml",
              "sr": 24000, "clone": True},
    "fish": {"repo": "fishaudio/s2-pro", "deploy": "fish_qwen3_omni_studio.yaml", "sr": 44100, "clone": True,
             "modules": ["fish_speech"]},
    "voxtral": {"repo": "mistralai/Voxtral-4B-TTS-2603", "deploy": "voxtral_tts_studio.yaml", "sr": 24000,
                "clone": False, "modules": ["mistral_common"]},
    "cosyvoice": {"repo": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "deploy": "cosyvoice3_studio.yaml", "sr": 24000,
                  "clone": True, "modules": ["s3tokenizer"]},
}
IDS = {"higgs-tts-3": "higgs", "fish-s2-pro": "fish", "voxtral-tts": "voxtral", "cosyvoice3": "cosyvoice"}


def _kind(model: dict) -> str:
    k = IDS.get(model.get("id", ""))
    if k:
        return k
    feats = model.get("features") or {}
    if feats.get("vllm_omni_kind") in SPECS:
        return feats["vllm_omni_kind"]
    for w in model.get("weights") or []:
        for k, s in SPECS.items():
            if isinstance(w, dict) and w.get("hf") == s["repo"]:
                return k
    raise InputError(f"vllm_omni backend does not know model {model.get('id')!r}")


def _hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _snapshot(repo: str) -> Path | None:
    """Local snapshot dir of a cached HF repo (the one `refs/main` points at, else the newest)."""
    d = _hub_cache() / ("models--" + repo.replace("/", "--"))
    ref = d / "refs" / "main"
    if ref.is_file():
        s = d / "snapshots" / ref.read_text().strip()
        if s.is_dir():
            return s
    snaps = sorted((d / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime) if (d / "snapshots").is_dir() else []
    return snaps[-1] if snaps else None


# ---------------------------------------------------------------- process helpers
def _children_map() -> dict[int, list[int]]:
    """ppid -> live child pids (zombies are reaped when they are ours, and never counted)."""
    kids: dict[int, list[int]] = {}
    me = os.getpid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / "stat").read_text()
            state, ppid = stat[stat.rindex(")") + 2:].split()[:2]
            ppid = int(ppid)
        except (OSError, ValueError, IndexError):
            continue
        if state in ("Z", "X"):
            if ppid == me:
                try:
                    os.waitpid(int(p.name), os.WNOHANG)
                except ChildProcessError:
                    pass
            continue
        kids.setdefault(ppid, []).append(int(p.name))
    return kids


def _descendants(pid: int | None = None) -> list[int]:
    pid = pid or os.getpid()
    kids = _children_map()
    out, todo = [], list(kids.get(pid, []))
    while todo:
        c = todo.pop()
        out.append(c)
        todo.extend(kids.get(c, []))
    return out


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def _engine_children() -> list[int]:
    """Our descendants except multiprocessing's resource tracker (tiny, CPU only, lives as long as we do)."""
    return [p for p in _descendants() if "resource_tracker" not in _cmdline(p)]


def _reap(timeout: float = 30.0):
    """Wait for engine children to exit after Omni.close(); SIGTERM then SIGKILL whatever is left.
    Only descendants of this worker are ever signalled."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        left = _engine_children()
        if not left:
            return
        time.sleep(0.5)
    left = _engine_children()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not left:
            return
        log(f"vllm_omni: {len(left)} engine process(es) still alive after close, sending {sig.name}: {left}")
        for p in left:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        t_end = time.time() + 10
        while time.time() < t_end and _engine_children():
            time.sleep(0.5)
        left = _engine_children()
    try:
        while os.waitpid(-1, os.WNOHANG)[0] > 0:   # collect zombies of direct children
            pass
    except ChildProcessError:
        pass


# ---------------------------------------------------------------- audio helpers
def _read_ref(path: str):
    import numpy as np
    import soundfile as sf

    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001
        raise InputError(f"Cannot read the reference clip {Path(path).name}: {e}") from e
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    if wav.size < sr * 0.5:
        raise InputError("The reference clip is shorter than 0.5 s")
    return wav, int(sr)


def _pcm(mm: dict):
    """Concatenate the final audio of one request into a 1-D float32 numpy array."""
    import numpy as np
    import torch

    audio = None
    for key in ("audio", "model_outputs"):
        if mm and mm.get(key) is not None:
            audio = mm[key]
            break
    if audio is None:
        raise RuntimeError(f"engine returned no audio (keys: {list((mm or {}).keys())})")
    if isinstance(audio, (list, tuple)):
        parts = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not parts:
            raise RuntimeError("engine returned an empty audio list")
        t = torch.cat(parts) if len(parts) > 1 else parts[0]
    else:
        t = torch.as_tensor(audio).float().cpu().reshape(-1)
    return np.nan_to_num(t.numpy().astype(np.float32))


def _mm_sr(mm: dict, default: int) -> int:
    sr = (mm or {}).get("sr")
    if sr is None or (isinstance(sr, (list, tuple)) and not sr):
        return default
    if isinstance(sr, (list, tuple)):
        sr = sr[-1]
    try:
        return int(sr.item() if hasattr(sr, "item") else sr)
    except (TypeError, ValueError):
        return default


def _write_wav(path: Path, pcm, sr: int):
    import numpy as np
    import soundfile as sf

    sf.write(str(path), np.clip(pcm, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")


# ---------------------------------------------------------------- the backend
class Engine:
    """One resident Omni engine plus the per-model prompt helpers."""

    def __init__(self, kind: str, omni, model_path: str, **extra):
        self.kind = kind
        self.omni = omni
        self.model_path = model_path
        self.extra = extra
        self.dead = False


class VllmOmniBackend(Backend):
    name = "vllm_omni"
    max_loaded = 1          # one engine resident at a time (each reserves 20-30 GB)

    # --- probe
    def probe(self, model):
        kind = _kind(model)
        spec = SPECS[kind]
        if importlib.util.find_spec("vllm_omni") is None:
            return False, "vllm_omni is not importable in this venv"
        for m in spec.get("modules", []):
            if importlib.util.find_spec(m) is None:
                return False, f"python module {m!r} missing (envs/vllm_omni_overlay on PYTHONPATH?)"
        dc = DEPLOY / spec["deploy"]
        if not dc.is_file():
            return False, f"deploy config missing: {dc}"
        snap = _snapshot(spec["repo"])
        if snap is None:
            return False, f"weights not cached: {spec['repo']}"
        if kind == "higgs" and _snapshot("bosonai/higgs-audio-v2-tokenizer") is None:
            return False, "weights not cached: bosonai/higgs-audio-v2-tokenizer"
        if kind == "cosyvoice" and not (snap / "speech_tokenizer_v3.onnx").exists():
            return False, "speech_tokenizer_v3.onnx missing from the CosyVoice3 snapshot"
        return True, ""

    # --- load / unload
    def load(self, model, ctx):
        kind = _kind(model)
        spec = SPECS[kind]
        ok, why = self.probe(model)
        if not ok:
            raise InputError(f"{model.get('name', model['id'])} cannot run: {why}")
        ctx.progress(0, 0, "loading", f"Starting the {model.get('name', kind)} engine (1-3 min)", interruptible=False)
        t0 = time.time()
        kw = {"deploy_config": str(DEPLOY / spec["deploy"]), "stage_init_timeout": 900, "log_stats": False}
        extra = {}
        if kind == "voxtral":
            model_path = str(_snapshot(spec["repo"]))
        elif kind == "cosyvoice":
            model_path = str(self._cosy_dir())
            kw["tokenizer"] = str(Path(model_path) / "CosyVoice-BlankEN")
        else:
            model_path = spec["repo"]
            kw["trust_remote_code"] = True

        from vllm_omni import Omni

        before = set(_descendants())
        try:
            omni = Omni(model=model_path, **kw)
        except BaseException:
            # a failed start can leave some stage processes behind
            if set(_descendants()) - before:
                _reap(20)
            raise
        try:
            if kind == "higgs":
                from transformers import AutoTokenizer
                from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
                    HiggsAudioV3TokenizerAdapter,
                )
                tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                extra["adapter"] = HiggsAudioV3TokenizerAdapter(tok)
            elif kind == "fish":
                from transformers import AutoTokenizer
                extra["tok"] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            elif kind == "voxtral":
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
                extra["tok"] = MistralTokenizer.from_file(str(Path(model_path) / "tekken.json")).instruct_tokenizer
                extra["voices"] = sorted(p.stem for p in (Path(model_path) / "voice_embedding").glob("*.pt"))
            elif kind == "cosyvoice":
                from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
                from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config
                cfg = CosyVoice3Config()
                extra["cfg"] = cfg
                extra["tok"] = get_qwen_tokenizer(token_path=kw["tokenizer"], skip_special_tokens=cfg.skip_special_tokens,
                                                  version=cfg.version)
        except BaseException:
            self._close(Engine(kind, omni, model_path))
            raise
        ctx.log(f"{model.get('name', kind)} engine started in {time.time() - t0:.0f}s "
                f"({len(_engine_children())} engine processes)")
        return Engine(kind, omni, model_path, **extra)

    def _cosy_dir(self) -> Path:
        src = _snapshot(SPECS["cosyvoice"]["repo"])
        COSY_DIR.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.name == "config.json":
                continue
            link = COSY_DIR / f.name
            if link.is_symlink() or link.exists():
                if link.is_symlink() and os.readlink(link) == str(f):
                    continue
                link.unlink()
            link.symlink_to(f)
        cfg = COSY_DIR / "config.json"
        want = {"model_type": "cosyvoice3", "architectures": ["CosyVoice3Model"]}
        if not cfg.is_file() or cfg.is_symlink() or json.loads(cfg.read_text() or "{}") != want:
            if cfg.is_symlink():
                cfg.unlink()
            cfg.write_text(json.dumps(want))
        # s3tokenizer.load_model("speech_tokenizer_v3_25hz") would download 925 MB from modelscope; the cached
        # speech_tokenizer_v3.onnx is byte-identical
        s3 = Path.home() / ".cache" / "s3tokenizer" / "speech_tokenizer_v3_25hz.onnx"
        if not s3.exists():
            s3.parent.mkdir(parents=True, exist_ok=True)
            if s3.is_symlink():
                s3.unlink()
            s3.symlink_to((src / "speech_tokenizer_v3.onnx").resolve())
        return COSY_DIR

    def _close(self, eng: Engine):
        try:
            eng.omni.close()
        except Exception as e:  # noqa: BLE001
            log(f"vllm_omni: close failed: {e}")
        eng.omni = None
        eng.dead = True
        if eng.kind == "higgs":   # the clone codec is cached on the GPU of this process
            try:
                from vllm_omni.model_executor.models.higgs_audio_v2 import higgs_audio_v2_tokenizer as h2
                enc = h2._ENCODER_CACHE
                h2._ENCODER_CACHE = None
                # transformers' HiggsAudioV2TokenizerModel._get_conv1d_layers is an @lru_cache keyed on self:
                # without clearing it the ~0.8 GB codec stays on the GPU forever
                fn = getattr(type(enc), "_get_conv1d_layers", None) if enc is not None else None
                if hasattr(fn, "cache_clear"):
                    fn.cache_clear()
                del enc
            except Exception:  # noqa: BLE001
                pass
        if eng.kind == "cosyvoice":   # the mm processor runs here and caches s3tokenizer (~0.9 GB) on the GPU
            try:
                from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3 as c3
                from vllm_omni.model_executor.models.cosyvoice3 import utils as c3u
                c3.CosyVoice3MultiModalProcessor._s3_model = None
                c3._RUNTIME_COMPONENTS_CACHE.clear()
                for v in list(vars(c3u).values()):   # lru-cached mel bases / hann windows / mel filters
                    if hasattr(v, "cache_clear"):
                        v.cache_clear()
            except Exception as e:  # noqa: BLE001
                log(f"vllm_omni: cosyvoice cache cleanup failed: {e}")
        import gc

        gc.collect()
        _reap()

    def unload(self, model_id):
        eng = self.models.pop(model_id, None)
        if isinstance(eng, Engine) and not eng.dead:
            t = time.time()
            self._close(eng)
            log(f"vllm_omni: {model_id} engine closed in {time.time() - t:.1f}s, "
                f"{len(_engine_children())} engine processes left")

    def _drop_dead(self, model_id: str, why: str):
        eng = self.models.pop(model_id, None)
        self.last_used.pop(model_id, None)
        if isinstance(eng, Engine):
            self._close(eng)
        send(ev="unloaded", model=model_id, reason="error")
        log(f"vllm_omni: dropped {model_id} engine ({why})")

    # --- run
    def run(self, job, ctx):
        eng: Engine = self.handle(job.model_id)
        if eng.omni is None or eng.omni.errored:
            # the engine died since the last job: restart it in place
            self.models.pop(job.model_id, None)
            self._close(eng)
            send(ev="unloaded", model=job.model_id, reason="error")
            eng = self.load(job.model, ctx)
            self.models[job.model_id] = eng
            send(ev="loaded", id=job.id, model=job.model_id, seconds=0, gpu_gb=None)

        p = job.params
        text = str(p.get("text") or p.get("prompt") or "").strip()
        if not text:
            raise InputError("Type the line to speak")
        n = max(1, min(8, int(p.get("batch", 1) or 1)))
        seed0 = int(p.get("seed", -1) if p.get("seed") is not None else -1)
        if seed0 < 0:
            seed0 = random.randrange(2**31 - 16)

        ref = self._ref_input(job)
        if job.task == "voice_clone" and ref is None:
            raise InputError("Voice clone needs a reference clip")
        if ref is not None and not SPECS[eng.kind]["clone"]:
            if job.task == "voice_clone":
                raise InputError(f"{job.model.get('name', eng.kind)} has no voice cloning with the open weights "
                                 "(presets only): pick a voice preset or another model")
            ref = None   # tts with a stray audio input: ignore it
        ref_text = str(p.get("ref_text") or "").strip()
        if ref is not None and not ref_text:
            ref_text = str(((ref.get("meta") or {}).get("x_voice") or {}).get("ref_text") or "").strip()

        prep = getattr(self, f"_prep_{eng.kind}")(eng, job, ctx, text, ref, ref_text)
        meta_common = prep.pop("meta", {})
        files = []
        for i in range(n):
            ctx.check()
            seed = seed0 + i
            ctx.progress(i, n, "generating", f"take {i + 1}/{n}" if n > 1 else "generating")
            prompt, sps = prep["build"](seed)
            t = time.time()
            try:
                outs = eng.omni.generate([prompt], sps, use_tqdm=False)
            except Exception as e:
                # Omni.generate closes the whole engine when a request fails
                self._drop_dead(job.model_id, f"{type(e).__name__}: {e}")
                raise
            dt = time.time() - t
            if not outs:
                raise RuntimeError("engine returned no output")
            mm = outs[-1].multimodal_output
            if not mm and getattr(outs[-1], "request_output", None) is not None:
                ro = outs[-1].request_output
                mm = getattr(ro, "multimodal_output", None) or (ro.outputs[0].multimodal_output if ro.outputs else None)
            pcm = _pcm(mm)
            sr = prep.get("sr") or _mm_sr(mm, SPECS[eng.kind]["sr"])
            dur = len(pcm) / sr
            if dur < 0.05:
                raise RuntimeError(f"engine returned {len(pcm)} samples (empty take)")
            f = job.out_dir / f"take_{i + 1}.wav"
            _write_wav(f, pcm, sr)
            peak = float(abs(pcm).max()) if len(pcm) else 0.0
            ctx.log(f"take {i + 1}/{n}: {dur:.2f}s of audio in {dt:.1f}s (seed {seed}, peak {peak:.2f})")
            files.append(ctx.output(f, "audio", seed=seed, index=i, name=f"{text[:40]} #{i + 1}",
                                    final_prompt=prep["final_prompt"], sample_rate=sr, duration=round(dur, 3),
                                    gen_s=round(dt, 2), **meta_common))
            ctx.progress(i + 1, n, "generating", f"take {i + 1}/{n} done")
        return {"meta": {"seed": seed0, "engine": eng.kind, **meta_common}}

    @staticmethod
    def _ref_input(job):
        r = job.input(role="ref")
        if r is None:
            for i in job.inputs:
                if i.get("type") in ("audio", "voice") or Path(str(i.get("path") or "")).suffix.lower() in AUDIO_EXT:
                    r = i
                    break
        if r is None:
            return None
        path = str(r.get("path") or "")
        if path and Path(path).is_dir():   # a voice asset may be a folder: take its first audio file
            cands = sorted(q for q in Path(path).iterdir() if q.suffix.lower() in AUDIO_EXT)
            path = str(cands[0]) if cands else ""
        if not path or not Path(path).is_file():
            raise InputError(f"Reference clip not found: {r.get('name') or path}")
        return {**r, "path": path}

    @staticmethod
    def _sps(eng: Engine, seed: int, **stage0):
        """Deploy-config defaults for every stage, with the seed set everywhere and overrides on stage 0."""
        sps = [copy.deepcopy(sp) for sp in eng.omni.default_sampling_params_list]
        for sp in sps:
            if hasattr(sp, "seed"):
                sp.seed = seed
        for k, v in stage0.items():
            if v is not None:
                setattr(sps[0], k, v)
        return sps

    # --- per-model prompt builders: return {"build": seed -> (prompt, sampling_params_list), "final_prompt"}
    def _prep_higgs(self, eng, job, ctx, text, ref, ref_text):
        import torch

        p = job.params
        temp = float(p["temperature"]) if p.get("temperature") is not None else None
        top_k = int(p["top_k"]) if p.get("top_k") is not None else None
        adapter = eng.extra["adapter"]
        if ref is None:
            prompt = {"prompt_token_ids": adapter.build_prompt(text)}
            final = text
        else:
            from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
                apply_delay_pattern,
                encode_reference_audio,
            )
            wav, sr = _read_ref(ref["path"])
            ctx.progress(0, 0, "encoding", "Encoding the reference clip", interruptible=False)
            codes = apply_delay_pattern(encode_reference_audio(wav, sr))
            prompt = {
                "prompt_token_ids": adapter.build_prompt(text, num_ref_tokens=int(codes.shape[0]),
                                                         reference_text=ref_text or None),
                "additional_information": {
                    "audio_input_ids": codes.to(torch.long),
                    "audio_input_ids_mask": torch.ones(codes.shape[0], dtype=torch.bool),
                },
            }
            final = text
        return {"build": lambda seed: (prompt, self._sps(eng, seed, temperature=temp, top_k=top_k)),
                "final_prompt": final, "meta": {"ref_text": ref_text} if ref is not None else {}}

    def _prep_fish(self, eng, job, ctx, text, ref, ref_text):
        import numpy as np
        import torch
        from vllm_omni.model_executor.models.fish_speech.dac_utils import DAC_HOP_LENGTH, DAC_SAMPLE_RATE
        from vllm_omni.model_executor.models.fish_speech.prompt_utils import (
            build_fish_text_only_prompt_ids,
            estimate_fish_voice_clone_prompt_len_from_normalized,
            normalize_fish_voice_clone_texts,
        )

        tok = eng.extra["tok"]
        if ref is None:
            ids, norm = build_fish_text_only_prompt_ids(tok, text)
            prompt = {"prompt_token_ids": ids, "additional_information": {"text": [norm]}}
        else:
            if not ref_text:
                raise InputError("Fish S2 Pro cloning needs the reference transcript (ref_text)")
            norm, norm_ref = normalize_fish_voice_clone_texts(text, ref_text)
            wav, sr = _read_ref(ref["path"])
            sem_len = max(1, math.ceil(max(1, math.ceil(len(wav) * DAC_SAMPLE_RATE / sr)) / DAC_HOP_LENGTH))
            ph_len = estimate_fish_voice_clone_prompt_len_from_normalized(tok, norm, norm_ref, sem_len)
            prompt = {
                "prompt_token_ids": [1] * ph_len,
                "additional_information": {
                    "text": norm, "ref_text": norm_ref,
                    "ref_audio_wav": torch.from_numpy(np.asarray(wav, dtype=np.float32)),
                    "ref_audio_sr": int(sr), "fish_structured_voice_clone": True,
                },
            }
        p = job.params   # optional: the registry entry may expose temperature / top_k (deploy: 0.8 / 30)
        temp = float(p["temperature"]) if p.get("temperature") is not None else None
        top_k = int(p["top_k"]) if p.get("top_k") is not None else None
        return {"build": lambda seed: (prompt, self._sps(eng, seed, temperature=temp, top_k=top_k)),
                "final_prompt": norm,
                "meta": {"ref_text": ref_text} if ref is not None else {}}

    def _prep_voxtral(self, eng, job, ctx, text, ref, ref_text):
        from mistral_common.protocol.speech.request import SpeechRequest
        from vllm import SamplingParams

        voices = eng.extra["voices"]
        voice = str(job.params.get("voice") or "neutral_male")
        if voice not in voices:
            raise InputError(f"Unknown Voxtral voice {voice!r}; presets: {', '.join(voices)}")
        tokenized = eng.extra["tok"].encode_speech_request(SpeechRequest(input=text, voice=voice))
        prompt = {"prompt_token_ids": tokenized.tokens, "additional_information": {"voice": [voice]}}

        def build(seed):
            # mirrors examples/.../voxtral_tts/end2end.py (fresh SamplingParams per stage, max_tokens 2500)
            sp = SamplingParams(max_tokens=2500, seed=seed)
            return prompt, [sp, copy.deepcopy(sp)]
        return {"build": build, "final_prompt": text, "meta": {"voice": voice}}

    def _prep_cosyvoice(self, eng, job, ctx, text, ref, ref_text):
        from vllm import SamplingParams
        from vllm_omni.model_executor.models.cosyvoice3.utils import extract_text_token

        if ref is None:
            raise InputError("CosyVoice 3 needs a reference clip (it always clones a voice)")
        wav, sr = _read_ref(ref["path"])
        if sr < 16000:
            raise InputError(f"Reference clip sample rate {sr} Hz is too low (CosyVoice 3 needs >= 16 kHz)")
        instruct = str(job.params.get("instruct") or "").strip()
        head = f"{COSY_PROMPT_HEAD} {instruct}" if instruct else COSY_PROMPT_HEAD
        prompt_text = f"{head}{COSY_EOP}{ref_text}"
        if not ref_text:
            ctx.log("no reference transcript: CosyVoice 3 clones much better with one (ref_text)")
        prompt = {
            "prompt": text,
            "multi_modal_data": {"audio": (wav, sr)},
            "mm_processor_kwargs": {"prompt_text": prompt_text, "sample_rate": sr},
        }
        cfg = eng.extra["cfg"]
        _, tlen = extract_text_token(text, eng.extra["tok"], cfg.allowed_special)
        base = int(tlen)
        min_len, max_len = int(base * cfg.min_token_text_ratio), max(8, int(base * cfg.max_token_text_ratio))

        def build(seed):
            gpt = SamplingParams(temperature=1.0, top_p=0.8, top_k=25, repetition_penalty=2.0, min_tokens=min_len,
                                 max_tokens=max_len, stop_token_ids=[6561 + 1], detokenize=False, seed=seed)
            s2 = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=2.0, max_tokens=256,
                                detokenize=False, seed=seed)
            return prompt, [gpt, s2]
        meta = {"ref_text": ref_text}
        if instruct:
            meta["instruct"] = instruct
        return {"build": build, "final_prompt": text, "sr": 24000, "meta": meta}


if __name__ == "__main__":
    serve(VllmOmniBackend())
