"""IndexTTS backend (envs/cloneb_index: torch 2.9.1+cu130, transformers 4.52.1, numpy 2.2).

  indextts-2     task voice_clone   zero-shot cloning from a reference clip; emotion from a separate reference clip,
                                    an 8-value emotion vector or a text description (Qwen emotion model). zh / en.
  indextts-2.5   same + `lang` (zh en ja es ar) and `duration_factor` (0.5-2.0x length control)

Upstream: github.com/index-tts/index-tts (cloned in vendor/index-tts, put on sys.path, not pip-installed).
The upstream wants a "model dir" with config + weights + hf_cache/ (w2v-bert, codec, campplus, bigvgan). We build
that dir from symlinks into the HF cache (vendor/index-tts/_run/<variant>), so nothing is downloaded at runtime.
Text normalisation: WeTextProcessing needs pynini (no aarch64 wheel), so `wetext` is used instead.
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "index-tts"
EMO_ORDER = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
AUX = {
    "w2v": "facebook/w2v-bert-2.0",
    "codec": "amphion/MaskGCT",
    "campplus": "funasr/campplus",
    "bigvgan": "nvidia/bigvgan_v2_22khz_80band_256x",
}


def seed0(p):
    s = int(p.get("seed", -1) if p.get("seed") is not None else -1)
    return s if s >= 0 else random.randrange(2**31)


def snap(repo):
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, local_files_only=True))


def link_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        t = dst / rel
        if f.is_dir():
            t.mkdir(parents=True, exist_ok=True)
        else:
            if t.is_symlink() or t.exists():
                t.unlink()
            t.symlink_to(f)


def build_model_dir(variant: str, repo: str) -> Path:
    """<vendor>/_run/<variant>: the model repo + hf_cache/ laid out the way upstream expects."""
    d = VENDOR / "_run" / variant
    link_tree(snap(repo), d)
    hc = d / "hf_cache"
    hc.mkdir(exist_ok=True)

    def ln(target: Path, name: str):
        t = hc / name
        if t.is_symlink() or t.exists():
            t.unlink()
        t.parent.mkdir(parents=True, exist_ok=True)
        t.symlink_to(target)

    ln(snap(AUX["w2v"]), "w2v-bert-2.0")
    ln(snap(AUX["codec"]) / "semantic_codec" / "model.safetensors", "semantic_codec_model.safetensors")
    ln(snap(AUX["campplus"]) / "campplus_cn_common.bin", "campplus_cn_common.bin")
    bg = snap(AUX["bigvgan"])
    for n in ("config.json", "bigvgan_generator.pt"):
        ln(bg / n, f"bigvgan/{n}")
    return d


def patch_normalizer():
    """Linux upstream uses WeTextProcessing (pynini, no aarch64 wheel): use the `wetext` package like macOS does."""
    from indextts.utils import front

    def load(self):
        if self.zh_normalizer is not None and self.en_normalizer is not None:
            return
        from wetext import Normalizer

        self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
        self.en_normalizer = Normalizer(lang="en", operator="tn")

    front.TextNormalizer.load = load


def patch_torchaudio():
    """torchaudio 2.9 save/load need torchcodec (no aarch64 wheel): route them through soundfile."""
    import soundfile as sf
    import torch
    import torchaudio

    def save(uri, src, sample_rate, channels_first=True, **kw):
        a = src.detach().cpu().numpy()
        if a.ndim == 2 and channels_first:
            a = a.T
        sf.write(str(uri), a, int(sample_rate))

    def load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kw):
        a, sr = sf.read(str(uri), dtype="float32", always_2d=True, start=frame_offset, frames=num_frames)
        t = torch.from_numpy(a.T.copy() if channels_first else a)
        return t, sr

    torchaudio.save = save
    torchaudio.load = load


class IndexBackend(Backend):
    name = "cloneb_index"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("torch", "transformers", "wetext", "tiktoken", "omegaconf", "librosa") if importlib.util.find_spec(m) is None]
        if not (VENDOR / "indextts").is_dir():
            missing.append("vendor/index-tts (git clone https://github.com/index-tts/index-tts)")
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        import torch

        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        os.environ.setdefault("USE_MODELSCOPE", "0")
        patch_normalizer()
        patch_torchaudio()
        v25 = model["id"].endswith("2.5")
        repo = model["weights"][0]["hf"]
        mdir = build_model_dir("v2_5" if v25 else "v2", repo)
        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        cfg = str(mdir / "config.yaml")
        if v25:
            from indextts.infer_v2_5 import IndexTTS2

            tts = IndexTTS2(cfg_path=cfg, model_dir=str(mdir), use_bf16=True, use_cuda_kernel=False, device="cuda:0")
        else:
            from indextts.infer_v2 import IndexTTS2

            aux = {
                "w2v_bert": str(mdir / "hf_cache" / "w2v-bert-2.0"),
                "semantic_codec": str(mdir / "hf_cache" / "semantic_codec_model.safetensors"),
                "campplus": str(mdir / "hf_cache" / "campplus_cn_common.bin"),
                "bigvgan": str(mdir / "hf_cache" / "bigvgan"),
            }
            tts = IndexTTS2(cfg_path=cfg, model_dir=str(mdir), use_fp16=True, use_cuda_kernel=False, device="cuda:0",
                            use_qwen_emo=True, aux_paths=aux)
        return {"tts": tts, "v25": v25}

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import soundfile as sf
        import torch

        h = self.handle(job.model_id)
        tts, v25 = h["tts"], h["v25"]
        p = job.params
        text = (p.get("text") or p.get("prompt") or "").strip()
        if not text:
            raise InputError("Write the line to speak first.")
        ref = job.input(role="ref") or next((i for i in job.inputs if i.get("type") in ("voice", "audio") and i.get("role") != "emo"), None)
        if not ref:
            raise InputError("IndexTTS clones a voice: pick a reference clip (3-15 s of clean speech).")
        kw = {}
        mode = p.get("emotion_mode") or "voice"
        extra = {"emotion_mode": mode}
        if mode == "audio":
            emo = job.input(role="emo")
            if not emo:
                raise InputError("Emotion mode 'reference clip' needs an emotion clip (role emo). Pick one or change the mode.")
            kw["emo_audio_prompt"] = emo["path"]
            kw["emo_alpha"] = float(p.get("emo_alpha", 0.8))
            extra["emo_asset"] = emo.get("asset_id")
        elif mode == "vector":
            vec = [float(p.get(f"emo_{k}", 0.0) or 0.0) for k in EMO_ORDER]
            if not any(vec):
                raise InputError("Emotion mode 'vector': set at least one emotion above 0.")
            kw["emo_vector"] = vec
            kw["emo_alpha"] = float(p.get("emo_alpha", 1.0))
            extra["emo_vector"] = dict(zip(EMO_ORDER, vec))
        elif mode == "text":
            kw["use_emo_text"] = True
            et = (p.get("emo_text") or "").strip()
            if et:
                kw["emo_text"] = et
            kw["emo_alpha"] = float(p.get("emo_alpha", 0.6))
            extra["emo_text"] = et or "(the line itself)"
        if v25:
            kw["lang"] = (p.get("language") or "EN").upper()
            kw["duration_factor"] = float(p.get("duration_factor", 1.0))
            extra["lang"] = kw["lang"]
            extra["duration_factor"] = kw["duration_factor"]
        for k in ("temperature", "top_p", "top_k", "num_beams", "repetition_penalty"):
            if p.get(k) is not None:
                kw[k] = int(p[k]) if k in ("top_k", "num_beams") else float(p[k])
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        outs = []
        for i in range(n):
            ctx.check()
            seed = s0 + i
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}", interruptible=False)
            f = job.out_dir / f"take_{i + 1}.wav"
            try:
                res = tts.infer(spk_audio_prompt=ref["path"], text=text, output_path=str(f),
                                use_random=bool(p.get("use_random", False)),
                                max_text_tokens_per_segment=int(p.get("max_segment_tokens", 120)),
                                interval_silence=int(p.get("pause_ms", 200)), verbose=False, **kw)
            except RuntimeError as e:
                raise InputError(str(e)) if "QwenEmotion" in str(e) else e
            if not res or not f.exists():
                raise RuntimeError("the model returned no audio")
            y, sr = sf.read(str(f), dtype="float32")
            if not np.isfinite(y).all() or len(y) < 0.1 * sr or float(np.abs(y).max()) < 1e-4:
                raise RuntimeError("the model returned silent or invalid audio")
            outs.append(f)
            ctx.output(f, "audio", seed=seed, index=i, name=(text[:48] + ("…" if len(text) > 48 else "") + (f" #{i + 1}" if n > 1 else "")),
                       final_prompt=text, sample_rate=int(sr), duration=round(len(y) / sr, 3), ref_asset=ref.get("asset_id"), **extra)
        return {"meta": {"seed": s0}}


if __name__ == "__main__":
    serve(IndexBackend())
