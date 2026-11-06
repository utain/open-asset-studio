"""Kimi-Audio backend (envs/omni_kimi: transformers 4.44.1, torch 2.9 cu130; upstream code in vendor/Kimi-Audio).

  kimi-audio-7b   task s2s   Whisper-large-v3 continuous features + discrete semantic tokens -> Qwen2.5-7B-based LM
                             emitting text and audio tokens in parallel -> flow-matching DiT + BigVGAN detokenizer (24 kHz).
Upstream hard-imports `flash_attn` (no aarch64 wheel, needs nvcc): a small SDPA-based stand-in is injected into
sys.modules before kimia_infer is imported. It is deliberately NOT installed into site-packages, so transformers
does not think FlashAttention-2 exists. One turn per job.
"""
from __future__ import annotations

import gc
import importlib.util
import math
import random
import re
import sys
import types
from pathlib import Path

import numpy as np

from studio_worker import Backend, InputError, serve

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "Kimi-Audio"
REPO = "moonshotai/Kimi-Audio-7B-Instruct"
TOKENIZER_REPO = "zai-org/glm-4-voice-tokenizer"


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


def install_flash_attn_shim():
    """SDPA implementations of the four flash_attn entry points Kimi-Audio uses."""
    if "flash_attn" in sys.modules:
        return
    import torch
    import torch.nn.functional as F

    def sdpa(q, k, v, scale, causal):  # [B,T,H,D] -> [B,T,H,D]; flash-attn semantics (GQA, bottom-right aligned causal mask)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if k.shape[1] != q.shape[1]:  # grouped-query attention: repeat the kv heads
            rep = q.shape[1] // k.shape[1]
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        tq, tk = q.shape[2], k.shape[2]
        if not causal or tq == 1:
            o = F.scaled_dot_product_attention(q, k, v, scale=scale)
        elif tq == tk:
            o = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)
        else:
            mask = torch.ones(tq, tk, dtype=torch.bool, device=q.device).tril(diagonal=tk - tq)
            o = F.scaled_dot_product_attention(q, k, v, scale=scale, attn_mask=mask)
        return o.transpose(1, 2)

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False, **kw):
        o = sdpa(q, k, v, softmax_scale, causal)
        return (o, None) if return_attn_probs else o

    def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
        q, k, v = qkv.unbind(2)
        return sdpa(q, k, v, softmax_scale, causal)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0,
                               softmax_scale=None, causal=False, return_attn_probs=False, **kw):
        outs = []
        cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
        for i in range(len(cq) - 1):
            o = sdpa(q[None, cq[i]:cq[i + 1]], k[None, ck[i]:ck[i + 1]], v[None, ck[i]:ck[i + 1]], softmax_scale, causal)
            outs.append(o[0])
        o = torch.cat(outs, 0)
        return (o, None) if return_attn_probs else o

    def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=None, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
        q, k, v = qkv.unbind(1)
        return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, softmax_scale=softmax_scale, causal=causal)

    m = types.ModuleType("flash_attn")
    m.__version__ = "0.0.0+sdpa-shim"
    import importlib.machinery

    m.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)  # transformers calls find_spec("flash_attn")
    m.flash_attn_func, m.flash_attn_qkvpacked_func = flash_attn_func, flash_attn_qkvpacked_func
    m.flash_attn_varlen_func, m.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func
    sys.modules["flash_attn"] = m

    import torch.nn.functional as F  # noqa: F811

    def index_first_axis(x, idx):
        return x.reshape(x.shape[0], -1)[idx].reshape(len(idx), *x.shape[1:])

    def pad_input(hidden, indices, batch, seqlen):
        out = torch.zeros(batch * seqlen, *hidden.shape[1:], dtype=hidden.dtype, device=hidden.device)
        out[indices] = hidden
        return out.view(batch, seqlen, *hidden.shape[1:])

    def unpad_input(hidden, attention_mask, *a, **k):
        seqlens = attention_mask.sum(-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        cu = F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))
        flat = hidden.reshape(-1, *hidden.shape[2:])
        return index_first_axis(flat, indices), indices, cu, int(seqlens.max())

    bp = types.ModuleType("flash_attn.bert_padding")
    bp.index_first_axis, bp.pad_input, bp.unpad_input = index_first_axis, pad_input, unpad_input
    bp.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", None)
    sys.modules["flash_attn.bert_padding"] = bp
    m.bert_padding = bp
    # the remote modeling code refuses to import unless transformers claims FlashAttention-2 exists
    import transformers.utils
    import transformers.utils.import_utils as iu

    iu.is_flash_attn_2_available = transformers.utils.is_flash_attn_2_available = lambda: True


def patch_torchaudio():
    import soundfile as sf
    import torch
    import torchaudio

    def load(path, *a, **k):
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(y.T.copy()), sr

    torchaudio.load = load


def patch_safetensors_metadata():
    """The whisper-large-v3 checkpoint bundled with Kimi-Audio has no safetensors metadata; transformers 4.44 then crashes on
    `metadata.get("format")`. Report format=pt for such files (in-process only, the downloaded file is untouched)."""
    import transformers.modeling_utils as mu

    real = mu.safe_open

    class _Open:
        def __init__(self, *a, **k):
            self._f = real(*a, **k)

        def __enter__(self):
            self._f.__enter__()
            return self

        def __exit__(self, *e):
            return self._f.__exit__(*e)

        def metadata(self):
            return self._f.metadata() or {"format": "pt"}

        def __getattr__(self, n):
            return getattr(self._f, n)

    mu.safe_open = _Open


def patch_rotary():
    """Kimi's remote modeling code already gathers cos/sin by position_ids and expects the transformers >= 4.48 helper that does
    not index again; 4.44's qwen2 helper does (double indexing -> shape mismatch / OOB gather). Swap in the non-indexing one."""
    import transformers.models.qwen2.modeling_qwen2 as q2

    def rotate_half(x):
        import torch

        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    q2.apply_rotary_pos_emb = apply_rotary_pos_emb


class KimiBackend(Backend):
    name = "omni_kimi"
    max_loaded = 1

    def probe(self, model):
        missing = [m for m in ("transformers", "loguru", "torchdyn", "timm", "librosa", "torch") if importlib.util.find_spec(m) is None]
        if not VENDOR.exists():
            missing.append(f"vendor/Kimi-Audio ({VENDOR})")
        return (not missing, f"missing: {', '.join(missing)}" if missing else "")

    def load(self, model, ctx):
        from huggingface_hub import snapshot_download

        ctx.progress(0, 0, "loading", f"Loading {model['name']}", interruptible=False)
        install_flash_attn_shim()
        patch_torchaudio()
        patch_safetensors_metadata()
        patch_rotary()
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from kimia_infer.api.kimia import KimiAudio
        import kimia_infer.api.prompt_manager as pm
        import kimia_infer.models.detokenizer.bigvgan_wrapper as bw

        # BigVGAN is built with its custom CUDA anti-alias kernel (JIT-compiled with nvcc, which this box lacks): use the
        # pure-PyTorch path instead (same maths, slower).
        orig_bigvgan = bw.BigVGAN
        bw.BigVGAN = lambda h, use_cuda_kernel=False: orig_bigvgan(h, False)

        # upstream hard-codes the old "THUDM/glm-4-voice-tokenizer" id (renamed to zai-org/); resolve it from the local cache
        orig_tok = pm.Glm4Tokenizer
        pm.Glm4Tokenizer = lambda _path: orig_tok(snapshot_download(TOKENIZER_REPO, local_files_only=True))
        local = snapshot_download(REPO, local_files_only=True)
        return KimiAudio(model_path=local, load_detokenizer=True)

    def unload(self, model_id):
        self.models.pop(model_id, None)

    def run(self, job, ctx):
        import torch

        km = self.handle(job.model_id)
        p = job.params
        src = job.input(role="audio") or job.input(role="ref") or next(
            (i for i in job.inputs if i.get("type") in ("audio", "voice")), None)
        typed = (p.get("text") or "").strip()
        instruction = (p.get("instruction") or "").strip()
        if not src:
            raise InputError("Pick the audio clip Kimi-Audio should answer.")
        msgs = []
        if instruction or typed:
            msgs.append({"role": "user", "message_type": "text", "content": instruction or typed})
        msgs.append({"role": "user", "message_type": "audio", "content": src["path"]})
        n, s0 = max(1, int(p.get("batch", 1))), seed0(p)
        secs = min(110.0, float(p.get("seconds") or 20))
        sampling = dict(audio_temperature=float(p.get("temperature") or 0.8), audio_top_k=10, text_temperature=0.0, text_top_k=5,
                        audio_repetition_penalty=1.0, audio_repetition_window_size=64, text_repetition_penalty=1.0,
                        text_repetition_window_size=16)
        for i in range(n):
            ctx.check()
            seed = s0 + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ctx.progress(i, n, "speaking", f"take {i + 1}/{n}: generating text + audio tokens", interruptible=False)
            wav, reply = self._generate(km, msgs, secs, sampling)
            reply = (reply or "").strip()
            f = job.out_dir / f"take_{i + 1}.wav"
            fm = save_wav(f, wav.detach().float().cpu().view(-1).numpy(), 24000)
            (job.out_dir / f"take_{i + 1}.txt").write_text(reply, encoding="utf-8")
            ctx.output(f, "audio", seed=seed, index=i, name=name_of(reply or "Reply", i, n), final_prompt=reply,
                       reply_text=reply, input_asset=src.get("asset_id"), attach=[f"take_{i + 1}.txt"], **fm)
            gc.collect()
        return {"meta": {"seed": s0}}

    @staticmethod
    def _generate(km, chats, secs, sampling):
        """KimiAudio.generate(output_type='both') with our own max_new_tokens (upstream fixes it at ~120 s)."""
        import torch

        dev = torch.cuda.current_device()
        history = km.prompt_manager.get_prompt(chats, output_type="both")
        a_ids, t_ids, cont_mask, _, _ = history.to_tensor()
        feats = [f.to(dev) for f in history.continuous_feature]
        max_new = int(12.5 * secs) + km.kimia_text_audiodelaytokens + 10
        with torch.inference_mode():
            wav_tokens, text_tokens = km._generate_loop(
                audio_input_ids=a_ids.to(dev), text_input_ids=t_ids.to(dev), max_new_tokens=max_new,
                audio_top_k=sampling["audio_top_k"], audio_temperature=sampling["audio_temperature"],
                audio_repetition_penalty=sampling["audio_repetition_penalty"],
                audio_repetition_window_size=sampling["audio_repetition_window_size"],
                text_top_k=sampling["text_top_k"], text_temperature=sampling["text_temperature"],
                text_repetition_penalty=sampling["text_repetition_penalty"],
                text_repetition_window_size=sampling["text_repetition_window_size"],
                is_continuous_mask=cont_mask.to(dev), continous_feature=feats, output_type="both")
            wav_tokens = [t for t in wav_tokens if t >= km.kimia_token_offset]
            if not wav_tokens:
                raise RuntimeError("the model produced no audio tokens (try again)")
            wav_tokens = torch.tensor(wav_tokens).unsqueeze(0) - km.kimia_token_offset
            text = km.detokenize_text([t for t in text_tokens if t < km.kimia_token_offset])
            wav = km.detokenize_audio(wav_tokens)
        return wav, text


if __name__ == "__main__":
    serve(KimiBackend())
