"""Shared loaders for the image studio (tested in envs/qwen: torch 2.9.0 cu130, diffusers 0.42.1, transformers 5.22.1).

1. Fast direct-to-GPU loading. On this DGX Spark `from_pretrained(...).to("cuda")` copies mmap'ed weights
   at ~170 MB/s; reading straight to the GPU is ~10-20x faster. This applies to EVERY diffusers / transformers
   model here (measured: Qwen-Image-2.1 50 s vs ~3 min, FLUX.1-Kontext see image_models.md).
     - diffusers ModelMixin:   from_pretrained(..., device_map="cuda", disable_mmap=True)
     - transformers models:    fast_te() below (lazy mmap model, then swap tensors for safetensors device="cuda" reads)
2. unsloth "prequant" INT8 checkpoints -> plain bf16 state dict, WITHOUT torchao / unsloth:
     - load_unsloth_safetensors(): unsloth_prequant_transformer_state_dict_v1 (torchao Int8Tensor flattened:
       `<mod>._weight_qdata` int8, `<mod>._weight_scale` [N,1], optional all-zero `_weight_zero_point`) and v2 ConvRot
       (weights were stored as W @ blockdiag(H)^T, H = 256x256 "regular" Hadamard; we undo it: W = Wdeq @ H).
       Root tensors are stored as `unsloth_root::<name>`.
     - load_unsloth_legacy_pt(): torch.save'd torchao AffineQuantizedTensor/LinearActivationQuantizedTensor (.pt).
       torchao classes are replaced by inert stand-ins (weights_only=True stays on), then int_data * scale.
   bf16 dequant = weight-only int8 accuracy (slightly *better* than unsloth's W8A8 runtime), bf16 memory/speed.
   Keep the int8 savings instead with torchao>=0.16 (test venv) if memory ever matters more than speed.
"""
from __future__ import annotations

import enum
import glob
import json
import os
import struct

import torch

HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))


def snap(repo: str) -> str:
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"))
    ref = os.path.join(d, "refs", "main")
    if os.path.exists(ref):
        return os.path.join(d, "snapshots", open(ref).read().strip())
    return sorted(glob.glob(d + "/snapshots/*"))[-1]


def fast_diffusers(cls, path: str, subfolder: str, dtype=torch.bfloat16):
    return cls.from_pretrained(path, subfolder=subfolder, dtype=dtype, device_map="cuda", disable_mmap=True)


def fast_te(cls, path: str, dtype=torch.bfloat16, **kw):
    """transformers model straight to the GPU (same trick as workers/qwen_worker.py)."""
    from safetensors.torch import load_file
    m = cls.from_pretrained(path, dtype=dtype, **kw)  # lazy mmap view: cheap
    params = dict(m.named_parameters()); params.update(dict(m.named_buffers()))
    for f in sorted(glob.glob(path + "/*.safetensors")):
        sd = load_file(f, device="cuda")
        for k, v in sd.items():
            for kk in (k, k.removeprefix("model."), "model." + k):
                t = params.get(kk)
                if t is not None and t.shape == v.shape:
                    t.data = v if v.dtype == t.dtype else v.to(t.dtype)
                    break
        del sd
    return m.to("cuda")


def _convrot_h256() -> torch.Tensor:
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
    h = h4
    for _ in range(3):
        h = torch.kron(h, h4)
    return (h / 16.0).float()  # symmetric, orthogonal


def load_unsloth_safetensors(path: str, device="cuda"):
    """-> (bf16 state dict, unsloth_metadata dict)"""
    from safetensors import safe_open
    with open(path, "rb") as f:
        meta = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))["__metadata__"]
    um = json.loads(meta.get("unsloth_metadata", "{}"))
    rot = set(um.get("activation_rotation_fqns") or [])
    g = int(um.get("activation_rotation_group", 256))
    H = _convrot_h256().to(device) if rot else None
    assert not rot or g == 256, "only group 256 ConvRot implemented"
    sd = {}
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # un-rotate in true fp32
    try:
        with safe_open(path, "pt", device=device) as f:
            keys = set(f.keys())
            for k in keys:
                if k.startswith("unsloth_root::"):
                    sd[k.removeprefix("unsloth_root::")] = f.get_tensor(k)
            for name, desc in meta.items():
                if not desc.startswith('{"_type"'):
                    continue
                d = json.loads(desc)
                if d["_type"] == "Tensor":
                    if name in keys:
                        sd[name] = f.get_tensor(name)
                    continue
                mod, leaf = name.rsplit(".", 1)
                q = f.get_tensor(f"{mod}._{leaf}_qdata").float()
                if f"{mod}._{leaf}_zero_point" in keys:
                    q = q - f.get_tensor(f"{mod}._{leaf}_zero_point").float()
                w = q * f.get_tensor(f"{mod}._{leaf}_scale").float().reshape(q.shape[0], -1)
                if mod in rot:
                    n, k = w.shape
                    w = (w.reshape(n, k // g, g) @ H).reshape(n, k)
                sd[name] = w.to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return sd, um


def load_unsloth_legacy_pt(path: str, device="cuda"):
    """Legacy unsloth .pt (HunyuanImage-2.1-INT8.pt, Lumina-Image-2.0-INT8.pt, Qwen-Image-INT8.pt) -> bf16 state dict."""
    class _Inert(torch.Tensor):
        __torch_function__ = torch._C._disabled_torch_function_impl

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise RuntimeError(f"stand-in cannot run {func}")

    LAQ = type("LinearActivationQuantizedTensor", (_Inert,), {})
    AQT = type("AffineQuantizedTensor", (_Inert,), {})
    IMP = type("PlainAQTTensorImpl", (_Inert,), {})
    PL = type("PlainLayout", (), {})

    def _act(*a, **k):
        raise RuntimeError("never called")

    ZPD = enum.Enum("ZeroPointDomain", {"INT": 1, "FLOAT": 2, "NONE": 3})
    from torch.torch_version import TorchVersion
    safe = [(LAQ, "torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor"),
            (AQT, "torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor"),
            (IMP, "torchao.dtypes.uintx.plain_layout.PlainAQTTensorImpl"),
            (PL, "torchao.dtypes.utils.PlainLayout"),
            (_act, "torchao.quantization.quant_api._int8_symm_per_token_reduced_range_quant"),
            (ZPD, "torchao.quantization.quant_primitives.ZeroPointDomain"), TorchVersion]
    with torch.serialization.safe_globals(safe):
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    sdq = obj.get("state_dict", obj)
    sd = {}
    for k, v in sdq.items():
        if isinstance(v, _Inert):
            impl = v.__dict__["original_weight_tensor"].__dict__["tensor_impl"].__dict__
            q = impl["int_data"].to(device).float()
            zp = impl.get("zero_point")
            if zp is not None and torch.count_nonzero(zp):
                q = q - zp.to(device).float().reshape(q.shape[0], -1)
            sd[k] = (q * impl["scale"].to(device).float().reshape(q.shape[0], -1)).to(torch.bfloat16)
        else:
            sd[k] = v.to(device)
    return sd, obj.get("metadata", {})


def transformer_from_sd(cls, base_path: str, sd: dict, dtype=torch.bfloat16):
    """Build the diffusers transformer from the base repo config and assign the (dequantised) weights."""
    from accelerate import init_empty_weights
    with init_empty_weights():
        m = cls.from_config(cls.load_config(base_path, subfolder="transformer"))
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return m.to(device="cuda", dtype=dtype).eval()


def free(pipe=None):
    """Drop a pipeline and give the unified memory back (or simply exit the worker process: most reliable)."""
    import gc
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
