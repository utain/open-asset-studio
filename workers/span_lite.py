"""Inference-only SPAN (Swift Parameter-free Attention Network) for the PBRify models (CC0 weights, 1x maps and the 4x upscaler).

Architecture after spandrel's SPAN (MIT) and hongyuanyu/SPAN (Apache-2.0), re-implemented here from the checkpoints' keys:
c = 48 features, six SPAB blocks (three 3x3 convs with SiLU, `att = sigmoid(out3) - 0.5`, `(out3 + x) * att`), `conv_cat` (1x1 over
[f, b6, b1, b5_2]), a conv and PixelShuffle(scale). The checkpoints carry a `no_norm` buffer: NO mean / range normalisation.

Every Conv3XC is stored as its TRAINING branches (`conv.0/1/2`, `sk`) next to a STALE `eval_conv`; `load_span` rebuilds the 3x3 kernel from
the branches with material_common.reparam_conv3xc (numpy). With the saved eval_conv the 4x upscaler outputs black and the height /
roughness nets are wrong (docs/research/materials.md). `circular=True` switches every 3x3 conv to wrap-around padding: the receptive field is
21 px, so a tileable input gives a tileable map, and tiled inference with a wrap-pad of 32 px is exact.
torch + numpy only (envs/qwen). Weights are loaded with torch.load(weights_only=True) only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from material_common import reparam_conv3xc

PAD = 32          # wrap overlap of tiled inference (> receptive field 21 px)


class C3(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.eval_conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x):
        return self.eval_conv(x)


class SPAB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1_r, self.c2_r, self.c3_r = C3(c, c), C3(c, c), C3(c, c)

    def forward(self, x):
        out1 = self.c1_r(x)
        out2 = self.c2_r(F.silu(out1))
        out3 = self.c3_r(F.silu(out2))
        att = torch.sigmoid(out3) - 0.5
        return (out3 + x) * att, out1


class SPAN(nn.Module):
    def __init__(self, cin=3, cout=3, c=48, scale=1):
        super().__init__()
        self.conv_1 = C3(cin, c)
        for i in range(1, 7):
            setattr(self, f"block_{i}", SPAB(c))
        self.conv_cat = nn.Conv2d(c * 4, c, 1)
        self.conv_2 = C3(c, c)
        self.upsampler = nn.Sequential(nn.Conv2d(c, cout * scale * scale, 3, padding=1), nn.PixelShuffle(scale))

    def forward(self, x):
        f = self.conv_1(x)
        b1, _ = self.block_1(f)
        b = b1
        for i in range(2, 6):
            b, _ = getattr(self, f"block_{i}")(b)
        b6, b5_2 = self.block_6(b)
        b6 = self.conv_2(b6)
        return self.upsampler(self.conv_cat(torch.cat([f, b6, b1, b5_2], 1)))


def rebuild_state(sd: dict) -> dict:
    """Checkpoint state dict -> the inference state dict of SPAN (eval_conv rebuilt from the training branches, numpy)."""
    keep = {k: v for k, v in sd.items() if k.startswith(("conv_cat.", "upsampler."))}
    g = lambda pre, k: sd[f"{pre}.{k}"].detach().float().cpu().numpy()  # noqa: E731
    for pre in sorted({k.rsplit(".eval_conv", 1)[0] for k in sd if ".eval_conv." in k}):
        w, b = reparam_conv3xc(g(pre, "conv.0.weight"), g(pre, "conv.0.bias"), g(pre, "conv.1.weight"), g(pre, "conv.1.bias"),
                               g(pre, "conv.2.weight"), g(pre, "conv.2.bias"), g(pre, "sk.weight"), g(pre, "sk.bias"))
        keep[f"{pre}.eval_conv.weight"], keep[f"{pre}.eval_conv.bias"] = torch.from_numpy(w), torch.from_numpy(b)
    return keep


def load_span(path, device="cuda", circular=True):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("params_ema", sd.get("params", sd))
    keep = rebuild_state(sd)
    c = keep["conv_1.eval_conv.weight"].shape[0]
    scale = int((keep["upsampler.0.weight"].shape[0] // 3) ** 0.5)
    m = SPAN(keep["conv_1.eval_conv.weight"].shape[1], 3, c, scale)
    missing, unexpected = m.load_state_dict(keep, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"PBRify checkpoint does not match the SPAN layout: missing={missing[:3]} unexpected={unexpected[:3]}")
    if circular:
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d) and mod.padding != (0, 0):
                mod.padding_mode = "circular"
    return m.to(device).eval()


@torch.no_grad()
def run_span(model, rgb01: np.ndarray, device="cuda", fp16: bool = False, tile: int = 0) -> np.ndarray:
    """rgb01: HxWx3 float32 in [0, 1] (periodic) -> HxWx3 float32 in [0, 1] at the model's scale. tile > 0 and a larger image: tiles of that
    size with a 32 px wrap overlap (exact for a circular model; keeps the peak memory bounded)."""
    h, w = rgb01.shape[:2]
    scale = model.upsampler[1].upscale_factor

    def one(a):
        x = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].to(device, torch.float32)
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(fp16)):
            y = model(x).clamp(0, 1)
        return y[0].permute(1, 2, 0).float().cpu().numpy()

    if not tile or max(h, w) <= tile:
        return one(rgb01)
    padded = np.pad(rgb01, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="wrap")
    out = np.zeros((h * scale, w * scale, 3), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
            crop = padded[y0:y1 + 2 * PAD, x0:x1 + 2 * PAD]
            r = one(crop)
            out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = r[PAD * scale:(PAD + (y1 - y0)) * scale, PAD * scale:(PAD + (x1 - x0)) * scale]
    return out
