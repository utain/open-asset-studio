"""RIFE v4 frame interpolation (pure PyTorch, no CUDA extension; runs in envs/video).

The IFNet architecture below is a condensed port of hzwer/ECCV2022-RIFE `model/IFNet.py` (MIT licence,
Copyright (c) Megvii Inc.), in the `torch.compile`-clean form published with the safetensors weights at
https://huggingface.co/TensorForger/RIFE-safetensors (MIT). The only changes: `warp` builds its grid on the device /
dtype of its input (the original read module globals), and `interpolate()` / `rife_double()` are new.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(i, o, k=3, s=1, p=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, d, bias=True), nn.PReLU(o))


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(_conv(in_planes, c // 2, 3, 2, 1), _conv(c // 2, c, 3, 2, 1))
        self.convblock0 = nn.Sequential(_conv(c, c), _conv(c, c))
        self.convblock1 = nn.Sequential(_conv(c, c), _conv(c, c))
        self.convblock2 = nn.Sequential(_conv(c, c), _conv(c, c))
        self.convblock3 = nn.Sequential(_conv(c, c), _conv(c, c))
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.PReLU(c // 2), nn.ConvTranspose2d(c // 2, 4, 4, 2, 1))
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.PReLU(c // 2), nn.ConvTranspose2d(c // 2, 1, 4, 2, 1))

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask


def warp(img, flow):
    n, _, h, w = flow.shape
    gx = torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype).view(1, 1, 1, w).expand(n, -1, h, -1)
    gy = torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([gx, gy], 1)
    flow = torch.cat([flow[:, 0:1] / ((img.shape[3] - 1.0) / 2.0), flow[:, 1:2] / ((img.shape[2] - 1.0) / 2.0)], 1)
    return F.grid_sample(img, (grid + flow).permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True)


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 4, c=90)
        self.block1 = IFBlock(7 + 4, c=90)
        self.block2 = IFBlock(7 + 4, c=90)
        self.block_tea = IFBlock(10 + 4, c=90)

    def forward(self, x):
        scale_list = [4, 2, 1]
        img0, img1 = x[:, :3], x[:, 3:]
        merged, masks = [], []
        w0, w1 = img0, img1
        flow = x[:, :4].detach() * 0
        mask = x[:, :1].detach() * 0
        blocks = [self.block0, self.block1, self.block2]
        for i in range(3):
            f0, m0 = blocks[i](torch.cat((w0[:, :3], w1[:, :3], mask), 1), flow, scale=scale_list[i])
            f1, m1 = blocks[i](torch.cat((w1[:, :3], w0[:, :3], -mask), 1), torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2
            masks.append(mask)
            w0, w1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
            merged.append((w0, w1))
        m = torch.sigmoid(masks[2])
        return merged[2][0] * m + merged[2][1] * (1 - m)


def load_rife(weights_file: str, device="cuda", dtype=torch.float16) -> IFNet:
    from safetensors.torch import load_file
    net = IFNet()
    net.load_state_dict(load_file(weights_file))
    return net.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def rife_double(net: IFNet, frames: np.ndarray, device="cuda", dtype=torch.float16, batch=4, check=None, report=None) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (2T-1, H, W, 3) uint8: one new frame between every pair (midpoint, t = 0.5)."""
    t, h, w, _ = frames.shape
    if t < 2:
        return frames
    ph, pw = (-h) % 32, (-w) % 32   # the flow blocks downsample by up to 16 x 2: pad to a multiple of 32

    def prep(a):
        x = torch.from_numpy(a).to(device).permute(0, 3, 1, 2).to(dtype) / 255.0
        return F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x

    out = np.empty((2 * t - 1, h, w, 3), dtype=np.uint8)
    out[0::2] = frames
    pairs = list(range(t - 1))
    for s in range(0, len(pairs), batch):
        if check:
            check()
        idx = pairs[s:s + batch]
        a, b = prep(frames[idx]), prep(frames[[i + 1 for i in idx]])
        mid = net(torch.cat([a, b], 1))[:, :, :h, :w]
        out[[2 * i + 1 for i in idx]] = (mid.float().clamp(0, 1) * 255.0 + 0.5).permute(0, 2, 3, 1).byte().cpu().numpy()
        if report:
            report(min(s + batch, len(pairs)), len(pairs))
    return out


def interpolate(net: IFNet, frames: np.ndarray, factor: int = 2, **kw) -> np.ndarray:
    """factor 2 or 4 (2 = one pass, 4 = two passes)."""
    out = frames
    for _ in range({2: 1, 4: 2}.get(int(factor), 1)):
        out = rife_double(net, out, **kw)
    return out
