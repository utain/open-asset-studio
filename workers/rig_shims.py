"""Stand-ins for the CUDA / Blender packages UniRig imports (no nvcc, no aarch64 wheels): installed in sys.modules BEFORE
vendor/UniRig is imported.

  spconv.pytorch, flash_attn(.modules.mha)   pure PyTorch (workers/trellis_shims.py + an MHA module with flash-attn's layout)
  torch_scatter.segment_csr                  torch.segment_reduce
  torch_cluster.fps                          farthest point sampling in plain torch
  bpy, open3d, pyrender, fast_simplification importable stubs (only the Blender extraction / FBX export paths use them,
                                             which the rig worker replaces)
  vertex_group.voxelization                  CPU re-implementation (the original renders depth maps with pyrender / OpenGL)
"""
from __future__ import annotations

import math
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import m3d_common as C
import trellis_shims as TS


# ----------------------------------------------------------------------------------------- torch_scatter / torch_cluster
def segment_csr(src, indptr, out=None, reduce="sum"):
    reduce = {"add": "sum"}.get(reduce, reduce)
    lengths = (indptr[1:] - indptr[:-1]).to(torch.int64)
    return torch.segment_reduce(src, reduce, lengths=lengths, axis=0)


def fps(pos, batch=None, ratio=0.5, random_start=True):
    """Farthest point sampling per batch element; returns global indices (same contract as torch_cluster.fps)."""
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    out = []
    for b in torch.unique(batch):
        idx = torch.nonzero(batch == b).squeeze(1)
        p = pos[idx].float()
        n = p.shape[0]
        k = max(1, int(math.ceil(ratio * n)))
        sel = torch.empty(k, dtype=torch.long, device=pos.device)
        dist = torch.full((n,), float("inf"), device=pos.device)
        cur = int(torch.randint(0, n, (1,))) if random_start else 0
        for i in range(k):
            sel[i] = cur
            d = ((p - p[cur]) ** 2).sum(-1)
            dist = torch.minimum(dist, d)
            cur = int(torch.argmax(dist))
        out.append(idx[sel])
    return torch.cat(out)


# ------------------------------------------------------------------------------------------- flash_attn.modules.mha.MHA
class MHA(nn.Module):
    """Cross / self attention with the parameter layout of flash_attn.modules.mha.MHA (Wq, Wkv / Wqkv, out_proj)."""

    def __init__(self, embed_dim, num_heads, cross_attn=False, qkv_proj_bias=True, out_proj_bias=True, **kw):
        super().__init__()
        self.embed_dim, self.num_heads, self.cross_attn = embed_dim, num_heads, cross_attn
        self.head_dim = embed_dim // num_heads
        if cross_attn:
            self.Wq = nn.Linear(embed_dim, embed_dim, bias=qkv_proj_bias)
            self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_proj_bias)
        else:
            self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_proj_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)

    def forward(self, x, x_kv=None, **kw):
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        if self.cross_attn:
            q = self.Wq(x).view(B, L, H, D)
            kv = self.Wkv(x if x_kv is None else x_kv)
            k, v = kv.view(B, -1, 2, H, D).unbind(2)
        else:
            q, k, v = self.Wqkv(x).view(B, L, 3, H, D).unbind(2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(o.reshape(B, L, H * D))


# ------------------------------------------------------------------------------------------------- CPU voxelization
def voxelization_cpu(vertices, faces, grid=256, scale=1.0, backend="pyrender"):
    """Occupied-voxel centres (M, 3) of a mesh normalised into [-1, 1]: surface voxels from dense surface sampling, then
    the interior is filled where at least two of the three axis-aligned column tests agree (same rule as UniRig's open3d
    branch). Output convention: centres at -1 + (i + 0.5) * 2 / grid."""
    import trimesh
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    m = trimesh.Trimesh(v, f, process=False)
    vs = 2.0 / grid
    n_pts = int(min(8_000_000, max(500_000, m.area / (vs * vs * 0.25))))
    pts, _ = trimesh.sample.sample_surface(m, n_pts)
    idx = np.floor((np.vstack([pts, v]) + 1.0) / vs).astype(np.int64).clip(0, grid - 1)
    vox = np.zeros((grid, grid, grid), dtype=bool)
    vox[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    ar = np.arange(grid)

    def span(axis):
        has = vox.any(axis=axis)
        first = np.where(has, vox.argmax(axis=axis), grid)
        last = np.where(has, grid - 1 - np.flip(vox, axis=axis).argmax(axis=axis), -1)
        return first, last

    x_min, x_max = span(0)          # (grid, grid) over (y, z)
    y_min, y_max = span(1)          # over (x, z)
    z_min, z_max = span(2)          # over (x, y)
    in_x = (ar[:, None, None] >= x_min[None]) & (ar[:, None, None] <= x_max[None])
    in_y = (ar[None, :, None] >= y_min[:, None, :]) & (ar[None, :, None] <= y_max[:, None, :])
    in_z = (ar[None, None, :] >= z_min[:, :, None]) & (ar[None, None, :] <= z_max[:, :, None])
    vox |= (in_x.astype(np.int8) + in_y + in_z) >= 2
    x, y, z = np.nonzero(vox)
    return (np.stack([x, y, z], 1) + 0.5) * vs - 1.0


def patch_voxelization():
    import src.data.vertex_group as vg
    vg.voxelization = voxelization_cpu


# ------------------------------------------------------------------------------------------------------------ install
def install():
    TS.install()
    spp = sys.modules["spconv.pytorch"]
    spp.modules = types.SimpleNamespace(
        is_spconv_module=lambda m: isinstance(m, (TS.SubMConv3d, TS.SparseConv3d)))
    sys.modules["spconv.pytorch.modules"] = spp.modules
    fa = sys.modules["flash_attn"]
    mod = types.ModuleType("flash_attn.modules")
    mha = types.ModuleType("flash_attn.modules.mha")
    mha.MHA = MHA
    mod.mha = mha
    fa.modules = mod
    sys.modules["flash_attn.modules"] = mod
    sys.modules["flash_attn.modules.mha"] = mha
    ts = types.ModuleType("torch_scatter")
    ts.segment_csr = segment_csr
    sys.modules["torch_scatter"] = ts
    tc = types.ModuleType("torch_cluster")
    tc.fps = fps
    sys.modules["torch_cluster"] = tc
    C.shim_missing("bpy", "open3d", "pyrender", "fast_simplification", "wandb")
    # transformers probes optional packages with importlib.util.find_spec, which raises for a module whose __spec__ is None
    import importlib.machinery as _m
    for name in ("flash_attn.modules", "flash_attn.modules.mha", "torch_scatter", "torch_cluster"):
        sys.modules[name].__spec__ = _m.ModuleSpec(name, None)
