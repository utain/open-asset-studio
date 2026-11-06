"""Pure-PyTorch stand-ins for the CUDA-extension packages TRELLIS needs (no nvcc, no aarch64 wheels here).

install() registers fake modules in sys.modules BEFORE trellis is imported:

  spconv.pytorch   SparseConvTensor + SubMConv3d (submanifold 3x3x3 convolution, hash lookup of the 27 neighbours).
                   TRELLIS' inference path only uses stride-1 submanifold convolutions; strided / inverse sparse convs
                   (the VAE *encoder*, not needed to generate) raise NotImplementedError.
  flash_attn       flash_attn_qkvpacked_func / flash_attn_varlen_{qkvpacked,kvpacked,}_func on torch SDPA
                   (variable-length sequences are padded into batches with a key mask).

Weight layout of SubMConv3d matches spconv ([out, k, k, k, in]) so the released checkpoints load unchanged, and the
neighbour rule is spconv's: out(p) = sum_k W[k] * in(p + k - centre).
"""
from __future__ import annotations

import itertools
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- spconv
class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, grid=None, voxel_num=None, indice_dict=None,
                 benchmark=False, permanent_thrust_allocator=False, enable_timer=False, force_algo=None):
        self.features = features
        self.indices = indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = int(batch_size)
        self.grid = grid
        self.voxel_num = voxel_num
        self.indice_dict = indice_dict if indice_dict is not None else {}
        self.benchmark = benchmark
        self.benchmark_record = {}
        self.thrust_allocator = None
        self._timer = None
        self.force_algo = force_algo
        self.int8_scale = None

    @property
    def _features(self):
        return self.features

    @_features.setter
    def _features(self, v):
        self.features = v

    def replace_feature(self, feature):
        new = SparseConvTensor(feature, self.indices, self.spatial_shape, self.batch_size, self.grid, self.voxel_num, self.indice_dict)
        return new

    @property
    def spatial_size(self):
        return self.spatial_shape

    def dense(self, channels_first=True):
        D = self.spatial_shape
        feats = self.features
        out = feats.new_zeros((self.batch_size, *D, feats.shape[1]))
        idx = self.indices.long()
        out[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = feats
        if channels_first:
            out = out.permute(0, 4, 1, 2, 3).contiguous()
        return out


_NB: dict = {}


class SubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 indice_key=None, algo=None, **kw):
        super().__init__()
        k = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * 3
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = tuple(k)
        self.dilation = dilation if isinstance(dilation, (tuple, list)) else (dilation,) * 3
        self.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        # same init as spconv (kaiming uniform), irrelevant for inference
        nn.init.kaiming_uniform_(self.weight.view(out_channels, -1), a=math.sqrt(5))

    def _neighbours(self, indices, spatial_shape):
        """[N, K] index of the neighbour feature row for every kernel tap (-1 = empty), cached per coordinate set."""
        key = (id(indices), tuple(spatial_shape), self.kernel_size, self.dilation)
        hit = _NB.get(key)
        if hit is not None and hit[0] is indices:      # same coordinate tensor object (kept alive by the entry)
            return hit[1]
        idx = indices.long()
        S = torch.tensor([s + 2 * max(self.kernel_size) * max(self.dilation) for s in spatial_shape], device=idx.device)
        pad = max(self.kernel_size) * max(self.dilation)

        def lin(c):  # c: [..., 4] (b, x, y, z) -> int64 key
            return ((c[..., 0] * S[0] + c[..., 1] + pad) * S[1] + c[..., 2] + pad) * S[2] + c[..., 3] + pad

        keys = lin(idx)
        order = torch.argsort(keys)
        skeys = keys[order]
        cols = []
        for off in itertools.product(*[range(k) for k in self.kernel_size]):
            d = torch.tensor([0] + [(o - k // 2) * dl for o, k, dl in zip(off, self.kernel_size, self.dilation)], device=idx.device)
            q = lin(idx + d)
            pos = torch.searchsorted(skeys, q).clamp_(max=skeys.numel() - 1)
            hit = skeys[pos] == q
            cols.append(torch.where(hit, order[pos], torch.full_like(pos, -1)))
        nb = torch.stack(cols, 1)
        _NB[key] = (indices, nb)
        while len(_NB) > 6:
            _NB.pop(next(iter(_NB)))
        return nb

    def forward(self, x: SparseConvTensor):
        feats = x.features
        nb = self._neighbours(x.indices, x.spatial_shape)               # [N, K]
        N, K = nb.shape
        w = self.weight.reshape(self.out_channels, K, self.in_channels)   # [O, K, I]
        out = feats.new_zeros(N, self.out_channels)
        chunk = max(1, 4_000_000 // max(1, self.in_channels))
        for s in range(0, N, chunk):
            n = nb[s:s + chunk]
            valid = n >= 0
            g = feats[n.clamp(min=0)]                                    # [n, K, I]
            g = g * valid.unsqueeze(-1).to(g.dtype)
            out[s:s + chunk] = torch.einsum("nki,oki->no", g, w.to(g.dtype))
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return x.replace_feature(out)


class SparseConv3d(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        raise NotImplementedError("strided sparse convolution is not available in the pure-torch spconv stand-in (encoder only)")


class SparseInverseConv3d(SparseConv3d):
    pass


class ConvAlgo:
    Native = 0
    MaskImplicitGemm = 1


# ------------------------------------------------------------------------ flash_attn
def _sdpa(q, k, v, mask=None):
    # q [B, Lq, H, C], k/v [B, Lk, H, C]
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return o.transpose(1, 2)


def flash_attn_qkvpacked_func(qkv, *a, **k):
    q, kk, v = qkv.unbind(dim=2)
    return _sdpa(q, kk, v)


def _pad(x, cu, lens, L):
    B = lens.numel()
    out = x.new_zeros((B, L) + tuple(x.shape[1:]))
    b = torch.repeat_interleave(torch.arange(B, device=x.device), lens)
    p = torch.arange(x.shape[0], device=x.device) - cu[:-1][b]
    out[b, p] = x
    return out, b, p


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, *a, **kw):
    cq = cu_seqlens_q.long().to(q.device)
    ck = cu_seqlens_k.long().to(q.device)
    lq, lk = cq[1:] - cq[:-1], ck[1:] - ck[:-1]
    Lq, Lk = int(lq.max()), int(lk.max())
    B = lq.numel()
    out = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
    qp, qb, qpos = _pad(q, cq, lq, Lq)
    kp, _, _ = _pad(k, ck, lk, Lk)
    vp, _, _ = _pad(v, ck, lk, Lk)
    mask = (torch.arange(Lk, device=q.device)[None] < lk[:, None])[:, None, None, :]
    chunk = max(1, 262144 // max(Lq, Lk, 1))
    res = []
    for s in range(0, B, chunk):
        res.append(_sdpa(qp[s:s + chunk], kp[s:s + chunk], vp[s:s + chunk], mask[s:s + chunk]))
    res = torch.cat(res, 0)
    out[:] = res[qb, qpos]
    return out


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=None, *a, **kw):
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, *a, **kw):
    k, v = kv.unbind(dim=1)
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k)


def install():
    sp = types.ModuleType("spconv")
    spp = types.ModuleType("spconv.pytorch")
    for n, o in dict(SparseConvTensor=SparseConvTensor, SubMConv3d=SubMConv3d, SparseConv3d=SparseConv3d,
                     SparseInverseConv3d=SparseInverseConv3d, ConvAlgo=ConvAlgo).items():
        setattr(spp, n, o)
    sp.pytorch = spp
    sys.modules["spconv"] = sp
    sys.modules["spconv.pytorch"] = spp
    fa = types.ModuleType("flash_attn")
    fa.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
    fa.flash_attn_varlen_func = flash_attn_varlen_func
    fa.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    fa.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    sys.modules["flash_attn"] = fa
    # transformers probes optional packages with importlib.util.find_spec, which raises for a module with __spec__ None
    import importlib.machinery
    for name in ("spconv", "spconv.pytorch", "flash_attn"):
        sys.modules[name].__spec__ = importlib.machinery.ModuleSpec(name, None)
