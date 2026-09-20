"""fftconv — conv-stem + FFT global convolution trunk: a global-convolution
sequence model in the spirit of S4D / Hyena, built from official torch.fft ops
only.

Global branch: per-channel implicit kernel over RELATIVE position p ∈ [-1, 1]
(2L-1 taps, regenerated per forward for the batch's L):
    K[c, t] = w1[c]·exp(-s1[c]·|p|)·cos(pi·f[c]·p) + w2[c]·exp(-s2[c]·|p|)
— damped-oscillator basis, 5 scalars/channel. Because p is normalized and the
frame always spans the full 0.5 q range, the kernel's PHYSICAL (q-unit) shape
is L-invariant by construction. Centered linear conv via rfft/irfft with
n = 3L zero-padding (exact, non-circular). Gapless global receptive field at
O(L log L); both-side context inherent (symmetric support).

Block = local depthwise k7 + pointwise + global FFT conv -> GN -> residual,
then channel MLP (1x1 64->128->64) -> GN -> residual. 4 blocks, C=64, ~109k
parameters at the default config.
Kernel weights init small (1e-2): the global branch starts near-silent.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import ArchModel  # noqa: E402

ARCH = "fftconv"
CFG = dict(in_ch=3, ch=64, kernel=7, n_blocks=4, mlp_ratio=2, gn_groups=8)


class FFTGlobalConv(nn.Module):
    """Per-channel damped-oscillator kernel, centered linear conv via FFT."""

    def __init__(self, ch: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(ch) * 1e-2)
        self.w2 = nn.Parameter(torch.randn(ch) * 1e-2)
        # softplus(s_raw) in ~[1, 60]: decay lengths from full-range to ~1/60
        self.s1_raw = nn.Parameter(torch.linspace(0.5, 4.0, ch))
        self.s2_raw = nn.Parameter(torch.linspace(4.0, 0.5, ch))
        self.f = nn.Parameter(torch.linspace(0.0, 24.0, ch))   # oscillation cycles
        self._Kf_cache = None        # rfft(kernel), frame-invariant at inference
        self._Kf_key = None          # (L, device, dtype) the cache was built for

    def _kernel_rfft(self, L: int, n: int, device, dtype) -> torch.Tensor:
        """rfft of the damped-oscillator kernel. Frame-invariant at inference (frozen params + L),
        so it is cached (computed once per geometry) -- the per-frame recompute (linspace + exp/cos
        + rfft of k) is pure redundant work. The cache key includes the parameter ._version
        counters, so ANY in-place change (an optimizer step OR load_state_dict) invalidates it; the
        cache is only used/filled when grad is off (inference), never during training."""
        cacheable = not self.training and not torch.is_grad_enabled()
        key = (L, device, dtype, self.w1._version, self.w2._version,
               self.s1_raw._version, self.s2_raw._version, self.f._version)
        if cacheable and self._Kf_key == key and self._Kf_cache is not None:
            return self._Kf_cache
        p = torch.linspace(-1.0, 1.0, 2 * L - 1, device=device, dtype=dtype)  # (2L-1,)
        ap = p.abs().unsqueeze(0)
        pc = p.unsqueeze(0)
        s1 = F.softplus(self.s1_raw).unsqueeze(1) * 8.0
        s2 = F.softplus(self.s2_raw).unsqueeze(1) * 8.0
        k = (self.w1.unsqueeze(1) * torch.exp(-s1 * ap)
             * torch.cos(math.pi * self.f.unsqueeze(1) * pc)
             + self.w2.unsqueeze(1) * torch.exp(-s2 * ap))      # (C, 2L-1)
        Kf = torch.fft.rfft(k, n=n)                             # (C, n//2+1)
        if cacheable:
            self._Kf_cache, self._Kf_key = Kf, key
        return Kf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        n = 3 * L
        Kf = self._kernel_rfft(L, n, x.device, x.dtype)        # cached at inference
        Xf = torch.fft.rfft(x, n=n)                            # (B, C, n//2+1)
        y = torch.fft.irfft(Xf * Kf.unsqueeze(0), n=n)         # (B, C, n)
        return y[..., L - 1:2 * L - 1]                          # centered "same"


class FFTConvBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, mlp_ratio: int, gn_groups: int):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, kernel, padding=kernel // 2, groups=ch,
                            bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
        self.glob = FFTGlobalConv(ch)
        self.gn1 = nn.GroupNorm(gn_groups, ch)
        hidden = ch * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv1d(ch, hidden, 1), nn.GELU(), nn.Conv1d(hidden, ch, 1))
        self.gn2 = nn.GroupNorm(gn_groups, ch)

    def forward(self, x):
        h = F.gelu(x + self.gn1(self.pw(self.dw(x)) + self.glob(x)))
        return F.gelu(h + self.gn2(self.mlp(h)))


def build(cfg: dict | None = None) -> ArchModel:
    c = dict(CFG)
    if cfg:
        c.update(cfg)
    trunk = nn.Sequential(
        nn.Conv1d(c["in_ch"], c["ch"], c["kernel"],
                  padding=c["kernel"] // 2, bias=False),
        nn.GroupNorm(c["gn_groups"], c["ch"]),
        nn.GELU(),
        *[FFTConvBlock(c["ch"], c["kernel"], c["mlp_ratio"], c["gn_groups"])
          for _ in range(c["n_blocks"])],
    )
    return ArchModel(trunk, c["ch"], c["gn_groups"])
