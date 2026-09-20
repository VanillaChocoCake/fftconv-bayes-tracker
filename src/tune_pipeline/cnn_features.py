"""CNN-input feature builder for the deep-learning tune estimators.

A single implementation of the per-frame ``x``/``w`` preprocessing
(``FeatureBuilder``) serves both the two-channel and the three-channel input
layouts on the full folded-tune frame ``[0, 0.5)``:
    ``build(x, w)``      -> (B, 2, L)  [MAD-z(x), w/max(w)]
    ``build(x, w, qg)``  -> (B, 3, L)  [MAD-z(x), w/max(w), q_grid]
The 2-channel output is the strict subset ``build(x, w, qg)[:, :2]`` (identical
ch0/ch1 via the shared ``mad_z`` / ``w_norm`` primitives).

torch-dependent; import explicitly
(``from tune_pipeline.cnn_features import FeatureBuilder``) so torch-free
modules do not pull in torch.

Channels:
    ch0 : MAD-robust z-score of the linear-magnitude PSD ``x`` (unclipped).
    ch1 : ``w / max(w)``, clipped to ``[0, 1]``.
    ch2 : ``q_grid`` (3-channel ``build(x, w, qg)`` only) — absolute tune coords, a CNN input channel.
ch0/ch1 are zeroed below the low-cut tune mask (``q_grid > low_cut_q``). Per-frame
normalisation removes absolute-amplitude information from the trunk's input.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .conventions import LOW_CUT_Q


@dataclass(frozen=True)
class FeatureConfig:
    """Per-frame input builder config.

    CNN input channels:
        ch0 : MAD-z-score(x) (unclipped)
        ch1 : w / max(w), clipped to [0, 1]

    Per-frame normalisation means the trunk learns no absolute amplitude
    invariance. No clip anywhere — MAD-z is self-scaled and the denominator is
    floored, so ch0 is numerically safe unclipped.
    """

    low_cut_q: float = LOW_CUT_Q       # single source: tune_pipeline.conventions.LOW_CUT_Q


# ----------------------------------------------------------------------------
# Shared per-frame channel primitives for ch0 (MAD-z signal) and ch1 (weight),
# used by both the 2-channel and the 3-channel build() paths, so the eps and
# the formula are identical in both.
# ----------------------------------------------------------------------------
# eps 1e-20: physical-scale PSDs have mad ~1e-8, so a larger absolute eps such
# as 1e-6 would dominate the denominator, defeat the MAD scale-invariance and
# flatten high-SNR sidebands. 1e-20 is a pure div-by-zero guard (above the
# float32 subnormal range) and must stay << any real mad.
_MAD_Z_EPS = 1e-20


def mad_z(x: torch.Tensor) -> torch.Tensor:
    """MAD-robust z-score of a linear-magnitude PSD ``(B, L)``.

    med/mad are taken over the full last axis (the caller masks the low-cut
    region afterwards). Unclipped: MAD-z is self-scaled and the denominator is
    floored, so it is numerically safe; a fixed clamp would flatten the
    sidebands of high-SNR frames, whose z-scores reach ~10^3.
    """
    med = x.median(dim=-1, keepdim=True).values
    mad = (x - med).abs().median(dim=-1, keepdim=True).values
    return (x - med) / (1.4826 * mad + _MAD_Z_EPS)


def w_norm(w: torch.Tensor) -> torch.Tensor:
    """Weight channel ``(B, L)`` normalised to ``[0, 1]`` by its per-frame max."""
    return (w / w.amax(dim=-1, keepdim=True).clamp(min=1e-6)).clamp(min=0.0, max=1.0)


class FeatureBuilder:
    """Caches the q-low-cut mask for a fixed config.

    Stateless w.r.t. frames — can be reused across batches and frames. The
    mask is created on first call based on the input tensor's device and
    dtype, and cached thereafter.
    """

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self._low_cut_masks: dict = {}

    # ---------- caches ---------------------------------------------
    def _get_low_cut_mask(self, L: int, device, dtype) -> torch.Tensor:
        """Per-(L, dtype) cached `q_grid > low_cut_q` mask."""
        cached = self._low_cut_masks.get((L, dtype))
        if cached is None or cached.device != torch.device(device):
            q_grid = torch.linspace(0.0, 0.5, L + 1, device=device)[:-1]
            cached = (q_grid > self.cfg.low_cut_q).to(dtype=dtype)
            self._low_cut_masks[(L, dtype)] = cached
        return cached

    # ---------- primitives -----------------------------------------
    def _ch0_linear_z(self, x: torch.Tensor) -> torch.Tensor:
        """MAD-based robust z-score on linear-magnitude PSD (shared ``mad_z``)."""
        return mad_z(x)

    # ---------- public --------------------------------------------
    def build(self, x: torch.Tensor, w: torch.Tensor,
              q_grid: torch.Tensor | None = None) -> torch.Tensor:
        """CNN-input feature tensor. ``q_grid=None`` -> 2 channels ``(B, 2, L)`` [MAD-z(x), w/max];
        ``q_grid`` given -> 3 channels ``(B, 3, L)`` [MAD-z(x), w/max, q_grid]
        (ch2 is an absolute-tune coordinate channel the first conv reads).

        ch0/ch1 are zeroed below the low-cut tune mask (``q_grid > low_cut_q``): synthesised from a
        canonical ``linspace(0, 0.5, L, endpoint=False)`` in the 2-channel path, or from the passed
        ``q_grid`` in the 3-channel path (passed, not synthesised, so ch2 is bit-identical to what the
        trained model consumed). The input is always the full frame ``[0, 0.5)``.
        """
        assert x.ndim == 2 and w.ndim == 2 and x.shape == w.shape, (
            f"expected matching (B, L); got x{x.shape} w{w.shape}"
        )
        B, L = x.shape
        ch0 = self._ch0_linear_z(x)                 # MAD-z(x)
        ch1 = w_norm(w)                             # w / max(w)
        if q_grid is None:                          # 2-channel — canonical cached low-cut mask
            feats = torch.cat([ch0.unsqueeze(1), ch1.unsqueeze(1)], dim=1)
            mask = self._get_low_cut_mask(L, feats.device, feats.dtype)
            return feats * mask.view(1, 1, L)
        assert q_grid.shape == x.shape, (           # 3-channel — mask from the passed q_grid
            f"expected q_grid (B, L)={tuple(x.shape)}; got {tuple(q_grid.shape)}"
        )
        q_grid = q_grid.to(x.dtype)
        mask = (q_grid > self.cfg.low_cut_q).to(x.dtype)
        return torch.stack([ch0 * mask, ch1 * mask, q_grid], dim=1)
