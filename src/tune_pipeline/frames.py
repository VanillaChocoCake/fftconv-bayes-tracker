"""Immutable per-frame data contracts exchanged between the spectral
preprocessing stage and a tune estimator.

Flow::

    spectral preprocessing ──► FrontendFrame ──► estimator ──► MeasurementResult

- `FrontendFrame` is the output of the spectral preprocessing stage: the
  PSD folded onto a uniform tune grid in [0, 0.5), together with the
  per-bin sampling weights and the validity mask.
- `MeasurementResult` is emitted by an estimator once per frame.

Both are `frozen` dataclasses, so no hidden mutable state leaks between
frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

import numpy as np


class QualityFlag(IntFlag):
    """Bit flags carried by a FrontendFrame describing upstream health.

    Estimators may veto or down-weight a frame based on these.
    """

    OK = 0
    FEW_PEAKS = 1 << 0          # fewer than 2 longitudinal harmonic peaks found
    FREV_FALLBACK = 1 << 1      # frev_est came from frev_prev fallback, not current data
    SATURATED = 1 << 2          # input PSD shows ADC saturation signature
    LOW_SNR = 1 << 3            # adaptive threshold bumped against the noise floor


@dataclass(frozen=True)
class FrontendFrame:
    """Output of the spectral preprocessing stage for one measurement frame.

    Shapes are fixed by the number of tune bins ``n_bins`` (typically 1024).

    dtype notes
    -----------
    - ``q_grid``, ``peak_indices``, ``valid_mask`` are always float32 /
      int32 / bool respectively.
    - ``mapped_psd`` and ``weight_sum`` may be float32 or float64 depending
      on the preprocessing path. Estimators handle both without casting.
    - ``frev_est`` is always Python float.
    """

    q_grid: np.ndarray           # [n_bins] float32, uniform in [0, 0.5)
    mapped_psd: np.ndarray       # [n_bins] float32 or float64, soft-binned PSD
    weight_sum: np.ndarray       # [n_bins] float32 or float64, accumulated bin weights
    peak_indices: np.ndarray     # [N_peaks] int32, longitudinal harmonic peak indices in the original f_absolute array
    frev_est: float              # Hz, estimated revolution frequency (may fall back to frev_prev)
    valid_mask: np.ndarray       # [n_bins] bool, True where q_grid is within usable range
    quality_flags: int = int(QualityFlag.OK)


@dataclass(frozen=True)
class MeasurementResult:
    """Output of a tune estimator for one frame."""

    q: float
    sigma: float = 0.0           # uncertainty if available, else 0
    failed: bool = False         # frame vetoed (out-of-band, no peaks, etc.)
    source: str = ""             # estimator identifier
