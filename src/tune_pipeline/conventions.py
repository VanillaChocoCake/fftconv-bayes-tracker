"""Numeric conventions shared by the spectral preprocessing stage and the
tune estimators.

Estimator configs default their fields to the names defined here instead of
hard-coding the values, so the preprocessing stage and every estimator always
agree.

torch-free / numpy-free on purpose: importable from any stage without pulling
heavy dependencies.
"""
from __future__ import annotations

# Low-tune cutoff of the folded tune axis. The valid tune search band is
# ``q_grid > LOW_CUT_Q`` (``FrontendFrame.valid_mask``). The preprocessing
# stage builds the mask and every estimator reads or rebuilds the same cutoff.
#
# The operating tunes span [0.05, 0.49], so 0.04 leaves a 0.01 guard below the
# lowest tune while keeping the q = 0.05 bin valid (0.05 > 0.04). The cutoff
# excludes the skirt of the folded longitudinal Schottky harmonics, which on
# low-revolution-frequency, wide-bandwidth (coasting-beam) frames can broaden
# into the low-tune region.
LOW_CUT_Q: float = 0.04
