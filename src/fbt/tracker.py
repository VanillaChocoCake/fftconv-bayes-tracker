"""Physical-units (q, v) Bayes tracker for variable-L observations.

Wraps the grid-agnostic ``QVBayesTracker`` (``fbt.bayes_tracker``) on a state grid
parameterised in PHYSICAL tune units (q/frame), so the velocity limits track real
tune-velocity bounds rather than a fixed bin geometry.

Design
------
- Physical parameters (the single source of truth, q units):
      v_max_q   = 0.02       q/frame   (see ``V_MAX_Q``)
      sigma_v_q = 5e-4       q/frame   (see ``SIGMA_V_Q``)
  On a Q_state grid these map to v_max_bins = v_max_q / (0.5/Q_state), etc.; ``v_stride``
  then coarsens only the velocity axis (a latency lever, full range preserved).
- Q_state is set by the caller = the fixed-per-run input length L. The per-step q-blur
  (velocity-lattice quantisation residual, σ = v_stride·√(1/12) fine bins — one lattice
  quantum) shrinks with Q at stride 1, lowering the posterior-width floor into the
  (1–3)e-4 q region that has operational value (drift early-warning).
- Observation: with Q_state == L the native-L map is consumed DIRECTLY as the per-frame
  likelihood; ``VLTracker.project_obs`` is then identity (it only repeat-interleaves a native-L
  input when Q_state > L).

GAUSSIAN_TRUNCATE_SIGMA = 4: the mass-loss criterion n > Φ⁻¹(1 − 1/(2L)) gives 3.89 at
L = 8192, so the 4σ truncation used here still satisfies it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fbt.bayes_tracker import (TrackerConfig, QVBayesTracker,
                                TrackerOutput)
from tune_pipeline.conventions import LOW_CUT_Q

# Physical parameter values (q units) — the SINGLE SOURCE OF TRUTH. VLTracker
# converts each to bins per-L (bin_step = 0.5/Q_state), so all are L-invariant.
# No physical quantity is hardcoded in bins anywhere downstream (the q-blur in
# bayes_tracker is a floor-quantisation residual = 1 bin by definition, not a
# physical param, so it legitimately stays in bins).
V_MAX_Q = 0.02                     # tracked velocity limit (q/frame); covers the fastest ramp of the dynamic-tune benchmark (peak |v|≈41 bins @ L=1024).
SIGMA_V_Q = 5e-4                   # velocity-process-noise std (q/frame) -- a MANUAL, PER-FACILITY calibration.
# It is the beam's tune-velocity process noise: a PHYSICAL q/frame quantity, therefore L-INVARIANT -- do NOT scale
# it with FFT resolution (the beam does not move slower because you took a longer FFT), and it is NOT a parameter
# to auto-adapt; set it from the target machine's beam-velocity characteristics. With this default,
# 4*sigma_v = 2e-3 q/frame bounds the per-frame tune innovation of the dynamic-tune benchmark trajectories (max ~1.1e-3);
# the tracker is insensitive to the exact value over roughly 4-7e-4.
PRIOR_SIGMA_Q = 1e-2               # reset Gaussian-prior width (q) = 20.48 bins @ L=1024; physical, so the prior keeps the same width at every L.


@dataclass(frozen=True)
class VLTrackerConfig:
    q_state_bins: int = 8192
    v_max_q: float = V_MAX_Q
    sigma_v_q: float = SIGMA_V_Q
    prior_sigma_q: float = PRIOR_SIGMA_Q   # reset Gaussian-prior width (q units), L-invariant
    transition_v_decay: float = 1.0
    low_cut_q: float = LOW_CUT_Q
    v_stride: int = 1          # velocity-grid stride (fine q-bins/state); >1 = latency lever, V∝1/stride


class VLTracker:
    """Wraps ``QVBayesTracker`` on a state grid in physical tune units."""

    def __init__(self, cfg: VLTrackerConfig, device: str = "cuda"):
        self.cfg = cfg
        Q = int(cfg.q_state_bins)
        bin_step = 0.5 / Q
        S = max(1, int(cfg.v_stride))
        # coarse velocity-state count; range preserved (stride·v_max_bins ≈ v_max_q/bin_step)
        v_max_bins = int(round(cfg.v_max_q / (S * bin_step)))
        # σ_v stays in FINE bins — v_axis carries strided fine-bin values, so the
        # transition Gaussian is the SAME physical kernel, just sampled coarser.
        # Floor at 1 grid step: a sub-grid σ_v freezes velocity diffusion (the transition
        # kernel becomes narrower than one velocity state, so the filter stops following
        # tune motion). Binds only when bin_step > σ_v_q (L=512 at the default σ_v:
        # 0.512 -> 1.0); identity at L>=1024.
        sigma_v_bins = max(1.0, cfg.sigma_v_q / bin_step)
        self.tcfg = TrackerConfig(
            n_q_bins=Q,
            v_max_bins=v_max_bins,
            transition_v_decay=cfg.transition_v_decay,
            sigma_v_bins=sigma_v_bins,
            v_stride=S,
        )
        self.core = QVBayesTracker(self.tcfg, device=device)
        self.device = torch.device(device)
        q_grid = torch.linspace(0.0, 0.5, Q + 1, device=self.device)[:-1]
        self.valid_mask = q_grid > cfg.low_cut_q
        self.Q = Q
        self.bin_step = bin_step           # q per state bin = 0.5/Q (for q->bin conversions)

    def reset(self, initial_q: float | None = None) -> None:
        # prior_sigma_q (physical q) -> bins per-L: the reset Gaussian keeps the
        # SAME physical width at every L. initial_q=None -> uniform (the core then
        # ignores the width).
        self.core.reset(initial_q,
                        prior_sigma_bins=self.cfg.prior_sigma_q / self.bin_step)

    def project_obs(self, map_native: torch.Tensor) -> torch.Tensor:
        """(L,) native logits -> (Q_state,) piecewise-constant logits."""
        L = map_native.shape[-1]
        assert self.Q % L == 0, f"Q_state={self.Q} not a multiple of L={L}"
        r_up = self.Q // L
        if r_up == 1:
            return map_native
        return map_native.repeat_interleave(r_up, dim=-1)


    def step(self, map_native: torch.Tensor, w_obs=None) -> TrackerOutput:
        """One frame: native-L logits in, (q_t, σ_t) out. ``w_obs`` tempers the observation update
        -- the estimator passes the confidence-gate weight sigmoid(conf_logit). A bare call
        (w_obs=None) applies no gate (w_obs=1)."""
        obs = self.project_obs(map_native.to(self.device))
        if w_obs is None:
            w_obs = 1.0
        new_post, scalars, p_q = self.core.step_core(
            self.core.log_post, obs, self.valid_mask, w_obs=w_obs)
        self.core.log_post = new_post
        # q/sigma come from `scalars` only — no per-frame (Q,) p_q / valid_mask
        # host transfers needed.
        return self.core.output_from(scalars.detach().cpu().numpy())
