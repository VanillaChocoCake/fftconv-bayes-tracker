"""2D (q, v) discrete Bayes filter — single linear step.

State
-----
log-posterior over ``(Q, V)`` where ``Q = n_q_bins``, ``V = 2·v_max+1``
and ``v`` is bins-per-frame. Log-domain keeps the step numerically safe
over long uncertain stretches.

Per-frame step (``step_core``, the tensor primitive the estimator inlines
into its fused cudagraph) is one straight-line tensor program, no Python
branches in the math:

    1. Observe     invalid bins → −∞; log_softmax(CNN map logits) → log_p_obs
    2. Update      log_post += log_p_obs (broadcast over v); normalise
    3. Readout     posterior-mean q + marginal σ_q from the post-update
                   q-marginal
    4. Predict     v-transition + q-shift-by-v + q-blur; normalise; the
                   result is stored as ``self.log_post`` for the *next*
                   step's update

Frame-0 / predict ordering
--------------------------
``log_post`` starts uniform (set in ``__init__``). Predict runs at the
*end* of every step, so the first step's update sees the uniform prior
and the readout is a pure-observation posterior. Predict at the end of
frame *i* prepares the prior for frame *i+1*; under the alternative
"skip-predict-on-frame-0, predict-at-start-otherwise" ordering exactly
one predict separates each pair of consecutive observations, so the two
orderings produce bit-identical readouts (modulo CUDA reduction order).
This lets the step body be branch-free — no ``frame_idx`` guard.

Observation (CNN map only)
--------------------------
    ℓ_t(k)    = log_softmax(map_logits(k))
    log_p_obs = ℓ_t

Transition
----------
    v_t = ρ · v_{t-1} + η_v        (σ_v bin/frame, v-transition kernel)
    q_t = q_{t-1} + floor(v_t)     (integer q-shift gather + wrap mask)
followed by a q-blur of σ_q = sqrt(1/12) ≈ 0.289 bins at ``v_stride = 1``
(the floor-residual quantisation variance only; the velocity noise σ_v is
already carried by the v-transition + q-shift). On a strided velocity grid
the blur widens with the stride — see ``_build_q_blur_kernel``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


GAUSSIAN_TRUNCATE_SIGMA = 4.0   # Gaussian-kernel truncation half-width, in σ


@dataclass(frozen=True)
class TrackerConfig:
    """2D (q, v) discrete Bayes filter.

    State grid (n_q_bins, 2*v_max+1). Transition: v_t = ρ·v_{t-1} + η_v,
    q_t = q_{t-1} + floor(v_t); the floor-quantisation noise (Var(U[0,1))
    = 1/12) is added by a separate q-blur kernel, NOT folded into σ_v.
    Observation = log_softmax(CNN map logits) only. Readout = posterior-mean q.
    """

    n_q_bins: int = 1024
    # Covers the empirical 99th-percentile tune velocity (the fast and medium
    # ramps reach |v| > 20 bins/frame at turn-points; slow ramps stay below 4).
    v_max_bins: int = 25
    # ρ ≈ 1 = near-random-walk on v (real trajectories hold ~constant v).
    transition_v_decay: float = 1.0
    # Velocity-process-noise std (bins/frame) in the v-transition kernel
    # only; the q-blur uses the constant σ_q = sqrt(1/12).
    sigma_v_bins: float = 1.0
    # Velocity-grid STRIDE (fine q-bins per velocity state). v_stride>1 coarsens
    # the velocity grid (V = 2·v_max_bins+1 shrinks ∝ 1/stride) while preserving
    # the SAME velocity range AND the SAME physical σ_v: v_axis carries STRIDED
    # FINE-bin values (stride·[-v_max..v_max]), so the transition Gaussian (σ in
    # fine bins) is the SAME physical kernel, just sampled coarser in v.
    # Latency lever: step_core cost ∝ V. stride=1 = the full fine grid.
    v_stride: int = 1


@dataclass
class TrackerOutput:
    """Estimator output: posterior-mean tune + marginal σ_q."""
    q_bin: float          # continuous posterior-mean bin index
    q: float              # posterior-mean tune in [0, 0.5)
    sigma_q: float        # std of marginal q posterior, in q units


class QVBayesTracker:
    """(q, v) discrete Bayes filter holding running log-posterior state.

    Parameters
    ----------
    cfg : TrackerConfig
    device : torch.device
        Where the state lives. Inference runs on cuda; cpu is supported and
        used by the tests.
    """

    def __init__(self, cfg: TrackerConfig,
                 device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.Q = cfg.n_q_bins
        self.V = 2 * cfg.v_max_bins + 1
        # v_axis carries STRIDED fine-bin velocity values (stride·integer): the
        # q-shift reads (q - v_axis[v]) so shifts are exact fine-bin displacements,
        # while the velocity grid has only V = 2·v_max_bins+1 states. stride=1 =
        # the contiguous fine grid.
        self.v_axis = int(cfg.v_stride) * torch.arange(
            -cfg.v_max_bins, cfg.v_max_bins + 1,
            dtype=torch.float32, device=self.device)
        # Precomputed kernels / tables. The v-transition is stored in
        # prob space (_Tv = exp(log_Tv), rows sum to 1) so the per-frame
        # predict can use a (Q,V)@(V,V) matmul instead of materialising a
        # (Q,V,V) logsumexp intermediate.
        self._Tv = self._build_v_transition().exp()           # (V, V), from the fixed cfg.sigma_v_bins
        self._q_blur_kernel = self._build_q_blur_kernel()     # (1, 1, K)
        self._q_blur_pad = (self._q_blur_kernel.shape[-1] - 1) // 2
        self._shift_idx, self._wrap_mask = self._build_q_shift_tables()
        # Uniform log-posterior: initial state + underflow fallback target.
        self._uniform_log_post = torch.full(
            (self.Q, self.V), -math.log(self.Q * self.V),
            device=self.device)
        self._q_idx_f = torch.arange(self.Q, device=self.device,
                                      dtype=self._uniform_log_post.dtype)
        self.log_post = self._uniform_log_post.clone()

    # ---------------- init / reset ---------------------------------
    def reset(self, initial_q: float | None = None,
              prior_sigma_bins: float | None = None) -> None:
        """Reset state to uniform, or to a wide Gaussian around initial_q.

        A true delta prior would be too strong — a single-frame
        observation could not override it. The Gaussian prior with
        ``prior_sigma_bins`` biases early frames toward the hint while
        letting the observation take over.

        ``prior_sigma_bins`` is a GRID quantity (state bins): the PHYSICAL
        width lives in q units one layer up (``tracker.PRIOR_SIGMA_Q``) and
        is converted per-L by ``VLTracker.reset``. This grid-agnostic core
        therefore holds NO bin-hardcoded physical default — when a prior is
        requested the caller must supply the (q-sourced) width.
        """
        if initial_q is None:
            self.log_post = self._uniform_log_post.clone()
            return
        if prior_sigma_bins is None:
            raise ValueError(
                "prior_sigma_bins is required when initial_q is set: the "
                "physical prior width (tracker.PRIOR_SIGMA_Q, q units) must be "
                "converted to bins by the caller (VLTracker.reset). The "
                "grid-agnostic core keeps no bin-hardcoded physical default.")
        k_star = initial_q / 0.5 * self.Q
        k_star = min(max(k_star, 0.0), self.Q - 1.0)
        log_q = -((self._q_idx_f - k_star) ** 2) / (2.0 * prior_sigma_bins ** 2)
        log_v = torch.zeros(self.V, device=self.device)
        lp = log_q.view(self.Q, 1) + log_v.view(1, self.V)
        Z = torch.logsumexp(lp.reshape(-1), dim=0)
        self.log_post = torch.where(torch.isfinite(Z), lp - Z,
                                    self._uniform_log_post)

    # ---------------- single frame step ---------------------------
    def step_core(self, log_post: torch.Tensor, map_logits: torch.Tensor,
                  valid_mask: torch.Tensor, w_obs=1.0):
        """Pure-tensor frame core: fuse → update → readout → predict.

        ``w_obs`` (scalar in [0,1], default 1.0) tempers the observation update:
        ``log_post += w_obs · log_softmax(map)``. w_obs=1 is the standard full
        update; a beam-loss gate drives w_obs→0 on uninformative (noise) frames so
        the filter coasts on the motion model instead of locking onto a noise peak.
        A plain multiply -> CUDA-graph / fused-path safe (no data-dependent branch).

        Fused-path contract: this is the tensor primitive that the estimator's
        fused CUDA-graph step (``FBTEstimator._step_fn``) inlines. No
        host syncs (no ``.item()``, no NumPy), so it is fullgraph-safe.
        Stateless w.r.t. ``self`` except the precomputed kernels — the
        caller owns ``log_post`` and passes it in / stores the result.
        Returns ``(new_log_post, scalars, p_q)`` where ``scalars`` is
        ``[mu_bin, sigma_bin]`` from the post-update marginal and
        ``new_log_post`` is the post-predict prior for the next frame.
        """
        Q = self.Q
        invalid = ~valid_mask

        # ---- observation: CNN-map log-likelihood over the valid bins ----
        # ell_t = log_softmax(map_logits); invalid bins -> -inf -> zero mass.
        ml = map_logits.masked_fill(invalid, float("-inf"))
        log_p_obs = F.log_softmax(ml, dim=-1)                 # (Q,)

        # ---- update + normalise (post-update lp drives the readout) ----
        # w_obs tempers the observation (beam-loss gate); w_obs=1.0 -> standard update.
        lp = log_post + w_obs * log_p_obs.view(Q, 1)
        Z = torch.logsumexp(lp.reshape(-1), dim=0)
        lp = torch.where(torch.isfinite(Z), lp - Z, self._uniform_log_post)

        # ---- readout: marginal q-posterior over the valid range ----
        log_q = torch.logsumexp(lp, dim=-1)                   # (Q,)
        p_q = torch.softmax(log_q, dim=-1) * valid_mask.to(log_q.dtype)
        p_q = p_q / p_q.sum().clamp(min=1e-30)
        mu_bin = (p_q * self._q_idx_f).sum()
        var_bin = ((p_q * (self._q_idx_f - mu_bin) ** 2).sum()).clamp(min=0.0)
        sigma_bin = torch.sqrt(var_bin)

        # ---- predict: prepare the prior for the next step ----
        # v-transition lp_pred[q,v_now] = logsumexp_v_prev(lp + log_Tv) done
        # as a prob-space matmul with a per-row max-shift (avoids the
        # (Q,V,V) logsumexp intermediate). Equivalent because log_Tv ≤ 0 ⇒
        # the row max-shift dominates the internal per-v_now max. Dead
        # (all -inf) q-rows are restored to -inf before the q-shift.
        mv = lp.amax(dim=1, keepdim=True)                      # (Q, 1)
        finite = torch.isfinite(mv)
        mv_safe = torch.where(finite, mv, torch.zeros_like(mv))
        lp_pred = torch.log(
            (torch.exp(lp - mv_safe) @ self._Tv).clamp(min=1e-30)) + mv_safe
        lp_pred = torch.where(finite, lp_pred,
                              torch.full_like(lp_pred, float("-inf")))
        lp_pred = torch.gather(lp_pred, 0, self._shift_idx).masked_fill(
            self._wrap_mask, float("-inf"))
        m = lp_pred.amax(dim=0, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p_blur = torch.exp(lp_pred - m).permute(1, 0).unsqueeze(1)
        p_blur = F.conv1d(p_blur, self._q_blur_kernel, padding=self._q_blur_pad)
        p_blur = p_blur.squeeze(1).permute(1, 0)
        lp_pred = torch.log(p_blur.clamp(min=1e-30)) + m
        Zp = torch.logsumexp(lp_pred.reshape(-1), dim=0)
        new_log_post = torch.where(torch.isfinite(Zp), lp_pred - Zp,
                                   self._uniform_log_post)

        scalars = torch.stack([mu_bin, sigma_bin])
        return new_log_post, scalars, p_q

    def output_from(self, scalars_host: np.ndarray) -> TrackerOutput:
        """Assemble a TrackerOutput from already-on-host ``step_core`` scalars.

        Split from ``step_core`` so the tensor work stays in the cudagraph
        and the host-side assembly stays out of it.
        """
        Q = self.Q
        bin_step_q = 0.5 / Q
        k_frac = float(scalars_host[0])
        sigma_q_bin = float(scalars_host[1])
        return TrackerOutput(
            q_bin=k_frac, q=k_frac / Q * 0.5,
            sigma_q=sigma_q_bin * bin_step_q)

    # ---------------- precompute ----------------------------------
    def _build_v_transition(self) -> torch.Tensor:
        """Gaussian T(v_now | v_prev) with mean ρ·v_prev, std σ_v = fixed cfg.sigma_v_bins.

        Returns log_T: (V, V) — log_T[v_prev_idx, v_now_idx], row-normalised.
        """
        cfg = self.cfg
        sv = cfg.sigma_v_bins
        v_prev = self.v_axis.view(self.V, 1)
        v_now = self.v_axis.view(1, self.V)
        mu = cfg.transition_v_decay * v_prev
        log_T = -((v_now - mu) ** 2) / (2.0 * sv ** 2)
        log_T = log_T - torch.logsumexp(log_T, dim=-1, keepdim=True)
        return log_T

    def _build_q_blur_kernel(self) -> torch.Tensor:
        """Gaussian kernel for the post-shift q blur.

        σ_q = v_stride·sqrt(1/12) fine bins — the LATTICE quantisation residual:
        the true (continuous) velocity falls up to ±stride/2 fine bins from the
        nearest velocity state, so the un-modelled per-frame displacement is
        U over ONE lattice quantum (= v_stride fine bins) → var = stride²/12.
        At stride 1 this is the classic integer-floor sqrt(1/12). Without the
        stride scaling the predict step is under-dispersed by ×stride and the
        filter cannot interpolate between lattice velocities. The velocity
        process noise σ_v is already propagated by the v-transition + q-shift,
        so it is NOT added here — adding it would double-count it. Half-width =
        GAUSSIAN_TRUNCATE_SIGMA·σ_q (4σ).
        """
        sigma = max(1, int(self.cfg.v_stride)) * math.sqrt(1.0 / 12.0)
        h = max(1, int(GAUSSIAN_TRUNCATE_SIGMA * sigma))
        xs = torch.arange(-h, h + 1, dtype=torch.float32, device=self.device)
        k = torch.exp(-(xs ** 2) / (2.0 * sigma ** 2))
        k = k / k.sum()
        return k.view(1, 1, -1)

    def _build_q_shift_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute (Q, V) tables for the per-v q-shift in step_core().

        ``shift_idx[q, v] = (q - v_axis[v]) mod Q`` is the source row to
        read for target slot ``[q, v]``; ``wrap_mask`` is True where the
        unwrapped shift would have crossed the [0, Q) boundary so the
        gather result is masked to −∞.
        """
        Q, V = self.Q, self.V
        v_signed = self.v_axis.to(torch.long)
        q = torch.arange(Q, device=self.device).unsqueeze(1)
        v = v_signed.view(1, V)
        shift_idx = (q - v) % Q
        wrap_mask = ((v > 0) & (q < v)) | ((v < 0) & (q >= Q + v))
        return shift_idx, wrap_mask
