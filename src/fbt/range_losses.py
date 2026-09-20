"""Loss on the native-L map of a [q_lo,q_hi] slice — windowed-NLL (sole loss).

Per-sample bin scale Δq = (q_hi - q_lo)/L varies, so every bin-unit constant is
expressed in q units and converted per sample:

  k*_native   = (q_true - q_lo)/Δq                   true-peak continuous bin
  conf radius = 2·sigma_loc                          self-measured (see total_loss_fbt)

L_map is a localized Laplace NLL inside a FIXED physics support window around k*
plus a mass lower-bound barrier; the per-frame width is self-measured from the
windowed map (no template, no target width, no SNR oracle).

All targets are per-sample tensors (q_lo, dq vary within a batch even at fixed L).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

SIGMA_Q_PHYS = 0.5 / 1024     # Gaussian target width, in q units
TAU_READOUT_BINS = 0.5        # UNIFORM readout temperature in BINS, L-independent. The inference
                              # readout is softmax(logits/0.5); validation must use the SAME
                              # constant so the checkpoint is selected under the readout it runs
                              # with. (A q-unit temperature would scale with L: 0.5/1/2/4 bins @
                              # L=1024/2048/4096/8192 — an unintended val<->test mismatch.)

# windowed-NLL: a localized Laplace NLL inside a FIXED physics support window around k*, with
# the width SELF-MEASURED from the windowed map (no template, no target width, no SNR oracle).
# The window is a maximum-support assumption ("a real bunched sideband fits here"), NOT a target
# width — it only excludes DISTANT distractors so the map stays robust to fixed/random narrowband
# interference; the per-frame sigma_loc supplies all the width adaptivity. Half-widths are
# physical (q-units, scaled to the bunched-beam upper bound) and converted to bins per sample, so
# the window is L-invariant. The window is anchored at k* (the label), NOT at any map quantity,
# so it is a constant mask w.r.t. the logits — this is what avoids the window<->sigma circular
# dependency an adaptive window would create.
WIN_SIGMA_MAX_Q = 0.025       # bunched-beam sideband upper bound = widest peak's sigma (q)
# Window factors (env-overridable). Defaults = the 1sigma/2sigma window: a=1 core out to
# 1*sigma_max, cos-taper to 0 at 2*sigma_max. Tight on purpose: a wider window only admits
# more noise and distractors, with no benefit at low SNR.
WIN_R_CORE_FACTOR = float(os.environ.get("FBT_WIN_CORE", "1.0"))  # core half-width = factor*sigma_max/dq bins
WIN_R_OUT_FACTOR = float(os.environ.get("FBT_WIN_OUT", "2.0"))    # cos-taper reaches 0 at factor*sigma_max/dq
# Peak-dominance contrastive-loss weight (out-of-window top-1; pushes the in-window peak to be the
# GLOBAL map max). 0.1 enables it (the default); FBT_W_DOM=0 disables it.
W_DOM = float(os.environ.get("FBT_W_DOM", "0.1"))
# min in-window mass fraction (log-barrier lower bound) = the support window's share of the [0,0.5)
# q-axis = the mass a UNIFORM (no-peak) map would put inside it. DERIVED from the window geometry so it
# auto-tracks WIN_SIGMA_MAX_Q / WIN_R_OUT_FACTOR and cannot fall out of sync when either changes: window
# full-extent = 2*WIN_R_OUT_FACTOR*WIN_SIGMA_MAX_Q q, over the 0.5 Nyquist span. The barrier then means
# "the map must beat a flat no-peak spectrum inside the window." = 0.20 at the default window
# (WIN_R_OUT_FACTOR = 2.0, WIN_SIGMA_MAX_Q = 0.025).
WIN_MASS_MIN = 2.0 * WIN_R_OUT_FACTOR * WIN_SIGMA_MAX_Q / 0.5
WIN_LAMBDA_MASS = 1.0         # mass-barrier weight


def soft_argmax(map_logits: torch.Tensor, tau_bins: torch.Tensor) -> torch.Tensor:
    """(B,L) logits, per-sample τ_bins -> (B,) continuous bin index."""
    L = map_logits.shape[-1]
    xs = torch.arange(L, device=map_logits.device,
                      dtype=map_logits.dtype).view(1, L)
    p = torch.softmax(map_logits / tau_bins.view(-1, 1).clamp(min=1e-3), dim=-1)
    return (p * xs).sum(dim=-1)


def loss_conf(conf_logit: torch.Tensor, map_logits: torch.Tensor,
              k_star: torch.Tensor, radius_bins: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        pred = map_logits.argmax(dim=-1).to(map_logits.dtype)
        good = ((pred - k_star).abs() < radius_bins).to(conf_logit.dtype)
    return F.binary_cross_entropy_with_logits(conf_logit, good)


def select_best_by_rho(rows):
    """Best-ckpt selection rule (val-only, no test leakage).

    rows: list of (epoch, mae, rho, tail) per validated epoch, where ``tail`` is the global
      catmiss rate (fraction of frames with global-readout err > 1e-3 = a distractor outranked
      the true peak / missed the precision spec). Rule: among the 10 epochs with the highest
      Spearman rho(|err|, sigma), keep those with val MAE <= 1.1 * (min MAE among those 10),
      then pick the one with the LOWEST catmiss tail. The rho-top-10 are the best-CALIBRATED
      epochs; the 1.1*min_mae gate keeps only those within 10% of the best accuracy among them;
      the catmiss tiebreak rewards peak-dominance. With W_DOM=0 it simply prefers fewer
      global misses among the already best-calibrated, accurate epochs.
    Returns ((epoch, mae, rho, tail), min_mae_top10, n_candidates).
    """
    top10 = sorted(rows, key=lambda r: r[2], reverse=True)[:10]   # highest rho (best-calibrated)
    min_mae = min(r[1] for r in top10)
    cand = [r for r in top10 if r[1] <= 1.1 * min_mae]            # within 10% of best accuracy
    chosen = min(cand, key=lambda r: r[3])                        # lowest catmiss tail (dominance)
    return chosen, min_mae, len(cand)


def _cos_taper_window(k_star: torch.Tensor, dq: torch.Tensor, L: int,
                      *, r_core_factor: float = WIN_R_CORE_FACTOR,
                      r_out_factor: float = WIN_R_OUT_FACTOR) -> torch.Tensor:
    """(B,) k*, (B,) dq -> (B,L) cos-tapered support window a in [0,1], centered at k*.

    a = 1 for |i-k*| <= R_core, a smooth cos-taper down to 0 at R_out, a = 0 beyond.
    R_core/R_out are physical (factor * WIN_SIGMA_MAX_Q in q) converted to bins per sample
    (/dq), so the window spans the bunched-beam upper bound at every L. C1-smooth (no
    hard-clip gradient jump at the edge). The mask depends only on k* and dq (both labels),
    NOT on the logits -> it is a constant w.r.t. the map; gradient flows only through the
    softmax probabilities it multiplies. R_core>=R_out is impossible (factors differ); both
    are floored so a degenerate dq cannot collapse the window.
    """
    xs = torch.arange(L, device=k_star.device, dtype=k_star.dtype).view(1, L)
    d = (xs - k_star.view(-1, 1)).abs()                          # (B,L) distance from truth (bins)
    r_core = (r_core_factor * WIN_SIGMA_MAX_Q / dq).clamp(min=1.0).view(-1, 1)   # (B,1) bins
    r_out = (r_out_factor * WIN_SIGMA_MAX_Q / dq).clamp(min=2.0).view(-1, 1)     # (B,1) bins
    frac = ((d - r_core) / (r_out - r_core).clamp(min=1e-9)).clamp(0.0, 1.0)     # 0@core -> 1@out
    return 0.5 * (1.0 + torch.cos(math.pi * frac))               # 1 at core, 0 at/after R_out


def total_loss_fbt(outputs: dict[str, torch.Tensor],
                    batch: dict[str, torch.Tensor],
                    w_conf: float | None = None,
                    ) -> tuple[torch.Tensor, dict[str, float]]:
    """fbt training loss at native L: windowed-NLL map term + w_conf·conf term.

    L_map = localized Laplace NLL (|k_hat-k*|/sigma + log sigma) computed inside a
    FIXED physics support window around k* (cos-taper, WIN_* constants) + a mass
    lower-bound barrier. The per-frame width sigma_loc is self-measured from the
    windowed map (no template, no target width, no SNR oracle); it also sets the
    conf radius (2·sigma_loc) and is the sigma paired with |err| for the rho-based
    ckpt selection. Computed at tau=0.5 -> matches the inference readout AND the
    val/rho metric.

    ``w_conf=None`` uses the default log2/logL schedule. NOTE: conf_logit is the
    tracker's observation gate (w_obs = sigmoid(conf_logit), its only gate), so the
    conf head has to be trained -- do not set w_conf=0 for a checkpoint that will be
    used with the tracker.
    """
    logits = outputs["map_native"]
    conf = outputs["conf_logit"]
    _, L = logits.shape
    k_star = batch["k_star"]
    dq = batch["dq"]

    # windowed-NLL (see WIN_* constants above): localized Laplace NLL in a FIXED support
    # window around k* + a mass lower-bound barrier. Width is self-measured (sigma_loc),
    # not a template. Computed at tau=0.5 -> matches the inference readout AND the val/rho
    # ckpt-selection metric; the tracker's log_softmax (tau=1) then sees a slightly blunter
    # = more conservative likelihood (helps fast-tune, avoids over-commitment).
    pr = F.softmax(logits / TAU_READOUT_BINS, dim=-1)        # tau=0.5
    ks = torch.arange(L, device=logits.device, dtype=logits.dtype).view(1, L)
    a = _cos_taper_window(k_star, dq, L)                    # (B,L) fixed support window
    p_loc = pr * a                                           # masked probabilities
    m_win = p_loc.sum(dim=-1)                                # (B,) mass inside the window
    r = p_loc / (m_win.view(-1, 1) + 1e-12)                 # (B,L) localized distribution
    k_hat = (r * ks).sum(dim=-1)                            # (B,) localized soft-argmax mean
    sigma = ((r * (ks - k_hat.view(-1, 1)) ** 2).sum(dim=-1)
             + 1e-6).sqrt().clamp(min=0.5)                   # (B,) localized spread, 0.5-bin floor
    nll_loc = (k_hat - k_star).abs() / sigma + torch.log(sigma)   # (B,) localized Laplace NLL
    # mass lower-bound barrier: 0 when m_win >= WIN_MASS_MIN, steep (log) as m_win -> 0.
    # blocks "epsilon mass at k*, dump the rest outside"; leaves >=1-MASS_MIN free for
    # legitimate multi-modal distractor peaks, so the map stays robust to interference.
    mass_pen = WIN_LAMBDA_MASS * F.relu(
        torch.log(WIN_MASS_MIN / (m_win + 1e-9))).pow(2)     # (B,)
    nll_ps = nll_loc + mass_pen
    lm = nll_ps.mean()
    conf_radius = 2.0 * sigma                               # conf "within 2sigma" uses sigma_loc
    lcf = loss_conf(conf, logits, k_star, conf_radius)

    if w_conf is None:
        w_conf = math.log(2.0) / math.log(max(2.0, float(L)))   # guard L=1 -> div-by-zero
    total = lm + w_conf * lcf
    # peak-dominance (out-of-window contrastive): rank logits[k*] above EVERY out-of-window bin so the
    # true sideband becomes the GLOBAL map max. It fights only out-of-window mass (never the peak's own
    # shoulders, so it does not over-sharpen). Gated by m_win (in-window mass) -> enforced only where a
    # real peak exists (no low-SNR overconfidence). Targets the single-frame/static readout against
    # structured distractors; the tracker already disambiguates via the predicted location.
    # W_DOM=0 turns the term off.
    l_dom = logits.new_zeros(())
    if W_DOM > 0.0:
        neg = torch.finfo(logits.dtype).min
        out = logits.masked_fill(a >= 1e-3, neg)                       # keep ONLY out-of-window bins
        peak = logits.gather(1, k_star.long().view(-1, 1))             # (B,1) logit at k*
        denom = torch.logsumexp(torch.cat([peak, out], dim=1), dim=1)  # (B,) logsumexp over {k*} U outside
        l_dom = ((denom - peak.squeeze(1)) * m_win).mean()             # -log p(k*|k*+outside), mass-gated
        total = total + W_DOM * l_dom
    return total, {"map": float(lm.detach()), "conf": float(lcf.detach()), "dom": float(l_dom.detach()),
                   "w_conf": float(w_conf), "total": float(total.detach())}
