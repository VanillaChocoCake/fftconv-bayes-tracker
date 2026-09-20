"""fbt estimator -- fftconv likelihood map + VLTracker.

Accepts any input length L, fixed per run.

Two pipeline modes (``mode``, default "tracker"):
  - "none"    : raw map@L -> tau=0.5 soft-argmax readout. No tracker
                (per-frame estimate, no temporal smoothing).
  - "tracker" : map@L -> VLTracker. The (q,v) Bayes tracker consumes the raw
                calibrated map as the per-frame likelihood. ``q_state == L``, so
                the tracker runs on the native map directly -- no resampling.

Per frame: ``FeatureBuilder.build`` (3ch x/w/q_grid) -> fftconv map CNN ->
[VLTracker]  or  [native tau=0.5 readout].

NOTE: the validity gate is the static ``q_grid > low_cut_q`` (baked into the
tracker at init and applied in the native readout); per-frame ``frame.valid_mask``
is NOT consumed -- a preprocessing stage with a non-trivial mask must keep
``valid_mask == (q_grid > low_cut_q)``.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from tune_pipeline.frames import FrontendFrame, MeasurementResult
from tune_pipeline.conventions import LOW_CUT_Q

from .arch_registry import build_from_ckpt
from tune_pipeline.cnn_features import FeatureBuilder, FeatureConfig
from .tracker import VLTracker, VLTrackerConfig

# <repo-root>/ckpt/* (estimator.py is src/fbt/estimator.py, so 3x dirname).
_CKPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ckpt")
DEFAULT_CKPT = os.path.join(_CKPT_DIR, "production.pth")
MODES = ("none", "tracker")
TAU_READOUT = 0.5        # native soft-argmax temperature (the readout the map is trained for)


class FBTEstimator:
    """fftconv map CNN + VLTracker, behind the common ``process(frame)`` API.
    See the module docstring for ``mode``.

    ``q_state`` is the tracker state-grid size and MUST equal the (fixed-per-run)
    input length ``L``: the (q,v) Bayes tracker then consumes the native map
    directly, with no resampling. Pass ``q_state=L`` for the operating length.
    """

    def __init__(self, ckpt_path: str | None = None, *, device: str = "cuda",
                 low_cut_q: float = LOW_CUT_Q, q_state: int = 1024,
                 mode: str = "tracker",
                 v_max_q: float | None = None, v_stride: int | None = None,
                 fuse: bool = False, compile_model: bool = False):
        assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
        self.mode = mode
        self.device = torch.device(device)
        self.low_cut_q = low_cut_q
        self.q_state = q_state
        self.ckpt_path = ckpt_path or DEFAULT_CKPT
        ck = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.model, self.arch = build_from_ckpt(ck, device=device)
        self.model.eval()
        self.n_params = self.model.num_parameters()
        self.fb = FeatureBuilder(FeatureConfig(low_cut_q=low_cut_q))
        self.tracker = None
        if mode == "tracker":
            if v_stride is None:
                # Default = the native velocity grid (stride 1), i.e. the full
                # fine-grained filter. v_stride>1 is an EXPLICIT opt-in latency lever
                # (V ∝ 1/stride; the q-blur auto-scales with the stride in
                # bayes_tracker). It trades velocity-grid resolution for step cost, so
                # stride 1 is the accuracy default and a coarser stride is an explicit
                # choice made against a per-frame latency budget.
                v_stride = 1
            tcfg_kw = dict(q_state_bins=q_state, low_cut_q=low_cut_q, v_stride=v_stride)
            if v_max_q is not None:            # (optional) also narrow the tracked velocity range
                tcfg_kw["v_max_q"] = v_max_q
            self.tracker = VLTracker(VLTrackerConfig(**tcfg_kw), device=device)
        # --- inference-latency paths (mutually exclusive) ---
        # fuse=True : ONE manual CUDA graph over the WHOLE chain (fb.build -> model ->
        #   step_core), captured at the first-seen L, tracker log_post in a static in-place
        #   buffer, step_core inductor-fused. Removes ALL kernel-launch overhead -> the floor
        #   is pure compute (model chain) + tracker bandwidth (∝ V; V = 83 at L=1024 on the
        #   native stride-1 grid, and v_stride>1 shrinks it for larger L).
        # compile_model : torch.compile(reduce-overhead) the model only; tracker eager.
        self.fuse = bool(fuse) and mode == "tracker"
        self._graph = None
        self._fused_L = None
        self._step_fn = None        # compiled lazily in _build_fused_graph (fused path only)
        if fuse and compile_model:
            import warnings
            warnings.warn("fuse=True ignores compile_model (the fused CUDA graph already "
                          "captures the whole chain); pass exactly one.", stacklevel=2)
        self.compiled = bool(compile_model) and not self.fuse
        if self.compiled:
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)
        self.reset()

    def reset(self, initial_q: float | None = None) -> None:
        """Reset the tracker to a fresh run (uniform prior when ``initial_q`` is None; no-op in 'none')."""
        if self.tracker is not None:
            self.tracker.reset(initial_q)
            if self.fuse and self._graph is not None:
                self._glp.copy_(self.tracker.core.log_post)   # sync the captured state buffer

    @torch.no_grad()
    def process(self, frame: FrontendFrame) -> MeasurementResult:
        """Consume one ``FrontendFrame`` (mapped_psd/weight_sum/q_grid), emit q/sigma."""
        x_np = np.asarray(frame.mapped_psd, dtype=np.float32)
        L = x_np.shape[0]
        if self.fuse:
            return self._process_fused(frame, x_np, L)
        x = torch.as_tensor(x_np, device=self.device).view(1, L)
        w = torch.as_tensor(np.asarray(frame.weight_sum, dtype=np.float32),
                            device=self.device).view(1, L)
        qg = torch.as_tensor(np.asarray(frame.q_grid, dtype=np.float32),
                             device=self.device).view(1, L)
        m = self._native_map(x, w, qg)                           # (L,) map logits
        if self.mode == "none":
            return self._readout_native(m, qg.squeeze(0))
        obs = self._observation(m, L)
        assert self._last_conf is not None, "the conf-gate requires the CNN conf head (conf_logit)"
        w_obs = torch.sigmoid(self._last_conf).reshape(())       # beam-loss gate = CNN per-frame reliability
        o = self.tracker.step(obs, w_obs=w_obs)
        return MeasurementResult(q=float(o.q), sigma=float(o.sigma_q), source="fbt")

    def _native_map(self, x: torch.Tensor, w: torch.Tensor, qg: torch.Tensor) -> torch.Tensor:
        """(1,L) x/w/q_grid -> (L,) map logits. Shared by the eager + fused paths."""
        out = self.model(self.fb.build(x, w, qg))
        self._last_conf = out.get("conf_logit")                  # (1,) conf head -> the conf-gate (w_obs = sigmoid)
        return out["map_native"].squeeze(0)

    def _observation(self, m: torch.Tensor, L: int) -> torch.Tensor:
        """(L,) native map logits -> tracker observation. With ``q_state == L`` the tracker
        runs on the native map directly (VLTracker.project_obs is identity; step_core's
        log_softmax normalises the raw logits). Shared by the eager + fused paths."""
        if L != self.q_state:
            raise ValueError(f"q_state must equal the input length L (got q_state={self.q_state}, L={L}); "
                             f"construct FBTEstimator(q_state=L) for the fixed deployment length.")
        return m

    @torch.no_grad()
    def _process_fused(self, frame: FrontendFrame, x_np: np.ndarray,
                       L: int) -> MeasurementResult:
        """Replay the captured full-chain CUDA graph (lazy capture at first L); q/sigma out.
        Copies straight into the static buffers and reads q+sigma in ONE D2H (sync)."""
        if self._graph is None or self._fused_L != L:
            self._build_fused_graph(L, np.asarray(frame.q_grid, dtype=np.float32))
        dev = self.device
        self._gx.view(L).copy_(torch.from_numpy(x_np).to(dev, non_blocking=True))
        self._gw.view(L).copy_(torch.from_numpy(
            np.asarray(frame.weight_sum, dtype=np.float32)).to(dev, non_blocking=True))
        # q_grid is the canonical arange(L)/(2L) for EVERY FrontendFrame at this L (the
        # folded-tune mapping onto [0, 0.5) is L-determined); _gq is already that (set in
        # _build_fused_graph), so the per-frame np.asarray + H2D is redundant -> skip it
        # (math-equivalent).
        self._graph.replay()
        q_bin, sigma_bin = self._gscalars.tolist()             # single D2H for both scalars
        return MeasurementResult(q=q_bin / self.q_state * 0.5,
                                 sigma=sigma_bin * (0.5 / self.q_state), source="fbt")

    @torch.no_grad()
    def _build_fused_graph(self, L: int, q_grid_np: np.ndarray) -> None:
        """Capture fb.build -> model -> step_core into ONE CUDA graph at input length ``L``.
        Static buffers; the tracker log_post lives in a static buffer updated in-place so the
        running state persists across replays. step_core is inductor-compiled.

        Tracker state is PRESERVED across (re)build: the authoritative pre-build state -- ``_glp``
        if a graph already ran (it, not tracker.core, holds the live state after replays), else the
        current tracker state (uniform, or an ``initial_q`` prior) -- is snapshotted, the warmup
        runs on it leaving garbage, then it is restored into the captured buffer. NOTE: the fused
        path targets a FIXED operating L; a per-frame L change forces a full rebuild + state
        transfer (slow) -- use the eager path for genuinely varying-L streams.
        """
        dev = self.device
        # authoritative state to carry into the new graph (survives rebuild AND initial_q)
        saved_post = (self._glp if self._graph is not None
                      else self.tracker.core.log_post).clone()
        if self._step_fn is None:                             # compile step_core once, on first build
            self._step_fn = torch.compile(self.tracker.core.step_core)
        self._gx = torch.zeros(1, L, device=dev)
        self._gw = torch.ones(1, L, device=dev)
        # q_grid from the ACTUAL frame (canonical arange(L)/(2L) by the folded-tune mapping, but we
        # use the frame's values, not an assumption) -> the per-frame q_grid H2D skip in
        # _process_fused is then exact for this L.
        self._gq = torch.from_numpy(q_grid_np).to(dev).view(1, L)
        self._glp = saved_post.clone()                        # static state buffer (authoritative)
        self._gscalars = torch.zeros(2, device=dev)
        self._fused_L = L

        # operator-fuse the model INSIDE the captured graph: mode="default" = inductor op-fusion
        # (NOT reduce-overhead, which nests its own CUDA graph). The memory-bound conv/GN/GELU/MLP
        # chains collapse to fewer Triton kernels -> less DRAM traffic. cuFFT ops stay eager but are
        # CUDA-graph-capturable. Compile once (re-capture at a new L reuses the compiled module).
        if not getattr(self, "_fused_nets_compiled", False):
            self.model = torch.compile(self.model, mode="default", dynamic=False)
            self._fused_nets_compiled = True

        def body():
            m = self._native_map(self._gx, self._gw, self._gq)
            obs = self.tracker.project_obs(self._observation(m, L))
            # beam-loss gate INSIDE the captured graph = the CNN conf head (self._last_conf, set by
            # _native_map above): w = sigmoid(conf_logit), a 0-dim tensor fed to step_core's w_obs.
            # CUDA-graph-safe (scalar op on the model's captured conf output).
            w = torch.sigmoid(self._last_conf).reshape(())
            new_lp, scalars, _ = self._step_fn(self._glp, obs, self.tracker.valid_mask, w_obs=w)
            self._glp.copy_(new_lp)                            # in-place state update
            self._gscalars.copy_(scalars)

        for _ in range(5):                                     # main-stream warmup: compile + alloc
            body()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                             # side-stream warmup, then capture
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            body()
        # warmup ran on dummy frames -> restore the authoritative pre-build state into the captured
        # buffer (replay reads it) AND tracker.core (so a future rebuild's snapshot is correct).
        self._glp.copy_(saved_post)
        self.tracker.core.log_post.copy_(saved_post)

    def _readout_native(self, m: torch.Tensor, qg: torch.Tensor) -> MeasurementResult:
        """'none' mode: tau=0.5 masked soft-argmax on the native map (no tracker)."""
        ml = m.masked_fill(qg <= self.low_cut_q, float("-inf"))
        p = torch.softmax(ml / TAU_READOUT, dim=-1)
        q = (p * qg).sum()
        sigma = torch.sqrt((p * (qg - q) ** 2).sum().clamp(min=0.0))
        return MeasurementResult(q=float(q), sigma=float(sigma), source="fbt")
