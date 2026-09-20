# fftconv-bayes-tracker

Core implementation of the deep-learning betatron-tune estimator described in

> P. Sun, M. Zhang, R. Yuan, D. Li and J. Dong, *Robust betatron-tune measurement from Schottky spectra: complementary classical and deep-learning paradigms*, JINST **21** (2026) P08005, [doi:10.1088/1748-0221/21/08/P08005](https://doi.org/10.1088/1748-0221/21/08/P08005)

(section "Deep-learning estimator: likelihood map and discrete tune-velocity Bayes tracker"). The classical estimator of the same paper is in [temporal-matched-filter](https://github.com/VanillaChocoCake/temporal-matched-filter).

## Method

The FFT-convolution Bayes tracker (FBT) separates static spectral recognition from temporal inference.

1. **Input.** Three channels of length L: a robust per-frame z-score of the folded PSD, the normalised soft-binning weight, and the tune coordinate of every bin.
2. **Likelihood-map CNN.** A convolutional stem followed by four blocks. Each block sums a local branch (depthwise and pointwise convolutions) and a global branch, a convolution over the whole tune axis whose kernel is generated from five scalars per channel as a damped-oscillator profile and evaluated with the FFT. No component depends on a fixed length, so one set of weights serves any L. A map head returns per-bin logits, and a confidence head returns one logit per frame.
3. **Training objective.** A windowed Laplace negative log-likelihood around the labelled tune with a mass barrier, a dominance term against out-of-window peaks, and a binary cross-entropy for the confidence head.
4. **Bayes tracker.** A discrete filter on a (q, v) grid of tune and tune velocity. The observation update is tempered by the sigmoid of the confidence logit, the prediction applies a Gaussian velocity transition, a tune shift by v and a blur that accounts for the grid quantisation. The output is the posterior mean of q and its standard deviation. All tracker parameters are specified in tune units.

## Files

| File | Content |
|---|---|
| `archs/fftconv/model.py` | `FFTGlobalConv`, `FFTConvBlock` and the network builder |
| `archs/_common.py` | Map head, confidence head and the model wrapper |
| `src/fbt/bayes_tracker.py` | `QVBayesTracker`: observe, update, read-out and predict steps on the (q, v) grid |
| `src/fbt/tracker.py` | `VLTracker`: the tracker parameterised in tune units for any L |
| `src/fbt/estimator.py` | `FBTEstimator` (`process(frame) -> MeasurementResult`), including a fused CUDA Graph path |
| `src/fbt/range_losses.py` | Training objective and the soft-argmax read-out |
| `src/fbt/arch_registry.py` | Loads the architecture named in a checkpoint |
| `src/tune_pipeline/cnn_features.py` | `FeatureBuilder` for the three input channels |
| `src/tune_pipeline/frames.py`, `src/tune_pipeline/conventions.py` | Data contracts and the low-tune cutoff |

`src/` has to be on `sys.path`; `arch_registry.py` finds `archs/` relative to the repository root.

## Scope

This repository contains the network, the training objective, the tracker and the inference path. The spectral preprocessing that produces a `FrontendFrame` (paper section "Upstream preprocessing pipeline"), the training loop, the datasets and the trained weights are not included, so a checkpoint path has to be passed to `FBTEstimator`.

## Dependencies

PyTorch, NumPy.

## Citation

```bibtex
@article{sun2026robust,
  title     = {Robust betatron-tune measurement from Schottky spectra: complementary classical and deep-learning paradigms},
  author    = {Sun, Peihan and Zhang, Manzhou and Yuan, Renxian and Li, Deming and Dong, Jian},
  journal   = {Journal of Instrumentation},
  volume    = {21},
  number    = {08},
  pages     = {P08005},
  year      = {2026},
  publisher = {IOP Publishing}
}
```

## License

MIT, see [LICENSE](LICENSE).
