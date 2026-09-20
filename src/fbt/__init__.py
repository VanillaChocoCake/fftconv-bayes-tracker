"""fbt — deep-learning betatron-tune estimator.

An ``fftconv`` likelihood-map CNN followed by ``VLTracker``, a discrete
(q, v) Bayes tracker parameterised in physical tune units. The map CNN has no
fixed-length component, so it accepts any input length L (fixed per run;
trained over L in [512, 8192]).

Inference:  ``from fbt.estimator import FBTEstimator``.

Layout: ``estimator.py`` (per-frame pipeline), ``tracker.py`` (VLTracker, the
physical-units wrapper), ``bayes_tracker.py`` (the grid-agnostic (q, v) filter
core), ``arch_registry.py`` (architecture loader), ``range_losses.py`` (training
objective), ``archs/fftconv/model.py`` (the network).

This package init stays light (no eager torch import); submodules are imported
directly by their consumers.
"""
