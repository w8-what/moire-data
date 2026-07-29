# Rough-screened phase heatmap visualizer

This folder is self-contained. It reuses the repository's current smoothing,
feature extraction, 3-pass × 5-iteration rough screening, and `get_fit_range`
logic without changing those source files.

The right heatmap applies adaptive multiscale smoothing to raw resistivity,
then fits a local power law inside each `get_fit_range` window:

`ρ(T) = ρ₀ + aTⁿ`

It classifies the fitted exponent relative to linear `n = 1`, requiring the
one-standard-error interval to clear a nonlinear threshold:

- `n - σₙ > 1 + A`: superlinear
- `n + σₙ < 1 + B`: sublinear
- otherwise: linear-compatible

The default threshold offsets are `A = 0.2` and `B = -0.2`. A is adjustable
from 0 to 1 and B from -1 to 0. Every local window must contain at least 9
points and span at least 1 K by default; both constraints are adjustable.
Accepted ranges that cannot satisfy both constraints are labeled as unlabeled.
The visualizer reports a one-standard-error uncertainty for n from the local
nonlinear least-squares covariance. This prevents noisy central estimates from
being labeled nonlinear when their uncertainty still overlaps the linear band.

Build the standalone HTML:

```sh
.venv/bin/python scripts/phase_visualizer/build_visualizer.py
```

Then open `phase_visualizer.html` in a browser. The generated file has no server
or external dependency.

Hover over either heatmap to select a filling linecut. The linked detail section
shows raw resistivity, adaptive multiscale smoothed resistivity with retained
features and the accepted behavior range and local power-law fit, dρ/dT, and
fitted n ± σₙ with the current A/B thresholds.
