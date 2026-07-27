# Rough-screened phase heatmap visualizer

This folder is self-contained. It reuses the repository's current smoothing,
feature extraction, 3-pass × 5-iteration rough screening, and `get_fit_range`
logic without changing those source files.

The right heatmap classifies only samples inside a `get_fit_range` extraction
window:

- `x > A`: superlinear
- `x < B`: sublinear
- `B ≤ x ≤ A`: linear

where `x = d ln(dρ/dT) / dT`. The default thresholds are `A = 0.2` and
`B = -0.2`. A is adjustable from 0 to 1 and B from -1 to 0.

Build the standalone HTML:

```sh
.venv/bin/python scripts/phase_visualizer/build_visualizer.py
```

Then open `phase_visualizer.html` in a browser. The generated file has no server
or external dependency.

Hover over either heatmap to select a filling linecut. The linked detail section
shows raw resistivity, smoothed resistivity with retained features and the
accepted behavior range, dρ/dT, and x with the current A/B thresholds.
