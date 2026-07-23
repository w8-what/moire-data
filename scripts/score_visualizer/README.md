# Score update visualizer

This tool displays one scoring method: **original-confidence anchoring**.
Support is recalculated from the preceding iteration, but every new score is
blended with the feature's original confidence.

## Scoring rule

For iteration `i`:

```text
support_i = neighborhood support calculated from score_(i-1)
score_i   = (1 - lambda) * original_confidence + lambda * support_i
```

`lambda` is the support weight:

- `lambda = 0` preserves original confidence.
- `lambda = 1` replaces original confidence with support.
- `lambda = 0.5` gives them equal weight.

Temperature separation attenuates a neighbor's score using:

```text
exp(-0.5 * temperature_index_separation^2 / tau)
```

Larger `tau` allows support to travel farther in temperature. The neighborhood
control independently chooses how many linecuts are inspected on each side.

## Interactive controls

- **Iterations:** choose which update round is displayed.
- **Temperature spread tau:** adjust `tau` from `0.1` to `20`.
- **Linecuts per side:** inspect between `1` and `20` neighboring linecuts.
- **Support weight lambda:** choose how strongly support updates confidence.
- **Display scores:** hide markers below a score threshold. This only changes
  the visualization; it does not change score calculations.
- **Sigmoid support:** optionally multiply raw support by
  `sigmoid((raw_support - center) / width)`.

## Build

Run from the repository root:

```bash
.venv/bin/python scripts/score_visualizer/build_visualizer.py
```

Then open `scripts/score_visualizer/score_visualizer.html`. It is a
self-contained file and does not need a server.

Build-time defaults can be changed when needed:

```bash
.venv/bin/python scripts/score_visualizer/build_visualizer.py \
  --fields 87 96.2 176 \
  --iterations 15 \
  --n-hood 3 \
  --max-n-hood 20 \
  --support-weight 0.5 \
  --tau 3
```
