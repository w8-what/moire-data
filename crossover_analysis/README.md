# Standalone Tcoh and T-prime candidate analysis

Everything added for this experiment lives in this folder. Existing project
modules are not modified. The graph generator reads the repository CSV files
and imports the existing adaptive smoother in read-only fashion.

The extractor uses one linecut at a time. It does not use neighboring fillings,
heatmap geometry, field identity, or published phase-boundary positions in any
plausibility score. Published Figure 3 points are optionally drawn as gray
reference markers only after scoring.

## Run

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python crossover_analysis/generate_graphs.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest crossover_analysis/tests
```

Outputs are written to `crossover_analysis/output/`.

To render the saved candidates using the same visual format as the repository's
existing heatmaps and linecuts:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python crossover_analysis/generate_matching_style.py
```

These figures are written to `crossover_analysis/matching_style/`.

The original project Tcoh extractor remains untouched. The standalone
strict-cluster improvement and four-field paper comparison can be run with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python crossover_analysis/improved_tcoh.py
```

Its code, candidate JSON, validation metrics, and four figures live under
`crossover_analysis/tcoh_improved/`.

## Candidate output

Each linecut returns up to five candidates with:

- `plausibility`: combined 0--1 local score;
- `B`: baseline-fit quality, including PRESS leave-one-out prediction at the
  converged robust weights;
- `D`: persistent 10% departure strength;
- `R`: agreement among fit windows within that same linecut;
- `C`: power-law regime contrast;
- the representative fit window and fit parameters.

Scores do not sum to one. A top score of at least 0.65 is called
`locally_supported`; otherwise the linecut remains `not_identifiable` while its
lower-scoring candidates are retained.

For the contrast term, one free power law is fit over the entire available
departure side. This is the Occam choice: the extractor does not search for a
smaller exponent window that happens to support a preferred interpretation.
