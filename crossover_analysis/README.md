# Standalone Tcoh and T-prime candidate analysis

Everything added for this experiment lives in this folder. Existing project
modules are not modified. The graph generator reads the repository CSV files
and imports the existing adaptive smoother in read-only fashion.

The extractor uses one linecut at a time. It does not use neighboring fillings,
heatmap geometry, field identity, or published phase-boundary positions in any
plausibility score. Published Figure 3 points are optionally drawn as black
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

Its code, candidate JSON, validation metrics, and staged figures live under
`crossover_analysis/paper_match_stages/`. Stage 1 shows the strict top-five
baseline, Stage 2 applies the final candidate-retention rule, and Stage 3 uses
the same candidates with a clearer visual hierarchy. No stage draws or finds a
path.

## Candidate output

Each linecut returns zero to four candidates with:

- `plausibility`: combined 0--1 local score;
- `B`: baseline-fit quality, including PRESS leave-one-out prediction at the
  converged robust weights;
- `D`: persistent 10% departure strength;
- `R`: agreement among fit windows within that same linecut;
- `C`: power-law regime contrast;
- a linecut-local temperature uncertainty and support interval;
- the representative fit window and fit parameters.

Scores do not sum to one. A candidate is retained only if its score is at least
0.65 and no more than 0.10 below that linecut's best score. At most four are
kept. If the best score is below 0.65, the linecut is `not_identifiable` and
returns no candidate. A leave-one-field-out threshold study on the four paper
fields motivated one universal rule; the published markers are never score
features and no field-specific threshold is used.

For the contrast term, one free power law is fit over the entire available
departure side. This is the Occam choice: the extractor does not search for a
smaller exponent window that happens to support a preferred interpretation.
