# Phase-diagram publication handoff

Last updated: 2026-08-04 11:06 America/New_York

## Goal and non-negotiable definitions

Build paper-ready WSe2 moiré transport phase diagrams for every repository dataset while avoiding
publication-coordinate overfitting.

- `Tcoh` is retained only after a genuine positive-`A` low-temperature `rho0 + A*T^2` regime.
- `Tprime` is a persistent 10% departure from a validated high-temperature linear-`T` regime; guarded
  log ratios may be used for fractional residuals.
- Every significant resistance upturn/minimum may be considered as a `Tneel` transport candidate, but
  it must be labeled `T_N^rho`, not independent magnetic confirmation.
- Published coordinates are an immutable source-backed overlay and positive-control diagnostic only.
  They never tune or seed automatic extraction.
- The -96 mV/nm left side contains two competing minima. Both are displayed as unresolved candidates;
  neither is promoted into an AFM line.

## Current stopping-point verdict

The source-backed core figure is publication-ready. The extended figures are publication-ready as
**exploratory transport-candidate maps**, not calibrated phase diagrams. This distinction is explicit
in titles, panel badges, legends, captions, README, and `qa_summary.json`.

`output/phase_diagrams/qa_summary.json` currently reports:

- `status: software_qa_passed`
- `physical_calibration.status: not_established`
- all software acceptance gates true
- 1,536 adversarial controls, 0 false positives
- field-blind scale/grid generalization passed
- all displayed automatic topology/model invariants passed

Physical calibration is deliberately not declared because the strict automatic extractor recovers
zero raw `Tcoh` and zero raw `Tprime` points on all four core matrices. This is not hidden or loosened
away: direct fits around many published core `Tcoh` coordinates have exponents around 1.4-1.8 rather
than 2. The source-backed curves therefore remain separate from automatic candidates.

## Primary artifacts

All are under `output/phase_diagrams/`.

- `phase_core_published.png/.pdf`: -103, -99, -96, -87 source-backed series.
- `phase_extended_datasets.png/.pdf`: -96.2, -74, -151, -176 independent exploratory maps.
- `phase_core_overlay.png`, `phase_extended_overlay.png`: panel-log resistance overlays.
- `phase_atlas.png/.pdf`: 7.15-inch-wide, four-row evidence atlas; bottom four panels explicitly
  independent, not a continuation of the top field sweep.
- `phase_overlay_atlas.png`: combined resistance-overlay diagnostic.
- `phase_minus96_tneel_linecuts.png/.pdf`: unresolved -96 competing-minimum evidence.
- `phase_tcoh_linecuts.png/.pdf`: exact stored accepted T-squared fits, fit windows, distinct crossing
  brackets, 10% envelopes, fitted exponents, and residuals.
- `phase_E-*.png`: individual field overlays.
- `transitions.csv`: displayed/published selected coordinates and full provenance.
- `candidates.csv`: complete raw automatic-candidate audit, including rejections.
- `qa_summary.json`: numerical gates, calibration disclosure, source positive controls, topology,
  adversarial results, dataset ranges, and caveats.

Latest PDF page sizes from Poppler:

- core: 500.255 x 497.505 pt
- extended: 514.56 x 497.505 pt
- combined atlas: 514.56 x 882.813 pt
- all one page, unencrypted, PDF 1.4

The latest PDFs were rendered with Poppler at 160 dpi into `tmp/pdfs/final_phase_qa/` and visually
inspected. `pdffonts` is not installed, but Matplotlib exports with `pdf.fonttype=42`; the visual
reviewer previously confirmed CIDFontType2/Type0 rather than Type 3.

Final layout fix: redundant top-row x-axis labels are suppressed in all multirow atlases, eliminating
the last extended-figure inter-row title collision. The latest extended PDF and overlay were visually
rechecked after this change.

## Code and architecture

Main implementation:

- `src/moire/phase_diagram.py`
- `src/moire/phase_outputs.py`
- `scripts/build_phase_diagrams.py`
- `tests/test_phase_diagram.py`
- `reference_data/wse2_fig3_reference.json`
- `README.md`

Automatic extraction is entirely in `phase_diagram.py` and contains no publication coordinates,
reference imports, field IDs, or per-field thresholds. Every field receives one frozen
`PhaseExtractionConfig`.

`Tcoh` validation includes:

- contiguous positive raw low-temperature data beginning at the lowest defensible temperature
- positive fixed quadratic coefficient and >=4-sigma signal
- median/p90 fit-error gates
- free exponent 1.82-2.18, `abs(n-2)<=0.18`, finite `sigma_n<=0.25`
- fixed-vs-free delta-BIC gate
- parameter-stability and holdout validation
- persistent 10% crossing at >=2-sigma over a physical temperature span
- global single-power rejection for well-fit nonquadratic curves
- branch components with >=4 points and <=350 K/filling local slope

Exact accepted fit offset/coefficient are stored in every automatic `Tcoh` row and used in the
diagnostic figure; the diagnostic no longer refits or extrapolates across the full temperature range.

`Tprime` uses a high-temperature linear baseline, exponent/model guards, persistence, and global
curved-power rejection. No real repository linecut currently survives this strict automatic test.

`T_N^rho` candidates require significant local minima, reject any basin containing invalid/nonpositive
raw resistance, use feature-local noise significance, and are selected by a physical-coordinate
dynamic-programming path with visible gaps preserved.

Plotting rules:

- published points: filled markers and connected source curves
- automatic candidates: open evidence points
- automatic robust centerline: rolling median only for components with >=7 points
- components with 4-6 points remain unjoined
- vertical whiskers: extraction uncertainty
- `Tcoh` crossing bracket is orange and distinct from the cyan validated T-squared fit window
- noncore panel badge: `exploratory transport candidates`
- no noncore panel is described as a continuous displacement-field sweep

## Generalization and anti-overfit evidence

- Automatic extractor has no access to source coordinates or field-specific IDs.
- E176 copied/relabelled as E999 produced identical raw/selected automatic results in an independent
  reviewer check.
- Unit tests require identical crossover results under resistance scales 0.2, 1, and 7, arbitrary
  field labels, and nonuniform/uniform/dense-low-T grids.
- Full QA runs on all eight measured temperature grids with noise sigma 0.15, 0.5, and 1.0:
  - false `Tcoh` on pure powers n=1.6,1.8,2.2,2.4
  - false `Tprime` on curved powers n=1.2,1.4
  - false `T_N^rho` on noisy monotonic powers n=1.2,2.0
  - 8 trials per cell; 1,536 total; 0 false positives
- Invalid-resistance regression requires zero selected minima when a synthetic basin contains
  nonpositive points.
- QA checks every displayed automatic crossing bracket, exponent/sigma, stored model, component size,
  and branch slope.

The -96 diagnostic is intentionally publication-guided and field-specific in the **output layer**
(`0.985 <= nu < 0.988`). Those points are marked `automatic_multimodal_candidate`, never selected,
and must not be described as generic extraction evidence.

## Exact selected automatic counts

- -96.2: 19 `Tcoh`, 77 `T_N^rho`
- -74: 93 `Tcoh`, 5 `T_N^rho`
- -151: 31 `Tcoh`, 14 `T_N^rho`
- -176: 4 `Tcoh`, 0 `T_N^rho`

Visible component sizes are saved under `selected_automatic_topology.component_sizes` in QA.

## Required physical caveats

- Core automatic positive-control recall:
  - `Tcoh`: 0 automatic points in -103, -99, -96, -87 despite 45 published coordinates
  - `Tprime`: 0 automatic points despite 21 published coordinates
  - `T_N^rho` published-coordinate match coverage: 26.3% at -103, 38.1% at -99, 90.9% at -96;
    -87 correctly has none
- -96 left AFM side remains unresolved. Competing modes near 1.9 K and 2.8 K are shown, not joined.
- -96.2 has four disconnected upturn ridges, not one AFM dome.
- -74 has five `T_N^rho` points locked at 1.1 K near filling 1.353-1.363; likely grid/background
  structure and not a resolved magnetic boundary.
- -151 has two incomplete upturn components that do not form a closed dome.
- -176 has only four unjoined `Tcoh` points and no complementary branch; insufficient for phase
  topology.
- Extended `Tcoh` points are locally plausible strict T-squared crossovers (selected exponent range
  about 1.823-2.138; representative median residuals about 1-2.1%) but not globally calibrated phases.

## Source provenance

Published comparison source:

- Xia et al., Nature 650, 585-591 (2026)
- https://www.nature.com/articles/s41586-025-10049-3
- Official Fig. 3 source workbook URL is recorded in
  `reference_data/wse2_fig3_reference.json`.

Exact frozen source counts:

- -103: `Tcoh` 12, `Tneel` 19, `Tprime` 7
- -99: `Tcoh` 12, `Tneel` 21, `Tprime` 7
- -96: `Tcoh` 11, `Tneel` 11, `Tprime` 7
- -87: `Tcoh` 10, `Tneel` 0, `Tprime` 0

## Verification commands

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/moire-mpl-cache PYTHONPATH=src \
  .venv/bin/python -m pytest -q tests

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/moire-mpl-cache PYTHONPATH=src \
  .venv/bin/python scripts/build_phase_diagrams.py

pdfinfo output/phase_diagrams/phase_core_published.pdf
pdfinfo output/phase_diagrams/phase_extended_datasets.pdf
pdfinfo output/phase_diagrams/phase_atlas.pdf

pdftoppm -f 1 -singlefile -png -r 160 \
  output/phase_diagrams/phase_core_published.pdf tmp/pdfs/final_phase_qa/phase_core_published
```

Use scoped pytest exactly as above. Bare `pytest -q` collects legacy GUI-generating script tests and
can crash the macOS backend.

## Workspace/git caution

The worktree was already heavily dirty with many staged/unrelated deletions before this task. Do not
reset, restore, stage, or commit broadly. Preserve user changes. The phase implementation files are
mostly untracked, so ordinary `git diff --stat` does not list them. No commit, branch, staging, or push
was performed.

## Good next steps

1. Read this handoff, `README.md`, and `output/phase_diagrams/qa_summary.json` first.
2. Run the scoped 15-test suite after any source change.
3. Rebuild the complete outputs if extraction/QA changes; for caption-only plotting iteration, existing
   `transitions.csv` may be rerendered through `_make_atlas`, but finish with the full build.
4. Do not relax `Tcoh`/`Tprime` until core coordinates match. Any change must be justified on synthetic
   and all-field evidence and rerun the 1,536-control matrix.
5. If a future paper needs calibrated automatic phases, acquire/identify the exact processed resistance
   data or methodology used to create the published boundaries; current core matrices do not satisfy
   the strict requested model.
6. Keep the extended deliverable labeled exploratory unless independent physical calibration is added.
