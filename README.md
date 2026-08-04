# moire-data

Analysis tools and scripts for extracting features from moire resistivity measurements.

## Setup

Create a virtual environment and install the package with its development tools:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Run the analysis

The general pipeline reads CSV files from `source_data/` and writes generated figures to
`output/`:

```bash
.venv/bin/python scripts/general_pipeline.py
```

Use `--fields` to process only selected electric fields and `--linecuts` to control how many
linecuts are plotted:

```bash
.venv/bin/python scripts/general_pipeline.py --fields 74 103 --linecuts 20
```

## Build the phase diagrams

Generate the complete eight-field phase atlas, individual panels, a PDF, the selected transition
table, and the rejected-candidate audit table with:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_phase_diagrams.py
```

The paper-facing outputs are split by evidentiary family:

- `phase_core_published.png/.pdf`: the comparable −103, −99, −96, and −87 mV nm⁻¹ source-backed
  series. Competing −96 transport minima left of the published endpoint are shown distinctly as
  unresolved candidates, not promoted to an AFM boundary.
- `phase_extended_datasets.png/.pdf`: the −96.2, −74, −151, and −176 maps shown as independent
  datasets because device/cooldown/geometry metadata do not establish one continuous field sweep.
- `phase_core_overlay.png` and `phase_extended_overlay.png`: resistance-map diagnostics with a
  panel-specific logarithmic scale and the complete measured temperature range.
- `phase_minus96_tneel_linecuts.png/.pdf` and `phase_tcoh_linecuts.png/.pdf`: representative raw
  linecut evidence.
- `transitions.csv`, `candidates.csv`, and `qa_summary.json`: selected coordinates, the complete
  candidate audit trail, and computed acceptance gates.

The extractor uses these operational definitions:

- `Tcoh`: persistent 10% departure from a validated low-temperature `rho0 + A*T^2` regime with
  `A > 0`.
- `Tprime`: persistent 10% departure from a validated high-temperature linear-in-`T` regime; log
  ratios are used only to guard the fractional residual calculation.
- `Tneel`: a significant local resistance minimum. Automatically extracted points are reported as
  the transport proxy `T_N^rho`, since magnetic order needs an independent probe.

The four 4 K paper panels use the exact author-provided Fig. 3 source-data coordinates. Their local
matrices do not contain the full high-temperature baseline used for every published `Tprime` point.
The −96 linecuts contain competing minima left of the source endpoint; both modes remain in the
reproducible candidate audit rather than being forced into one phase path. All wider-temperature
datasets use the same automatic extraction, invalid-basin exclusion, and physical-coordinate
branch-selection code, but they are labeled exploratory because the 4 K core matrices do not provide
a successful end-to-end positive control for every published boundary.
`qa_summary.json` reports software QA separately from physical calibration. Software QA passes only
when topology, provenance, files, field-blind scale/grid checks, and adversarial controls pass. The
current automatic extractor is explicitly marked `physical_calibration: not_established`: it does not
recover a strict `T^2` candidate in the core matrices and is not loosened to imitate the published
coordinates. This keeps the same criterion applicable to every input dataset without publication-data
tuning.

## Run the tests

```bash
MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python -m pytest -q tests
```

Generated figures, notebook outputs, bytecode, and local environments are intentionally excluded
from version control.
