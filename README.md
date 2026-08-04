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

## Run the tests

```bash
.venv/bin/python -m pytest
```

Generated figures, notebook outputs, bytecode, and local environments are intentionally excluded
from version control.
