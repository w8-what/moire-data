# Moire data analysis

Python tools for analyzing temperature-dependent longitudinal resistance in moire
materials. The project cleans resistance matrices, smooths individual filling-factor
linecuts, estimates local noise, extracts transport features, fits local power laws, and
creates linecut and phase-diagram visualizations.

## Features

- Load resistance matrices indexed by temperature and filling factor.
- Apply Hampel filtering and adaptive multiscale smoothing.
- Detect upturns, downturns, superconducting transitions (`Tc`), and coherence
  crossovers (`Tcoh`).
- Fit the local power law `rho(T) = rho0 + A*T^n`.
- Refine feature confidence using neighboring linecuts.
- Produce static plots and explore fits in a local interactive visualizer.

## Setup

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

The `dev` and `notebook` extras are optional. For only the analysis library and its
runtime dependencies, use `python -m pip install -e .`.

## Data format

Input files belong in `source_data/` and use this naming convention:

```text
Rxx_matrix_E-<field>mV_nm.csv
```

The first column contains temperatures, the remaining column headers are filling
factors, and the remaining cells form the resistance matrix. For example:

```python
from pathlib import Path

from moire.io import clean_sort_data, load_field

temperature, filling, resistance = load_field(87, Path("source_data"))
temperature, filling, resistance = clean_sort_data(
    temperature, filling, resistance
)
```

## Run the analysis

The main pipeline reads the configured fields from `source_data/` and writes generated
figures beneath `output/`:

```bash
python scripts/general_pipeline.py
```

To generate the current comparison figures instead:

```bash
python scripts/generate_figures.py
```

The selected fields and processing parameters are currently configured near the top of
each script.

## Interactive visualizer

Build the visualizer data bundle, then start its local fitting server:

```bash
python scripts/phase_visualizer/build_visualizer.py
python scripts/phase_visualizer/serve_visualizer.py
```

Open <http://127.0.0.1:8765>. See
[`scripts/phase_visualizer/README.md`](scripts/phase_visualizer/README.md) for the fit
controls and classification rules.

## Tests and formatting

```bash
pytest
black --check src tests scripts
```

Run `black src tests scripts` to apply formatting.

## Repository layout

```text
src/moire/               Analysis and plotting library
tests/                   Automated tests
scripts/                 Pipelines, figure generation, and visualizer
notebooks/               Exploratory notebooks
source_data/             Input resistance matrices
output/                  Generated analysis artifacts (ignored by Git)
```
