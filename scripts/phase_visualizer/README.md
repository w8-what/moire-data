# Transport linecut explorer

Everything for this visualizer lives in this folder. The interface is now
linecut-first; the raw and classified heatmaps are secondary and remain closed
until requested.

## Run

Build the preprocessed data bundle:

```sh
.venv/bin/python scripts/phase_visualizer/build_visualizer.py
```

Start the local Python analysis server:

```sh
.venv/bin/python scripts/phase_visualizer/serve_visualizer.py
```

Open <http://127.0.0.1:8765>.

The server is required because local power-law fitting and noise calculations
remain in Python instead of being duplicated in browser JavaScript.
It uses up to eight worker processes for full-field phase calculations; override
that with `--workers N` when needed.

## Linecut plots

The primary explorer follows the current `general_pipeline.py` presentation:

1. Raw resistivity.
2. Adaptive-multiscale smoothed resistivity with rough-screened features,
   `get_fit_range`, and the selected local fit.
3. Local exponent `n` with its approximate `n_sigma`.

Move across any plot to inspect the local fit at that temperature. Fits use:

```text
rho(T) = rho0 + A*T^n
```

Every local window stays inside `get_fit_range` and must satisfy the selected
minimum point count and temperature span.

## Fit choices

- Source: adaptive-smoothed or raw resistivity.
- Loss: ordinary squared (`linear`), `soft_l1`, or `cauchy`.
- No normalization: residuals stay in resistivity units. Robust losses use
  `f_scale=1`.
- Existing `local_noise`: each residual is divided by the project’s
  temperature-dependent raw-minus-smoothed noise estimate.
- Fit-residual MAD: each window first receives an unweighted linear-loss power
  law; the final fit uses one robust scale calculated as
  `1.4826 * MAD(raw rho - preliminary fit)`.
- Pooled estimate: each temperature’s noise is estimated from detrended
  neighboring-temperature residuals pooled across all fillings.

The pooled option is useful when one linecut’s physical structure would
otherwise contaminate its own noise estimate.

## Phase overview

The secondary phase section calculates only when opened. It uses the current
fit configuration and the uncertainty-aware rules:

```text
n - n_sigma > 1 + A  -> superlinear
n + n_sigma < 1 + B  -> sublinear
otherwise             -> linear-compatible
```

Full-field calculations are cached by the local server after the first request.
Changing A or B reclassifies the cached `n` and `n_sigma` values immediately in
the browser and does not run the fits again.
