"""Build the self-contained rough-screened phase heatmap visualizer.

Run from anywhere in the repository with:

    .venv/bin/python scripts/phase_visualizer/build_visualizer.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from hampel import hampel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth
from moire.extract_features import (
    extract_Tc,
    extract_downturns,
    extract_upturns,
    get_fit_range,
)
from moire.io import clean_sort_data, load_field
from moire.signal_helpers import local_noise, moving_average
from moire.update_scoring import update_score

DEFAULT_FIELDS = [74, 87, 96, 96.2, 99, 103, 151, 176]


def _round(value, digits=6):
    """Return a compact JSON-safe float, or None for a non-finite value."""
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _phase_metrics(temperatures, smoothed):
    """Calculate dρ/dT and x = d ln(dρ/dT) / dT.

    get_fit_range already requires predominantly positive dρ/dT. Remaining
    non-positive samples are short gaps in most accepted windows, so their log
    derivative is filled by linear interpolation before the 1 K moving average.
    """
    dpdt = np.asarray(hampel(np.gradient(smoothed, temperatures)).filtered_data)
    positive = np.isfinite(dpdt) & (dpdt > 0)
    if np.count_nonzero(positive) < 2:
        return dpdt, np.full_like(temperatures, np.nan, dtype=float)

    log_dpdt = np.interp(
        temperatures,
        temperatures[positive],
        np.log(dpdt[positive]),
    )
    exponent = np.gradient(log_dpdt, temperatures)
    return dpdt, moving_average(exponent, temperatures, 1.0)


def _extract_field(field):
    """Run the repository's rough screen and serialize one electric field."""
    load_value = int(field) if float(field).is_integer() else field
    temperatures, fillings, resistivity = clean_sort_data(
        *load_field(load_value, ROOT / "source_data")
    )

    linecuts = []
    for index, filling in enumerate(fillings):
        rho = resistivity[:, index]
        smoothed = adaptive_multiscale_smooth(temperatures, rho, z_threshold=3)
        linecut = {
            "E": load_value,
            "nu": filling,
            "T": temperatures,
            "rho": rho,
            "rho_smoothed": smoothed,
            "local_noise": local_noise(temperatures, rho, smoothed),
            "behaviors": [],
        }
        features = extract_upturns(temperatures, linecut)
        features += extract_downturns(temperatures, linecut)
        features += extract_Tc(temperatures, linecut)
        linecut["features"] = features
        linecuts.append(linecut)

    # Match the project's rough screening defaults: 3 passes × 5 iterations,
    # followed by the positive-slope extraction-range screen.
    update_score(linecuts)
    for linecut in linecuts:
        get_fit_range(temperatures, linecut)

    score_name = "score_15"
    features = []
    phase_columns = []
    linecut_series = []
    accepted_count = 0

    for linecut_index, linecut in enumerate(linecuts):
        for feature in linecut["features_new"]:
            features.append(
                {
                    "linecut": linecut_index,
                    "nu": _round(linecut["nu"]),
                    "T": _round(feature["T"]),
                    "type": feature["type"],
                    "score": _round(feature[score_name]),
                }
            )

        dpdt, exponent = _phase_metrics(temperatures, linecut["rho_smoothed"])
        extraction = next(
            (
                behavior
                for behavior in linecut["behaviors"]
                if behavior.get("type") == "extraction_range"
            ),
            None,
        )

        linecut_series.append(
            {
                "rho": [_round(value, 4) for value in linecut["rho"]],
                "smoothed": [_round(value, 4) for value in linecut["rho_smoothed"]],
                "dpdt": [_round(value, 4) for value in dpdt],
                "x": [_round(value) for value in exponent],
            }
        )

        if extraction is None:
            phase_columns.append(
                {
                    "nu": _round(linecut["nu"]),
                    "lower": None,
                    "upper": None,
                    "x": [None] * len(temperatures),
                }
            )
            continue

        accepted_count += 1
        lower = float(extraction["T_lower"])
        upper = float(extraction["T_upper"])
        inside = (temperatures >= lower) & (temperatures <= upper)
        values = [
            _round(value) if keep else None
            for value, keep in zip(exponent, inside, strict=True)
        ]
        phase_columns.append(
            {
                "nu": _round(linecut["nu"]),
                "lower": _round(lower),
                "upper": _round(upper),
                "x": values,
            }
        )

    positive_rho = resistivity[np.isfinite(resistivity) & (resistivity > 0)]
    low, high = np.nanpercentile(positive_rho, [1, 99])
    stride = max(1, math.ceil(len(fillings) / 300))

    return {
        "field": _round(load_value),
        "temperatures": [_round(value) for value in temperatures],
        "fillings": [_round(value) for value in fillings],
        "heatFillings": [_round(value) for value in fillings[::stride]],
        "resistivity": [
            [_round(value, 4) for value in row]
            for row in resistivity[:, ::stride]
        ],
        "logMin": math.log10(float(low)),
        "logMax": math.log10(float(high)),
        "features": features,
        "phaseColumns": phase_columns,
        "linecuts": linecut_series,
        "acceptedCount": accepted_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", nargs="+", type=float, default=DEFAULT_FIELDS)
    parser.add_argument("--upper", type=float, default=0.2)
    parser.add_argument("--lower", type=float, default=-0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "phase_visualizer.html",
    )
    args = parser.parse_args()

    if not 0 <= args.upper <= 1:
        parser.error("--upper must be between 0 and 1")
    if not -1 <= args.lower <= 0:
        parser.error("--lower must be between -1 and 0")
    if args.lower > args.upper:
        parser.error("--lower must not exceed --upper")

    fields = [_extract_field(field) for field in args.fields]
    payload = {
        "upper": args.upper,
        "lower": args.lower,
        "screen": {
            "passes": 3,
            "iterationsPerPass": 5,
            "filter": 0.1,
            "positiveSlopeFraction": 0.8,
            "phaseSmoothingWindowK": 1.0,
        },
        "fields": fields,
    }

    template_path = Path(__file__).resolve().parent / "visualizer_template.html"
    template = template_path.read_text()
    output = template.replace("__PHASE_VISUALIZER_DATA__", json.dumps(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KiB)")
    for field in fields:
        total = len(field["phaseColumns"])
        print(
            f"  E={field['field']}: {field['acceptedCount']}/{total} "
            "linecuts accepted by get_fit_range"
        )


if __name__ == "__main__":
    main()
