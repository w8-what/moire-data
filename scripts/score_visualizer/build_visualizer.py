"""Build a self-contained interactive score-update heatmap visualizer.

Run from anywhere in the repository with:

    .venv/bin/python scripts/score_visualizer/build_visualizer.py
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth
from moire.extract_features import extract_downturns, extract_upturns
from moire.io import clean_sort_data, load_field
from moire.signal_helpers import local_noise


DEFAULT_FIELDS = [74, 87, 96, 96.2, 99, 103, 151, 176]


def _round(value, digits=6):
    return round(float(value), digits)


def extract_field(field):
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
        }
        linecut["features"] = extract_upturns(temperatures, linecut)
        linecut["features"] += extract_downturns(temperatures, linecut)
        linecuts.append(linecut)

    features = []
    for linecut_index, original_line in enumerate(linecuts):
        for original in original_line["features"]:
            features.append(
                {
                    "linecut": linecut_index,
                    "nu": _round(original_line["nu"]),
                    "T": _round(original["T"]),
                    "tIndex": int(np.argmin(np.abs(temperatures - original["T"]))),
                    "type": original["type"],
                    "confidence": _round(original["confidence"]),
                }
            )

    positive = resistivity[np.isfinite(resistivity) & (resistivity > 0)]
    low, high = np.nanpercentile(positive, [1, 99])
    stride = max(1, math.ceil(len(fillings) / 280))
    sampled_fillings = fillings[::stride]
    sampled_resistivity = resistivity[:, ::stride]

    return {
        "field": _round(load_value),
        "linecutCount": len(linecuts),
        "temperatures": [_round(value) for value in temperatures],
        "fillings": [_round(value) for value in sampled_fillings],
        "resistivity": [
            [_round(value, 4) for value in row] for row in sampled_resistivity
        ],
        "logMin": math.log10(float(low)),
        "logMax": math.log10(float(high)),
        "features": features,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", nargs="+", type=float, default=DEFAULT_FIELDS)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--n-hood", type=int, default=3)
    parser.add_argument("--max-n-hood", type=int, default=12)
    parser.add_argument("--retain", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--sigmoid-center", type=float, default=0.6)
    parser.add_argument("--sigmoid-width", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "score_visualizer.html",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if not 1 <= args.n_hood <= args.max_n_hood:
        parser.error("--n-hood must be between 1 and --max-n-hood")
    if args.tau <= 0:
        parser.error("--tau must be positive")
    if args.sigmoid_width <= 0:
        parser.error("--sigmoid-width must be positive")

    fields = [extract_field(field) for field in args.fields]
    payload = {
        "iterations": args.iterations,
        "retain": args.retain,
        "supportWeight": 1.0 - args.retain,
        "nHood": args.n_hood,
        "maxNHood": args.max_n_hood,
        "tau": args.tau,
        "sigmoidCenter": args.sigmoid_center,
        "sigmoidWidth": args.sigmoid_width,
        "fields": fields,
    }
    template = (Path(__file__).resolve().parent / "visualizer_template.html").read_text()
    output = template.replace("__SCORE_VISUALIZER_DATA__", json.dumps(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
