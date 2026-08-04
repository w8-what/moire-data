#!/usr/bin/env python3
"""Generate the final WSe2 phase-diagram atlas and tabular boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from moire.phase_outputs import FIELD_ORDER, build_phase_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "source_data",
        help="Directory containing Rxx_matrix_E-*mV_nm.csv files.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "reference_data" / "wse2_fig3_reference.json",
        help="Official Fig. 3 source-data coordinates for the paper panels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "phase_diagrams",
        help="Output directory.",
    )
    parser.add_argument("--fields", nargs="+", type=float, default=list(FIELD_ORDER))
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = build_phase_outputs(args.input, args.reference, args.output, fields=args.fields)
    print(f"Phase atlas: {outputs['atlas']}")
    print(f"Transitions: {outputs['csv']}")
    print(f"Candidate audit: {outputs['candidates']}")
    print(f"QA summary: {outputs['qa']}")


if __name__ == "__main__":
    main()
