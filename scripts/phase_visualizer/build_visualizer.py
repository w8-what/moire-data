"""Build the Python-backed linecut explorer in this folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualizer_math import DEFAULT_FIELDS, build_dataset

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", nargs="+", type=float, default=DEFAULT_FIELDS)
    parser.add_argument("--output", type=Path, default=HERE / "phase_visualizer.html")
    parser.add_argument("--data", type=Path, default=HERE / "phase_visualizer_data.json")
    args = parser.parse_args()

    dataset = build_dataset(args.fields)
    args.data.write_text(json.dumps(dataset, separators=(",", ":")))
    args.output.write_text((HERE / "visualizer_template.html").read_text())

    print(f"Wrote {args.output}")
    print(f"Wrote {args.data} ({args.data.stat().st_size / 1024:.0f} KiB)")
    for field in dataset["fields"]:
        print(
            f"  E={field['field']}: {field['acceptedCount']}/"
            f"{len(field['linecuts'])} accepted by get_fit_range"
        )


if __name__ == "__main__":
    main()
