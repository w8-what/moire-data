import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hampel import hampel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth
from moire.draw_lines import generate_layout, overlay_behaviors, overlay_features, plot_general_line
from moire.extract_behaviors import extract_fit_range
from moire.extract_features import extract_downturns, extract_Tc, extract_upturns
from moire.extract_power_law import extract_local_fits
from moire.io import clean_sort_data, fmt4, load_field
from moire.signal_helpers import local_noise
from moire.update_scoring import update_score

DEFAULT_FIELDS = (87, 96, 99, 103, 74, 96.2, 151, 176)


def _field_value(value):
    """Keep integral field names compatible with files such as ``E-96mV``."""
    number = float(value)
    return int(number) if number.is_integer() else number


def process_field(field, input_dir, output_dir, num_linecuts=30):
    """Process one electric field and write a representative set of linecut plots."""
    T, nu, R = load_field(field, input_dir)
    T, nu, R = clean_sort_data(T, nu, R)

    linecuts = [
        {"E": field, "nu": filling, "T": T, "rho": R[:, index]} for index, filling in enumerate(nu)
    ]

    for linecut in linecuts:
        rho = linecut["rho"]
        rho_hampel = np.asarray(hampel(rho).filtered_data, dtype=float)
        rho_smoothed = adaptive_multiscale_smooth(T, rho_hampel, z_threshold=3)
        linecut["rho_smoothed"] = rho_smoothed
        linecut["local_noise"] = local_noise(T, rho, rho_smoothed)

        features = []
        features.extend(extract_upturns(T, linecut))
        features.extend(extract_downturns(T, linecut))
        features.extend(extract_Tc(T, linecut))
        linecut["features"] = features
        linecut["behaviors"] = []

    update_score(linecuts)

    for linecut in linecuts:
        linecut["behaviors"].extend(extract_fit_range(T, linecut))
        linecut["local_power_fit"] = extract_local_fits(T, linecut)

    count = min(max(num_linecuts, 0), len(linecuts))
    selected_indices = np.linspace(0, len(linecuts) - 1, count, dtype=int) if count else []
    linecut_dir = output_dir / "linecuts"
    linecut_dir.mkdir(parents=True, exist_ok=True)

    for index in selected_indices:
        linecut = linecuts[index]
        param_string = "  ".join(
            f"{key} = {fmt4(value)}" for key, value in linecut.items() if key in {"E", "nu"}
        )

        rho = linecut["rho"]
        rho_smoothed = linecut["rho_smoothed"]
        derivative = np.gradient(rho_smoothed, T)
        fit = linecut["local_power_fit"]

        fig, axes = generate_layout(4, title=param_string)
        linecut_axis_kwargs = {
            "xlabel": "Temperature (K)",
            "ylabel": "Resistivity (Ω*cm)",
            "xlim": (0, None),
            "ylim": (0, None),
        }

        plot_general_line(axes[0], T, rho, title="Raw Data", **linecut_axis_kwargs)
        plot_general_line(
            axes[1],
            T,
            rho_smoothed,
            title="Smoothed Data, Features, Behaviors",
            **linecut_axis_kwargs,
        )
        plot_general_line(
            axes[2],
            T,
            derivative,
            title="Smoothed First Derivative",
            xlabel="Temperature (K)",
            ylabel="dρ/dT",
            xlim=(0, np.max(T)),
            shaded=True,
            fill_alpha=0.5,
        )
        plot_general_line(
            axes[3],
            T,
            fit["n"],
            error=np.asarray(fit["n_sigma"], dtype=float),
            title="Local Power-Law Exponent",
            xlabel="Temperature (K)",
            ylabel="n",
            xlim=(0, np.max(T)),
            ylim=(0, 8),
            shaded=True,
            fill_alpha=0.5,
        )
        for value in (0, 0.8, 1.2):
            axes[3].axhline(y=value, alpha=0.5, linestyle="-", color="grey")

        overlay_features(axes[1], linecut, score_name="score_15", feature_name="features_new")
        overlay_behaviors(axes[1], linecut)
        fig.tight_layout()
        fig.savefig(linecut_dir / f"{param_string}.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

    return linecuts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the moire resistivity analysis pipeline")
    parser.add_argument(
        "--fields",
        nargs="+",
        type=_field_value,
        default=list(DEFAULT_FIELDS),
        help="electric fields to process (default: all source fields)",
    )
    parser.add_argument(
        "--linecuts",
        type=int,
        default=30,
        help="maximum number of representative linecuts to plot per field",
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "source_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args(argv)
    if args.linecuts < 0:
        parser.error("--linecuts must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    for field in args.fields:
        process_field(field, args.input_dir, args.output_dir, args.linecuts)


if __name__ == "__main__":
    main()
