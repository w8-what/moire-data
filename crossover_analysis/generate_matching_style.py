"""Render crossover results in the repository's established plot style."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth, estimate_noise_1d

FIELDS = (103, 99, 96, 87)
KINDS = ("Tcoh", "Tprime")
LABELS = {"Tcoh": r"$T_{coh}$", "Tprime": r"$T'$"}
LINECUT_FILLINGS = {"Tcoh": 0.850, "Tprime": 0.918}
BLUE = np.asarray([0.0, 0.22, 0.82, 1.0])


def load_field(field):
    frame = pd.read_csv(ROOT / "source_data" / f"Rxx_matrix_E-{field}mV_nm.csv")
    temperatures = frame.iloc[:, 0].to_numpy(float)
    fillings = np.asarray([float(column) for column in frame.columns[1:]])
    resistance = frame.iloc[:, 1:].to_numpy(float)
    row_mask = np.isfinite(temperatures) & np.all(np.isfinite(resistance), axis=1)
    column_mask = np.isfinite(fillings) & np.all(np.isfinite(resistance), axis=0)
    temperatures = temperatures[row_mask]
    resistance = resistance[row_mask][:, column_mask]
    fillings = fillings[column_mask]
    row_order = np.argsort(temperatures)
    column_order = np.argsort(fillings)
    return (temperatures[row_order], fillings[column_order], resistance[row_order][:, column_order])


def load_results(field):
    path = HERE / "output" / f"field_{field}_candidates.json"
    return json.loads(path.read_text())["linecuts"]


def candidate_coordinates(fillings, rows, kind):
    steps = np.diff(np.unique(fillings))
    steps = steps[steps > 0]
    jitter = 0.4 * float(steps.min()) if len(steps) else 0.0
    x_values, y_values, scores = [], [], []
    for row in rows:
        candidates = row[kind]["candidates"]
        groups = {}
        for index, candidate in enumerate(candidates):
            groups.setdefault(float(candidate["T"]), []).append(index)
        offsets = np.zeros(len(candidates))
        for indices in groups.values():
            if len(indices) > 1:
                offsets[indices] = np.linspace(-jitter, jitter, len(indices))
        x_values.extend(float(row["nu"]) + offsets)
        y_values.extend(float(candidate["T"]) for candidate in candidates)
        scores.extend(float(candidate["plausibility"]) for candidate in candidates)
    return np.asarray(x_values), np.asarray(y_values), np.asarray(scores)


def score_legend():
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=(BLUE[0], BLUE[1], BLUE[2], score),
            markeredgecolor="none",
            label=f"{score:.1f}",
        )
        for score in (0.2, 0.5, 0.8, 1.0)
    ]


def plot_candidate_map(field, kind, temperatures, fillings, resistance, rows, output):
    positive = resistance[np.isfinite(resistance) & (resistance > 0)]
    color_norm = LogNorm(*np.percentile(positive, [1, 99]))
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.pcolormesh(
        fillings,
        temperatures,
        resistance,
        shading="nearest",
        cmap="coolwarm",
        norm=color_norm,
        alpha=0.55,
    )

    x_values, y_values, scores = candidate_coordinates(fillings, rows, kind)
    order = np.argsort(scores)
    colors = np.tile(BLUE, (len(scores), 1))
    colors[:, 3] = np.clip(scores, 0, 1)
    axis.scatter(
        x_values[order],
        y_values[order],
        s=12,
        marker="o",
        facecolors=colors[order],
        edgecolors="none",
    )

    axis.legend(handles=score_legend(), title="Plausibility", loc="upper left", framealpha=0.9)
    axis.set_xlabel("Filling ν")
    axis.set_ylabel("Temperature (K)")
    figure.suptitle(rf"All {LABELS[kind]} candidates — {field:g} mV/nm", y=0.98)
    figure.text(
        0.5,
        0.925,
        r"Opacity = linecut-only plausibility " r"$B^{0.30}D^{0.30}R^{0.25}C^{0.15}$",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_linecut(field, kind, temperatures, fillings, resistance, rows, output):
    index = int(np.argmin(np.abs(fillings - LINECUT_FILLINGS[kind])))
    raw = resistance[:, index]
    sigma = estimate_noise_1d(temperatures, raw)
    smoothed = adaptive_multiscale_smooth(temperatures, raw, sigma, z_threshold=3)
    result = rows[index][kind]

    figure, axes = plt.subplots(1, 2, squeeze=False, figsize=(15, 6), dpi=250)
    raw_axis, smooth_axis = axes.flatten()
    common_style = {
        "color": "blue",
        "linewidth": 1.0,
        "marker": "o",
        "markersize": 3,
        "markerfacecolor": "none",
        "markeredgecolor": "navy",
    }
    raw_axis.plot(temperatures, raw, **common_style)
    smooth_axis.plot(temperatures, smoothed, **common_style, label="Smoothed data")
    smooth_axis.fill_between(
        temperatures, smoothed - sigma, smoothed + sigma, color="tab:blue", alpha=0.15, linewidth=0
    )

    for rank, candidate in enumerate(result["candidates"]):
        plausibility = candidate["plausibility"]
        fit_x = temperatures**2 if kind == "Tcoh" else temperatures
        fit = candidate["rho0"] + candidate["A"] * fit_x
        if rank == 0:
            smooth_axis.plot(
                temperatures,
                fit,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="Representative fit",
            )
            smooth_axis.fill_between(
                temperatures,
                0.9 * fit,
                1.1 * fit,
                color="grey",
                alpha=0.12,
                linewidth=0,
                label="±10%",
            )
        candidate_rho = smoothed[np.argmin(np.abs(temperatures - candidate["T"]))]
        smooth_axis.scatter(
            candidate["T"],
            candidate_rho,
            s=35,
            color="blue",
            edgecolor="black",
            linewidth=0.4,
            alpha=plausibility,
            zorder=5,
        )
        smooth_axis.axvline(
            candidate["T"], linewidth=1, linestyle="--", color="grey", alpha=plausibility
        )
        if rank == 0:
            smooth_axis.annotate(
                f"P={plausibility:.3f}\nB/D/R/C="
                f"{candidate['B']:.2f}/{candidate['D']:.2f}/"
                f"{candidate['R']:.2f}/{candidate['C']:.2f}",
                xy=(candidate["T"], candidate_rho),
                xytext=(0.55, 0.12),
                textcoords="axes fraction",
                bbox={"boxstyle": "round", "fc": "0.8", "alpha": 0.8},
                arrowprops={
                    "arrowstyle": "->",
                    "shrinkA": 0,
                    "shrinkB": 8,
                    "connectionstyle": "angle,angleA=0,angleB=90,rad=10",
                    "alpha": 0.8,
                },
            )

    for axis, title in zip((raw_axis, smooth_axis), ("Raw Data", "Smoothed Data")):
        axis.set_xlabel("Temperature (K)")
        axis.set_ylabel("Resistivity (Ω*cm)")
        axis.set_xlim(0, None)
        axis.set_ylim(0, None)
        axis.set_title(title)
    if result["candidates"]:
        smooth_axis.legend(loc="lower right", frameon=True, framealpha=1.0, fontsize=8)
    else:
        smooth_axis.text(
            0.05,
            0.9,
            "No retained candidates",
            transform=smooth_axis.transAxes,
            bbox={"boxstyle": "round", "fc": "0.8", "alpha": 0.8},
        )
    figure.suptitle(f"E = {field:g}     nu = {fillings[index]:.3f}     {LABELS[kind]}")
    figure.tight_layout()
    figure.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", nargs="+", type=float, default=FIELDS)
    parser.add_argument("--output", type=Path, default=HERE / "matching_style")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for value in args.fields:
        field = int(value) if float(value).is_integer() else value
        temperatures, fillings, resistance = load_field(field)
        rows = load_results(field)
        for kind in KINDS:
            suffix = "Tcoh" if kind == "Tcoh" else "Tprime"
            plot_candidate_map(
                field,
                kind,
                temperatures,
                fillings,
                resistance,
                rows,
                args.output / f"{field:g}_{suffix}_candidates.png",
            )
            plot_linecut(
                field,
                kind,
                temperatures,
                fillings,
                resistance,
                rows,
                args.output / f"{field:g}_{suffix}_linecut.png",
            )


if __name__ == "__main__":
    main()
