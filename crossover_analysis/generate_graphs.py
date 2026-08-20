"""Generate standalone diagnostic graphs for local Tcoh and T-prime candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from crossover_candidates import analyze_crossover
from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth, estimate_noise_1d

COLORS = {"Tcoh": "#1769aa", "Tprime": "#c02f77"}
LABELS = {"Tcoh": r"$T_{\mathrm{coh}}$", "Tprime": r"$T'$"}
DEFAULT_FIELDS = (103, 99, 96, 87)
DIAGNOSTIC_FILLINGS = {"Tcoh": 0.85, "Tprime": 0.918}


def load_field(field):
    frame = pd.read_csv(ROOT / "source_data" / f"Rxx_matrix_E-{field}mV_nm.csv")
    T = frame.iloc[:, 0].to_numpy(float)
    fillings = np.asarray([float(column) for column in frame.columns[1:]])
    resistance = frame.iloc[:, 1:].to_numpy(float)
    row_mask = np.isfinite(T) & np.all(np.isfinite(resistance), axis=1)
    column_mask = np.isfinite(fillings) & np.all(np.isfinite(resistance), axis=0)
    T, resistance = T[row_mask], resistance[row_mask][:, column_mask]
    fillings = fillings[column_mask]
    row_order, column_order = np.argsort(T), np.argsort(fillings)
    return T[row_order], fillings[column_order], resistance[row_order][:, column_order]


def prepare_linecut(T, raw, filling):
    # Both the noise estimate and smoother see only this linecut.
    sigma = estimate_noise_1d(T, raw)
    smooth = adaptive_multiscale_smooth(T, raw, sigma, z_threshold=3)
    return {"rho": raw, "rho_smoothed": smooth, "local_noise": sigma, "nu": float(filling)}


def published_boundaries(field):
    path = ROOT / "annotations" / "crossover_labels.json"
    if not path.exists():
        return {"Tcoh": [], "Tprime": []}
    data = json.loads(path.read_text())
    return data.get("fields", {}).get(str(field), {}).get("boundaries", {"Tcoh": [], "Tprime": []})


def analyze_field(field):
    T, fillings, resistance = load_field(field)
    linecuts = []
    results = {"Tcoh": [], "Tprime": []}
    for index, filling in enumerate(fillings):
        linecut = prepare_linecut(T, resistance[:, index], filling)
        linecuts.append(linecut)
        for kind in results:
            results[kind].append(analyze_crossover(T, linecut, kind))
    return T, fillings, resistance, linecuts, results


def _overlay_candidates(axis, fillings, results, kind):
    color = COLORS[kind]
    for filling, result in zip(fillings, results[kind]):
        for rank, candidate in enumerate(result["candidates"]):
            score = candidate["plausibility"]
            axis.scatter(
                filling,
                candidate["T"],
                s=12 + 80 * score,
                facecolors=color if rank == 0 else "none",
                edgecolors=color,
                linewidths=0.5 + score,
                alpha=0.18 + 0.82 * score,
                zorder=4,
            )


def _overlay_published(axis, boundaries, kind):
    points = boundaries.get(kind, [])
    segments = sorted({point.get("segment", "published") for point in points})
    for segment in segments:
        selected = sorted(
            (point for point in points if point.get("segment", "published") == segment),
            key=lambda point: point["nu"],
        )
        if not selected:
            continue
        axis.plot(
            [point["nu"] for point in selected],
            [point["T"] for point in selected],
            color="#30343b",
            marker="x",
            markersize=4,
            linewidth=0.8,
            alpha=0.75,
            zorder=5,
        )


def plot_candidate_map(field, T, fillings, resistance, results, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    positive = resistance[resistance > 0]
    low, high = np.percentile(positive, (1, 99))
    boundaries = published_boundaries(field)
    for axis, kind in zip(axes, ("Tcoh", "Tprime")):
        mesh = axis.pcolormesh(
            fillings,
            T,
            resistance,
            shading="nearest",
            cmap="magma",
            norm=LogNorm(vmin=low, vmax=high),
            rasterized=True,
        )
        _overlay_candidates(axis, fillings, results, kind)
        _overlay_published(axis, boundaries, kind)
        axis.set_title(f"{LABELS[kind]} candidates")
        axis.set_xlabel(r"Filling $\nu$")
        axis.grid(False)
        colorbar = figure.colorbar(mesh, ax=axis, pad=0.02)
        colorbar.set_label(r"$\rho_{xx}$")
    axes[0].set_ylabel("Temperature (K)")
    figure.suptitle(
        rf"Linecut-only crossover candidates — $|E|={field:g}$ mV nm$^{{-1}}$", fontsize=14
    )
    figure.text(
        0.5,
        0.015,
        "Circle size/opacity = local plausibility; gray × = published Figure 3 reference (not scored)",
        ha="center",
        fontsize=9,
        color="#555b66",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _candidate_curve(T, candidate, kind):
    x = T**2 if kind == "Tcoh" else T
    return candidate["rho0"] + candidate["A"] * x


def plot_linecut_diagnostics(field, T, fillings, linecuts, results, output):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, kind in enumerate(("Tcoh", "Tprime")):
        index = int(np.argmin(np.abs(fillings - DIAGNOSTIC_FILLINGS[kind])))
        linecut, result = linecuts[index], results[kind][index]
        raw = np.asarray(linecut["rho"])
        smooth = np.asarray(linecut["rho_smoothed"])
        top, bottom = axes[0, column], axes[1, column]
        top.scatter(T, raw, s=18, facecolors="white", edgecolors="#2f3338", linewidths=0.8)
        top.plot(T, smooth, color="#2f3338", linewidth=1.4, label="linecut smooth")
        metallic = result["metallic_interval"]
        if metallic:
            top.axvspan(metallic["T_lower"], metallic["T_upper"], color="#aeb6c2", alpha=0.12)
        bottom.axhline(10, color="#3b4048", linestyle="--", linewidth=1, label="10%")

        for rank, candidate in enumerate(result["candidates"]):
            score = candidate["plausibility"]
            curve = _candidate_curve(T, candidate, kind)
            relative = 100 * np.abs(smooth - curve) / np.maximum(np.abs(curve), 1e-12)
            if rank == 0:
                top.plot(T, curve, color=COLORS[kind], linewidth=1.6, label="representative fit")
                top.fill_between(
                    T,
                    0.9 * curve,
                    1.1 * curve,
                    color=COLORS[kind],
                    alpha=0.10,
                    label="±10% envelope",
                )
                top.axvspan(
                    candidate["fit_T_lower"],
                    candidate["fit_T_upper"],
                    color=COLORS[kind],
                    alpha=0.08,
                )
                bottom.plot(T, relative, color=COLORS[kind], linewidth=1.4)
            for axis in (top, bottom):
                axis.axvline(
                    candidate["T"],
                    color=COLORS[kind],
                    alpha=0.18 + 0.75 * score,
                    linewidth=0.7 + 1.4 * score,
                )

        best = result["candidates"][0] if result["candidates"] else None
        summary = result["status"].replace("_", " ")
        if best:
            summary += (
                f"\nT={best['T']:.2f} K, P={best['plausibility']:.2f}"
                f"\nB/D/R/C={best['B']:.2f}/{best['D']:.2f}/{best['R']:.2f}/{best['C']:.2f}"
            )
        top.text(
            0.03,
            0.96,
            summary,
            transform=top.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d6dae0", "alpha": 0.92},
        )
        top.set_title(rf"{LABELS[kind]} at $\nu={fillings[index]:.3f}$")
        top.set_ylabel(r"$\rho_{xx}$")
        bottom.set_ylabel("Departure (%)")
        bottom.set_xlabel("Temperature (K)")
        top.legend(loc="lower right", fontsize=8, frameon=False)
        top.grid(color="#e6e8eb", linewidth=0.6)
        bottom.grid(color="#e6e8eb", linewidth=0.6)
    figure.suptitle(
        rf"Representative linecut diagnostics — $|E|={field:g}$ mV nm$^{{-1}}$", fontsize=14
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_candidates(field, fillings, results, output):
    payload = {"field": field, "linecuts": []}
    for index, filling in enumerate(fillings):
        payload["linecuts"].append(
            {
                "nu": float(filling),
                "Tcoh": results["Tcoh"][index],
                "Tprime": results["Tprime"][index],
            }
        )
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", nargs="+", type=float, default=DEFAULT_FIELDS)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for value in args.fields:
        field = int(value) if float(value).is_integer() else value
        print(f"Analyzing {field} mV/nm", flush=True)
        T, fillings, resistance, linecuts, results = analyze_field(field)
        stem = str(field).replace(".", "p")
        plot_candidate_map(
            field, T, fillings, resistance, results, args.output / f"field_{stem}_candidate_map.png"
        )
        plot_linecut_diagnostics(
            field,
            T,
            fillings,
            linecuts,
            results,
            args.output / f"field_{stem}_linecut_diagnostics.png",
        )
        save_candidates(field, fillings, results, args.output / f"field_{stem}_candidates.json")


if __name__ == "__main__":
    main()
