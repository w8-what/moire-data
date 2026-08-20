"""Build Tcoh candidate-score figures for every source field."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from hampel import hampel
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth  # noqa: E402
from moire.extract_behaviors import extract_fit_range  # noqa: E402
from moire.extract_features import (  # noqa: E402
    extract_Tc,
    extract_Tcoh,
    extract_downturns,
    extract_upturns,
)
from moire.io import clean_sort_data, load_field  # noqa: E402
from moire.signal_helpers import local_noise  # noqa: E402
from moire.update_scoring import update_extrema  # noqa: E402

FIELDS = (74, 87, 96, 96.2, 99, 103, 151, 176)
OUT = Path(__file__).resolve().parent / "selector_comparisons"
WEIGHTED_OUT = Path(__file__).resolve().parent / "weighted_all_candidates"
FIT_SPAN_REFERENCE_K = 1.0
FIT_POINTS_REFERENCE = 10
HILL_REFERENCE_SCORE = 0.8
HILL_POWER = 2


def prepare_field(field):
    temperatures, fillings, resistance = clean_sort_data(*load_field(field, ROOT / "source_data"))
    linecuts = []

    for index, filling in enumerate(fillings):
        rho = resistance[:, index]
        filtered = hampel(rho).filtered_data
        smoothed = adaptive_multiscale_smooth(temperatures, filtered, z_threshold=3)
        linecut = {
            "nu": filling,
            "rho": rho,
            "rho_smoothed": smoothed,
            "local_noise": local_noise(temperatures, rho, smoothed),
        }
        linecut["features"] = (
            extract_upturns(temperatures, linecut)
            + extract_downturns(temperatures, linecut)
            + extract_Tc(temperatures, linecut)
        )
        linecuts.append(linecut)

    update_extrema(temperatures, linecuts)
    for linecut in linecuts:
        linecut["behaviors"] = extract_fit_range(temperatures, linecut)
        linecut["tcoh"] = extract_Tcoh(temperatures, linecut)

    return temperatures, fillings, resistance, linecuts


def _hill_score(values, reference):
    """Return a Hill score with ``reference`` mapping to 0.8."""
    values = np.asarray(values, float)
    constant = reference**HILL_POWER * (1 - HILL_REFERENCE_SCORE) / HILL_REFERENCE_SCORE
    powered = values**HILL_POWER
    return powered / (powered + constant)


def candidate_scores(temperatures, candidates):
    """Geometric mean of p-value, fit-span support, and point support."""
    spans = np.asarray(
        [candidate["fit_T_upper"] - candidate["fit_T_lower"] for candidate in candidates]
    )
    points = np.asarray(
        [
            np.count_nonzero(
                (temperatures >= candidate["fit_T_lower"])
                & (temperatures <= candidate["fit_T_upper"])
            )
            for candidate in candidates
        ]
    )
    span_score = _hill_score(spans, FIT_SPAN_REFERENCE_K)
    point_score = _hill_score(points, FIT_POINTS_REFERENCE)
    pvalues = np.asarray([candidate["pvalue"] for candidate in candidates])
    return np.cbrt(pvalues * span_score * point_score)


def length_weighted_candidate(temperatures, candidates):
    scores = candidate_scores(temperatures, candidates)
    return candidates[int(np.argmax(scores))]


def extraction_range(linecut):
    return next(
        (
            behavior
            for behavior in linecut["behaviors"]
            if behavior.get("type") == "extraction_range"
        ),
        None,
    )


def build_comparison(field, prepared=None):
    if prepared is None:
        prepared = prepare_field(field)
    temperatures, fillings, resistance, linecuts = prepared
    linecuts = [linecut for linecut in linecuts if linecut["tcoh"]]

    positive = resistance[np.isfinite(resistance) & (resistance > 0)]
    color_norm = LogNorm(*np.percentile(positive, [1, 99]))
    figure, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True)

    for axis in axes:
        axis.pcolormesh(
            fillings,
            temperatures,
            resistance,
            shading="nearest",
            cmap="coolwarm",
            norm=color_norm,
            alpha=0.55,
        )
        axis.set_xlabel("Filling ν")
    axes[0].set_ylabel("Temperature (K)")

    for linecut in linecuts:
        candidates = linecut["tcoh"]
        axes[0].scatter(
            [linecut["nu"]] * len(candidates),
            [candidate["T"] for candidate in candidates],
            s=12,
            color="#0047ff",
        )

        best_pvalue = max(candidates, key=lambda candidate: candidate["pvalue"])
        axes[1].scatter(linecut["nu"], best_pvalue["T"], s=14, color="#0047ff")

        weighted = length_weighted_candidate(temperatures, candidates)
        axes[2].scatter(linecut["nu"], weighted["T"], s=14, color="#0047ff")

        valid_range = extraction_range(linecut)
        if valid_range is not None and np.isclose(valid_range["T_lower"], temperatures[0]):
            longest = max(
                candidates,
                key=lambda candidate: (
                    candidate["fit_T_upper"] - candidate["fit_T_lower"],
                    candidate["pvalue"],
                ),
            )
            axes[3].scatter(linecut["nu"], longest["T"], s=14, color="#0047ff")

    titles = (
        "All candidates",
        "Highest p-value only",
        "p × normalized span/points",
        "Longest fit; valid full-low-T range",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title)

    figure.suptitle(f"T_coh selector comparison — {field:g} mV/nm")
    figure.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{field:g}_mV_nm.png"
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def _candidate_coordinates(temperatures, fillings, linecuts):
    """Return every candidate with small x offsets for exact overlaps."""
    filling_steps = np.diff(np.unique(fillings))
    filling_steps = filling_steps[filling_steps > 0]
    jitter = 0.4 * float(filling_steps.min()) if len(filling_steps) else 0.0

    plotted_fillings = []
    plotted_temperatures = []
    plotted_scores = []

    for linecut in linecuts:
        candidates = linecut["tcoh"]
        if not candidates:
            continue

        scores = candidate_scores(temperatures, candidates)
        groups = {}
        for index, candidate in enumerate(candidates):
            groups.setdefault(float(candidate["T"]), []).append(index)

        offsets = np.zeros(len(candidates))
        for indices in groups.values():
            if len(indices) > 1:
                offsets[indices] = np.linspace(-jitter, jitter, len(indices))

        plotted_fillings.extend(float(linecut["nu"]) + offsets)
        plotted_temperatures.extend(float(candidate["T"]) for candidate in candidates)
        plotted_scores.extend(float(score) for score in scores)

    return (
        np.asarray(plotted_fillings),
        np.asarray(plotted_temperatures),
        np.asarray(plotted_scores),
    )


def build_weighted_all_candidates(field, prepared=None):
    if prepared is None:
        prepared = prepare_field(field)
    temperatures, fillings, resistance, linecuts = prepared

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

    x, y, scores = _candidate_coordinates(temperatures, fillings, linecuts)
    order = np.argsort(scores)
    colors = np.tile(np.asarray([0.0, 0.22, 0.82, 1.0]), (len(scores), 1))
    colors[:, 3] = np.clip(scores, 0.0, 1.0)
    axis.scatter(x[order], y[order], s=12, marker="o", facecolors=colors[order], edgecolors="none")

    legend_scores = (0.2, 0.5, 0.8, 1.0)
    legend = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=(0.0, 0.22, 0.82, score),
            markeredgecolor="none",
            label=f"{score:.1f}",
        )
        for score in legend_scores
    ]
    axis.legend(handles=legend, title="Combined score", loc="upper left", framealpha=0.9)
    axis.set_xlabel("Filling ν")
    axis.set_ylabel("Temperature (K)")
    figure.suptitle(rf"All $T_{{coh}}$ candidates — {field:g} mV/nm", y=0.98)
    figure.text(
        0.5,
        0.925,
        r"Opacity = $(p\,T_{score}\,N_{score})^{1/3}$; "
        r"2nd-power Hill scores: 1.0 K and 10 points $\mapsto 0.8$",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))

    WEIGHTED_OUT.mkdir(parents=True, exist_ok=True)
    destination = WEIGHTED_OUT / f"{field:g}_mV_nm.png"
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


if __name__ == "__main__":
    for selected_field in FIELDS:
        field_data = prepare_field(selected_field)
        print(build_comparison(selected_field, field_data))
        print(build_weighted_all_candidates(selected_field, field_data))
