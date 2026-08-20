"""Standalone strict-cluster Tcoh improvement and four-field validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from crossover_candidates import (
    DEFAULT_CONFIG,
    _metallic_interval,
    _prepare_linecut,
    _weighted_median,
    _window_proposal,
)
from generate_graphs import load_field, prepare_linecut

FIELDS = (103, 99, 96, 87)
BLUE = np.asarray([0.0, 0.22, 0.82, 1.0])


def strict_clusters(proposals, maximum_diameter=2):
    """Cluster snapped indices with total diameter <= two indices."""
    ordered = sorted(proposals, key=lambda proposal: proposal["index"])
    clusters = []
    for proposal in ordered:
        if not clusters or proposal["index"] - clusters[-1][0]["index"] > maximum_diameter:
            clusters.append([proposal])
        else:
            clusters[-1].append(proposal)
    return clusters


def analyze_tcoh(temperatures, linecut, max_candidates=5):
    temperatures, raw, smooth, sigma = _prepare_linecut(temperatures, linecut)
    metallic = _metallic_interval(temperatures, smooth, DEFAULT_CONFIG)
    if metallic is None:
        return {"status": "not_identifiable", "candidates": [], "metallic_interval": None}

    left, right = metallic["left"], metallic["right"]
    proposals = []
    eligible_weight = 0.0
    eligible_windows = 0
    for end in range(left + DEFAULT_CONFIG.min_fit_points - 1, right):
        indices = np.arange(left, end + 1)
        if temperatures[end] - temperatures[left] < DEFAULT_CONFIG.min_fit_span:
            continue
        proposal, baseline = _window_proposal(
            "Tcoh", temperatures, raw, smooth, sigma, indices, right, DEFAULT_CONFIG
        )
        eligible_windows += 1
        eligible_weight += baseline
        if proposal is not None:
            proposals.append(proposal)

    candidates = []
    for cluster in strict_clusters(proposals, DEFAULT_CONFIG.cluster_indices):
        if len(cluster) < DEFAULT_CONFIG.min_supporting_windows:
            continue
        baseline = np.asarray([proposal["B"] for proposal in cluster], float)
        departure = np.asarray([proposal["D"] for proposal in cluster], float)
        contrast = np.asarray([proposal["C"] for proposal in cluster], float)
        indices = np.asarray([proposal["index"] for proposal in cluster], float)
        crossing_temperatures = np.asarray([proposal["T"] for proposal in cluster], float)
        weights = np.maximum(baseline, np.finfo(float).eps)

        candidate_temperature = _weighted_median(
            crossing_temperatures, np.maximum(baseline * departure, np.finfo(float).eps)
        )
        score_B = float(np.average(baseline, weights=weights))
        score_D = float(np.average(departure, weights=weights))
        score_C = float(np.average(contrast, weights=weights))
        vote = min(1.0, float(np.sum(baseline) / eligible_weight))
        median_index = _weighted_median(indices, weights)
        index_mad = _weighted_median(np.abs(indices - median_index), weights)
        spread = math.exp(-0.5 * (index_mad / 2) ** 2)
        score_R = math.sqrt(vote * spread)
        plausibility = score_B**0.30 * score_D**0.30 * score_R**0.25 * score_C**0.15
        representative = max(
            cluster, key=lambda proposal: (proposal["B"] * proposal["D"], proposal["B"])
        )
        candidates.append(
            {
                "T": float(candidate_temperature),
                "plausibility": float(np.clip(plausibility, 0, 1)),
                "B": score_B,
                "D": score_D,
                "R": score_R,
                "C": score_C,
                "supporting_windows": len(cluster),
                "vote_fraction": vote,
                "index_mad": float(index_mad),
                "fit_T_lower": float(temperatures[representative["fit_left_index"]]),
                "fit_T_upper": float(temperatures[representative["fit_right_index"]]),
                "rho0": representative["rho0"],
                "A": representative["A"],
            }
        )

    candidates.sort(
        key=lambda candidate: (candidate["plausibility"], candidate["supporting_windows"]),
        reverse=True,
    )
    candidates = candidates[:max_candidates]
    status = (
        "locally_supported"
        if candidates and candidates[0]["plausibility"] >= DEFAULT_CONFIG.identifiable_threshold
        else "not_identifiable"
    )
    return {
        "status": status,
        "candidates": candidates,
        "metallic_interval": {
            "T_lower": float(temperatures[left]),
            "T_upper": float(temperatures[right]),
            "positive_fraction": metallic["positive_fraction"],
        },
        "proposal_count": len(proposals),
        "eligible_window_count": eligible_windows,
    }


def analyze_field(field):
    temperatures, fillings, resistance = load_field(field)
    rows = []
    for index, filling in enumerate(fillings):
        linecut = prepare_linecut(temperatures, resistance[:, index], filling)
        rows.append({"nu": float(filling), "Tcoh": analyze_tcoh(temperatures, linecut)})
    return temperatures, fillings, resistance, rows


def published_points(field):
    path = ROOT / "annotations" / "crossover_labels.json"
    data = json.loads(path.read_text())
    return data["fields"][str(field)]["boundaries"]["Tcoh"]


def validation_metrics(rows, anchors):
    top_errors, oracle_errors = [], []
    comparisons = []
    for anchor in anchors:
        row = min(rows, key=lambda item: abs(item["nu"] - anchor["nu"]))
        candidates = row["Tcoh"]["candidates"]
        top_temperature = candidates[0]["T"] if candidates else None
        top_error = abs(top_temperature - anchor["T"]) if candidates else None
        oracle_error = (
            min(abs(candidate["T"] - anchor["T"]) for candidate in candidates)
            if candidates
            else None
        )
        top_errors.append(top_error)
        oracle_errors.append(oracle_error)
        comparisons.append(
            {
                "nu_published": anchor["nu"],
                "T_published": anchor["T"],
                "segment": anchor.get("segment"),
                "nu_measured": row["nu"],
                "T_top": top_temperature,
                "top_absolute_error": top_error,
                "oracle_absolute_error": oracle_error,
            }
        )
    finite_top = np.asarray([value for value in top_errors if value is not None])
    finite_oracle = np.asarray([value for value in oracle_errors if value is not None])
    return {
        "top_mae_K": float(np.mean(finite_top)),
        "top_median_absolute_error_K": float(np.median(finite_top)),
        "oracle_mae_K": float(np.mean(finite_oracle)),
        "anchor_count": len(anchors),
        "comparisons": comparisons,
    }


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


def plot_field(field, temperatures, fillings, resistance, rows, anchors, metrics, output):
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

    x_values, y_values, scores = [], [], []
    for row in rows:
        for candidate in row["Tcoh"]["candidates"]:
            x_values.append(row["nu"])
            y_values.append(candidate["T"])
            scores.append(candidate["plausibility"])
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    scores = np.asarray(scores)
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

    axis.scatter(
        [anchor["nu"] for anchor in anchors],
        [anchor["T"] for anchor in anchors],
        color="#222222",
        marker="x",
        s=30,
        linewidth=1,
        zorder=5,
    )

    legend = axis.legend(
        handles=score_legend()
        + [
            Line2D(
                [],
                [],
                color="#222222",
                marker="x",
                markersize=5,
                linestyle="none",
                label="Published Figure 3",
            )
        ],
        title="Plausibility",
        loc="upper left",
        framealpha=0.9,
    )
    legend.get_title().set_fontsize(9)
    axis.set_xlabel("Filling ν")
    axis.set_ylabel("Temperature (K)")
    figure.suptitle(rf"Improved $T_{{coh}}$ candidates — {field:g} mV/nm", y=0.98)
    figure.text(
        0.5,
        0.925,
        f"Five strict local candidates · top-candidate MAE {metrics['top_mae_K']:.2f} K · "
        f"nearest retained-candidate MAE {metrics['oracle_mae_K']:.2f} K",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", nargs="+", type=float, default=FIELDS)
    parser.add_argument("--output", type=Path, default=HERE / "tcoh_improved")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary = {"method": "strict_two_index_clusters", "fields": {}}
    for value in args.fields:
        field = int(value) if float(value).is_integer() else value
        print(f"Analyzing improved Tcoh for {field} mV/nm", flush=True)
        temperatures, fillings, resistance, rows = analyze_field(field)
        anchors = published_points(field)
        metrics = validation_metrics(rows, anchors)
        summary["fields"][str(field)] = metrics
        (args.output / f"{field:g}_candidates.json").write_text(
            json.dumps({"field": field, "linecuts": rows}, indent=2, allow_nan=False)
        )
        plot_field(
            field,
            temperatures,
            fillings,
            resistance,
            rows,
            anchors,
            metrics,
            args.output / f"{field:g}_Tcoh_improved.png",
        )
    (args.output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
