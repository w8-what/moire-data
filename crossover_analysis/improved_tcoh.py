"""Standalone strict-cluster Tcoh improvement and four-field validation."""

from __future__ import annotations

import argparse
import copy
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
ABSOLUTE_PLAUSIBILITY = 0.65
RELATIVE_PLAUSIBILITY_MARGIN = 0.10
MAX_RETAINED_CANDIDATES = 4


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


def select_candidates(
    candidate_pool,
    mode="competitive",
    max_candidates=MAX_RETAINED_CANDIDATES,
    absolute_threshold=ABSOLUTE_PLAUSIBILITY,
    relative_margin=RELATIVE_PLAUSIBILITY_MARGIN,
):
    """Select candidates using only evidence from the current linecut."""
    if mode == "strict_top5":
        return [copy.deepcopy(candidate) for candidate in candidate_pool[:5]]
    if mode != "competitive":
        raise ValueError(f"Unknown candidate selection mode: {mode}")
    if not candidate_pool or candidate_pool[0]["plausibility"] < absolute_threshold:
        return []

    best = candidate_pool[0]["plausibility"]
    floor = max(absolute_threshold, best - relative_margin)
    selected = []
    for candidate in candidate_pool:
        if candidate["plausibility"] < floor:
            continue
        item = copy.deepcopy(candidate)
        item["plausibility_gap_from_best"] = float(best - item["plausibility"])
        item["relative_competitiveness"] = float(
            np.clip(1 - item["plausibility_gap_from_best"] / relative_margin, 0, 1)
        )
        selected.append(item)
        if len(selected) == max_candidates:
            break
    return selected


def analyze_tcoh(temperatures, linecut):
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

    candidate_pool = []
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
        crossing_weights = np.maximum(baseline * departure, np.finfo(float).eps)
        temperature_mad = 1.4826 * _weighted_median(
            np.abs(crossing_temperatures - candidate_temperature), crossing_weights
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
        candidate_pool.append(
            {
                "T": float(candidate_temperature),
                "T_uncertainty_K": float(temperature_mad),
                "T_support_lower_K": float(np.min(crossing_temperatures)),
                "T_support_upper_K": float(np.max(crossing_temperatures)),
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

    candidate_pool.sort(
        key=lambda candidate: (candidate["plausibility"], candidate["supporting_windows"]),
        reverse=True,
    )
    candidates = select_candidates(candidate_pool)
    status = "locally_supported" if candidates else "not_identifiable"
    return {
        "status": status,
        "candidates": candidates,
        "candidate_pool": candidate_pool,
        "candidate_count_before_retention": len(candidate_pool),
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


def rows_for_selection(rows, mode):
    """Materialize a plotting/output view without recomputing linecuts."""
    selected_rows = []
    for row in rows:
        selected_row = {"nu": row["nu"], "Tcoh": copy.deepcopy(row["Tcoh"])}
        analysis = selected_row["Tcoh"]
        pool = analysis.pop("candidate_pool", [])
        analysis["candidates"] = select_candidates(pool, mode=mode)
        analysis["status"] = "locally_supported" if analysis["candidates"] else "not_identifiable"
        selected_rows.append(selected_row)
    return selected_rows


def published_points(field):
    path = ROOT / "annotations" / "crossover_labels.json"
    data = json.loads(path.read_text())
    return data["fields"][str(field)]["boundaries"]["Tcoh"]


def validation_metrics(rows, anchors, tolerance_K=0.20):
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
    counts = np.asarray([len(row["Tcoh"]["candidates"]) for row in rows])
    return {
        "top_mae_K": float(np.mean(finite_top)) if finite_top.size else None,
        "top_median_absolute_error_K": float(np.median(finite_top)) if finite_top.size else None,
        "nearest_candidate_mae_K": (float(np.mean(finite_oracle)) if finite_oracle.size else None),
        "anchor_recall_within_tolerance": float(
            sum(value is not None and value <= tolerance_K for value in oracle_errors)
            / len(anchors)
        ),
        "recall_tolerance_K": tolerance_K,
        "mean_candidates_per_linecut": float(np.mean(counts)),
        "median_candidates_per_linecut": float(np.median(counts)),
        "zero_candidate_fraction": float(np.mean(counts == 0)),
        "linecut_count": len(rows),
        "anchor_count": len(anchors),
        "comparisons": comparisons,
    }


def aggregate_metrics(field_metrics):
    """Aggregate field validation without giving small fields extra weight."""
    comparisons = [
        comparison for metrics in field_metrics.values() for comparison in metrics["comparisons"]
    ]
    nearest_errors = [
        comparison["oracle_absolute_error"]
        for comparison in comparisons
        if comparison["oracle_absolute_error"] is not None
    ]
    top_errors = [
        comparison["top_absolute_error"]
        for comparison in comparisons
        if comparison["top_absolute_error"] is not None
    ]
    linecut_count = sum(metrics["linecut_count"] for metrics in field_metrics.values())
    weighted_candidate_count = sum(
        metrics["mean_candidates_per_linecut"] * metrics["linecut_count"]
        for metrics in field_metrics.values()
    )
    weighted_zero_count = sum(
        metrics["zero_candidate_fraction"] * metrics["linecut_count"]
        for metrics in field_metrics.values()
    )
    tolerance = next(iter(field_metrics.values()))["recall_tolerance_K"]
    return {
        "top_mae_K": float(np.mean(top_errors)) if top_errors else None,
        "nearest_candidate_mae_K": (float(np.mean(nearest_errors)) if nearest_errors else None),
        "anchor_recall_within_tolerance": float(
            sum(error is not None and error <= tolerance for error in nearest_errors)
            / len(comparisons)
        ),
        "recall_tolerance_K": tolerance,
        "mean_candidates_per_linecut": weighted_candidate_count / linecut_count,
        "zero_candidate_fraction": weighted_zero_count / linecut_count,
        "anchor_count": len(comparisons),
        "linecut_count": linecut_count,
    }


def score_legend(scores):
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=(BLUE[0], BLUE[1], BLUE[2], score),
            markeredgecolor="none",
            label=f"{score:.2f}",
        )
        for score in scores
    ]


def plot_field(
    field,
    temperatures,
    fillings,
    resistance,
    rows,
    anchors,
    metrics,
    output,
    stage_label,
    visual_hierarchy=False,
    legend_scores=(0.65, 0.75, 0.85, 0.95),
):
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

    x_values, y_values, scores, competitiveness = [], [], [], []
    for row in rows:
        for candidate in row["Tcoh"]["candidates"]:
            x_values.append(row["nu"])
            y_values.append(candidate["T"])
            scores.append(candidate["plausibility"])
            competitiveness.append(candidate.get("relative_competitiveness", 1.0))
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    scores = np.asarray(scores)
    competitiveness = np.asarray(competitiveness)
    order = np.argsort(scores)
    colors = np.tile(BLUE, (len(scores), 1))
    if visual_hierarchy:
        colors[:, 3] = np.clip(0.20 + 0.80 * scores * competitiveness, 0, 1)
        sizes = 9 + 8 * competitiveness
    else:
        colors[:, 3] = np.clip(scores, 0, 1)
        sizes = np.full(len(scores), 12)
    axis.scatter(
        x_values[order],
        y_values[order],
        s=sizes[order],
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
        handles=score_legend(legend_scores)
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
    figure.suptitle(rf"$T_{{coh}}$ candidates — {field:g} mV/nm", y=0.98)
    recall = 100 * metrics["anchor_recall_within_tolerance"]
    nearest = metrics["nearest_candidate_mae_K"]
    figure.text(
        0.5,
        0.925,
        f"{stage_label} · published-marker recall within 0.20 K: {recall:.0f}% · "
        f"nearest-candidate MAE {nearest:.2f} K",
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
    parser.add_argument("--output", type=Path, default=HERE / "paper_match_stages")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    stage_one = args.output / "stage_01_strict_top5"
    stage_two = args.output / "stage_02_competitive_candidates"
    stage_three = args.output / "stage_03_final_visual_hierarchy"
    for directory in (stage_one, stage_two, stage_three):
        directory.mkdir(parents=True, exist_ok=True)

    summary = {
        "method": "strict_two_index_clusters_with_linecut_local_competitive_retention",
        "selection": {
            "absolute_plausibility_minimum": ABSOLUTE_PLAUSIBILITY,
            "maximum_gap_from_linecut_best": RELATIVE_PLAUSIBILITY_MARGIN,
            "maximum_candidates_per_linecut": MAX_RETAINED_CANDIDATES,
        },
        "stages": {"stage_01_strict_top5": {}, "stage_02_competitive_candidates": {}},
    }
    for value in args.fields:
        field = int(value) if float(value).is_integer() else value
        print(f"Analyzing improved Tcoh for {field} mV/nm", flush=True)
        temperatures, fillings, resistance, rows = analyze_field(field)
        anchors = published_points(field)
        top_five_rows = rows_for_selection(rows, "strict_top5")
        competitive_rows = rows_for_selection(rows, "competitive")
        top_five_metrics = validation_metrics(top_five_rows, anchors)
        competitive_metrics = validation_metrics(competitive_rows, anchors)
        summary["stages"]["stage_01_strict_top5"][str(field)] = top_five_metrics
        summary["stages"]["stage_02_competitive_candidates"][str(field)] = competitive_metrics

        (stage_one / f"{field:g}_candidates.json").write_text(
            json.dumps({"field": field, "linecuts": top_five_rows}, indent=2, allow_nan=False)
        )
        (stage_two / f"{field:g}_candidates.json").write_text(
            json.dumps({"field": field, "linecuts": competitive_rows}, indent=2, allow_nan=False)
        )
        (stage_three / f"{field:g}_candidates.json").write_text(
            json.dumps({"field": field, "linecuts": competitive_rows}, indent=2, allow_nan=False)
        )
        plot_field(
            field,
            temperatures,
            fillings,
            resistance,
            top_five_rows,
            anchors,
            top_five_metrics,
            stage_one / f"{field:g}_Tcoh_candidates.png",
            "Stage 1: up to five strict local candidates",
            legend_scores=(0.2, 0.5, 0.8, 1.0),
        )
        plot_field(
            field,
            temperatures,
            fillings,
            resistance,
            competitive_rows,
            anchors,
            competitive_metrics,
            stage_two / f"{field:g}_Tcoh_candidates.png",
            "Stage 2: variable 0–4 competitive candidates",
        )
        plot_field(
            field,
            temperatures,
            fillings,
            resistance,
            competitive_rows,
            anchors,
            competitive_metrics,
            stage_three / f"{field:g}_Tcoh_candidates.png",
            "Final: 0–4 candidates; local competition controls emphasis",
            visual_hierarchy=True,
        )
    for stage in summary["stages"].values():
        stage["overall"] = aggregate_metrics(stage)
    (args.output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
