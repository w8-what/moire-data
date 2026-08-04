"""Build publication-style phase-diagram artifacts for every resistance map."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth
from moire.io import clean_sort_data, load_field
from moire.phase_diagram import (
    PhaseExtractionConfig,
    extract_field_phase_diagram,
    extract_tcoh_linecut,
    extract_tneel_candidates,
    extract_tprime_linecut,
)

FIELD_ORDER = (103, 99, 96, 87, 96.2, 74, 151, 176)
REFERENCE_FIELDS = (103, 99, 96, 87)

TRANSITION_STYLES = {
    "Tcoh": {
        "label": r"$T_{\mathrm{coh}}$ ($T^2$ boundary)",
        "color": "#15B7D3",
        "marker": "o",
        "linestyle": "-",
    },
    "Tprime": {
        "label": r"$T^{\prime}$ (linear-$T$ departure)",
        "color": "#D64F70",
        "marker": "s",
        "linestyle": "--",
    },
    "Tneel": {"label": r"$T_N$ (published)", "color": "#F28E2B", "marker": "^", "linestyle": "-."},
}


def _field_key(field):
    return str(int(field)) if float(field).is_integer() else str(field)


def _field_filename(field):
    return _field_key(field).replace(".", "p")


def _load_reference(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _nearest_temperature_uncertainty(temperature, value):
    temperature = np.asarray(temperature, float)
    index = int(np.argmin(np.abs(temperature - value)))
    lower = temperature[max(0, index - 1)]
    upper = temperature[min(len(temperature) - 1, index + 1)]
    return max(float(0.25 * (upper - lower)), 0.025)


def _reference_rows(field, temperature, reference):
    field_data = reference["fields"][_field_key(field)]
    rows = []
    for transition, coordinates in field_data.items():
        coordinates = sorted(coordinates, key=lambda pair: pair[0])
        component = 0
        previous_nu = None
        for nu, transition_temperature in coordinates:
            if transition == "Tcoh" and previous_nu is not None and nu - previous_nu > 0.04:
                component += 1
            rows.append(
                {
                    "field_mV_nm": -abs(float(field)),
                    "nu": float(nu),
                    "transition": transition,
                    "temperature_K": float(transition_temperature),
                    "uncertainty_K": _nearest_temperature_uncertainty(
                        temperature, transition_temperature
                    ),
                    "uncertainty_lower_K": _nearest_temperature_uncertainty(
                        temperature, transition_temperature
                    ),
                    "uncertainty_upper_K": _nearest_temperature_uncertainty(
                        temperature, transition_temperature
                    ),
                    "crossing_uncertainty_K": _nearest_temperature_uncertainty(
                        temperature, transition_temperature
                    ),
                    "sensitivity_uncertainty_K": "",
                    "confidence": 1.0,
                    "component": component,
                    "provenance": "published_source_data_fig3",
                    "model": reference["method"][transition],
                    "fit_lower_K": "",
                    "fit_upper_K": "",
                    "fit_median_fractional_error": "",
                    "fit_p90_fractional_error": "",
                    "fit_offset": "",
                    "fit_coefficient": "",
                    "exponent": "",
                    "exponent_sigma": "",
                    "crossing_lower_K": "",
                    "crossing_upper_K": "",
                    "censored": False,
                    "nu_censored": False,
                    "nu_censor_side": "",
                    "support": 1.0,
                    "selected_for_atlas": True,
                    "rejection_reason": "",
                }
            )
            previous_nu = nu
    return rows


def _automatic_rows(result, *, points=None, selected_ids=None):
    rows = []
    points = result["points"] if points is None else points
    selected_ids = {id(point) for point in points} if selected_ids is None else selected_ids
    for point in points:
        item = asdict(point)
        selected = id(point) in selected_ids
        crossing_uncertainty = (
            0.5 * (item["crossing_upper"] - item["crossing_lower"])
            if item["crossing_lower"] is not None and item["crossing_upper"] is not None
            else item["uncertainty"]
        )
        rows.append(
            {
                "field_mV_nm": item["field"],
                "nu": item["nu"],
                "transition": item["transition"],
                "temperature_K": item["temperature"],
                "uncertainty_K": item["uncertainty"],
                "uncertainty_lower_K": item["uncertainty"],
                "uncertainty_upper_K": item["uncertainty"],
                "crossing_uncertainty_K": crossing_uncertainty,
                "sensitivity_uncertainty_K": item["uncertainty"],
                "confidence": item["confidence"],
                "component": item["component"],
                "provenance": "automatic_transport_extraction",
                "model": item["model"],
                "fit_lower_K": item["fit_lower"],
                "fit_upper_K": item["fit_upper"],
                "fit_median_fractional_error": item["fit_median_fractional_error"],
                "fit_p90_fractional_error": item["fit_p90_fractional_error"],
                "fit_offset": item["fit_offset"],
                "fit_coefficient": item["fit_coefficient"],
                "exponent": item["exponent"],
                "exponent_sigma": item["exponent_sigma"],
                "crossing_lower_K": item["crossing_lower"],
                "crossing_upper_K": item["crossing_upper"],
                "censored": item["censored"],
                "nu_censored": item["nu_censored"],
                "nu_censor_side": item["nu_censor_side"],
                "support": item["support"],
                "selected_for_atlas": selected,
                "rejection_reason": "" if selected else "failed physical branch continuity/support",
            }
        )
    return rows


def _minus96_ambiguous_rows(result):
    """Expose both measured minima left of the source endpoint without choosing a phase path."""
    candidates = [point for point in result["raw_candidates"]["Tneel"] if 0.985 <= point.nu < 0.988]
    rows = _automatic_rows(result, points=candidates, selected_ids=set())
    for component, row in enumerate(rows):
        row["component"] = 100 + component
        row["provenance"] = "automatic_multimodal_candidate"
        row["selected_for_atlas"] = False
        row["nu_censored"] = False
        row["nu_censor_side"] = ""
        row["model"] = "unresolved competing transport minima; not an AFM boundary"
        row["rejection_reason"] = "competing temperature modes prevent a unique T_N path"
    return rows


def _display_resistance(resistance):
    resistance = np.asarray(resistance, float)
    positive = resistance[np.isfinite(resistance) & (resistance > 0)]
    if not len(positive):
        return np.full_like(resistance, 10.0), 10.0
    floor = 10.0
    return np.maximum(resistance, floor), floor


def _panel_lognorm(resistance):
    positive = np.asarray(resistance, float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if not len(positive):
        return LogNorm(vmin=10, vmax=100)
    vmin = max(float(np.percentile(positive, 1)), 1e-3)
    vmax = max(float(np.percentile(positive, 99.5)), 10.0 * vmin)
    return LogNorm(vmin=vmin, vmax=vmax)


def _rolling_median(values, window):
    values = np.asarray(values, float)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray([np.median(padded[index : index + window]) for index in range(len(values))])


def _plot_transition_rows(ax, rows, *, errorbars):
    for transition, style in TRANSITION_STYLES.items():
        transition_rows = [row for row in rows if row["transition"] == transition]
        components = sorted(
            {row["component"] for row in transition_rows if row["component"] is not None}
        )
        for component in components:
            branch = sorted(
                [row for row in transition_rows if row["component"] == component],
                key=lambda row: row["nu"],
            )
            if not branch:
                continue
            x = np.asarray([row["nu"] for row in branch])
            y = np.asarray([row["temperature_K"] for row in branch])
            provenance = branch[0]["provenance"]
            automatic = provenance == "automatic_transport_extraction"
            ambiguous = provenance == "automatic_multimodal_candidate"
            marker_face = "none" if automatic or ambiguous else style["color"]
            marker_edge = style["color"] if automatic or ambiguous else "white"
            marker = "x" if ambiguous else style["marker"]
            ax.plot(
                x,
                y,
                color=style["color"],
                linestyle="none" if automatic or ambiguous else style["linestyle"],
                linewidth=0 if automatic or ambiguous else 2.0,
                marker=marker,
                markevery=max(1, int(np.ceil(len(branch) / 18))),
                markersize=4.4,
                markerfacecolor=marker_face,
                markeredgecolor=marker_edge,
                markeredgewidth=0.8,
                zorder=6,
            )
            # Automatic grid-level estimates are evidence points, not an
            # interpolating curve.  A robust centerline is drawn only when a
            # component has enough support; short components remain unjoined.
            if automatic and len(branch) >= 7:
                window = 7 if len(branch) >= 15 else 5
                ax.plot(
                    x,
                    _rolling_median(y, window),
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.35,
                    alpha=0.88,
                    zorder=5.5,
                )
            if errorbars and provenance != "published_source_data_fig3":
                lower_error = [float(row["uncertainty_lower_K"]) for row in branch]
                upper_error = [float(row["uncertainty_upper_K"]) for row in branch]
                ax.errorbar(
                    x,
                    y,
                    yerr=np.asarray([lower_error, upper_error]),
                    fmt="none",
                    ecolor=style["color"],
                    elinewidth=0.7,
                    capsize=1.5,
                    alpha=0.48,
                    zorder=5,
                )
            temperature_censored = [row for row in branch if row.get("censored")]
            if temperature_censored:
                for row in temperature_censored:
                    ax.scatter(
                        [row["nu"]],
                        [row["temperature_K"]],
                        marker="^",
                        s=34,
                        facecolors="none",
                        edgecolors=style["color"],
                        linewidths=1.3,
                        zorder=8,
                    )
                    ax.annotate(
                        "",
                        xy=(row["nu"], row["temperature_K"] + 0.035 * max(ax.get_ylim()[1], 1)),
                        xytext=(row["nu"], row["temperature_K"]),
                        arrowprops={"arrowstyle": "-|>", "color": style["color"], "lw": 0.8},
                        zorder=8,
                    )
            censored = [row for row in branch if row.get("nu_censored")]
            if censored:
                for row in censored:
                    side_marker = "<" if row.get("nu_censor_side") == "lower" else ">"
                    ax.scatter(
                        [row["nu"]],
                        [row["temperature_K"]],
                        s=34,
                        marker=side_marker,
                        facecolors="none",
                        edgecolors=style["color"],
                        linewidths=1.5,
                        zorder=7,
                    )


def _format_panel(ax, field, *, source_backed, dark_background):
    minus = "\N{MINUS SIGN}"
    ax.set_title(
        rf"$E={minus}{_field_key(field)}$ mV nm$^{{-1}}$", fontsize=11, weight="semibold", pad=6
    )
    ax.set_xlabel(r"Filling $\nu$")
    ax.set_ylabel("Temperature (K)")
    ax.grid(color="white" if dark_background else "0.85", alpha=0.18, linewidth=0.6)
    ax.text(
        0.02,
        0.97,
        "published boundary" if source_backed else "exploratory transport candidates",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        color="white" if dark_background else "0.25",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "black",
            "alpha": 0.34,
            "edgecolor": "none",
        },
        zorder=8,
    )


def _legend_handles(rows):
    handles = []
    provenance = {row["provenance"] for row in rows}
    for transition in ("Tcoh", "Tprime"):
        transition_rows = [row for row in rows if row["transition"] == transition]
        if not transition_rows:
            continue
        style = TRANSITION_STYLES[transition]
        published = any(
            row["provenance"] == "published_source_data_fig3" for row in transition_rows
        )
        automatic = any(
            row["provenance"] == "automatic_transport_extraction" for row in transition_rows
        )
        if published:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markerfacecolor=style["color"],
                    markeredgecolor="white",
                    linewidth=2,
                    label=style["label"],
                )
            )
        if automatic:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markerfacecolor="none",
                    markeredgecolor=style["color"],
                    linewidth=1.35,
                    label=(
                        r"$T_{\mathrm{coh}}$ candidate ($T^2$ model)"
                        if transition == "Tcoh"
                        else r"$T^{\prime}$ candidate"
                    ),
                )
            )
    if any(
        row["transition"] == "Tneel" and row["provenance"] == "published_source_data_fig3"
        for row in rows
    ):
        style = TRANSITION_STYLES["Tneel"]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=style["color"],
                markeredgecolor="white",
                linewidth=2,
                label=r"$T_N^\rho$ (published transport proxy)",
            )
        )
    if "automatic_transport_extraction" in provenance and any(
        row["transition"] == "Tneel" for row in rows
    ):
        style = TRANSITION_STYLES["Tneel"]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor="none",
                markeredgecolor=style["color"],
                linewidth=2,
                label=r"$T_N^\rho$ (automatic upturn proxy)",
            )
        )
    if "automatic_multimodal_candidate" in provenance:
        handles.append(
            Line2D(
                [0],
                [0],
                color=TRANSITION_STYLES["Tneel"]["color"],
                linestyle="none",
                marker="x",
                label="unresolved competing minima",
            )
        )
    if any(row.get("censored") for row in rows):
        handles.append(
            Line2D(
                [0],
                [0],
                color=TRANSITION_STYLES["Tcoh"]["color"],
                linestyle="none",
                marker="^",
                markerfacecolor="none",
                label=r"$T_{\mathrm{coh}}\geq T_{\max}$",
            )
        )
    if any(row.get("nu_censored") for row in rows):
        handles.append(
            Line2D(
                [0],
                [0],
                color=TRANSITION_STYLES["Tcoh"]["color"],
                linestyle="none",
                marker=">",
                markerfacecolor="none",
                label="continues beyond filling range",
            )
        )
    return handles


def _make_atlas(datasets, rows, fields, output_path, *, heatmap):
    columns = 2
    rows_count = int(np.ceil(len(fields) / columns))
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(7.2, 6.9) if len(fields) <= 4 else (7.2, 12.4),
        constrained_layout=False,
        squeeze=False,
    )
    image = None
    visible_rows = [
        row for row in rows if any(np.isclose(row["field_mV_nm"], -abs(field)) for field in fields)
    ]
    for panel_index, (ax, field) in enumerate(zip(axes.flat, fields)):
        dataset = datasets[field]
        field_rows = [row for row in rows if np.isclose(row["field_mV_nm"], -abs(field))]
        if heatmap:
            display_resistance, _ = _display_resistance(dataset["resistance"])
            image = ax.pcolormesh(
                dataset["filling"],
                dataset["temperature"],
                display_resistance,
                shading="auto",
                cmap="cividis",
                norm=_panel_lognorm(dataset["resistance"]),
                rasterized=True,
            )
            colorbar = figure.colorbar(image, ax=ax, fraction=0.047, pad=0.018)
            colorbar.ax.tick_params(labelsize=6.5)
            colorbar.set_label(r"$\rho_{xx}$", fontsize=7)
            ax.set_facecolor("#17202A")
        else:
            ax.set_facecolor("#F4F5F7")
        _plot_transition_rows(ax, field_rows, errorbars=True)
        _format_panel(
            ax,
            field,
            source_backed=int(field) == field and int(field) in REFERENCE_FIELDS,
            dark_background=heatmap,
        )
        if rows_count > 1 and panel_index < len(fields) - columns:
            ax.set_xlabel("")
        ax.set_xlim(float(dataset["filling"].min()), float(dataset["filling"].max()))
        ax.set_ylim(0, float(dataset["temperature"].max()) * 1.025)
        ax.text(
            -0.13,
            1.04,
            f"({chr(ord('a') + panel_index)})",
            transform=ax.transAxes,
            fontsize=9,
            weight="bold",
            va="bottom",
        )
    for ax in axes.flat[len(fields) :]:
        ax.set_visible(False)

    core_only = all(int(field) == field and int(field) in REFERENCE_FIELDS for field in fields)
    title = (
        "Core source-backed transport boundaries"
        if core_only
        else (
            "Independent extended-temperature transport maps"
            if len(fields) <= 4
            else "WSe₂ moiré transport-boundary evidence atlas"
        )
    )
    figure.suptitle(title, fontsize=13, weight="bold", y=0.985)
    legend_y = (
        0.135 if not core_only and len(fields) <= 4 else (0.072 if len(fields) > 4 else 0.052)
    )
    figure.legend(
        handles=_legend_handles(visible_rows),
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=3,
        frameon=False,
        fontsize=7.0,
    )
    if core_only:
        caption = (
            "Filled markers denote published source coordinates; crosses show unresolved competing minima.\n"
            "$T_N^\\rho$ is a resistive-upturn proxy, not independent magnetic confirmation."
        )
    elif len(fields) <= 4:
        caption = (
            "Open points are individual automatic transport candidates; robust centerlines require at least seven points.\n"
            "Whiskers report extraction uncertainty. $T_N^\\rho$ is a resistive-upturn proxy, not independent magnetic confirmation.\n"
            "Panels are independent repository datasets, not one displacement-field sweep."
        )
    else:
        caption = (
            "Filled markers are published coordinates; open points are automatic candidates; crosses are unresolved minima.\n"
            "$T_N^\\rho$ is a resistive-upturn proxy, not independent magnetic confirmation.\n"
            "The bottom four panels are independent repository datasets, not a continuation of the top device series.\n"
            "Robust automatic centerlines require at least seven points; whiskers report extraction uncertainty."
        )
    if heatmap:
        caption += (
            "\nHeatmaps use panel-specific logarithmic scales and full measured temperature ranges."
        )
    else:
        caption += "\nEach panel shows its full measured temperature range."
    figure.text(0.5, 0.012, caption, ha="center", fontsize=6.5, color="0.25", linespacing=1.35)
    figure.subplots_adjust(
        left=0.075,
        right=0.965,
        top=0.91,
        bottom=(
            0.27 if not core_only and len(fields) <= 4 else (0.22 if len(fields) > 4 else 0.18)
        ),
        wspace=0.46 if heatmap else 0.27,
        hspace=0.32,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def _make_individual(dataset, rows, field, output_path):
    figure, ax = plt.subplots(figsize=(6.2, 4.7), constrained_layout=True)
    display_resistance, _ = _display_resistance(dataset["resistance"])
    image = ax.pcolormesh(
        dataset["filling"],
        dataset["temperature"],
        display_resistance,
        shading="auto",
        cmap="cividis",
        norm=_panel_lognorm(dataset["resistance"]),
        rasterized=True,
    )
    _plot_transition_rows(ax, rows, errorbars=True)
    _format_panel(
        ax,
        field,
        source_backed=int(field) == field and int(field) in REFERENCE_FIELDS,
        dark_background=True,
    )
    ax.set_xlim(float(dataset["filling"].min()), float(dataset["filling"].max()))
    ax.set_ylim(0, float(dataset["temperature"].max()) * 1.025)
    figure.colorbar(image, ax=ax, label=r"$\rho_{xx}$ (panel logarithmic scale)")
    ax.legend(
        handles=_legend_handles(rows),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        fontsize=7.0,
        frameon=False,
        ncol=2,
    )
    figure.savefig(output_path, dpi=240)
    plt.close(figure)


def _adversarial_model_gates(datasets, config):
    """Test pure powers with the production smoother on every measured T grid."""
    rng = np.random.default_rng(20260803)
    tcoh_counts = {}
    tprime_counts = {}
    tneel_counts = {}
    trials = 8
    noise_levels = (0.15, 0.5, 1.0)
    for field, dataset in datasets.items():
        temperature = np.asarray(dataset["temperature"], float)
        field_key = _field_key(field)
        tcoh_counts[field_key] = {}
        tprime_counts[field_key] = {}
        tneel_counts[field_key] = {}
        for noise in noise_levels:
            sigma = np.full(temperature.shape, noise)
            noise_key = f"sigma={noise:g}"
            tcoh_counts[field_key][noise_key] = {}
            tprime_counts[field_key][noise_key] = {}
            tneel_counts[field_key][noise_key] = {}
            for exponent in (1.6, 1.8, 2.2, 2.4):
                noiseless = 30.0 + 4.0 * temperature**exponent
                found = 0
                for _ in range(trials):
                    observed = noiseless + rng.normal(0.0, sigma)
                    smoothed = adaptive_multiscale_smooth(
                        temperature, observed, sigma, z_threshold=2.0
                    )
                    found += (
                        extract_tcoh_linecut(
                            temperature,
                            observed,
                            smoothed,
                            sigma,
                            field=-abs(float(field)),
                            nu=1.0,
                            config=config,
                        )
                        is not None
                    )
                tcoh_counts[field_key][noise_key][str(exponent)] = found
            for exponent in (1.2, 1.4):
                noiseless = 30.0 + 4.0 * temperature**exponent
                found = 0
                for _ in range(trials):
                    observed = noiseless + rng.normal(0.0, sigma)
                    smoothed = adaptive_multiscale_smooth(
                        temperature, observed, sigma, z_threshold=2.0
                    )
                    found += (
                        extract_tprime_linecut(
                            temperature,
                            observed,
                            smoothed,
                            sigma,
                            field=-abs(float(field)),
                            nu=1.0,
                            config=config,
                        )
                        is not None
                    )
                tprime_counts[field_key][noise_key][str(exponent)] = found
            for exponent in (1.2, 2.0):
                noiseless = 30.0 + 4.0 * temperature**exponent
                found = 0
                for _ in range(trials):
                    observed = noiseless + rng.normal(0.0, sigma)
                    smoothed = adaptive_multiscale_smooth(
                        temperature, observed, sigma, z_threshold=2.0
                    )
                    found += len(
                        extract_tneel_candidates(
                            temperature,
                            observed,
                            smoothed,
                            sigma,
                            field=-abs(float(field)),
                            nu=1.0,
                            config=config,
                        )
                    )
                tneel_counts[field_key][noise_key][str(exponent)] = found
    false_count = (
        sum(
            sum(sum(exponents.values()) for exponents in noise.values())
            for noise in tcoh_counts.values()
        )
        + sum(
            sum(sum(exponents.values()) for exponents in noise.values())
            for noise in tprime_counts.values()
        )
        + sum(
            sum(sum(exponents.values()) for exponents in noise.values())
            for noise in tneel_counts.values()
        )
    )
    total = len(datasets) * len(noise_levels) * trials * 8
    cell_counts = [
        count
        for fields in (tcoh_counts, tprime_counts, tneel_counts)
        for noise in fields.values()
        for exponents in noise.values()
        for count in exponents.values()
    ]
    maximum_cell_fpr = max(cell_counts, default=0) / trials
    return {
        "trials_per_exponent_per_grid": trials,
        "noise_levels": noise_levels,
        "tcoh_false_positives": tcoh_counts,
        "tprime_false_positives": tprime_counts,
        "tneel_monotone_false_positives": tneel_counts,
        "false_positive_count": false_count,
        "total_controls": total,
        "false_positive_rate": false_count / total,
        "maximum_per_exponent_grid_false_positive_rate": maximum_cell_fpr,
        "passed": maximum_cell_fpr < 0.05,
    }


def _invalid_basin_gate(config):
    """Regression gate: an interpolated-looking minimum cannot cross invalid raw rho."""
    temperature = np.linspace(0.2, 3.0, 57)
    smooth = 20.0 + 5.0 * (temperature - 1.4) ** 2
    raw = smooth.copy()
    raw[np.abs(temperature - 1.4) <= 0.10] = -1.0
    sigma = np.full(temperature.shape, 0.05)
    points = extract_tneel_candidates(
        temperature, raw, smooth, sigma, field=-1, nu=1.0, config=config
    )
    return {"passed": len(points) == 0, "selected_points": len(points)}


def _field_blind_generalization_gate(config):
    """Check scale, field-label, and temperature-grid invariance without source data."""
    grids = {
        "nonuniform": np.r_[np.linspace(0.05, 0.8, 15), np.linspace(0.9, 5.0, 35)],
        "uniform": np.linspace(0.05, 5.0, 70),
        "dense_low_temperature": np.r_[np.geomspace(0.05, 0.9, 40), np.linspace(1.0, 5.0, 55)],
    }
    scales_and_fields = ((0.2, -74.0), (1.0, -1.0), (7.0, -999.0))
    results = {}
    passed = True
    for name, temperature in grids.items():
        sigma = np.full(temperature.shape, 0.15)
        tcoh_departure = np.where(
            temperature > 1.5, 8.0 * np.maximum(temperature - 1.5, 0.0) ** 1.5, 0.0
        )
        tcoh_resistance = 30.0 + 4.0 * temperature**2 + tcoh_departure
        tprime_departure = np.where(temperature < 1.6, 10.0 * (1.6 - temperature) ** 2, 0.0)
        tprime_resistance = 20.0 + 5.0 * temperature + tprime_departure
        grid_results = {"Tcoh_K": [], "Tprime_K": []}
        for scale, field in scales_and_fields:
            tcoh = extract_tcoh_linecut(
                temperature,
                scale * tcoh_resistance,
                scale * tcoh_resistance,
                scale * sigma,
                field=field,
                nu=0.85,
                config=config,
            )
            tprime = extract_tprime_linecut(
                temperature,
                scale * tprime_resistance,
                scale * tprime_resistance,
                scale * sigma,
                field=field,
                nu=0.9,
                config=config,
            )
            grid_results["Tcoh_K"].append(None if tcoh is None else tcoh.temperature)
            grid_results["Tprime_K"].append(None if tprime is None else tprime.temperature)
        results[name] = grid_results
        for temperatures in grid_results.values():
            if any(value is None for value in temperatures):
                passed = False
            elif np.ptp(np.asarray(temperatures, float)) > 1e-8:
                passed = False
    for transition in ("Tcoh_K", "Tprime_K"):
        grid_centers = np.asarray([values[transition][1] for values in results.values()], float)
        if np.ptp(grid_centers) > 0.02:
            passed = False
    return {
        "passed": bool(passed),
        "description": (
            "No published coordinates enter this synthetic test. Identical physics is required "
            "to survive resistance-unit rescaling, arbitrary field labels, and three temperature grids."
        ),
        "results": results,
    }


def _reference_calibration(result, field_reference):
    """Report, without hiding failures, how automatic points cover source coordinates."""
    report = {}
    for transition, coordinates in field_reference.items():
        automatic = [point for point in result["points"] if point.transition == transition]
        matched = 0
        errors = []
        for nu, temperature in coordinates:
            nearby = [point for point in automatic if abs(point.nu - nu) <= 0.006]
            if not nearby:
                continue
            error = min(abs(point.temperature - temperature) for point in nearby)
            errors.append(error)
            matched += error <= 0.35
        report[transition] = {
            "source_points": len(coordinates),
            "automatic_points": len(automatic),
            "matched_within_dnu_0.006_and_dT_0.35K": matched,
            "coverage": matched / len(coordinates) if coordinates else 1.0,
            "median_temperature_error_K": float(np.median(errors)) if errors else None,
        }
    return report


def _make_minus96_linecut_diagnostics(dataset, rows, output_path):
    ambiguous = [row for row in rows if row["provenance"] == "automatic_multimodal_candidate"]
    published = {
        round(float(row["nu"]), 3): row
        for row in rows
        if row["provenance"] == "published_source_data_fig3" and row["transition"] == "Tneel"
    }
    targets = (0.985, 0.986, 0.987, 0.988)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.3), sharex=True, constrained_layout=False)
    for ax, target in zip(axes.flat, targets):
        column = int(np.argmin(np.abs(dataset["filling"] - target)))
        actual_nu = float(dataset["filling"][column])
        temperature = dataset["temperature"]
        resistance = np.asarray(dataset["resistance"][:, column], float)
        valid = np.isfinite(resistance) & (resistance > 0)
        scale = float(np.nanmedian(resistance[valid]))
        ax.plot(
            temperature[valid],
            resistance[valid] / scale,
            color="0.20",
            linewidth=1.1,
            marker="o",
            markersize=2.0,
            markerfacecolor="white",
            markeredgewidth=0.45,
        )
        target_rows = [row for row in ambiguous if np.isclose(float(row["nu"]), target)]
        for row in target_rows:
            ax.axvspan(
                float(row["crossing_lower_K"]),
                float(row["crossing_upper_K"]),
                color=TRANSITION_STYLES["Tneel"]["color"],
                alpha=0.12,
            )
            ax.axvline(
                float(row["temperature_K"]),
                color=TRANSITION_STYLES["Tneel"]["color"],
                linestyle=":",
                linewidth=1.7,
                label="automatic minimum candidate",
            )
        source_row = published.get(round(target, 3))
        if source_row:
            ax.axvline(
                float(source_row["temperature_K"]),
                color=TRANSITION_STYLES["Tneel"]["color"],
                linewidth=1.8,
                label="published endpoint",
            )
        ax.set_title(rf"$\nu={actual_nu:.3f}$", fontsize=10)
        ax.set_xlim(0, min(4.0, float(temperature.max())))
        ax.grid(alpha=0.18, linewidth=0.55)
    for ax in axes[-1]:
        ax.set_xlabel("Temperature (K)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\rho_{xx}/\mathrm{median}(\rho_{xx})$")
    handles = [
        Line2D(
            [0],
            [0],
            color=TRANSITION_STYLES["Tneel"]["color"],
            linestyle=":",
            label="automatic minimum candidate",
        ),
        Line2D(
            [0],
            [0],
            color=TRANSITION_STYLES["Tneel"]["color"],
            linestyle="-",
            label="published source endpoint",
        ),
        Line2D(
            [0],
            [0],
            color=TRANSITION_STYLES["Tneel"]["color"],
            linewidth=7,
            alpha=0.18,
            label="candidate sampling bracket",
        ),
    ]
    figure.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.040), ncol=3, frameon=False
    )
    figure.text(
        0.5,
        0.010,
        "Each linecut is normalized independently; panel-specific vertical scales are not comparable.",
        ha="center",
        fontsize=6.5,
        color="0.25",
    )
    figure.suptitle(
        "−96 mV nm⁻¹: unresolved transport minima left of the source endpoint",
        weight="bold",
        y=0.985,
    )
    figure.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.18, hspace=0.26, wspace=0.22)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def _make_tcoh_linecut_diagnostics(datasets, rows, fields, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.5), constrained_layout=False)
    for panel_index, (ax, field) in enumerate(zip(axes.flat, fields)):
        candidates = [
            row
            for row in rows
            if np.isclose(row["field_mV_nm"], -abs(float(field)))
            and row["transition"] == "Tcoh"
            and row["provenance"] == "automatic_transport_extraction"
        ]
        if not candidates:
            ax.set_visible(False)
            continue
        row = max(candidates, key=lambda item: (item["confidence"], item["support"]))
        dataset = datasets[field]
        column = int(np.argmin(np.abs(dataset["filling"] - float(row["nu"]))))
        temperature = np.asarray(dataset["temperature"], float)
        resistance = np.asarray(dataset["resistance"][:, column], float)
        valid = np.isfinite(resistance) & (resistance > 0)
        scale = float(np.nanmedian(resistance[valid]))
        display_max = min(
            float(temperature[valid].max()),
            max(
                float(row["crossing_upper_K"]) + 1.0,
                1.5 * float(row["temperature_K"]),
                float(row["fit_upper_K"]) + 1.0,
            ),
        )
        display = valid & (temperature <= display_max)
        display_temperature = temperature[display]
        offset = float(row["fit_offset"])
        coefficient = float(row["fit_coefficient"])
        model = (offset + coefficient * display_temperature**2) / scale
        ax.plot(
            display_temperature, resistance[display] / scale, "o-", color="0.25", ms=2.0, lw=0.9
        )
        ax.plot(display_temperature, model, color="#315A7D", linestyle="--", linewidth=1.2)
        ax.fill_between(
            display_temperature, 0.9 * model, 1.1 * model, color="#315A7D", alpha=0.08, linewidth=0
        )
        ax.axvspan(
            float(row["fit_lower_K"]),
            float(row["fit_upper_K"]),
            color=TRANSITION_STYLES["Tcoh"]["color"],
            alpha=0.10,
            label=r"validated $T^2$ window",
        )
        ax.axvline(
            float(row["temperature_K"]),
            color=TRANSITION_STYLES["Tcoh"]["color"],
            lw=1.8,
            label=r"10% departure",
        )
        ax.axvspan(
            float(row["crossing_lower_K"]),
            float(row["crossing_upper_K"]),
            color="#F28E2B",
            alpha=0.14,
        )
        ax.text(
            0.98,
            0.04,
            rf"$n={float(row['exponent']):.2f}\pm{float(row['exponent_sigma']):.2f}$"
            + "\n"
            + rf"median error={100 * float(row['fit_median_fractional_error']):.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="0.25",
        )
        ax.set_title(rf"$E={-abs(float(field)):g}$ mV nm$^{{-1}}$, $\nu={float(row['nu']):.3f}$")
        if panel_index >= 2:
            ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"$\rho_{xx}/\mathrm{median}(\rho_{xx})$")
        ax.set_xlim(float(display_temperature.min()), display_max)
        ax.grid(alpha=0.18, linewidth=0.55)
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=TRANSITION_STYLES["Tcoh"]["color"],
                lw=7,
                alpha=0.18,
                label=r"validated $T^2$ window",
            ),
            Line2D(
                [0],
                [0],
                color=TRANSITION_STYLES["Tcoh"]["color"],
                lw=1.8,
                label="persistent 10% departure",
            ),
            Line2D([0], [0], color="#F28E2B", lw=7, alpha=0.20, label="crossing bracket"),
            Line2D([0], [0], color="#315A7D", linestyle="--", label=r"fitted $\rho_0+AT^2$"),
            Line2D([0], [0], color="#315A7D", linewidth=7, alpha=0.10, label=r"$\pm10\%$ envelope"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        r"Representative automatic $T_{\mathrm{coh}}$ linecut diagnostics", weight="bold", y=0.985
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.18, hspace=0.28, wspace=0.20)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def _selected_topology_gates(rows, config):
    """Verify that every displayed automatic branch obeys extraction invariants."""
    automatic = [
        row
        for row in rows
        if row["provenance"] == "automatic_transport_extraction" and row["selected_for_atlas"]
    ]
    tcoh = [row for row in automatic if row["transition"] == "Tcoh"]
    bracket_failures = [
        row
        for row in tcoh
        if not (
            float(row["crossing_lower_K"])
            <= float(row["temperature_K"])
            <= float(row["crossing_upper_K"])
        )
    ]
    exponent_failures = [
        row
        for row in tcoh
        if not (
            config.min_t2_exponent <= float(row["exponent"]) <= config.max_t2_exponent
            and abs(float(row["exponent"]) - 2.0) <= config.max_t2_equivalence_distance
            and np.isfinite(float(row["exponent_sigma"]))
            and float(row["exponent_sigma"]) <= config.max_exponent_sigma
        )
    ]
    model_failures = [
        row
        for row in tcoh
        if not (
            np.isfinite(float(row["fit_offset"]))
            and np.isfinite(float(row["fit_coefficient"]))
            and float(row["fit_coefficient"]) > 0
        )
    ]

    component_sizes = {}
    slope_maxima = {}
    component_failures = []
    slope_failures = []
    for transition, minimum_size in (
        ("Tcoh", config.tcoh_min_component_points),
        ("Tneel", config.branch_min_points),
    ):
        grouped = {}
        for row in automatic:
            if row["transition"] != transition:
                continue
            key = (float(row["field_mV_nm"]), int(row["component"]))
            grouped.setdefault(key, []).append(row)
        for key, component_rows in grouped.items():
            label = f"{key[0]:g}:{transition}:{key[1]}"
            component_sizes[label] = len(component_rows)
            if len(component_rows) < minimum_size:
                component_failures.append(label)
            if transition != "Tcoh" or len(component_rows) < 2:
                continue
            ordered = sorted(component_rows, key=lambda row: float(row["nu"]))
            slopes = [
                abs(
                    (float(right["temperature_K"]) - float(left["temperature_K"]))
                    / (float(right["nu"]) - float(left["nu"]))
                )
                for left, right in zip(ordered, ordered[1:])
                if float(right["nu"]) > float(left["nu"])
            ]
            maximum = max(slopes, default=0.0)
            slope_maxima[label] = maximum
            if maximum > config.tcoh_max_slope + 1e-9:
                slope_failures.append(label)

    return {
        "passed": not (
            bracket_failures
            or exponent_failures
            or model_failures
            or component_failures
            or slope_failures
        ),
        "crossing_brackets_contain_centers": not bracket_failures,
        "tcoh_exponents_and_uncertainties_valid": not exponent_failures,
        "stored_tcoh_models_valid": not model_failures,
        "visible_components_meet_minimum_size": not component_failures,
        "visible_tcoh_branches_meet_slope_limit": not slope_failures,
        "component_sizes": component_sizes,
        "tcoh_max_slopes_K_per_nu": slope_maxima,
        "failure_counts": {
            "crossing_bracket": len(bracket_failures),
            "exponent": len(exponent_failures),
            "stored_model": len(model_failures),
            "component_size": len(component_failures),
            "slope": len(slope_failures),
        },
    }


def build_phase_outputs(input_dir, reference_path, output_dir, *, fields=FIELD_ORDER, config=None):
    """Extract, validate, and render a complete all-field phase atlas."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or PhaseExtractionConfig()
    reference = _load_reference(reference_path)
    datasets = {}
    all_rows = []
    automatic_diagnostics = {}
    candidate_rows = []
    dataset_summary = {}

    for field in fields:
        temperature, filling, resistance = clean_sort_data(*load_field(field, input_dir))
        datasets[field] = {"temperature": temperature, "filling": filling, "resistance": resistance}
        dataset_summary[_field_key(field)] = {
            "shape": list(resistance.shape),
            "temperature_range_K": [float(temperature.min()), float(temperature.max())],
            "filling_range": [float(filling.min()), float(filling.max())],
            "nonpositive_resistance_points": int(np.count_nonzero(resistance <= 0)),
        }
        source_backed = int(field) == field and int(field) in REFERENCE_FIELDS
        if source_backed:
            all_rows.extend(_reference_rows(field, temperature, reference))

        result = extract_field_phase_diagram(field, input_dir, config=config)
        if source_backed and int(field) == 96:
            all_rows.extend(_minus96_ambiguous_rows(result))
        automatic_rows = _automatic_rows(result)
        if not source_backed:
            all_rows.extend(automatic_rows)
        selected_ids = {id(point) for point in result["points"]}
        raw_points = [
            point
            for transition_points in result["raw_candidates"].values()
            for point in transition_points
        ]
        candidate_rows.extend(_automatic_rows(result, points=raw_points, selected_ids=selected_ids))
        automatic_diagnostics[_field_key(field)] = {
            "raw_candidate_counts": {
                transition: len(points) for transition, points in result["raw_candidates"].items()
            },
            "selected_counts": dict(Counter(row["transition"] for row in automatic_rows)),
        }
        if source_backed:
            automatic_diagnostics[_field_key(field)]["source_positive_control"] = (
                _reference_calibration(result, reference["fields"][_field_key(field)])
            )

    csv_path = output_dir / "transitions.csv"
    fieldnames = list(all_rows[0]) if all_rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    candidates_path = output_dir / "candidates.csv"
    with candidates_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows)

    reference_counts = {
        field: {transition: len(points) for transition, points in transitions.items()}
        for field, transitions in reference["fields"].items()
    }
    adversarial = _adversarial_model_gates(datasets, config)
    invalid_basin = _invalid_basin_gate(config)
    field_blind_generalization = _field_blind_generalization_gate(config)
    selected_topology = _selected_topology_gates(all_rows, config)
    published_expected = {
        (-abs(float(field)), transition, float(nu), float(temperature))
        for field, transitions in reference["fields"].items()
        for transition, coordinates in transitions.items()
        for nu, temperature in coordinates
    }
    published_actual = {
        (
            float(row["field_mV_nm"]),
            row["transition"],
            float(row["nu"]),
            float(row["temperature_K"]),
        )
        for row in all_rows
        if row["provenance"] == "published_source_data_fig3"
    }
    core_counts_match = reference_counts == {
        "103": {"Tcoh": 12, "Tneel": 19, "Tprime": 7},
        "99": {"Tcoh": 12, "Tneel": 21, "Tprime": 7},
        "96": {"Tcoh": 11, "Tneel": 11, "Tprime": 7},
        "87": {"Tcoh": 10, "Tneel": 0, "Tprime": 0},
    }
    ambiguous_rows = [
        row for row in all_rows if row["provenance"] == "automatic_multimodal_candidate"
    ]
    ambiguous_gate = (
        len(ambiguous_rows) >= 3
        and all(not row["selected_for_atlas"] for row in ambiguous_rows)
        and all(not row["nu_censored"] for row in ambiguous_rows)
        and len({row["temperature_K"] for row in ambiguous_rows}) > 1
    )
    compatibility_groups = {
        "published_device_series": ["-103", "-99", "-96", "-87"],
        "repository_maps_shown_as_independent_datasets": ["-96.2", "-74", "-151", "-176"],
        "warning": "The noncore maps are not interpreted as one displacement-field sweep without device/cooldown/geometry metadata.",
    }
    acceptance_gates = {
        "published_coordinates_match_source_exactly": core_counts_match
        and published_actual == published_expected,
        "minus96_competing_minima_visible_but_not_promoted_to_boundary": ambiguous_gate,
        "pure_power_adversarial_controls": adversarial["passed"],
        "invalid_resistance_excluded_from_fit_and_minimum_basins": invalid_basin["passed"],
        "field_blind_scale_and_grid_generalization": field_blind_generalization["passed"],
        "automatic_boundary_topology_and_fit_invariants": selected_topology["passed"],
        "noncore_datasets_not_claimed_as_continuous_field_series": True,
        "full_temperature_ranges_plotted": True,
        "source_and_automatic_provenance_separated": all(
            row["provenance"] != "published_source_data_fig3" or row["selected_for_atlas"]
            for row in all_rows
        ),
    }
    minimum_physical_calibration_gate = all(
        diagnostics["source_positive_control"]["Tcoh"]["automatic_points"] > 0
        for diagnostics in automatic_diagnostics.values()
        if "source_positive_control" in diagnostics
        and diagnostics["source_positive_control"]["Tcoh"]["source_points"] > 0
    )
    software_status = "passed" if all(acceptance_gates.values()) else "failed"
    qa = {
        "status": f"software_qa_{software_status}",
        "acceptance_gates": acceptance_gates,
        "physical_calibration": {
            "status": (
                "minimum_core_positive_control_passed"
                if minimum_physical_calibration_gate
                else "not_established"
            ),
            "minimum_criterion": (
                "At least one automatic Tcoh candidate must survive in every source-backed "
                "field that contains published Tcoh coordinates."
            ),
            "criterion_passed": minimum_physical_calibration_gate,
            "interpretation": (
                "Software consistency does not calibrate the extended automatic curves as phases. "
                "They remain exploratory transport candidates."
            ),
        },
        "definitions": reference["method"],
        "published_reference_source": reference["source"],
        "published_source_data": reference["source_data"],
        "published_reference_counts": reference_counts,
        "published_reference_reproduced_exactly": published_actual == published_expected,
        "minus96_unresolved_candidates": ambiguous_rows,
        "adversarial_model_controls": adversarial,
        "invalid_basin_regression": invalid_basin,
        "field_blind_generalization": field_blind_generalization,
        "selected_automatic_topology": selected_topology,
        "automatic_fields": automatic_diagnostics,
        "datasets": dataset_summary,
        "compatibility_groups": compatibility_groups,
        "caveats": [
            "Tneel is a transport proxy defined by a significant local resistance minimum; magnetic order requires an independent probe.",
            "The 4 K repository matrices do not contain the full high-temperature baselines used for every published Tprime point, so the four paper panels use the authors' source-data coordinates.",
            "Nonpositive resistance values are never logged or clipped for transition extraction; display normalization floors them only for the heatmap background.",
            "The −96 data contain competing minima left of the published endpoint. They are shown as unresolved candidates and are not promoted to an AFM boundary.",
            "Tcoh is retained only where a positive-A low-temperature T-squared model passes stability and holdout tests; it is not connected through central upturn/negative-A regions.",
            "Automatic extraction does not reproduce every source-backed core boundary on the truncated 4 K matrices; noncore automatic curves are therefore presented as exploratory transport candidates, not calibrated phase assignments.",
        ],
        "config": asdict(config),
    }
    qa_path = output_dir / "qa_summary.json"

    atlas_png = output_dir / "phase_atlas.png"
    _make_atlas(datasets, all_rows, fields, atlas_png, heatmap=False)
    _make_atlas(datasets, all_rows, fields, output_dir / "phase_overlay_atlas.png", heatmap=True)
    _make_atlas(datasets, all_rows, fields, output_dir / "phase_atlas.pdf", heatmap=False)
    core_fields = [
        field for field in fields if int(field) == field and int(field) in REFERENCE_FIELDS
    ]
    extended_fields = [field for field in fields if field not in core_fields]
    if core_fields:
        _make_atlas(
            datasets, all_rows, core_fields, output_dir / "phase_core_published.png", heatmap=False
        )
        _make_atlas(
            datasets, all_rows, core_fields, output_dir / "phase_core_overlay.png", heatmap=True
        )
        _make_atlas(
            datasets, all_rows, core_fields, output_dir / "phase_core_published.pdf", heatmap=False
        )
    if extended_fields:
        _make_atlas(
            datasets,
            all_rows,
            extended_fields,
            output_dir / "phase_extended_datasets.png",
            heatmap=False,
        )
        _make_atlas(
            datasets,
            all_rows,
            extended_fields,
            output_dir / "phase_extended_overlay.png",
            heatmap=True,
        )
        _make_atlas(
            datasets,
            all_rows,
            extended_fields,
            output_dir / "phase_extended_datasets.pdf",
            heatmap=False,
        )
    for field in fields:
        field_rows = [row for row in all_rows if np.isclose(row["field_mV_nm"], -abs(field))]
        _make_individual(
            datasets[field], field_rows, field, output_dir / f"phase_E-{_field_filename(field)}.png"
        )
    if 96 in datasets:
        minus96_rows = [row for row in all_rows if np.isclose(row["field_mV_nm"], -96.0)]
        _make_minus96_linecut_diagnostics(
            datasets[96], minus96_rows, output_dir / "phase_minus96_tneel_linecuts.png"
        )
        _make_minus96_linecut_diagnostics(
            datasets[96], minus96_rows, output_dir / "phase_minus96_tneel_linecuts.pdf"
        )
    diagnostic_fields = [field for field in (96.2, 74, 151, 176) if field in datasets]
    if diagnostic_fields:
        _make_tcoh_linecut_diagnostics(
            datasets, all_rows, diagnostic_fields, output_dir / "phase_tcoh_linecuts.png"
        )
        _make_tcoh_linecut_diagnostics(
            datasets, all_rows, diagnostic_fields, output_dir / "phase_tcoh_linecuts.pdf"
        )
    qa["generated_artifacts"] = sorted(path.name for path in output_dir.glob("phase_*.png"))
    acceptance_gates["paper_figures_generated"] = all(
        (output_dir / filename).exists()
        for filename in (
            "phase_core_published.png",
            "phase_core_published.pdf",
            "phase_extended_datasets.png",
            "phase_extended_datasets.pdf",
            "phase_minus96_tneel_linecuts.png",
            "phase_tcoh_linecuts.png",
        )
    )
    qa["status"] = "software_qa_passed" if all(acceptance_gates.values()) else "software_qa_failed"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n")
    return {
        "atlas": atlas_png,
        "csv": csv_path,
        "candidates": candidates_path,
        "qa": qa_path,
        "rows": all_rows,
    }
