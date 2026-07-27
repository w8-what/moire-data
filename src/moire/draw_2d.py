import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

from pathlib import Path
import numpy as np

from moire.plot_styles import (
    BEHAVIOR_COLORS,
    DEFAULT_BEHAVIOR_COLOR,
    FEATURE_LEGEND_STYLE,
    get_feature_style,
)


def draw_heatmap(
    fig,
    ax,
    col,
    row,
    data,
    title="heatmap",
    xlabel="Filling v",
    ylabel="Temperature T (K)",
    cbar_label="Resistivity",
):

    # Log rounded vmin & vmax
    vmin_raw, vmax_raw = np.nanpercentile(data[data > 0], [1, 99])

    emin = int(np.floor(np.log10(vmin_raw)))
    emax = int(np.ceil(np.log10(vmax_raw)))

    vmin = 10**emin
    vmax = 10**emax

    im = ax.pcolormesh(
        col, row, data, cmap="bwr", shading="nearest", norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    tick_exps = np.arange(emin, emax + 1)
    ticks = 10**tick_exps

    # Drawing colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", location="right", pad=0.03)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())

    cbar.set_label(cbar_label, rotation=90)
    cbar.set_ticks(ticks)
    cbar.set_label(r"$\rho_{xx}$ ($\Omega$)")

    # Axis titles and labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()


def overlay_features_heatmap(
    ax, linecuts, feature_name="features", score_name="confidence", filter=0
):

    used_labels = set()

    for linecut in linecuts:
        features = linecut.get(feature_name)
        nu = linecut.get("nu")

        for feat in features:

            feature_type = feat.get("type")
            T_transition = feat.get("T")
            score = feat.get(score_name)
            style = get_feature_style(feature_type)

            if style is None or score < filter:
                continue

            label = style["label"]

            if label in used_labels:
                style["label"] = None
            else:
                used_labels.add(label)

            ax.scatter(nu, T_transition, alpha=score, **style)

    if used_labels:
        legend = ax.legend(**FEATURE_LEGEND_STYLE)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)


def overlay_behaviors_heatmap(ax, linecuts, drawn_behaviors=["extraction_range"], alpha=0.2):
    # for each linecut, draw the interval and shade it with alpha = something like 0.2
    # do this for each drawn behavior

    nus = np.array([linecut["nu"] for linecut in linecuts])

    # Boundaries halfway between neighboring filling values
    edges = np.empty(len(nus) + 1)
    edges[1:-1] = (nus[:-1] + nus[1:]) / 2
    edges[0] = nus[0] - (nus[1] - nus[0]) / 2
    edges[-1] = nus[-1] + (nus[-1] - nus[-2]) / 2

    for i, linecut in enumerate(linecuts):
        behaviors = linecut.get("behaviors", {})

        for behavior in behaviors:

            if behavior.get("type") not in drawn_behaviors:
                continue

            T_lower, T_upper = behavior.get("T_upper"), behavior.get("T_lower")

            ax.fill_betweenx(
                [T_lower, T_upper],
                edges[i],
                edges[i + 1],
                color=BEHAVIOR_COLORS.get(behavior.get("type"), DEFAULT_BEHAVIOR_COLOR),
                alpha=alpha,
                linewidth=0,
            )

    return ax


# def draw_mosaic_diagrams(col, row, data, OUT=None, name="mosaic_phase_diagram", save=False):
#     phase_to_id = {"Unknown": 0, "AFM": 1, "Metal": 2, "Insulator": 3}
#     colors = ["silver", "maroon", "steelblue", "beige"]

#     phase = np.zeros(data.shape, dtype=int)
#     candidates_all = []

#     for j, filling in enumerate(col):
#         linecut = data[:, j]

#         candidates = extract_upturns(row, linecut)
#         # candidates = extract_metallic_transitions(row, linecut, candidates)
#         candidates = sorted(candidates, key=lambda c: c["T"])

#         candidates_all.append(candidates)

#         for k in range(len(candidates) - 1):
#             phase_name = candidates[k].get("phase_right")
#             T0 = candidates[k]["T"]
#             T1 = candidates[k + 1]["T"]

#             if phase_name is not None:
#                 mask = (row >= T0) & (row < T1)
#                 phase[mask, j] = phase_to_id.get(phase_name, 0)

#         if candidates:
#             phase[row >= candidates[-1]["T"], j] = phase_to_id.get(
#                 candidates[-1].get("phase_left"), 0
#             )

#     cmap = ListedColormap(colors)
#     norm = BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)

#     fig, ax = plt.subplots(figsize=(8, 6))

#     ax.pcolormesh(
#         col,
#         row,
#         phase,
#         cmap=cmap,
#         norm=norm,
#         shading="nearest",
#     )

#     ax.set_xlabel(r"Filling $\nu$")
#     ax.set_ylabel(r"Temperature $T$ (K)")
#     ax.set_title(name)

#     labels = ["Unknown", "AFM", "Metal", "Insulator"]
#     ax.legend(
#         handles=[Patch(facecolor=c, label=p) for p, c in zip(labels, colors)]
#     )

#     fig.tight_layout()

#     if save:
#         OUT = Path(OUT)
#         OUT.mkdir(parents=True, exist_ok=True)
#         fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")

#     return fig, ax, phase, candidates_all
