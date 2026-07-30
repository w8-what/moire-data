import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from moire.plot_styles import (
    BEHAVIOR_COLORS,
    DEFAULT_BEHAVIOR_COLOR,
    DEFAULT_LINE_PLOT_KWARGS,
    FEATURE_LEGEND_STYLE,
    get_feature_style,
)
from moire.io import fmt4


def generate_layout(numAxes, title="title"):

    layouts = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3)}

    nrows, ncols = layouts[numAxes]

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(7.5 * ncols, 6 * nrows), dpi=250)
    axes = axes.flatten()

    # Removes the bottom-right subplot for five plots
    for ax in axes[numAxes:]:
        ax.remove()

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_general_line(
    ax,
    x,
    y,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    ylim=None,
    shaded=False,
    error=None,
    fill_alpha=0.2,
    **plot_kwargs,
):
    """Draw and configure a line on an existing axis."""

    x = np.asarray(x)
    y = np.asarray(y)

    style = {**DEFAULT_LINE_PLOT_KWARGS, **plot_kwargs}
    ax.plot(x, y, **style)

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim)

    if shaded:
        ax.fill_between(x, 0, y, alpha=fill_alpha)

    if error is not None:
        ax.fill_between(x, y - error, y + error, alpha=fill_alpha)

    return ax


def overlay_features(ax, linecut, feature_name = "features", score_name = "confidence"):

    T = linecut.get("T")
    rho_smoothed = linecut.get("rho_smoothed")
    features = linecut.get(feature_name) or []
    used_labels = set()

    for feature in features:
        style = get_feature_style(feature.get("type"))
        if style is None:
            continue

        label = style["label"]
        if label in used_labels:
            style["label"] = None
        else:
            used_labels.add(label)

        T_feature = feature.get("T")
        rho_at_T = rho_smoothed[np.argmin(np.abs(T - T_feature))]

        conf = feature.get(score_name)
        conf_label = f"conf={float(conf):.4g}"

        ax.scatter(T_feature, rho_at_T, alpha=conf, **style)
        ax.axvline(T_feature, linewidth=1, linestyle="--", color="grey", zorder=3)

        max_rho = np.max(rho_smoothed)
        top_half = rho_at_T > (max_rho / 2)
        y_text = 0.8 * max_rho if top_half else 0.2 * max_rho
        ax.annotate(
            conf_label,
            xy=(T_feature, rho_at_T),
            xytext=(T_feature, y_text),
            bbox=dict(boxstyle="round", fc="0.8", alpha=0.8),
            arrowprops=dict(
                arrowstyle="->",
                shrinkA=0,
                shrinkB=10,
                connectionstyle="angle,angleA=0,angleB=90,rad=10",
                alpha=0.8,
            ),
        )

    if used_labels:
        legend = ax.legend(**FEATURE_LEGEND_STYLE)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)

    return ax


def overlay_behaviors(ax, linecut, drawn_behaviors = ["linear", "sublinear", "superlinear"]):
    """Shade each extracted behavior's temperature range on a line plot.

    A behavior's confidence is used directly as its opacity, up to a maximum
    alpha of 0.5. Behaviors without a usable confidence use an alpha of 0.2.
    """

    T = linecut.get("T")
    behaviors = linecut.get("behaviors") or []

    if T is None or len(T) == 0:
        return ax

    for behavior in behaviors:

        if behavior["type"] not in drawn_behaviors:
            continue 

        T_lower = behavior.get("T_lower")
        T_upper = behavior.get("T_upper")

        if T_lower is None or T_upper is None:
            continue

        confidence = behavior.get("confidence")
        try:
            alpha = float(confidence) if confidence is not None else 0.2
        except (TypeError, ValueError):
            alpha = 0.2

        if not np.isfinite(alpha):
            alpha = 0.2
        alpha = np.clip(alpha, 0.0, 0.5)

        behavior_type = behavior.get("type")
        color = BEHAVIOR_COLORS.get(behavior_type, DEFAULT_BEHAVIOR_COLOR)
        ax.axvspan(T_lower, T_upper, color=color, alpha=alpha, linewidth=0)

    return ax


def plot_linecut(T: list, linecut, OUT):
    """Plot the raw linecut, smoothed data, and first two derivatives."""

    param_string = "  ".join(
        f"{k} = {fmt4(v)}" for k, v in linecut.items() if k == "E" or k == "nu"
    )

    rho = linecut.get("rho")
    rho_smoothed = linecut.get("rho_smoothed")
    dpdT = np.gradient(rho_smoothed, T)
    d2pdT2 = np.gradient(dpdT, T)

    fig, axes = generate_layout(4, title=param_string)
    linecut_axis_kwargs = {
        "xlabel": "Temperature (K)",
        "ylabel": "Resistivity (Ω*cm)",
        "xlim": (0, None),
        "ylim": (0, None),
    }

    plot_general_line(axes[0], T, rho, title="Raw Data", **linecut_axis_kwargs)
    plot_general_line(axes[1], T, rho_smoothed, title="Smoothed Data, Features, Behaviors", **linecut_axis_kwargs)
    plot_general_line(axes[2], T, dpdT, title="First Derivative", shaded=True, fill_alpha=0.5)
    plot_general_line(axes[3], T, d2pdT2, title="Second Derivative", shaded=True, fill_alpha=0.5)

    overlay_features(axes[1], linecut, score_name="score_15", feature_name="features_new")
    overlay_behaviors(axes[1], linecut)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / Path(param_string + ".png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return fig, axes

