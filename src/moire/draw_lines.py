import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from moire.PLOTS_CONFIG import (
    BEHAVIOR_COLORS,
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


def plot_line_general(
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


def overlay_features(ax, linecut, drawn_features = "features", 
                     drawn_types =["upturn", "downturn", "Tcoh", "T\'"], 
                     score_name = "confidence", filter = 0):

    T = linecut.get("T")
    rho_smoothed = linecut.get("rho_smoothed")
    features = linecut.get(drawn_features) or []
    used_labels = set()

    # Draw each feature on the linecut
    for feature in features:

        if feature["type"] not in drawn_types:
            continue

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

        if conf < filter:
            continue

        ax.scatter(T_feature, rho_at_T, alpha=conf, **style)
        ax.axvline(T_feature, linewidth=1, linestyle="--", color="grey", zorder=3)

        ymin, ymax = ax.get_ylim()
        top_half = rho_at_T > ((ymax + ymin) / 2)
        y_text = 0.8 * (ymax-ymin) if top_half else 0.2 * (ymax-ymin)
        y_text += ymin
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


def overlay_behaviors(ax, linecut, drawn_behaviors = "behaviors", drawn_types = ["linear", "sublinear", "superlinear"]):
    """Shade each extracted behavior's temperature range on a line plot.

    A behavior's confidence is used directly as its opacity, up to a maximum
    alpha of 0.5. Behaviors without a usable confidence use an alpha of 0.2.
    """

    T = linecut.get("T")
    behaviors = linecut.get(drawn_behaviors) or []

    if T is None or len(T) == 0:
        return ax

    used_labels = set()

    for behavior in behaviors:
        if behavior["type"] not in drawn_types:
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
        color = BEHAVIOR_COLORS.get(behavior_type)
        label = behavior_type if behavior_type not in used_labels else None
        used_labels.add(behavior_type)
        ax.axvspan(
            T_lower,
            T_upper,
            color=color,
            alpha=alpha,
            linewidth=0,
            label=label,
        )

    if used_labels:
        legend = ax.legend(**FEATURE_LEGEND_STYLE)
        for handle, text in zip(legend.legend_handles, legend.get_texts()):
            handle.set_alpha(1.0)
            if text.get_text() in used_labels:
                handle.set_edgecolor("black")
                handle.set_linewidth(0.5)

    return ax


def plot_line_default(T, linecut):
    """Plot the raw linecut, smoothed data, and first two derivatives."""

    param_string = "  ".join(
        f"{k} = {fmt4(v)}" for k, v in linecut.items() if k == "E" or k == "nu"
    )

    rho = linecut["rho"]
    rho_smoothed = linecut["rho_smoothed"]
    fit = linecut["exponent_fit"]
    dpdT = np.gradient(rho_smoothed, T)
    noise = linecut["local_noise"]

    fig, axes = generate_layout(4, title=param_string)
    linecut_axis_kwargs = {
        "xlabel": "Temperature (K)",
        "ylabel": "Resistivity (Ω*cm)",
        "xlim": (0, None),
        "ylim": (0, None),
    }

    plot_line_general(axes[0], T, rho, error=noise, title="Raw Data", **linecut_axis_kwargs)
    plot_line_general(axes[1], T, rho_smoothed, error=noise, title=
                      "Smoothed Data, Features, Behaviors", **linecut_axis_kwargs)
    plot_line_general(axes[2], T, dpdT, title="First Derivative", shaded=True, fill_alpha=0.5)
    plot_line_general(axes[3], T, fit, title="Second Derivative", shaded=True, fill_alpha=0.5)

    overlay_features(axes[1], linecut, score_name="score_15", drawn_features="features_rescored")
    overlay_features(axes[1], linecut, drawn_features="features", drawn_types= ["Tcoh, T\'"])
    overlay_behaviors(axes[1], linecut)

    fig.tight_layout()

    return fig, axes
