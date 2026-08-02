"""Shared style configuration for 1D and 2D plots."""

BEHAVIOR_COLORS = {
    "extraction_range": "purple",
    "linear": "white",
    "superlinear": "skyblue",
    "sublinear": "grey",
    "unlabeled" : "orange"
}

DEFAULT_LINE_PLOT_KWARGS = {
    "color": "blue",
    "linewidth": 1.0,
    "marker": "o",
    "markersize": 3,
    "markerfacecolor": "none",
    "markeredgecolor": "navy",
}

FEATURE_STYLES = {
    "upturn": {"color": "yellow", "marker": "^", "label": "upturn"},
    "downturn": {"color": "green", "marker": "v", "label": "downturn"},
    "Tc": {"color": "navy", "marker": "o", "label": "Tc"},
    "Tcoh": {"color": "blue", "marker": "o", "label": "Tcoh"}
}

FEATURE_SCATTER_STYLE = {"s": 35, "edgecolor": "black", "linewidth": 0.4, "zorder": 5}

FEATURE_LEGEND_STYLE = {"frameon": True, "framealpha": 1.0, "facecolor": "white", "fontsize": 8}

BEHAVIOR_LEGEND_STYLE = {""}


def get_feature_style(feature_type):
    """Return a new combined style dictionary for a feature type."""

    if feature_type not in FEATURE_STYLES:
        return None

    return {**FEATURE_SCATTER_STYLE, **FEATURE_STYLES[feature_type]}
