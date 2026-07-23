"""Iterative, neighborhood-based confidence score updates."""

import math

import numpy as np


VALID_UPDATE_MODES = ("original_anchor", "recursive")


def _temperature_index(temperatures, value):
    return int(np.argmin(np.abs(temperatures - value)))


def _sigmoid(value):
    """Numerically stable logistic sigmoid."""
    if value >= 0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def _support_scores(
    linecuts,
    score_name,
    n_hood=3,
    tau=3.0,
    sigmoid_support=False,
    sigmoid_center=0.6,
    sigmoid_width=0.1,
):
    """Return synchronous support scores for every feature.

    Neighboring features only support features of the same type.  The strongest
    match on each side is attenuated by its separation along the temperature
    axis, then the two sides are combined with a geometric mean.
    """
    if n_hood < 1:
        raise ValueError("n_hood must be at least 1")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if sigmoid_width <= 0:
        raise ValueError("sigmoid_width must be positive")

    scores = []
    for i, linecut in enumerate(linecuts):
        temperatures = np.asarray(linecut["T"])
        features = linecut.get("features", [])
        left_hood = linecuts[max(0, i - n_hood) : i]
        right_hood = linecuts[i + 1 : i + n_hood + 1]

        for feature in features:
            feature_type = feature.get("type")
            feature_index = _temperature_index(temperatures, feature["T"])

            def strongest(neighborhood):
                maximum = 0.0
                for neighbor_linecut in neighborhood:
                    neighbor_temperatures = np.asarray(neighbor_linecut["T"])
                    for neighbor in neighbor_linecut.get("features", []):
                        if neighbor.get("type") != feature_type:
                            continue
                        neighbor_index = _temperature_index(
                            neighbor_temperatures, neighbor["T"]
                        )
                        separation = feature_index - neighbor_index
                        candidate = float(neighbor.get(score_name, 0.0)) * math.exp(
                            -0.5 * separation**2 / tau
                        )
                        maximum = max(maximum, candidate)
                return maximum

            left = strongest(left_hood)
            right = strongest(right_hood)

            # At a dataset boundary, do not punish a feature solely because one
            # side of its neighborhood does not exist.
            if len(left_hood) < n_hood and left == 0.0:
                left = right
            if len(right_hood) < n_hood and right == 0.0:
                right = left

            support = math.sqrt(left * right)
            if sigmoid_support:
                support *= _sigmoid((support - sigmoid_center) / sigmoid_width)
            scores.append((feature, support))

    return scores


def update_scores_iter(
    linecuts,
    num_iter,
    mode="original_anchor",
    n_hood=3,
    retain=0.5,
    tau=3.0,
    sigmoid_support=False,
    sigmoid_center=0.6,
    sigmoid_width=0.1,
):
    """Compute and store ``score_1`` through ``score_<num_iter>``.

    Parameters
    ----------
    linecuts:
        Linecut dictionaries containing ``T`` and a ``features`` list. Each
        feature must contain ``T``, ``type``, and ``confidence``.
    num_iter:
        Number of synchronous score-update rounds.
    mode:
        ``"original_anchor"`` blends every round with original ``confidence``.
        ``"recursive"`` blends every round with the preceding score.
    retain:
        Weight retained from the blend base; support receives ``1 - retain``.
    sigmoid_support:
        If true, multiply raw support ``x`` by
        ``sigmoid((x - sigmoid_center) / sigmoid_width)`` before blending.

    Notes
    -----
    In both modes, support at round *i* is calculated from neighbors' scores at
    round *i - 1*.  The input objects are updated in place and returned.
    """
    if mode not in VALID_UPDATE_MODES:
        raise ValueError(f"mode must be one of {VALID_UPDATE_MODES}")
    if num_iter < 0:
        raise ValueError("num_iter cannot be negative")
    if not 0.0 <= retain <= 1.0:
        raise ValueError("retain must be between 0 and 1")

    for iteration in range(1, num_iter + 1):
        previous_name = "confidence" if iteration == 1 else f"score_{iteration - 1}"
        support_name = f"support_{iteration}"
        score_name = f"score_{iteration}"

        # Calculate the entire round before writing it so results do not depend
        # on linecut traversal order.
        round_support = _support_scores(
            linecuts,
            previous_name,
            n_hood=n_hood,
            tau=tau,
            sigmoid_support=sigmoid_support,
            sigmoid_center=sigmoid_center,
            sigmoid_width=sigmoid_width,
        )
        for feature, support in round_support:
            feature[support_name] = support
            blend_base_name = (
                "confidence" if mode == "original_anchor" else previous_name
            )
            blend_base = float(feature[blend_base_name])
            feature[score_name] = retain * blend_base + (1.0 - retain) * support

    return linecuts


def update_scores(linecuts, n_hood=3):
    """Backward-compatible one-round update.

    This keeps the original ``support`` and ``score_1`` keys used by plotting
    code while delegating the calculation to :func:`update_scores_iter`.
    """
    update_scores_iter(linecuts, 1, mode="original_anchor", n_hood=n_hood)
    for linecut in linecuts:
        for feature in linecut.get("features", []):
            feature["support"] = feature["support_1"]
    return linecuts
