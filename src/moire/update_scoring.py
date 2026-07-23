"""Iteratively refine feature confidence using neighboring linecuts.

Each round has two steps:

1. Calculate support from the previous round's scores.
2. Blend that support with the feature's *original* confidence.

The original confidence is always the anchor:

    score_i = (1 - support_weight) * confidence
              + support_weight * support_i
"""

import math

import numpy as np


def _temperature_index(temperatures, value):
    """Return the grid index nearest to a feature temperature."""
    return int(np.argmin(np.abs(np.asarray(temperatures) - value)))


def _sigmoid(value):
    """Numerically stable logistic sigmoid."""
    if value >= 0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def _strongest_match(target_index, feature_type, neighbors, score_name, tau):
    """Find the strongest same-type feature in a group of linecuts."""
    strongest = 0.0

    for linecut in neighbors:
        temperatures = linecut["T"]
        for neighbor in linecut.get("features", []):
            if neighbor.get("type") != feature_type:
                continue

            neighbor_index = _temperature_index(temperatures, neighbor["T"])
            separation = target_index - neighbor_index

            # Gaussian-like attenuation in temperature-index space.
            attenuation = math.exp(-0.5 * separation**2 / tau)
            candidate = float(neighbor.get(score_name, 0.0)) * attenuation
            strongest = max(strongest, candidate)

    return strongest


def _calculate_support(
    linecuts,
    score_name,
    n_hood,
    tau,
    sigmoid_support,
    sigmoid_center,
    sigmoid_width,
):
    """Calculate one synchronous support round for every feature."""
    support_values = []

    for linecut_index, linecut in enumerate(linecuts):
        left_neighbors = linecuts[max(0, linecut_index - n_hood) : linecut_index]
        right_neighbors = linecuts[
            linecut_index + 1 : linecut_index + n_hood + 1
        ]

        for feature in linecut.get("features", []):
            target_index = _temperature_index(linecut["T"], feature["T"])
            feature_type = feature.get("type")

            left = _strongest_match(
                target_index, feature_type, left_neighbors, score_name, tau
            )
            right = _strongest_match(
                target_index, feature_type, right_neighbors, score_name, tau
            )

            # At a dataset edge, mirror the available side instead of treating
            # a physically missing neighborhood as zero support.
            if len(left_neighbors) < n_hood and left == 0.0:
                left = right
            if len(right_neighbors) < n_hood and right == 0.0:
                right = left

            raw_support = math.sqrt(left * right)
            support = raw_support
            if sigmoid_support:
                multiplier = _sigmoid(
                    (raw_support - sigmoid_center) / sigmoid_width
                )
                support *= multiplier

            support_values.append((feature, support))

    return support_values


def update_scores_iter(
    linecuts,
    num_iter,
    n_hood=3,
    tau=3.0,
    support_weight=0.5,
    sigmoid_support=False,
    sigmoid_center=0.6,
    sigmoid_width=0.1,
):
    """Store ``support_i`` and ``score_i`` for each requested iteration.

    Parameters
    ----------
    linecuts:
        Linecut dictionaries containing ``T`` and ``features``. Each feature
        must contain ``T``, ``type``, and ``confidence``.
    num_iter:
        Number of synchronous update rounds.
    n_hood:
        Number of linecuts inspected on each side of the current linecut.
    tau:
        Temperature-index spread in
        ``exp(-0.5 * temperature_separation**2 / tau)``.
    support_weight:
        Support blend weight λ. Zero keeps original confidence; one uses only
        support.
    sigmoid_support:
        Apply ``support *= sigmoid((support - center) / width)`` when true.

    Notes
    -----
    Support for round ``i`` is calculated from neighbors' ``score_(i-1)``.
    The blend itself always uses the original confidence as its anchor.
    Updates are synchronous, so traversal order cannot change the result.
    """
    if num_iter < 0:
        raise ValueError("num_iter cannot be negative")
    if n_hood < 1:
        raise ValueError("n_hood must be at least 1")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not 0.0 <= support_weight <= 1.0:
        raise ValueError("support_weight must be between 0 and 1")
    if sigmoid_width <= 0:
        raise ValueError("sigmoid_width must be positive")

    for iteration in range(1, num_iter + 1):
        previous_score = (
            "confidence" if iteration == 1 else f"score_{iteration - 1}"
        )
        support_name = f"support_{iteration}"
        score_name = f"score_{iteration}"

        # Calculate the complete support round before mutating any feature.
        support_round = _calculate_support(
            linecuts=linecuts,
            score_name=previous_score,
            n_hood=n_hood,
            tau=tau,
            sigmoid_support=sigmoid_support,
            sigmoid_center=sigmoid_center,
            sigmoid_width=sigmoid_width,
        )

        for feature, support in support_round:
            confidence = float(feature["confidence"])
            feature[support_name] = support
            feature[score_name] = (
                (1.0 - support_weight) * confidence
                + support_weight * support
            )

    return linecuts


def update_scores(linecuts, n_hood=3):
    """Run one update while preserving the legacy ``support`` key."""
    update_scores_iter(linecuts, 1, n_hood=n_hood)
    for linecut in linecuts:
        for feature in linecut.get("features", []):
            feature["support"] = feature["support_1"]
    return linecuts
