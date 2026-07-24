"""Iteratively refine feature confidence using neighboring linecuts.

Each round has two steps:

1. Calculate support from the previous round's scores.
2. Blend that support with the feature's *original* confidence.

The original confidence is always the anchor:

    score_i = (1 - support_weight) * confidence
              + support_weight * support_i
"""

import copy
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


def _strongest_match(
    target_index,
    feature_type,
    neighbors,
    score_name,
    tau,
    feature_key="features",
):
    """Find the strongest same-type feature in a group of linecuts."""
    strongest = 0.0

    for linecut in neighbors:
        temperatures = linecut["T"]
        for neighbor in linecut.get(feature_key, []):
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
    n_hood = 12,
    tau = 20,
    sigmoid_support = True,
    sigmoid_center = 0,
    sigmoid_width = 0.1,
    feature_key = "features",
):
    """Calculate one synchronous support round for every feature."""
    support_values = []

    for linecut_index, linecut in enumerate(linecuts):
        left_neighbors = linecuts[max(0, linecut_index - n_hood) : linecut_index]
        right_neighbors = linecuts[
            linecut_index + 1 : linecut_index + n_hood + 1
        ]

        for feature in linecut.get(feature_key, []):
            target_index = _temperature_index(linecut["T"], feature["T"])
            feature_type = feature.get("type")

            left = _strongest_match(
                target_index,
                feature_type,
                left_neighbors,
                score_name,
                tau,
                feature_key,
            )
            right = _strongest_match(
                target_index,
                feature_type,
                right_neighbors,
                score_name,
                tau,
                feature_key,
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


def _update_score_range(
    linecuts,
    first_iteration,
    last_iteration,
    feature_key,
    n_hood,
    tau,
    support_weight,
    sigmoid_support,
    sigmoid_center,
    sigmoid_width,
):
    """Run a consecutively numbered range of synchronous score updates."""
    for iteration in range(first_iteration, last_iteration + 1):
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
            feature_key=feature_key,
        )

        for feature, support in support_round:
            original_confidence = float(feature["confidence"])
            feature[support_name] = support
            feature[score_name] = (
                (1.0 - support_weight) * original_confidence
                + support_weight * support
            )


def update_scores_iter(
    linecuts,
    num_iter,
    n_hood = 12,
    tau = 20,
    support_weight = 0.8,
    sigmoid_support = True,
    sigmoid_center = 0,
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

    _update_score_range(
        linecuts=linecuts,
        first_iteration=1,
        last_iteration=num_iter,
        feature_key="features",
        n_hood=n_hood,
        tau=tau,
        support_weight=support_weight,
        sigmoid_support=sigmoid_support,
        sigmoid_center=sigmoid_center,
        sigmoid_width=sigmoid_width,
    )

    return linecuts


def update_score(linecuts, num_iter=5, num_passes=3, filter=0.10):
    """Iteratively score copied features and prune noise between passes.

    ``features_new`` starts as a deep copy of each linecut's ``features``.
    Every pass performs ``num_iter`` score updates and then permanently removes
    entries whose latest score is less than ``filter``. Score names remain
    consecutive across passes, so the defaults produce ``score_1`` through
    ``score_15`` on every surviving feature.

    The original ``features`` lists and their feature dictionaries are not
    modified.
    """
    if num_iter < 1:
        raise ValueError("num_iter must be at least 1")
    if num_passes < 1:
        raise ValueError("num_passes must be at least 1")
    if not 0.0 <= filter <= 1.0:
        raise ValueError("filter must be between 0 and 1")

    for linecut in linecuts:
        linecut["features_new"] = copy.deepcopy(linecut.get("features", []))

    for pass_index in range(num_passes):
        first_iteration = pass_index * num_iter + 1
        last_iteration = first_iteration + num_iter - 1
        _update_score_range(
            linecuts=linecuts,
            first_iteration=first_iteration,
            last_iteration=last_iteration,
            feature_key="features_new",
            n_hood=12,
            tau=20,
            support_weight=0.8,
            sigmoid_support=True,
            sigmoid_center=0,
            sigmoid_width=0.1,
        )

        score_name = f"score_{last_iteration}"
        for linecut in linecuts:
            linecut["features_new"] = [
                feature
                for feature in linecut["features_new"]
                if feature[score_name] >= filter
            ]

    return linecuts
