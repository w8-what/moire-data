"""Standalone local-only candidate extraction for Tcoh and T-prime crossovers.

The routines in this module deliberately use only one resistivity linecut at a
time.  They generate several 10%-departure candidates and attach interpretable
0--1 plausibility components instead of forcing one phase-boundary point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm, spearmanr


@dataclass(frozen=True)
class CrossoverConfig:
    """Numerical constants for the universal linecut-local algorithm."""

    deviation: float = 0.10
    min_metallic_points: int = 8
    min_metallic_span: float = 0.25
    positive_slope_fraction: float = 0.80
    min_fit_points: int = 6
    min_fit_span: float = 0.25
    persistence_points: int = 4
    persistence_span: float = 0.30
    exponent_points: int = 6
    exponent_span: float = 0.50
    cluster_indices: int = 2
    min_supporting_windows: int = 2
    max_candidates: int = 5
    min_plausibility: float = 0.05
    identifiable_threshold: float = 0.65


DEFAULT_CONFIG = CrossoverConfig()


def _temperature_weights(T: np.ndarray) -> np.ndarray:
    if len(T) == 1:
        return np.ones(1)
    weights = np.empty(len(T), float)
    weights[0] = 0.5 * (T[1] - T[0])
    weights[-1] = 0.5 * (T[-1] - T[-2])
    if len(T) > 2:
        weights[1:-1] = 0.5 * (T[2:] - T[:-2])
    return np.maximum(weights, np.finfo(float).eps)


def _weighted_fraction(mask: np.ndarray, T: np.ndarray) -> float:
    if not len(mask):
        return 0.0
    weights = _temperature_weights(T)
    return float(np.sum(weights * mask) / np.sum(weights))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * np.sum(weights)
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _prepare_linecut(T, linecut):
    T = np.asarray(T, float)
    raw = np.asarray(linecut["rho"], float)
    smooth = np.asarray(linecut.get("rho_smoothed", raw), float)
    sigma = np.asarray(linecut.get("local_noise", np.ones_like(T)), float)
    if not (T.ndim == raw.ndim == smooth.ndim == sigma.ndim == 1):
        raise ValueError("T, rho, rho_smoothed, and local_noise must be one-dimensional")
    if not (len(T) == len(raw) == len(smooth) == len(sigma)):
        raise ValueError("T and linecut arrays must have equal length")

    finite = np.isfinite(T) & np.isfinite(raw) & np.isfinite(smooth)
    T, raw, smooth, sigma = T[finite], raw[finite], smooth[finite], sigma[finite]
    if len(T) < 2:
        return T, raw, smooth, sigma

    order = np.argsort(T, kind="stable")
    T, raw, smooth, sigma = T[order], raw[order], smooth[order], sigma[order]
    unique = np.r_[True, np.diff(T) > 0]
    T, raw, smooth, sigma = T[unique], raw[unique], smooth[unique], sigma[unique]

    valid_sigma = np.isfinite(sigma) & (sigma > 0)
    if np.any(valid_sigma):
        median_sigma = float(np.median(sigma[valid_sigma]))
    else:
        residual = raw - smooth
        median_sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if not np.isfinite(median_sigma) or median_sigma <= 0:
            median_sigma = max(float(np.std(raw)) * 1e-3, np.finfo(float).eps)
    sigma[~valid_sigma] = median_sigma
    sigma = np.maximum(sigma, median_sigma / 4)
    return T, raw, smooth, sigma


def _metallic_interval(T, smooth, config):
    """Return the widest interval with positive slope over 80% of its span."""
    best = None
    positive = np.diff(smooth) > 0
    widths = np.diff(T)
    for left in range(len(T)):
        for right in range(left + config.min_metallic_points - 1, len(T)):
            span = T[right] - T[left]
            if span < config.min_metallic_span:
                continue
            fraction = float(np.sum(widths[left:right] * positive[left:right]) / span)
            if fraction + 1e-12 < config.positive_slope_fraction:
                continue
            key = (span, right - left + 1, -left)
            if best is None or key > best[0]:
                best = (key, left, right, fraction)
    if best is None:
        return None
    return {"left": best[1], "right": best[2], "positive_fraction": best[3]}


def _robust_linear_fit(x, y, sigma):
    """Huber IRLS for y = beta0 + beta1*x, weighted by local noise."""
    X = np.column_stack((np.ones(len(x)), x))
    base = 1 / np.maximum(sigma, np.finfo(float).eps) ** 2
    weights = base.copy()
    beta = np.linalg.lstsq(X * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
    for _ in range(30):
        residual_z = (y - X @ beta) / sigma
        robust = np.ones_like(residual_z)
        large = np.abs(residual_z) > 1.345
        robust[large] = 1.345 / np.abs(residual_z[large])
        new_weights = base * robust
        new_beta = np.linalg.lstsq(
            X * np.sqrt(new_weights[:, None]), y * np.sqrt(new_weights), rcond=None
        )[0]
        if np.linalg.norm(new_beta - beta) <= 1e-10 * (1 + np.linalg.norm(beta)):
            beta, weights = new_beta, new_weights
            break
        beta, weights = new_beta, new_weights
    return beta, weights


def _baseline_score(T, raw, sigma, model_x, config):
    beta, weights = _robust_linear_fit(model_x, raw, sigma)
    fitted = beta[0] + beta[1] * model_x
    residual_z = (raw - fitted) / sigma
    z_rms = float(np.sqrt(np.mean(residual_z**2)))
    score_residual = 1 / (1 + (z_rms / 2) ** 2)

    # PRESS is the exact leave-one-out identity for a weighted linear fit. We
    # evaluate it at the converged Huber weights, keeping this diagnostic both
    # robust and inexpensive enough to apply to every fit window.
    design = np.column_stack((np.ones(len(T)), model_x))
    inverse = np.linalg.pinv(design.T @ (weights[:, None] * design))
    leverage = weights * np.einsum("ij,jk,ik->i", design, inverse, design)
    loo_residual_z = residual_z / np.maximum(1 - leverage, 1e-6)
    z_cv = float(np.sqrt(np.mean(loo_residual_z**2)))
    score_cv = 1 / (1 + (z_cv / 2) ** 2)

    residual = raw - fitted
    if np.ptp(residual) <= np.finfo(float).eps * max(np.max(np.abs(raw)), 1.0):
        trend = 0.0
    else:
        trend = spearmanr(T, residual).statistic
        trend = 0.0 if not np.isfinite(trend) else abs(float(trend))
    score_trend = max(0.0, 1 - trend)
    count = len(T)
    span = float(T[-1] - T[0])
    score_count = count**2 / (count**2 + config.min_metallic_points**2 / 4)
    score_span = span**2 / (span**2 + 0.5**2 / 4)
    components = np.array([score_residual, score_cv, score_trend, score_count, score_span], float)
    baseline = float(np.prod(np.clip(components, 0, 1)) ** (1 / len(components)))
    return (
        beta,
        baseline,
        {
            "residual": float(score_residual),
            "cross_validation": float(score_cv),
            "residual_trend": float(score_trend),
            "point_support": float(score_count),
            "span_support": float(score_span),
            "z_rms": z_rms,
            "z_cv": z_cv,
        },
    )


def _first_crossing(T, departure, fit_edge, outer_edge, direction, threshold):
    index = fit_edge + direction
    while index <= outer_edge if direction > 0 else index >= outer_edge:
        if departure[index] >= threshold:
            previous = index - direction
            d0, d1 = departure[previous], departure[index]
            if d1 == d0:
                crossing = T[index]
            else:
                fraction = np.clip((threshold - d0) / (d1 - d0), 0, 1)
                crossing = T[previous] + fraction * (T[index] - T[previous])
            return float(crossing), int(index)
        index += direction
    return None


def _outward_indices(T, crossing, crossing_index, outer_edge, direction, min_points, min_span):
    indices = []
    index = crossing_index
    while index <= outer_edge if direction > 0 else index >= outer_edge:
        indices.append(index)
        span = abs(float(T[index] - crossing))
        if len(indices) >= min_points and span >= min_span:
            break
        index += direction
    return np.asarray(sorted(indices), int)


def _departure_score(
    T, smooth, sigma, predicted, crossing, crossing_index, outer_edge, direction, config
):
    indices = _outward_indices(
        T,
        crossing,
        crossing_index,
        outer_edge,
        direction,
        config.persistence_points,
        config.persistence_span,
    )
    if not len(indices):
        return 0.0, {}, indices
    absolute = np.abs(smooth - predicted)
    relative = absolute / np.maximum(np.abs(predicted), 1e-12)
    local_T = T[indices]
    f10 = _weighted_fraction(relative[indices] >= config.deviation, local_T)
    f_sigma = _weighted_fraction(absolute[indices] >= 1.5 * sigma[indices], local_T)
    score_magnitude = min(1.0, float(np.median(relative[indices])) / config.deviation)
    span = float(local_T[-1] - local_T[0]) if len(local_T) > 1 else 0.0
    score_coverage = min(
        1.0, len(indices) / config.persistence_points, span / config.persistence_span
    )
    terms = np.array([f10, f_sigma, score_magnitude, score_coverage], float)
    departure_score = float(np.prod(np.clip(terms, 0, 1)) ** 0.25)
    return (
        departure_score,
        {
            "ten_percent_fraction": float(f10),
            "noise_fraction": float(f_sigma),
            "magnitude": float(score_magnitude),
            "coverage": float(score_coverage),
            "points": int(len(indices)),
            "span": span,
        },
        indices,
    )


def _power_fit(T, raw, sigma):
    T_ref = float(np.median(T))
    scaled = T / T_ref
    design = np.column_stack((np.ones(len(T)), scaled))
    offset, scale = np.linalg.lstsq(design, raw, rcond=None)[0]
    start = np.array([offset, max(float(scale), np.finfo(float).eps), 1.0])

    def residual(parameters):
        rho0, coefficient, exponent = parameters
        return (rho0 + coefficient * scaled**exponent - raw) / sigma

    result = least_squares(
        residual,
        start,
        bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 4.0]),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return None
    exponent = float(result.x[2])
    dof = len(T) - 3
    normal_matrix = result.jac.T @ result.jac
    if dof <= 0 or np.linalg.matrix_rank(normal_matrix) < 3:
        exponent_sigma = math.nan
    else:
        covariance = np.linalg.inv(normal_matrix) * (2 * result.cost / dof)
        exponent_sigma = math.sqrt(max(float(covariance[2, 2]), 0.0))
    if np.isclose(exponent, (0.1, 4.0), rtol=0, atol=1e-4).any():
        exponent_sigma = math.nan
    return exponent, exponent_sigma


def _contrast_score(kind, T, raw, sigma, crossing, crossing_index, outer_edge, direction, config):
    # Occam choice: use one power law over the whole available departure side
    # instead of searching for a smaller exponent window that happens to agree.
    indices = np.arange(min(crossing_index, outer_edge), max(crossing_index, outer_edge) + 1)
    if len(indices) < config.exponent_points or np.ptp(T[indices]) < config.exponent_span:
        return 0.5, {"exponent": None, "exponent_sigma": None, "reliable": False}
    fitted = _power_fit(T[indices], raw[indices], sigma[indices])
    if fitted is None or not np.isfinite(fitted[1]) or fitted[1] <= 0:
        return 0.5, {"exponent": None, "exponent_sigma": None, "reliable": False}
    exponent, exponent_sigma = fitted
    if kind == "Tcoh":
        quadratic_probability = norm.cdf((2.5 - exponent) / exponent_sigma) - norm.cdf(
            (1.5 - exponent) / exponent_sigma
        )
        contrast = 1 - quadratic_probability
    else:
        contrast = norm.cdf((1 - exponent) / exponent_sigma)
    return float(np.clip(contrast, 0, 1)), {
        "exponent": exponent,
        "exponent_sigma": exponent_sigma,
        "reliable": True,
    }


def _window_proposal(kind, T, raw, smooth, sigma, fit_indices, outer_edge, config):
    direction = 1 if kind == "Tcoh" else -1
    fit_edge = int(fit_indices[-1] if direction > 0 else fit_indices[0])
    model_x = T[fit_indices] ** 2 if kind == "Tcoh" else T[fit_indices]
    beta, baseline, baseline_parts = _baseline_score(
        T[fit_indices], raw[fit_indices], sigma[fit_indices], model_x, config
    )
    if beta[1] <= 0:
        return None, baseline

    full_x = T**2 if kind == "Tcoh" else T
    predicted = beta[0] + beta[1] * full_x
    departure = np.abs(smooth - predicted) / np.maximum(np.abs(predicted), 1e-12)
    crossing = _first_crossing(T, departure, fit_edge, outer_edge, direction, config.deviation)
    if crossing is None:
        return None, baseline
    crossing_temperature, crossing_index = crossing
    departure_score, departure_parts, _ = _departure_score(
        T,
        smooth,
        sigma,
        predicted,
        crossing_temperature,
        crossing_index,
        outer_edge,
        direction,
        config,
    )
    contrast, contrast_parts = _contrast_score(
        kind, T, raw, sigma, crossing_temperature, crossing_index, outer_edge, direction, config
    )
    snapped_index = int(np.argmin(np.abs(T - crossing_temperature)))
    proposal = {
        "T": crossing_temperature,
        "index": snapped_index,
        "B": baseline,
        "D": departure_score,
        "C": contrast,
        "baseline_components": baseline_parts,
        "departure_components": departure_parts,
        "contrast_components": contrast_parts,
        "fit_left_index": int(fit_indices[0]),
        "fit_right_index": int(fit_indices[-1]),
        "rho0": float(beta[0]),
        "A": float(beta[1]),
    }
    return proposal, baseline


def _cluster_proposals(proposals, eligible_baseline_weight, config):
    if not proposals or eligible_baseline_weight <= 0:
        return []
    ordered = sorted(proposals, key=lambda item: item["index"])
    clusters = [[ordered[0]]]
    for proposal in ordered[1:]:
        if proposal["index"] - clusters[-1][-1]["index"] <= config.cluster_indices:
            clusters[-1].append(proposal)
        else:
            clusters.append([proposal])

    candidates = []
    for cluster in clusters:
        if len(cluster) < config.min_supporting_windows:
            continue
        B = np.asarray([proposal["B"] for proposal in cluster], float)
        D = np.asarray([proposal["D"] for proposal in cluster], float)
        C = np.asarray([proposal["C"] for proposal in cluster], float)
        temperatures = np.asarray([proposal["T"] for proposal in cluster], float)
        indices = np.asarray([proposal["index"] for proposal in cluster], float)
        cluster_weight = np.maximum(B, np.finfo(float).eps)
        representative_temperature = _weighted_median(
            temperatures, np.maximum(B * D, np.finfo(float).eps)
        )
        B_cluster = float(np.average(B, weights=cluster_weight))
        D_cluster = float(np.average(D, weights=cluster_weight))
        C_cluster = float(np.average(C, weights=cluster_weight))
        vote = min(1.0, float(np.sum(B) / eligible_baseline_weight))
        median_index = _weighted_median(indices, cluster_weight)
        index_mad = _weighted_median(np.abs(indices - median_index), cluster_weight)
        spread = math.exp(-0.5 * (index_mad / 2) ** 2)
        robustness = math.sqrt(vote * spread)
        plausibility = B_cluster**0.30 * D_cluster**0.30 * robustness**0.25 * C_cluster**0.15
        representative = max(
            cluster, key=lambda proposal: (proposal["B"] * proposal["D"], proposal["B"])
        )
        candidates.append(
            {
                "T": representative_temperature,
                "plausibility": float(np.clip(plausibility, 0, 1)),
                "B": B_cluster,
                "D": D_cluster,
                "R": float(robustness),
                "C": C_cluster,
                "supporting_windows": len(cluster),
                "index_mad": float(index_mad),
                "vote_fraction": float(vote),
                "fit_T_lower": float(representative["fit_left_index"]),
                "fit_T_upper": float(representative["fit_right_index"]),
                "fit_left_index": representative["fit_left_index"],
                "fit_right_index": representative["fit_right_index"],
                "rho0": representative["rho0"],
                "A": representative["A"],
                "baseline_components": representative["baseline_components"],
                "departure_components": representative["departure_components"],
                "contrast_components": representative["contrast_components"],
            }
        )
    candidates.sort(
        key=lambda item: (item["plausibility"], item["supporting_windows"]), reverse=True
    )
    return [
        candidate
        for candidate in candidates
        if candidate["plausibility"] >= config.min_plausibility
    ][: config.max_candidates]


def analyze_crossover(T, linecut, kind, config=DEFAULT_CONFIG):
    """Analyze one linecut and return candidates plus local identifiability."""
    if kind not in {"Tcoh", "Tprime"}:
        raise ValueError("kind must be 'Tcoh' or 'Tprime'")
    T, raw, smooth, sigma = _prepare_linecut(T, linecut)
    empty = {
        "kind": kind,
        "status": "not_identifiable",
        "candidates": [],
        "metallic_interval": None,
    }
    if len(T) < max(config.min_metallic_points, config.min_fit_points + 1):
        return empty
    metallic = _metallic_interval(T, smooth, config)
    if metallic is None:
        return empty

    left, right = metallic["left"], metallic["right"]
    proposals = []
    eligible_baseline_weight = 0.0
    if kind == "Tcoh":
        endpoints = range(left + config.min_fit_points - 1, right)
        windows = [np.arange(left, endpoint + 1) for endpoint in endpoints]
        outer_edge = right
    else:
        starts = range(left + 1, right - config.min_fit_points + 2)
        windows = [np.arange(start, right + 1) for start in starts]
        outer_edge = left

    for indices in windows:
        if T[indices[-1]] - T[indices[0]] < config.min_fit_span:
            continue
        proposal, baseline = _window_proposal(
            kind, T, raw, smooth, sigma, indices, outer_edge, config
        )
        eligible_baseline_weight += baseline
        if proposal is not None:
            proposals.append(proposal)

    candidates = _cluster_proposals(proposals, eligible_baseline_weight, config)
    for candidate in candidates:
        candidate["nu"] = linecut.get("nu")
        candidate["type"] = kind
        candidate["fit_T_lower"] = float(T[candidate["fit_left_index"]])
        candidate["fit_T_upper"] = float(T[candidate["fit_right_index"]])
    status = (
        "locally_supported"
        if candidates and candidates[0]["plausibility"] >= config.identifiable_threshold
        else "not_identifiable"
    )
    return {
        "kind": kind,
        "status": status,
        "candidates": candidates,
        "metallic_interval": {
            "T_lower": float(T[left]),
            "T_upper": float(T[right]),
            "left_index": left,
            "right_index": right,
            "positive_fraction": metallic["positive_fraction"],
        },
        "proposal_count": len(proposals),
        "eligible_window_count": len(windows),
    }


def extract_Tcoh_candidates(T, linecut, config=DEFAULT_CONFIG):
    return analyze_crossover(T, linecut, "Tcoh", config)["candidates"]


def extract_Tprime_candidates(T, linecut, config=DEFAULT_CONFIG):
    return analyze_crossover(T, linecut, "Tprime", config)["candidates"]
