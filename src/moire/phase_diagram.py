"""Literature-aligned transport crossover extraction for moire resistance maps.

The operational definitions follow Xia et al., Nature 650, 585--591 (2026):

* ``Tneel`` is a transport proxy at a local minimum of rho(T).
* ``Tprime`` is a persistent 10% departure below a high-T linear fit.
* ``Tcoh`` is a persistent 10% departure above a validated low-T T-squared fit.

The implementation deliberately separates per-linecut model evidence from
cross-linecut branch selection.  The latter uses physical temperature and
filling distances, so it behaves consistently on the repository's different
and non-uniform grids.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from moire.adaptive_multiscale_smooth import (
    adaptive_multiscale_smooth,
    estimate_noise_1d,
    estimate_noise_matrix,
)
from moire.io import clean_sort_data, load_field


@dataclass(frozen=True)
class PhaseExtractionConfig:
    deviation: float = 0.10
    min_fit_points: int = 7
    min_fit_span: float = 0.40
    min_crossing_points: int = 2
    min_crossing_span: float = 0.10
    min_crossing_fraction: float = 0.80
    min_signal_sigma: float = 4.0
    max_fit_median_fractional_error: float = 0.055
    max_fit_p90_fractional_error: float = 0.11
    min_t2_exponent: float = 1.82
    max_t2_exponent: float = 2.18
    max_t2_equivalence_distance: float = 0.18
    max_exponent_sigma: float = 0.25
    min_linear_exponent: float = 0.85
    max_linear_exponent: float = 1.15
    max_parameter_drift: float = 0.35
    max_fit_trials: int = 12
    tneel_min_temperature: float = 0.35
    tneel_min_prominence_sigma: float = 3.0
    tneel_min_prominence_fraction: float = 0.03
    branch_min_confidence: float = 0.36
    branch_max_slope: float = 180.0
    branch_max_missing_columns: int = 2
    branch_min_points: int = 5
    tcoh_min_confidence: float = 0.28
    tcoh_max_gap_fraction: float = 0.03
    tcoh_max_slope: float = 350.0
    tcoh_min_component_points: int = 4


@dataclass
class PhasePoint:
    field: float
    nu: float
    transition: str
    temperature: float
    uncertainty: float
    confidence: float
    model: str
    fit_lower: float | None = None
    fit_upper: float | None = None
    fit_median_fractional_error: float | None = None
    fit_p90_fractional_error: float | None = None
    fit_offset: float | None = None
    fit_coefficient: float | None = None
    exponent: float | None = None
    exponent_sigma: float | None = None
    crossing_lower: float | None = None
    crossing_upper: float | None = None
    censored: bool = False
    nu_censored: bool = False
    nu_censor_side: str | None = None
    support: float = 0.0
    component: int | None = None

    def to_dict(self):
        return asdict(self)


def _sigmoid_score(value, reference, power=3.0):
    value = max(float(value), 0.0)
    reference = max(float(reference), np.finfo(float).eps)
    return value**power / (value**power + reference**power)


def _safe_fractional_error(observed, predicted, scale_floor):
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    denominator = np.maximum(np.abs(predicted), float(scale_floor))
    return np.abs(observed - predicted) / denominator


def _robust_scale(values):
    values = np.asarray(values, float)
    center = np.nanmedian(values)
    scale = 1.4826 * np.nanmedian(np.abs(values - center))
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(values)
    return max(float(scale), np.finfo(float).eps)


def _fit_fixed_power(T, rho, sigma, power):
    x = np.asarray(T, float) ** power
    y = np.asarray(rho, float)
    s = np.maximum(np.asarray(sigma, float), np.finfo(float).eps)
    design = np.column_stack((np.ones_like(x), x))
    weighted = design / s[:, None]
    offset, coefficient = np.linalg.lstsq(weighted, y / s, rcond=None)[0]
    predicted = offset + coefficient * x
    residual = (y - predicted) / s
    rss = max(float(np.sum(residual**2)), np.finfo(float).eps)
    bic = len(y) * math.log(rss / len(y)) + 2 * math.log(len(y))
    return {
        "offset": float(offset),
        "coefficient": float(coefficient),
        "predicted": predicted,
        "rss": rss,
        "bic": bic,
    }


def _fit_global_power_grid(T, rho, sigma):
    """Numerically stable single-power screen used before local window search."""
    T = np.asarray(T, float)
    rho = np.asarray(rho, float)
    sigma = np.maximum(np.asarray(sigma, float), np.finfo(float).eps)
    best = None
    for exponent in np.linspace(0.6, 3.4, 113):
        fit = _fit_fixed_power(T, rho, sigma, float(exponent))
        if fit["coefficient"] <= 0:
            continue
        residual = np.abs(rho - fit["predicted"]) / sigma
        score = float(np.median(residual) + 0.2 * np.percentile(residual, 90))
        if best is None or score < best["score"]:
            best = {"exponent": float(exponent), "predicted": fit["predicted"], "score": score}
    return best


def _fit_free_power(T, rho, sigma):
    T = np.asarray(T, float)
    rho = np.asarray(rho, float)
    sigma = np.maximum(np.asarray(sigma, float), np.finfo(float).eps)
    reference = float(np.median(T))
    scaled = T / reference
    offset, coefficient = np.linalg.lstsq(
        np.column_stack((np.ones_like(scaled), scaled**2)), rho, rcond=None
    )[0]
    start = [offset, max(float(coefficient), np.finfo(float).eps), 2.0]

    def residuals(parameters):
        rho0, amplitude, exponent = parameters
        return (rho0 + amplitude * scaled**exponent - rho) / sigma

    result = least_squares(
        residuals,
        start,
        bounds=([-np.inf, 0.0, 0.2], [np.inf, np.inf, 4.0]),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=800,
    )
    # Long, densely sampled grids can hit ``max_nfev`` after converging to a
    # stable finite optimum.  Treat that as usable model evidence; rejecting
    # it would bypass the global single-power guard and create local-window
    # false crossovers.
    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.cost):
        return None

    exponent = float(result.x[2])
    exponent_sigma = np.nan
    dof = len(T) - 3
    normal = result.jac.T @ result.jac
    if dof > 0 and np.linalg.matrix_rank(normal) == 3:
        covariance = np.linalg.inv(normal) * (2 * result.cost / dof)
        variance = float(covariance[2, 2])
        if np.isfinite(variance) and variance >= 0:
            exponent_sigma = math.sqrt(variance)

    raw_residual = residuals(result.x)
    rss = max(float(np.sum(raw_residual**2)), np.finfo(float).eps)
    bic = len(T) * math.log(rss / len(T)) + 3 * math.log(len(T))
    predicted = result.x[0] + result.x[1] * scaled**exponent
    return {
        "offset": float(result.x[0]),
        "amplitude": float(result.x[1]),
        "reference_temperature": reference,
        "predicted": predicted,
        "exponent": exponent,
        "exponent_sigma": exponent_sigma,
        "bic": bic,
    }


def _fit_robust_linear(T, rho, sigma):
    T = np.asarray(T, float)
    rho = np.asarray(rho, float)
    sigma = np.maximum(np.asarray(sigma, float), np.finfo(float).eps)
    offset, slope = np.linalg.lstsq(np.column_stack((np.ones_like(T), T)), rho, rcond=None)[0]

    def residuals(parameters):
        return (parameters[0] + parameters[1] * T - rho) / sigma

    result = least_squares(
        residuals,
        [offset, max(float(slope), np.finfo(float).eps)],
        bounds=([-np.inf, 0.0], [np.inf, np.inf]),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=500,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return None
    predicted = result.x[0] + result.x[1] * T
    normalized_residual = (rho - predicted) / sigma
    rss = max(float(np.sum(normalized_residual**2)), np.finfo(float).eps)
    bic = len(T) * math.log(rss / len(T)) + 2 * math.log(len(T))
    return {
        "offset": float(result.x[0]),
        "slope": float(result.x[1]),
        "predicted": predicted,
        "rss": rss,
        "bic": bic,
    }


def _validate_t2_prefix(T, rho, sigma, start, end, scale_floor, config):
    """Validate T^2 on the complete pre-crossing interval with holdout checks."""
    if end - start + 1 < config.min_fit_points or T[end] - T[start] < config.min_fit_span:
        return None
    selection = slice(start, end + 1)
    fit = _fit_fixed_power(T[selection], rho[selection], sigma[selection], 2.0)
    if fit["coefficient"] <= 0:
        return None
    signal = fit["coefficient"] * (T[end] ** 2 - T[start] ** 2)
    signal_z = signal / max(float(np.median(sigma[selection])), np.finfo(float).eps)
    if signal_z < config.min_signal_sigma:
        return None
    error = _safe_fractional_error(rho[selection], fit["predicted"], scale_floor)
    median_error = float(np.median(error))
    p90_error = float(np.percentile(error, 90))
    if median_error > config.max_fit_median_fractional_error:
        return None
    if p90_error > config.max_fit_p90_fractional_error:
        return None

    free = _fit_free_power(T[selection], rho[selection], sigma[selection])
    if free is None or not (config.min_t2_exponent <= free["exponent"] <= config.max_t2_exponent):
        return None
    exponent_sigma = free["exponent_sigma"]
    if not np.isfinite(exponent_sigma):
        return None
    if exponent_sigma > config.max_exponent_sigma:
        return None
    if abs(free["exponent"] - 2.0) > config.max_t2_equivalence_distance:
        return None
    if fit["bic"] - free["bic"] > 2.0:
        return None

    count = end - start + 1
    train_end = start + max(config.min_fit_points - 1, int(math.floor(0.70 * count)) - 1)
    if train_end >= end or T[train_end] - T[start] < 0.65 * config.min_fit_span:
        return None
    train = slice(start, train_end + 1)
    train_fit = _fit_fixed_power(T[train], rho[train], sigma[train], 2.0)
    if train_fit["coefficient"] <= 0:
        return None
    coefficient_drift = abs(train_fit["coefficient"] - fit["coefficient"]) / max(
        abs(fit["coefficient"]), np.finfo(float).eps
    )
    if coefficient_drift > config.max_parameter_drift:
        return None
    holdout = np.arange(train_end + 1, end + 1)
    holdout_prediction = train_fit["offset"] + train_fit["coefficient"] * T[holdout] ** 2
    holdout_error = _safe_fractional_error(rho[holdout], holdout_prediction, scale_floor)
    if float(np.median(holdout_error)) > config.max_fit_median_fractional_error:
        return None
    if float(np.percentile(holdout_error, 90)) > config.max_fit_p90_fractional_error:
        return None
    return {
        "fit": fit,
        "free": free,
        "signal_z": signal_z,
        "median_error": median_error,
        "p90_error": p90_error,
        "coefficient_drift": coefficient_drift,
    }


def _crossing_temperature(T, relative_error, noise_z, threshold, start, direction, config):
    """Return a persistent crossing and its measured bracket.

    ``direction=1`` walks upward for Tcoh.  ``direction=-1`` walks downward
    for Tprime.  The crossing value is linearly interpolated at the requested
    relative-error threshold.
    """
    T = np.asarray(T, float)
    relative_error = np.asarray(relative_error, float)
    noise_z = np.asarray(noise_z, float)
    n = len(T)
    indices = range(start, n) if direction == 1 else range(start, -1, -1)

    for index in indices:
        if relative_error[index] < threshold or noise_z[index] < 2.0:
            continue
        if direction == 1:
            span_stop = int(np.searchsorted(T, T[index] + config.min_crossing_span, side="left"))
            stop = max(index + config.min_crossing_points - 1, span_stop)
            if stop >= n:
                continue
            run = np.arange(index, stop + 1)
            previous = index - 1
        else:
            span_stop = int(
                np.searchsorted(T, T[index] - config.min_crossing_span, side="right") - 1
            )
            stop = min(index - config.min_crossing_points + 1, span_stop)
            if stop < 0:
                continue
            run = np.arange(stop, index + 1)
            previous = index + 1

        if len(run) < config.min_crossing_points:
            continue
        persistent = (relative_error[run] >= threshold) & (noise_z[run] >= 2.0)
        if np.mean(persistent) < config.min_crossing_fraction:
            continue

        if previous < 0 or previous >= n:
            return float(T[index]), float(T[index]), float(T[index]), True
        low_index, high_index = sorted((index, previous))
        x0, x1 = T[low_index], T[high_index]
        e0, e1 = relative_error[low_index], relative_error[high_index]
        if np.isclose(e0, e1):
            crossing = 0.5 * (x0 + x1)
        else:
            crossing = x0 + (threshold - e0) * (x1 - x0) / (e1 - e0)
        crossing = float(np.clip(crossing, x0, x1))
        return crossing, float(x0), float(x1), False
    return None


def _cluster_estimates(estimates, tolerance):
    if not estimates:
        return []
    estimates = sorted(estimates, key=lambda item: item["temperature"])
    clusters = [[estimates[0]]]
    for estimate in estimates[1:]:
        center = np.median([item["temperature"] for item in clusters[-1]])
        if abs(estimate["temperature"] - center) <= tolerance:
            clusters[-1].append(estimate)
        else:
            clusters.append([estimate])
    return clusters


def _evenly_sample_indices(indices, maximum):
    """Keep a deterministic, representative subset of candidate fit bounds."""
    indices = np.asarray(indices, int)
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum)
    return np.unique(indices[np.rint(positions).astype(int)])


def extract_tcoh_linecut(T, rho, smoothed, sigma, *, field, nu, config=None):
    """Extract one strict, model-validated Tcoh candidate from a linecut."""
    config = config or PhaseExtractionConfig()
    T = np.asarray(T, float)
    rho = np.asarray(rho, float)
    smoothed = np.asarray(smoothed, float)
    sigma = np.maximum(np.asarray(sigma, float), np.finfo(float).eps)
    valid = (
        np.isfinite(T) & np.isfinite(rho) & np.isfinite(smoothed) & np.isfinite(sigma) & (rho > 0)
    )
    positive = rho[valid]
    if len(positive) < config.min_fit_points + config.min_crossing_points:
        return None
    scale_floor = max(float(np.percentile(positive, 10)) * 0.08, np.median(sigma), 1e-9)
    valid_indices = np.flatnonzero(valid)
    last_valid = int(valid_indices[-1])
    if np.all(np.diff(valid_indices) == 1):
        global_free = _fit_global_power_grid(T[valid], rho[valid], sigma[valid])
        if global_free is not None:
            global_prediction = global_free["predicted"]
            global_error = _safe_fractional_error(smoothed[valid], global_prediction, scale_floor)
            if (
                not 1.95 <= global_free["exponent"] <= 2.05
                # A single smooth power law can look locally quadratic on a
                # long grid.  Reject it before window search. A genuine
                # piecewise crossover fails this stringent 1.5% global test.
                and (global_free["score"] <= 2.5 or float(np.percentile(global_error, 90)) <= 0.015)
            ):
                return None
    local_step = float(np.median(np.diff(T[: min(len(T), 20)])))
    tolerance = max(0.12, 2.5 * local_step)

    first_valid = int(valid_indices[0])
    if T[first_valid] - T[0] > 0.10:
        return None
    # A coherence fit must begin at the lowest defensible temperature.  Allow
    # only tiny endpoint perturbations; never skip an upturn or low-T anomaly.
    max_start_T = T[first_valid] + min(0.08, 0.02 * np.ptp(T))
    starts = _evenly_sample_indices(
        np.flatnonzero(valid & (np.arange(len(T)) >= first_valid) & (T <= max_start_T)),
        min(config.max_fit_trials, 5),
    )
    estimates = []
    attempted_starts = 0
    for start in starts:
        if len(T) - start < config.min_fit_points:
            continue
        attempted_starts += 1
        ends = _evenly_sample_indices(
            np.arange(start + config.min_fit_points - 1, len(T)), 2 * config.max_fit_trials
        )
        for end in ends:
            if T[end] - T[start] < config.min_fit_span:
                continue
            selection = slice(start, end + 1)
            if not np.all(valid[selection]):
                continue
            fit = _fit_fixed_power(T[selection], rho[selection], sigma[selection], 2.0)
            if fit["coefficient"] <= 0:
                continue
            signal = fit["coefficient"] * (T[end] ** 2 - T[start] ** 2)
            signal_z = signal / max(float(np.median(sigma[selection])), np.finfo(float).eps)
            if signal_z < config.min_signal_sigma:
                continue

            fit_error = _safe_fractional_error(rho[selection], fit["predicted"], scale_floor)
            median_error = float(np.median(fit_error))
            p90_error = float(np.percentile(fit_error, 90))
            if median_error > config.max_fit_median_fractional_error:
                continue
            if p90_error > config.max_fit_p90_fractional_error:
                continue

            free = _fit_free_power(T[selection], rho[selection], sigma[selection])
            if free is None:
                continue
            exponent_ok = config.min_t2_exponent <= free["exponent"] <= config.max_t2_exponent
            exponent_ok = (
                exponent_ok
                and np.isfinite(free["exponent_sigma"])
                and free["exponent_sigma"] <= config.max_exponent_sigma
                and abs(free["exponent"] - 2.0) <= config.max_t2_equivalence_distance
            )
            bic_ok = fit["bic"] - free["bic"] <= 2.0
            if not exponent_ok or not bic_ok:
                continue

            predicted_all = fit["offset"] + fit["coefficient"] * T**2
            relative = _safe_fractional_error(smoothed, predicted_all, scale_floor)
            noise_z = np.abs(smoothed - predicted_all) / sigma
            relative[~valid] = -np.inf
            noise_z[~valid] = -np.inf
            crossing = (
                _crossing_temperature(T, relative, noise_z, config.deviation, end + 1, 1, config)
                if end + 1 < len(T)
                else None
            )
            if crossing is None:
                tail = np.arange(end + 1, last_valid + 1)
                if len(tail) and (
                    not np.all(valid[tail]) or np.any(relative[tail] >= config.deviation)
                ):
                    continue
                # The T^2 regime reaches the measurement ceiling: Tcoh is a
                # lower bound, not an absent feature.
                temperature = float(T[last_valid])
                lower = float(T[last_valid])
                upper = float(T[last_valid])
                censored = True
            else:
                temperature, lower, upper, censored = crossing
                if end > start and relative[end] >= config.deviation:
                    continue
            prefix_end = last_valid if censored else end
            if not np.all(valid[start : prefix_end + 1]):
                continue
            validated = _validate_t2_prefix(T, rho, sigma, start, prefix_end, scale_floor, config)
            if validated is None:
                continue
            fit = validated["fit"]
            free = validated["free"]
            signal_z = validated["signal_z"]
            median_error = validated["median_error"]
            p90_error = validated["p90_error"]
            score = (
                _sigmoid_score(signal_z, config.min_signal_sigma)
                * math.exp(-median_error / config.max_fit_median_fractional_error)
                * math.exp(-abs(free["exponent"] - 2.0) / 0.25)
                * math.exp(-validated["coefficient_drift"] / config.max_parameter_drift)
            ) ** (1 / 4)
            estimates.append(
                {
                    "temperature": temperature,
                    "lower": lower,
                    "upper": upper,
                    "score": score,
                    "fit_lower": float(T[start]),
                    "fit_upper": float(T[prefix_end]),
                    "median_error": median_error,
                    "p90_error": p90_error,
                    "fit_offset": fit["offset"],
                    "fit_coefficient": fit["coefficient"],
                    "exponent": free["exponent"],
                    "exponent_sigma": free["exponent_sigma"],
                    "censored": censored,
                }
            )

    clusters = _cluster_estimates(estimates, tolerance)
    if not clusters or attempted_starts == 0:
        return None
    cluster = max(clusters, key=lambda items: sum(item["score"] for item in items))
    recovery = min(1.0, len({item["fit_lower"] for item in cluster}) / max(attempted_starts, 1))
    if recovery < 0.25:
        return None
    weights = np.asarray([item["score"] for item in cluster], float)
    temperatures = np.asarray([item["temperature"] for item in cluster], float)
    center = float(np.average(temperatures, weights=weights))
    representative = min(cluster, key=lambda item: abs(item["temperature"] - center))
    spread = 1.4826 * np.median(np.abs(temperatures - np.median(temperatures)))
    crossing_lower = float(min(item["lower"] for item in cluster))
    crossing_upper = float(max(item["upper"] for item in cluster))
    center = float(np.clip(center, crossing_lower, crossing_upper))
    uncertainty = max(float(spread), center - crossing_lower, crossing_upper - center)
    confidence = float(np.clip(np.mean(weights) * math.sqrt(recovery), 0.0, 1.0))
    return PhasePoint(
        field=float(field),
        nu=float(nu),
        transition="Tcoh",
        temperature=center,
        uncertainty=uncertainty,
        confidence=confidence,
        model=(
            "rho0 + A*T^2 through measurement ceiling; lower-bound Tcoh"
            if representative["censored"]
            else "rho0 + A*T^2; persistent 10% departure"
        ),
        fit_lower=representative["fit_lower"],
        fit_upper=representative["fit_upper"],
        fit_median_fractional_error=representative["median_error"],
        fit_p90_fractional_error=representative["p90_error"],
        fit_offset=representative["fit_offset"],
        fit_coefficient=representative["fit_coefficient"],
        exponent=representative["exponent"],
        exponent_sigma=representative["exponent_sigma"],
        crossing_lower=crossing_lower,
        crossing_upper=crossing_upper,
        censored=representative["censored"],
    )


def extract_tprime_linecut(T, rho, smoothed, sigma, *, field, nu, config=None):
    """Extract Tprime from a validated high-T linear-in-T regime.

    Fractional deviations are evaluated with positive prediction guards.  This
    is equivalent to a guarded log-ratio check at the upper 10% boundary while
    retaining the paper's linear-in-T model.
    """
    config = config or PhaseExtractionConfig()
    T = np.asarray(T, float)
    rho = np.asarray(rho, float)
    smoothed = np.asarray(smoothed, float)
    sigma = np.maximum(np.asarray(sigma, float), np.finfo(float).eps)
    valid = (
        np.isfinite(T) & np.isfinite(rho) & np.isfinite(smoothed) & np.isfinite(sigma) & (rho > 0)
    )
    positive = rho[valid]
    if len(positive) < config.min_fit_points + config.min_crossing_points:
        return None
    scale_floor = max(float(np.percentile(positive, 10)) * 0.08, np.median(sigma), 1e-9)
    valid_indices = np.flatnonzero(valid)
    if np.all(np.diff(valid_indices) == 1):
        global_free = _fit_global_power_grid(T[valid], rho[valid], sigma[valid])
        if global_free is not None:
            global_prediction = global_free["predicted"]
            global_error = _safe_fractional_error(smoothed[valid], global_prediction, scale_floor)
            if not 0.90 <= global_free["exponent"] <= 1.10 and (
                global_free["score"] <= 2.5 or float(np.percentile(global_error, 90)) <= 0.015
            ):
                return None
    local_step = float(np.median(np.diff(T[: min(len(T), 20)])))
    tolerance = max(0.12, 2.5 * local_step)

    starts = _evenly_sample_indices(
        np.flatnonzero((T >= T[0] + 0.42 * np.ptp(T)) & (T <= T[0] + 0.78 * np.ptp(T))),
        config.max_fit_trials,
    )
    estimates = []
    attempted = 0
    for start in starts:
        if len(T) - start < config.min_fit_points or T[-1] - T[start] < config.min_fit_span:
            continue
        attempted += 1
        selection = slice(start, len(T))
        if not np.all(valid[selection]):
            continue
        fit = _fit_robust_linear(T[selection], rho[selection], sigma[selection])
        if fit is None or fit["slope"] <= 0:
            continue
        free = _fit_free_power(T[selection], rho[selection], sigma[selection])
        if free is None or not (
            config.min_linear_exponent <= free["exponent"] <= config.max_linear_exponent
        ):
            continue
        exponent_identified = np.isfinite(free["exponent_sigma"]) and (
            free["exponent_sigma"] <= config.max_exponent_sigma
        )
        if not exponent_identified:
            continue
        if fit["bic"] - free["bic"] > 2.0:
            continue
        count = len(T) - start
        train_start = start + max(2, int(math.ceil(0.30 * count)))
        if len(T) - train_start < config.min_fit_points:
            continue
        upper_fit = _fit_robust_linear(T[train_start:], rho[train_start:], sigma[train_start:])
        if upper_fit is None or upper_fit["slope"] <= 0:
            continue
        slope_drift = abs(upper_fit["slope"] - fit["slope"]) / max(
            abs(fit["slope"]), np.finfo(float).eps
        )
        if slope_drift > config.max_parameter_drift:
            continue
        holdout = np.arange(start, train_start)
        holdout_prediction = upper_fit["offset"] + upper_fit["slope"] * T[holdout]
        holdout_error = _safe_fractional_error(rho[holdout], holdout_prediction, scale_floor)
        if float(np.median(holdout_error)) > config.max_fit_median_fractional_error:
            continue
        if float(np.percentile(holdout_error, 90)) > config.max_fit_p90_fractional_error:
            continue
        predicted = fit["offset"] + fit["slope"] * T
        if np.any(predicted[: start + 1] <= scale_floor):
            continue
        fit_error = _safe_fractional_error(rho[selection], predicted[selection], scale_floor)
        median_error = float(np.median(fit_error))
        p90_error = float(np.percentile(fit_error, 90))
        if median_error > config.max_fit_median_fractional_error:
            continue
        if p90_error > config.max_fit_p90_fractional_error:
            continue

        # A T-sublinear state lies above the downward extrapolation of the
        # high-T line.  Guard the relative calculation rather than clipping rho.
        signed_fractional = (smoothed - predicted) / np.maximum(np.abs(predicted), scale_floor)
        noise_z = np.abs(smoothed - predicted) / sigma
        signed_fractional[~valid] = -np.inf
        noise_z[~valid] = -np.inf
        crossing = _crossing_temperature(
            T, signed_fractional, noise_z, config.deviation, start - 1, -1, config
        )
        if crossing is None:
            continue
        temperature, lower, upper, censored = crossing

        below = np.flatnonzero((T < temperature) & (T >= max(T[0], temperature - 0.8)))
        if len(below) < 2:
            continue
        slope_ratio = np.median(np.gradient(smoothed, T)[below]) / max(fit["slope"], 1e-12)
        if not np.isfinite(slope_ratio) or slope_ratio >= 0.90:
            continue
        dynamic = fit["slope"] * (T[-1] - T[start])
        signal_z = dynamic / max(float(np.median(sigma[selection])), np.finfo(float).eps)
        if signal_z < config.min_signal_sigma:
            continue
        score = (
            _sigmoid_score(signal_z, config.min_signal_sigma)
            * math.exp(-median_error / config.max_fit_median_fractional_error)
            * _sigmoid_score(0.9 - slope_ratio, 0.25)
            * math.exp(-abs(free["exponent"] - 1.0) / 0.25)
            * math.exp(-slope_drift / config.max_parameter_drift)
        ) ** (1 / 5)
        estimates.append(
            {
                "temperature": temperature,
                "lower": lower,
                "upper": upper,
                "score": score,
                "fit_lower": float(T[start]),
                "fit_upper": float(T[-1]),
                "median_error": median_error,
                "p90_error": p90_error,
                "exponent": free["exponent"],
                "exponent_sigma": free["exponent_sigma"],
                "censored": censored,
            }
        )

    clusters = _cluster_estimates(estimates, tolerance)
    if not clusters or attempted == 0:
        return None
    cluster = max(clusters, key=lambda items: sum(item["score"] for item in items))
    recovery = min(1.0, len({item["fit_lower"] for item in cluster}) / attempted)
    if recovery < 0.30:
        return None
    weights = np.asarray([item["score"] for item in cluster], float)
    temperatures = np.asarray([item["temperature"] for item in cluster], float)
    center = float(np.average(temperatures, weights=weights))
    representative = min(cluster, key=lambda item: abs(item["temperature"] - center))
    spread = 1.4826 * np.median(np.abs(temperatures - np.median(temperatures)))
    crossing_lower = float(min(item["lower"] for item in cluster))
    crossing_upper = float(max(item["upper"] for item in cluster))
    center = float(np.clip(center, crossing_lower, crossing_upper))
    uncertainty = max(float(spread), center - crossing_lower, crossing_upper - center)
    confidence = float(np.clip(np.mean(weights) * math.sqrt(recovery), 0.0, 1.0))
    return PhasePoint(
        field=float(field),
        nu=float(nu),
        transition="Tprime",
        temperature=center,
        uncertainty=uncertainty,
        confidence=confidence,
        model="rho0 + B*T; guarded fractional/log-ratio 10% departure",
        fit_lower=representative["fit_lower"],
        fit_upper=representative["fit_upper"],
        fit_median_fractional_error=representative["median_error"],
        fit_p90_fractional_error=representative["p90_error"],
        exponent=representative["exponent"],
        exponent_sigma=representative["exponent_sigma"],
        crossing_lower=crossing_lower,
        crossing_upper=crossing_upper,
        censored=representative["censored"],
    )


def extract_tneel_candidates(T, rho, smoothed, sigma, *, field, nu, config=None):
    """Return transport-proxy TN candidates at significant rho(T) minima."""
    config = config or PhaseExtractionConfig()
    T = np.asarray(T, float)
    rho = np.asarray(rho, float)
    smoothed = np.asarray(smoothed, float)
    sigma = np.maximum(np.asarray(sigma, float), np.finfo(float).eps)
    peaks, properties = find_peaks(-smoothed, prominence=(None, None), width=(None, None))
    candidates = []
    typical = max(float(np.nanmedian(np.abs(smoothed))), np.finfo(float).eps)
    for position, index in enumerate(peaks):
        if T[index] < config.tneel_min_temperature:
            continue
        left = int(properties["left_bases"][position])
        right = int(properties["right_bases"][position])
        if left >= index or right <= index:
            continue
        # Never infer a minimum through an interpolated invalid measurement.
        # Invalid blocks in these maps occur precisely in the low-temperature
        # regions that can otherwise mimic an ordered-state upturn.
        if not np.all(np.isfinite(rho[left : right + 1]) & (rho[left : right + 1] > 0)):
            continue
        prominence = float(properties["prominences"][position])
        # Estimate significance at the feature width, not across the full
        # prominence basin; the latter can span an entire insulating linecut
        # and misclassify real phase curvature as measurement noise.
        sigma_left = max(0, int(math.floor(properties["left_ips"][position])) - 1)
        sigma_right = min(len(T) - 1, int(math.ceil(properties["right_ips"][position])) + 1)
        local_sigma = max(
            float(np.median(sigma[sigma_left : sigma_right + 1])), np.finfo(float).eps
        )
        prominence_z = prominence / local_sigma
        prominence_fraction = prominence / max(abs(float(smoothed[index])), 0.05 * typical)
        if prominence_z < config.tneel_min_prominence_sigma:
            continue
        if prominence_fraction < config.tneel_min_prominence_fraction:
            continue
        width = float(T[right] - T[left])
        local_step = float(np.median(np.diff(T[max(0, left) : min(len(T), right + 1)])))
        if width < max(2 * local_step, 0.10):
            continue
        derivative = np.gradient(smoothed, T)
        left_slope = float(np.median(derivative[max(0, index - 2) : index]))
        right_slope = float(np.median(derivative[index + 1 : min(len(T), index + 3)]))
        if not left_slope < 0 < right_slope:
            continue
        confidence = (
            _sigmoid_score(prominence_z, config.tneel_min_prominence_sigma)
            * _sigmoid_score(prominence_fraction, config.tneel_min_prominence_fraction)
            * _sigmoid_score(width, max(4 * local_step, 0.20))
        ) ** (1 / 3)
        candidates.append(
            PhasePoint(
                field=float(field),
                nu=float(nu),
                transition="Tneel",
                temperature=float(T[index]),
                # The basin width is physical structure, not measurement error.
                # Bracket the sampled minimum by the adjacent temperature step.
                uncertainty=max(
                    local_step, 0.5 * (T[min(index + 1, len(T) - 1)] - T[max(index - 1, 0)])
                ),
                confidence=float(np.clip(confidence, 0, 1)),
                model="local rho(T) minimum; transport proxy for T_N",
                crossing_lower=float(T[max(index - 1, 0)]),
                crossing_upper=float(T[min(index + 1, len(T) - 1)]),
            )
        )
    return candidates


def _select_one_per_filling(points):
    selected = {}
    for point in points:
        current = selected.get(point.nu)
        if current is None or point.confidence > current.confidence:
            selected[point.nu] = point
    return sorted(selected.values(), key=lambda point: point.nu)


def select_physical_branches(points, fillings, config=None):
    """Continuity-score candidates using physical delta-nu and delta-T."""
    config = config or PhaseExtractionConfig()
    points = _select_one_per_filling(points)
    if not points:
        return []
    fillings = np.sort(np.asarray(fillings, float))
    dnu = float(np.median(np.diff(fillings))) if len(fillings) > 1 else 0.01
    nu_radius = max((config.branch_max_missing_columns + 2) * dnu, 0.008)
    selected = []
    for point in points:
        left = [candidate for candidate in points if 0 < point.nu - candidate.nu <= nu_radius]
        right = [candidate for candidate in points if 0 < candidate.nu - point.nu <= nu_radius]

        def strongest(neighbors):
            values = []
            for candidate in neighbors:
                delta_nu = abs(candidate.nu - point.nu)
                allowed_T = max(0.20, config.branch_max_slope * delta_nu)
                delta_T = abs(candidate.temperature - point.temperature)
                if delta_T > 2.5 * allowed_T:
                    continue
                weight = math.exp(
                    -0.5 * (delta_nu / nu_radius) ** 2 - 0.5 * (delta_T / allowed_T) ** 2
                )
                values.append(candidate.confidence * weight)
            return max(values, default=0.0)

        left_support = strongest(left)
        right_support = strongest(right)
        if not left:
            left_support = right_support
        if not right:
            right_support = left_support
        support = math.sqrt(left_support * right_support)
        point.support = float(support)
        point.confidence = float(math.sqrt(max(point.confidence, 0) * max(support, 0)))
        if point.confidence >= config.branch_min_confidence:
            selected.append(point)

    if not selected:
        return []
    selected.sort(key=lambda point: point.nu)
    components = [[selected[0]]]
    max_nu_gap = (config.branch_max_missing_columns + 1) * dnu * 1.05
    for point in selected[1:]:
        previous = components[-1][-1]
        delta_nu = point.nu - previous.nu
        allowed_T = max(0.25, config.branch_max_slope * delta_nu)
        if delta_nu <= max_nu_gap and abs(point.temperature - previous.temperature) <= allowed_T:
            components[-1].append(point)
        else:
            components.append([point])

    kept = []
    component_id = 0
    for component in components:
        if len(component) < config.branch_min_points:
            continue
        for point in component:
            point.component = component_id
            kept.append(point)
        component_id += 1
    return kept


def select_tcoh_envelopes(points, fillings, config=None):
    """Retain broad, physically smooth Tcoh support across filling.

    Per-linecut model validation happens before this function.  This stage is
    intentionally permissive about continuity: it removes only isolated
    temperature outliers, bridges modest candidate dropouts, and preserves
    measured-versus-censored state on every point.
    """
    config = config or PhaseExtractionConfig()
    points = sorted(
        [
            point
            for point in _select_one_per_filling(points)
            if point.confidence >= config.tcoh_min_confidence
        ],
        key=lambda point: point.nu,
    )
    if not points:
        return []
    fillings = np.sort(np.asarray(fillings, float))
    dnu = float(np.median(np.diff(fillings))) if len(fillings) > 1 else 0.01
    maximum_gap = config.tcoh_max_gap_fraction * float(np.ptp(fillings)) + 1.05 * dnu

    # Reject a lone vertical excursion only when several nearby linecuts
    # establish a substantially different local envelope.
    filtered = []
    for index, point in enumerate(points):
        neighborhood = points[max(0, index - 2) : min(len(points), index + 3)]
        nearby = [
            candidate.temperature
            for candidate in neighborhood
            if candidate is not point and abs(candidate.nu - point.nu) <= maximum_gap
        ]
        if len(nearby) >= 3 and not point.censored:
            center = float(np.median(nearby))
            mad = 1.4826 * float(np.median(np.abs(np.asarray(nearby) - center)))
            local_steps = [
                abs(candidate.nu - point.nu) for candidate in neighborhood if candidate is not point
            ]
            local_dnu = float(np.median(local_steps)) if local_steps else dnu
            tolerance = max(0.55, 3.0 * mad, 1.5 * config.tcoh_max_slope * local_dnu)
            if abs(point.temperature - center) > tolerance:
                continue
        filtered.append(point)

    if not filtered:
        return []
    components = [[filtered[0]]]
    for point in filtered[1:]:
        previous = components[-1][-1]
        delta_nu = point.nu - previous.nu
        allowed_temperature = max(0.60, config.tcoh_max_slope * delta_nu)
        if (
            delta_nu <= maximum_gap
            and abs(point.temperature - previous.temperature) <= allowed_temperature
        ):
            components[-1].append(point)
        else:
            components.append([point])

    kept = []
    component_id = 0
    for component in components:
        if len(component) < config.tcoh_min_component_points:
            continue
        for point in component:
            point.component = component_id
            point.support = point.confidence
            kept.append(point)
        component_id += 1
    return kept


def select_primary_physical_path(points, fillings, config=None):
    """Select one globally smooth, single-valued boundary from competing minima.

    This dynamic-programming path operates on physical filling and temperature,
    not array indices.  It is used for Tneel because one linecut can contain
    multiple genuine minima whose nearly tied local scores otherwise make a
    branch jump between unrelated temperatures.
    """
    config = config or PhaseExtractionConfig()
    nodes = sorted(points, key=lambda point: (point.nu, point.temperature))
    if not nodes:
        return []
    fillings = np.sort(np.asarray(fillings, float))
    dnu = float(np.median(np.diff(fillings))) if len(fillings) > 1 else 0.01
    maximum_gap = 0.05 * float(np.ptp(fillings)) + 1.05 * dnu
    score = np.asarray(
        [max(point.confidence - config.branch_min_confidence, 0.0) for point in nodes], float
    )
    previous = np.full(len(nodes), -1, int)

    for index, point in enumerate(nodes):
        reward = score[index]
        for prior_index in range(index):
            prior = nodes[prior_index]
            delta_nu = point.nu - prior.nu
            if delta_nu <= 0 or delta_nu > maximum_gap:
                continue
            allowed_temperature = max(0.15, config.branch_max_slope * delta_nu)
            uncertainty_allowance = point.uncertainty + prior.uncertainty
            slope_z = (
                max(0.0, abs(point.temperature - prior.temperature) - uncertainty_allowance)
                / allowed_temperature
            )
            if slope_z > 1.0:
                continue
            missing_penalty = 0.10 * max(delta_nu / dnu - 1.0, 0.0)
            edge_penalty = missing_penalty + 0.25 * slope_z**2
            candidate_score = score[prior_index] - edge_penalty + reward
            if candidate_score > score[index]:
                score[index] = candidate_score
                previous[index] = prior_index

    cursor = int(np.argmax(score))
    path = []
    while cursor >= 0:
        path.append(nodes[cursor])
        cursor = int(previous[cursor])
    path.reverse()
    if len(path) < config.branch_min_points:
        return []

    # Preserve visible data gaps even though the global scoring recognizes the
    # segments as one logical boundary.
    component = 0
    path[0].component = component
    path[0].support = path[0].confidence
    visible_gap = (config.branch_max_missing_columns + 1) * dnu * 1.05
    for prior, point in zip(path, path[1:]):
        delta_nu = point.nu - prior.nu
        allowed_temperature = max(0.25, config.branch_max_slope * delta_nu)
        if (
            delta_nu > visible_gap
            or abs(point.temperature - prior.temperature) > allowed_temperature
        ):
            component += 1
        point.component = component
        point.support = point.confidence
    component_counts = {
        component_id: sum(point.component == component_id for point in path)
        for component_id in {point.component for point in path}
    }
    kept_components = {
        component_id
        for component_id, count in component_counts.items()
        if count >= config.branch_min_points
    }
    kept = [point for point in path if point.component in kept_components]
    component_map = {
        old: new for new, old in enumerate(sorted({point.component for point in kept}))
    }
    for point in kept:
        point.component = component_map[point.component]
    return kept


def _prepare_resistance_for_phase_extraction(T, resistance):
    """Mask invalid rho for inference and estimate heteroscedastic linecut noise."""
    resistance = np.asarray(resistance, float)
    valid = np.isfinite(resistance) & (resistance > 0)
    prepared = resistance.copy()
    for column in range(prepared.shape[1]):
        column_valid = valid[:, column]
        if np.count_nonzero(column_valid) < 3:
            raise ValueError(f"linecut {column} has fewer than three positive resistance values")
        prepared[~column_valid, column] = np.interp(
            T[~column_valid], T[column_valid], prepared[column_valid, column]
        )

    pooled_sigma = estimate_noise_matrix(T, prepared)
    sigma = np.column_stack(
        [
            np.maximum(pooled_sigma, estimate_noise_1d(T, prepared[:, column]))
            for column in range(prepared.shape[1])
        ]
    )
    smoothed = np.column_stack(
        [
            adaptive_multiscale_smooth(T, prepared[:, column], sigma[:, column], z_threshold=2.0)
            for column in range(prepared.shape[1])
        ]
    )
    return prepared, smoothed, sigma, valid


def extract_field_phase_diagram(field, input_dir, config=None):
    """Load one field and return smoothed data plus selected phase boundaries."""
    config = config or PhaseExtractionConfig()
    T, fillings, resistance = clean_sort_data(*load_field(field, Path(input_dir)))
    prepared, smoothed, sigma, valid = _prepare_resistance_for_phase_extraction(T, resistance)
    raw_by_type = {name: [] for name in ("Tcoh", "Tprime", "Tneel")}

    for column, nu in enumerate(fillings):
        rho = resistance[:, column]
        smooth = smoothed[:, column]
        tcoh = extract_tcoh_linecut(
            T, rho, smooth, sigma[:, column], field=-abs(float(field)), nu=nu, config=config
        )
        if tcoh is not None:
            raw_by_type["Tcoh"].append(tcoh)
        tprime = extract_tprime_linecut(
            T, rho, smooth, sigma[:, column], field=-abs(float(field)), nu=nu, config=config
        )
        if tprime is not None:
            raw_by_type["Tprime"].append(tprime)
        raw_by_type["Tneel"].extend(
            extract_tneel_candidates(
                T, rho, smooth, sigma[:, column], field=-abs(float(field)), nu=nu, config=config
            )
        )

    # A strong upturn/minimum is incompatible with treating the same linecut
    # as an uninterrupted low-T FL or sublinear metal.
    tneel = select_primary_physical_path(raw_by_type["Tneel"], fillings, config)
    tneel_nu = np.asarray([point.nu for point in tneel], float)
    exclusion = max(2.5 * np.median(np.diff(fillings)), 0.004)

    selected_by_type = {"Tneel": tneel}
    for transition in ("Tcoh", "Tprime"):
        eligible = [
            point
            for point in raw_by_type[transition]
            if not len(tneel_nu) or np.min(np.abs(tneel_nu - point.nu)) > exclusion
        ]
        selected_by_type[transition] = (
            select_tcoh_envelopes(eligible, fillings, config)
            if transition == "Tcoh"
            else select_physical_branches(eligible, fillings, config)
        )

    # A branch that reaches the measured filling edge is open-ended in nu; it
    # must not be drawn as though the physical phase terminates there.
    filling_step = float(np.median(np.diff(fillings)))
    for transition, selected in selected_by_type.items():
        if transition != "Tcoh":
            continue
        components = {point.component for point in selected}
        for component in components:
            branch = sorted(
                [point for point in selected if point.component == component],
                key=lambda point: point.nu,
            )
            if branch[0].nu - fillings[0] <= 2.1 * filling_step:
                branch[0].nu_censored = True
                branch[0].nu_censor_side = "lower"
            if fillings[-1] - branch[-1].nu <= 2.1 * filling_step:
                branch[-1].nu_censored = True
                branch[-1].nu_censor_side = "upper"

    points = [point for values in selected_by_type.values() for point in values]
    points.sort(key=lambda point: (point.transition, point.component or 0, point.nu))
    return {
        "field": -abs(float(field)),
        "temperature": T,
        "filling": fillings,
        "resistance": resistance,
        "smoothed": smoothed,
        "noise": sigma,
        "valid_resistance": valid,
        "points": points,
        "raw_candidates": raw_by_type,
        "config": asdict(config),
    }
