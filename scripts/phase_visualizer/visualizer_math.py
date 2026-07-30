"""Python data preparation and fitting for the linecut explorer."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from moire.adaptive_multiscale_smooth import (  # noqa: E402
    adaptive_multiscale_smooth,
    estimate_noise_matrix,
)
from moire.extract_features import (  # noqa: E402
    extract_Tc,
    extract_downturns,
    extract_upturns,
    get_fit_range,
)
from moire.io import clean_sort_data, load_field  # noqa: E402
from moire.signal_helpers import local_noise  # noqa: E402
from moire.update_scoring import update_score  # noqa: E402

DEFAULT_FIELDS = [74, 87, 96, 96.2, 99, 103, 151, 176]
LOSSES = {"linear", "soft_l1", "cauchy"}
SOURCES = {"smoothed", "raw"}
NOISE_MODELS = {"none", "local", "fit_residual", "pooled"}
N_BOUNDS = (0.1, 4.0)


def _finite(value, digits=8):
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _values(array, digits=8):
    return [_finite(value, digits) for value in np.asarray(array)]


def build_field(field):
    """Prepare one field's linecuts, features, fit ranges, and noise estimates."""
    load_value = int(field) if float(field).is_integer() else field
    T, fillings, resistivity = clean_sort_data(*load_field(load_value, ROOT / "source_data"))
    pooled_noise = estimate_noise_matrix(T, resistivity)

    linecuts = []
    for index, filling in enumerate(fillings):
        raw = resistivity[:, index]
        smoothed = adaptive_multiscale_smooth(T, raw, z_threshold=3)
        linecut = {
            "E": load_value,
            "nu": filling,
            "T": T,
            "rho": raw,
            "rho_smoothed": smoothed,
            "local_noise": local_noise(T, raw, smoothed),
            "behaviors": [],
        }
        linecut["features"] = (
            extract_upturns(T, linecut) + extract_downturns(T, linecut) + extract_Tc(T, linecut)
        )
        linecuts.append(linecut)

    update_score(linecuts)
    for linecut in linecuts:
        get_fit_range(T, linecut)

    serialized = []
    accepted = 0
    for linecut in linecuts:
        extraction = next(
            (
                behavior
                for behavior in linecut["behaviors"]
                if behavior.get("type") == "extraction_range"
            ),
            None,
        )
        if extraction is not None:
            accepted += 1

        serialized.append(
            {
                "raw": _values(linecut["rho"], 6),
                "smoothed": _values(linecut["rho_smoothed"], 6),
                "localNoise": _values(linecut["local_noise"], 6),
                "features": [
                    {
                        "T": _finite(feature["T"]),
                        "type": feature["type"],
                        "score": _finite(feature["score_15"]),
                    }
                    for feature in linecut["features_new"]
                ],
                "range": (
                    {
                        "lower": _finite(extraction["T_lower"]),
                        "upper": _finite(extraction["T_upper"]),
                    }
                    if extraction is not None
                    else None
                ),
            }
        )

    positive = resistivity[np.isfinite(resistivity) & (resistivity > 0)]
    low, high = np.percentile(positive, [1, 99])
    stride = max(1, math.ceil(len(fillings) / 300))

    return {
        "field": _finite(load_value),
        "temperatures": _values(T),
        "fillings": _values(fillings),
        "pooledNoise": _values(pooled_noise, 6),
        "linecuts": serialized,
        "heatFillings": _values(fillings[::stride]),
        "heatmap": [_values(row, 6) for row in np.asarray(resistivity)[:, ::stride]],
        "logMin": math.log10(float(low)),
        "logMax": math.log10(float(high)),
        "acceptedCount": accepted,
    }


def build_dataset(fields=DEFAULT_FIELDS):
    return {"fields": [build_field(field) for field in fields]}


def _safe_scale(values):
    """Return a positive robust scale with an RMS fallback."""
    values = np.asarray(values, float)
    center = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - center))
    floor = np.finfo(float).eps * max(float(np.max(np.abs(values))), 1.0)
    if not np.isfinite(scale) or scale <= floor:
        scale = np.sqrt(np.mean((values - center) ** 2))
    return max(float(scale), floor)


def _fit_model(T, rho, sigma, loss, initial=None):
    """Run one bounded robust fit and return its curve and four outputs."""
    T_ref = float(np.median(T))
    t = T / T_ref
    log_t = np.log(t)
    design = np.column_stack((np.ones_like(t), t))
    rho0, coefficient = np.linalg.lstsq(design, rho, rcond=None)[0]
    coefficient = max(float(coefficient), np.finfo(float).eps)
    start = np.array([rho0, coefficient, 1.0], float)
    if initial is not None:
        candidate = np.array(
            [
                initial["rho0"],
                initial["A"] * T_ref ** initial["n"],
                initial["n"],
            ],
            float,
        )
        if np.all(np.isfinite(candidate)) and candidate[1] >= 0:
            candidate[1] = max(candidate[1], np.finfo(float).eps)
            candidate[2] = np.clip(candidate[2], N_BOUNDS[0], N_BOUNDS[1])
            start = candidate

    def residuals(parameters):
        offset, scale, exponent = parameters
        return (offset + scale * t**exponent - rho) / sigma

    def jacobian(parameters):
        _, scale, exponent = parameters
        power = t**exponent
        return np.column_stack(
            (
                1 / sigma,
                power / sigma,
                scale * power * log_t / sigma,
            )
        )

    result = least_squares(
        residuals,
        start,
        jac=jacobian,
        bounds=([-np.inf, 0.0, N_BOUNDS[0]], [np.inf, np.inf, N_BOUNDS[1]]),
        loss=loss,
        f_scale=1.0,
        x_scale="jac",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return None

    rho0, coefficient, n = (float(value) for value in result.x)
    A = coefficient / T_ref**n
    curve = rho0 + A * T**n

    n_sigma = np.nan
    dof = len(T) - 3
    normal = result.jac.T @ result.jac
    if dof > 0 and np.linalg.matrix_rank(normal) == 3:
        covariance = np.linalg.inv(normal) * (2 * result.cost / dof)
        variance = float(covariance[2, 2])
        if np.isfinite(variance) and variance >= 0:
            n_sigma = math.sqrt(variance)
    if np.isclose(n, N_BOUNDS).any():
        n_sigma = np.nan

    return {"rho0": rho0, "A": A, "n": n, "n_sigma": n_sigma, "curve": curve}


def _window(T, center, left_bound, right_bound, min_points, min_span):
    """Return the smallest nearby window satisfying both fit constraints."""
    if right_bound - left_bound + 1 < min_points:
        return None
    if T[right_bound] - T[left_bound] < min_span:
        return None

    left = right = center
    while right - left + 1 < min_points or T[right] - T[left] < min_span:
        left_distance = T[center] - T[left - 1] if left > left_bound else np.inf
        right_distance = T[right + 1] - T[center] if right < right_bound else np.inf
        if np.isinf(left_distance) and np.isinf(right_distance):
            return None
        if left_distance <= right_distance:
            left -= 1
        else:
            right += 1
    return left, right


def _fit_noise(mode, T, source, raw, local_sigma, pooled_sigma):
    if mode == "none":
        return np.ones_like(T), None
    if mode == "local":
        return local_sigma, None
    if mode == "pooled":
        return pooled_sigma, None

    preliminary = _fit_model(T, source, np.ones_like(T), "linear")
    if preliminary is None:
        return np.full_like(T, _safe_scale(raw)), None
    return np.full_like(T, _safe_scale(raw - preliminary["curve"])), preliminary


def fit_linecut(
    field_data,
    linecut_index,
    *,
    source="smoothed",
    loss="soft_l1",
    noise="none",
    min_points=9,
    min_span=1.0,
    include_curves=True,
):
    """Fit every valid local window for one linecut."""
    if source not in SOURCES or loss not in LOSSES or noise not in NOISE_MODELS:
        raise ValueError("unknown source, loss, or noise model")
    if min_points < 4 or min_span <= 0:
        raise ValueError("invalid fit-window constraints")

    T = np.asarray(field_data["temperatures"], float)
    pooled_sigma = np.asarray(field_data["pooledNoise"], float)
    linecut = field_data["linecuts"][linecut_index]
    raw = np.asarray(linecut["raw"], float)
    smoothed = np.asarray(linecut["smoothed"], float)
    local_sigma = np.asarray(linecut["localNoise"], float)
    fit_source = smoothed if source == "smoothed" else raw

    names = ("rho0", "A", "n", "n_sigma")
    local_fit = {name: [None] * len(T) for name in names}
    fit_windows = {"left": [None] * len(T), "right": [None] * len(T), "curve": [None] * len(T)}
    extraction = linecut["range"]
    if extraction is None:
        return {"localFit": local_fit, "fitWindows": fit_windows}

    indices = np.flatnonzero((T >= extraction["lower"]) & (T <= extraction["upper"]))
    if not len(indices):
        return {"localFit": local_fit, "fitWindows": fit_windows}

    left_bound, right_bound = int(indices[0]), int(indices[-1])
    cache = {}
    previous_fit = None
    for center in indices:
        center = int(center)
        window = _window(T, center, left_bound, right_bound, min_points, min_span)
        if window is None:
            continue

        if window not in cache:
            left, right = window
            selection = slice(left, right + 1)
            sigma, preliminary = _fit_noise(
                noise,
                T[selection],
                fit_source[selection],
                raw[selection],
                local_sigma[selection],
                pooled_sigma[selection],
            )
            cache[window] = _fit_model(
                T[selection],
                fit_source[selection],
                sigma,
                loss,
                initial=(previous_fit or preliminary) if loss == "linear" else None,
            )

        fit = cache[window]
        if fit is None:
            continue
        previous_fit = fit

        for name in names:
            local_fit[name][center] = _finite(fit[name])
        if include_curves:
            left, right = window
            fit_windows["left"][center] = left
            fit_windows["right"][center] = right
            fit_windows["curve"][center] = _values(fit["curve"], 6)

    return {"localFit": local_fit, "fitWindows": fit_windows}
