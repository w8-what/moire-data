"""Feature-preserving smoothing for unevenly sampled rho(T) linecuts.

The main routine uses robust local-linear fits over many physical temperature
scales and keeps the widest scale that is statistically consistent with the
finer-scale fits.  T must be finite, one-dimensional, and strictly increasing.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def _mad(x: np.ndarray, axis=None) -> np.ndarray:
    center = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - center), axis=axis)


def estimate_noise_matrix(T: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Estimate sigma(T) by pooling detrended neighbor residuals over linecuts."""
    T = np.asarray(T, float)
    R = np.asarray(R, float)
    sigma = np.empty(len(T))

    left = (T[2:] - T[1:-1]) / (T[2:] - T[:-2])
    right = (T[1:-1] - T[:-2]) / (T[2:] - T[:-2])
    residual = R[1:-1] - left[:, None] * R[:-2] - right[:, None] * R[2:]
    normalization = np.sqrt(1 + left**2 + right**2)
    sigma[1:-1] = _mad(residual / normalization[:, None], axis=1)
    sigma[0], sigma[-1] = sigma[1], sigma[-2]

    # Pool nearby temperatures on the log scale without letting a few large
    # transition residuals inflate the estimated measurement noise.
    log_sigma = np.log(np.maximum(sigma, np.nanmedian(sigma[sigma > 0]) * 1e-3))
    log_sigma = median_filter(log_sigma, size=5, mode="nearest")
    return np.exp(log_sigma)


def estimate_noise_1d(T: np.ndarray, y: np.ndarray, neighbors: int = 11) -> np.ndarray:
    """Fallback sigma(T) estimate when only one linecut is available."""
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    n = len(T)

    left = (T[2:] - T[1:-1]) / (T[2:] - T[:-2])
    right = (T[1:-1] - T[:-2]) / (T[2:] - T[:-2])
    residual = np.empty(n)
    residual[1:-1] = (
        y[1:-1] - left * y[:-2] - right * y[2:]
    ) / np.sqrt(1 + left**2 + right**2)
    residual[0], residual[-1] = residual[1], residual[-2]

    sigma = np.empty(n)
    k = min(neighbors, n)
    for i in range(n):
        idx = np.argpartition(np.abs(T - T[i]), k - 1)[:k]
        sigma[i] = _mad(residual[idx])

    floor = max(np.nanmedian(sigma[sigma > 0]) * 0.1, np.finfo(float).eps)
    return np.maximum(median_filter(sigma, size=5, mode="nearest"), floor)


def _isolated_point_weights(T: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Conservatively downweight only isolated, >5-sigma point glitches."""
    weights = np.ones(len(T))
    left = (T[2:] - T[1:-1]) / (T[2:] - T[:-2])
    right = 1 - left
    residual = y[1:-1] - left * y[:-2] - right * y[2:]
    residual_sigma = np.sqrt(
        sigma[1:-1] ** 2 + left**2 * sigma[:-2] ** 2 + right**2 * sigma[2:] ** 2
    )
    z = np.abs(residual) / np.maximum(residual_sigma, np.finfo(float).eps)
    weights[1:-1] = np.minimum(1.0, 5.0 / np.maximum(z, 1.0))
    return weights


def _local_linear_fit(
    T: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    point_weights: np.ndarray,
    i: int,
    radius: float,
    min_points: int,
) -> tuple[float, float, float]:
    distance = np.abs(T - T[i])
    idx = np.flatnonzero(distance <= radius)
    if len(idx) < min_points:
        idx = np.argpartition(distance, min_points - 1)[:min_points]

    h = max(distance[idx].max() * 1.000001, np.finfo(float).eps)
    kernel = (1 - (distance[idx] / h) ** 3) ** 3
    x = T[idx] - T[i]
    X = np.column_stack((np.ones(len(idx)), x))
    w = kernel * point_weights[idx] / np.maximum(sigma[idx] ** 2, np.finfo(float).eps)

    normal = X.T @ (w[:, None] * X)
    inverse = np.linalg.pinv(normal)
    beta = inverse @ (X.T @ (w * y[idx]))

    # Covariance of a kernel-weighted, heteroscedastic local-linear estimate.
    middle = X.T @ (((w * sigma[idx]) ** 2)[:, None] * X)
    variance = max((inverse @ middle @ inverse)[0, 0], 0.0)
    return beta[0], np.sqrt(variance), h


def adaptive_multiscale_smooth(
    T: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray | None = None,
    *,
    min_points: int = 5,
    scales: int = 12,
    z_threshold: float = 1.5,
    smooth_bandwidths: int = 3,
    return_diagnostics: bool = False,
):
    """Smooth an unevenly sampled linecut without choosing one fixed window.

    At every T, local-linear fits are made over a ladder of physical radii.
    The selected radius is the widest one still statistically consistent with
    all finer fits (a Lepski-style multiscale rule). Strong, narrow structure
    therefore selects a small radius; smooth/noisy regions select a large one.

    For a full R(T, parameter) matrix, pass ``estimate_noise_matrix(T, R)`` as
    ``sigma``.  The fallback single-linecut estimate is less reliable.
    """
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    sigma = estimate_noise_1d(T, y) if sigma is None else np.asarray(sigma, float)
    n = len(T)
    min_points = min(max(min_points, 3), n)

    minimum_radius = max(2 * np.min(np.diff(T)), np.ptp(T) / 500)
    maximum_radius = min(0.35 * np.ptp(T), 6.0)
    radii = np.geomspace(minimum_radius, maximum_radius, scales)
    point_weights = _isolated_point_weights(T, y, sigma)

    fits = np.empty((scales, n))
    errors = np.empty_like(fits)
    effective_radii = np.empty_like(fits)
    for level, radius in enumerate(radii):
        for i in range(n):
            fits[level, i], errors[level, i], effective_radii[level, i] = _local_linear_fit(
                T, y, sigma, point_weights, i, radius, min_points
            )

    selected = np.zeros(n, dtype=int)
    for i in range(n):
        for coarse in range(scales - 1, -1, -1):
            difference = np.abs(fits[coarse, i] - fits[: coarse + 1, i])
            uncertainty = z_threshold * np.sqrt(
                errors[coarse, i] ** 2 + errors[: coarse + 1, i] ** 2
            )
            if np.all(difference <= uncertainty):
                selected[i] = coarse
                break

    if smooth_bandwidths > 1:
        selected = median_filter(selected, size=smooth_bandwidths, mode="nearest")
    smoothed = fits[selected, np.arange(n)]

    if not return_diagnostics:
        return smoothed
    return smoothed, {
        "sigma": sigma,
        "selected_level": selected,
        "selected_radius": effective_radii[selected, np.arange(n)],
        "point_weights": point_weights,
        "candidate_radii": radii,
    }


def smooth_matrix(T: np.ndarray, R: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Smooth every column of R using one pooled, temperature-dependent noise estimate."""
    sigma = estimate_noise_matrix(T, R)
    smoothed = np.column_stack(
        [adaptive_multiscale_smooth(T, R[:, j], sigma, **kwargs) for j in range(R.shape[1])]
    )
    return smoothed, sigma
