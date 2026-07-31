"""Local power-law fits for screened resistivity linecuts."""

import numpy as np
from scipy.optimize import least_squares


def _fit(T, rho, sigma, n_bounds):
    """Fit rho = rho0 + A*T**n and return only its parameters and sigma_n."""
    T_ref = np.median(T)
    t = T / T_ref

    rho0, coefficient = np.linalg.lstsq(np.column_stack((np.ones_like(t), t)), rho, rcond=None)[0]
    coefficient = max(coefficient, np.finfo(float).eps)

    def residuals(parameters):
        rho0, coefficient, n = parameters
        return (rho0 + coefficient * t**n - rho) / sigma

    result = least_squares(
        residuals,
        [rho0, coefficient, 1.0],
        bounds=([-np.inf, 0.0, n_bounds[0]], [np.inf, np.inf, n_bounds[1]]),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
    )
    if not result.success:
        return None

    rho0, coefficient, n = result.x
    A = coefficient / T_ref**n

    n_sigma = np.nan
    dof = len(T) - 3
    normal = result.jac.T @ result.jac
    if dof > 0 and np.linalg.matrix_rank(normal) == 3:
        covariance = np.linalg.inv(normal) * (2 * result.cost / dof)
        n_sigma = np.sqrt(max(covariance[2, 2], 0.0))

    if np.isclose(n, n_bounds).any():
        n_sigma = np.nan

    return {"rho0": float(rho0), "A": float(A), "n": float(n), "n_sigma": float(n_sigma)}


def _window(T, center, left_bound, right_bound, min_pts, min_T):
    """Return the smallest centered window satisfying both constraints."""
    if right_bound - left_bound + 1 < min_pts:
        return None
    if T[right_bound] - T[left_bound] < min_T:
        return None

    left = right = center
    while right - left + 1 < min_pts or T[right] - T[left] < min_T:
        left_distance = T[center] - T[left - 1] if left > left_bound else np.inf
        right_distance = T[right + 1] - T[center] if right < right_bound else np.inf
        if left_distance == right_distance == np.inf:
            return None
        if left_distance <= right_distance:
            left -= 1
        else:
            right += 1

    return left, right


def extract_local_fits(T, linecut, rho = "rho", min_pts=10, min_T=1.0, n_bounds=(0.1, 4.0)):

    T = np.asarray(T, float)
    rho = np.asarray(linecut[rho], float)
    sigma = np.asarray(linecut["local_noise"], float)

    if T.ndim != 1 or rho.shape != T.shape or sigma.shape != T.shape:
        raise ValueError("T, rho_smoothed, and local_noise must be matching 1D arrays")
    if not np.all(np.isfinite(T)) or not np.all(T > 0) or not np.all(np.diff(T) > 0):
        raise ValueError("T must be finite, positive, and strictly increasing")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho_smoothed must contain only finite values")
    if not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError("local_noise must contain only finite positive values")
    if min_pts < 4 or min_T <= 0:
        raise ValueError("min_pts must be at least 4 and min_T must be positive")
    if len(n_bounds) != 2 or not 0 < n_bounds[0] < n_bounds[1]:
        raise ValueError("n_bounds must contain two positive increasing values")

    output = {name: [np.nan] * len(T) for name in ("rho0", "A", "n", "n_sigma")}
    cache = {}

    for behavior in linecut.get("behaviors", []):
        if behavior.get("type") != "extraction_range":
            continue

        lower, upper = sorted((behavior["T_lower"], behavior["T_upper"]))
        indices = np.flatnonzero((T >= lower) & (T <= upper))
        if not len(indices):
            continue

        left_bound, right_bound = indices[0], indices[-1]
        for center in indices:
            window = _window(T, center, left_bound, right_bound, min_pts, min_T)
            if window is None:
                continue

            left, right = window
            if window not in cache:
                selection = slice(left, right + 1)
                cache[window] = _fit(T[selection], rho[selection], sigma[selection], n_bounds)
            fit = cache[window]
            if fit is not None:
                for name, value in fit.items():
                    output[name][center] = value

    return output
