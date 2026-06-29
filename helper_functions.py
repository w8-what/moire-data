
import math
import numpy as np

from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

def fmt4(x):
    if x == 0:
        return "0.000"

    digits_before = 1 if abs(x) < 1 else int(math.log10(abs(x))) + 1
    decimals = max(0, 4 - digits_before)
    return f"{x:.{decimals}f}"

def contiguous_regions(mask) -> list:
    """
    Given a Boolean array, return inclusive index intervals where mask is True.

    Example:
        mask = [False, True, True, False, True]
        returns [(1, 2), (4, 4)]
    """
    mask = np.asarray(mask, dtype=bool)
    true_idx = np.flatnonzero(mask)

    if len(true_idx) == 0:
        return []

    breaks = np.where(np.diff(true_idx) > 1)[0]

    starts = np.r_[true_idx[0], true_idx[breaks + 1]]
    ends = np.r_[true_idx[breaks], true_idx[-1]]

    return list(zip(starts, ends))


def robust_snr(x, T, background_window_T=0.5, polyorder=1):
    """
    Robust residual-MAD SNR for a signal x(T).

    Intended use:
        x = d²ρ/dT²

    Method:
        1. Smooth x(T) with a broad temperature window to estimate background
        2. residual = x - background
        3. noise = 1.4826 * MAD(residual)
        4. snr = |residual| / noise

    Parameters
    ----------
    T : array
        Temperature values. Uneven spacing allowed.

    x : array
        Signal array, e.g. d²ρ/dT².

    background_window_T : float
        Smoothing window in Kelvin for the slow background.
        Should be wider than the physical transition width.

    polyorder : int
        Savitzky-Golay polynomial order used by smooth_rho.

    Returns
    -------
    snr : array
        Dimensionless SNR at each point.

    residual : array
        x - smooth background.

    background : array
        Smooth background of x.

    noise : float
        Robust noise estimate from residual MAD.
    """

    T = np.asarray(T, dtype=float)
    x = np.asarray(x, dtype=float)

    snr = np.full_like(x, np.nan, dtype=float)
    residual = np.full_like(x, np.nan, dtype=float)
    background = np.full_like(x, np.nan, dtype=float)

    mask = np.isfinite(T) & np.isfinite(x)

    if np.sum(mask) < polyorder + 3:
        return snr, residual, background, np.nan

    T_valid = T[mask]
    x_valid = x[mask]

    T_range = np.nanmax(T_valid) - np.nanmin(T_valid)

    if T_range <= 0:
        return snr, residual, background, np.nan

    # Convert absolute Kelvin window into your smooth_rho window_frac
    window_frac = background_window_T / T_range

    background_valid = smooth_rho(
        T_valid,
        x_valid,
        window_frac=window_frac,
        polyorder=polyorder,
    )

    residual_valid = x_valid - background_valid

    med = np.nanmedian(residual_valid)
    mad = np.nanmedian(np.abs(residual_valid - med))

    noise = 1.4826 * mad

    if noise == 0 or not np.isfinite(noise):
        noise = np.nanstd(residual_valid)

    if noise == 0 or not np.isfinite(noise):
        return snr, residual, background, noise

    snr_valid = np.abs(residual_valid) / noise

    snr[mask] = snr_valid
    residual[mask] = residual_valid
    background[mask] = background_valid

    return snr, residual, background, noise

def clean_boolean_mask(mask, max_gap=1, min_len=3):
    """
    Fill small False holes and remove short True islands.

    max_gap:
        Fill False gaps of length <= max_gap between True regions.

    min_len:
        Remove True regions shorter than min_len.
    """

    mask = np.asarray(mask, dtype=bool).copy()

    # Fill small holes: True False True -> True True True
    for left, right in contiguous_regions(~mask):
        touches_edge = left == 0 or right == len(mask) - 1
        gap_len = right - left + 1

        if not touches_edge and gap_len <= max_gap:
            mask[left:right + 1] = True

    # Remove tiny islands: False True False -> False False False
    for left, right in contiguous_regions(mask):
        region_len = right - left + 1

        if region_len < min_len:
            mask[left:right + 1] = False

    return mask


def smooth_rho(rho, T, window_frac=0.1, polyorder=1):
    """
    Lightly smooth rho(T) when T is unevenly spaced.

    Same signature as before:
        smooth_rho(T, rho, window_frac=0.08, polyorder=2)

    Difference:
        window_frac is now interpreted as a fraction of the T-range,
        not a fraction of the number of indices.
    """

    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)

    rho_out = np.full_like(rho, np.nan, dtype=float)

    mask = np.isfinite(T) & np.isfinite(rho)
    if np.sum(mask) < polyorder + 3:
        return rho.copy()

    T_valid = T[mask]
    rho_valid = rho[mask]

    order = np.argsort(T_valid)
    T_sorted = T_valid[order]
    rho_sorted = rho_valid[order]

    # Combine duplicate T values using median rho
    unique_T = np.unique(T_sorted)
    if len(unique_T) < len(T_sorted):
        rho_unique = np.array([
            np.median(rho_sorted[T_sorted == t]) for t in unique_T
        ])
        T_sorted = unique_T
        rho_sorted = rho_unique

    if len(T_sorted) < polyorder + 3:
        rho_out[mask] = rho_valid
        return rho_out

    dT = np.diff(T_sorted)
    dT = dT[dT > 0]

    if len(dT) == 0:
        rho_out[mask] = rho_valid
        return rho_out

    # Uniform grid spacing based on actual temperature spacing
    grid_step = np.median(dT)

    T_grid = np.arange(
        T_sorted.min(),
        T_sorted.max() + grid_step,
        grid_step
    )

    # Interpolate uneven data onto uniform T grid
    interp_raw = PchipInterpolator(T_sorted, rho_sorted)
    rho_grid = interp_raw(T_grid)

    # Convert fractional T-window into SavGol index-window
    T_range = T_sorted.max() - T_sorted.min()
    window_T = window_frac * T_range

    window_length = int(round(window_T / grid_step))

    # SavGol requirements
    window_length = max(window_length, polyorder + 3)
    if window_length % 2 == 0:
        window_length += 1

    if window_length >= len(T_grid):
        window_length = len(T_grid) - 1
        if window_length % 2 == 0:
            window_length -= 1

    if window_length <= polyorder:
        rho_out[mask] = rho_valid
        return rho_out

    # Smooth on uniform grid
    rho_grid_smooth = savgol_filter(
        rho_grid,
        window_length=window_length,
        polyorder=polyorder,
        mode="interp"
    )

    # Interpolate smoothed curve back to original uneven T points
    interp_smooth = PchipInterpolator(T_grid, rho_grid_smooth)
    rho_smooth_valid = interp_smooth(T_valid)

    rho_out[mask] = rho_smooth_valid

    return rho_out

# ----- HELPER FUNCTIONS FOR ADAPTIVE SMOOTHING -----


def mad(x):
    # Robust version of std; less affected by real transitions or spikes.
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def local_poly(T, y, x0, h, deg=2):
    # Use a temperature window, not a point-count window.
    idx = np.abs(T - x0) <= h

    # Local polynomial needs enough points to fit stably.
    if idx.sum() < deg + 2:
        idx = np.argsort(np.abs(T - x0))[:deg + 2]

    x = T[idx] - x0
    yy = y[idx]

    # Tricube weights: closer points matter more.
    u = np.abs(x) / h
    w = (1 - u**3)**3
    w[u >= 1] = 0

    X = np.vstack([x**p for p in range(deg + 1)]).T

    # Weighted least squares.
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], yy * sw, rcond=None)[0]

    # beta[0] is the fitted value at x0 because x = T - x0.
    return beta[0]


def adaptive_smooth(rho, T, deg=1, h_min=0.5, h_max=None, sensitivity=2):
    dT = np.median(np.diff(T))
    Tr = T[-1] - T[0]

    # h_min prevents the smoother from chasing noise.
    # h_max is the broad window used in smooth/background regions.
    h_min = 3 * dT if h_min is None else h_min
    h_max = 0.15 * Tr if h_max is None else h_max

    # First pass: broad smooth so curvature is not dominated by raw noise.
    rough = np.array([local_poly(T, rho, t, h_max, deg) for t in T])

    # Curvature estimates where the curve bends sharply.
    d1 = np.gradient(rough, T)
    d2 = np.gradient(d1, T)
    sharp = np.abs(d2)

    # MAD-score = how unusually sharp this point is compared to this linecut.
    score = (sharp - np.median(sharp)) / mad(sharp)
    score = np.clip(score, 0, 8)

    # Large score -> shrink smoothing window to preserve transition.
    # Small score -> use large window to smooth boring regions.
    h = h_max / (1 + score / sensitivity)
    h = np.clip(h, h_min, h_max)

    # Final pass: adaptive smoothing with local temperature window.
    smooth = np.array([local_poly(T, rho, T[i], h[i], deg) for i in range(len(T))])

    return smooth
