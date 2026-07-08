
import math
import numpy as np
import pandas as pd 
from hampel import hampel 


def load_field(E, IN):
    df = pd.read_csv(IN / f'Rxx_matrix_E-{E}mV_nm.csv')

    T  = df.iloc[:, 0].astype(float).to_numpy()
    nu = np.array([float(c) for c in df.columns[1:]])
    R  = df.iloc[:, 1:].astype(float).to_numpy()

    # # Keep only rows where T is finite and the whole rho linecut is finite
    # mask = np.isfinite(T) & np.all(np.isfinite(R), axis=1)
    # T, R = T[mask], R[mask]

    # Sort rows by increasing temperature; R rows must follow T
    # idx = np.argsort(T)
    # T, R = T[idx], R[idx]

    return T, nu, R

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


# ----- HELPER FUNCTIONS FOR METALLIC EXTRACTION -----


def smooth_mask(mask, min_len=3):
    mask = np.asarray(mask, dtype=bool).copy()

    while True:
        # Run boundaries: starts inclusive, ends exclusive
        edges = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1, len(mask)]
        lengths = np.diff(edges)

        bad = np.flatnonzero(lengths < min_len)
        if len(bad) == 0 or len(edges) <= 2:
            return mask

        # Remove the shortest bad run first; this avoids weird simultaneous flips
        i = bad[np.argmin(lengths[bad])]
        s, e = edges[i], edges[i + 1]

        if i == 0:
            mask[s:e] = mask[e]          # left edge: merge right
        elif i == len(lengths) - 1:
            mask[s:e] = mask[s - 1]      # right edge: merge left
        else:
            mask[s:e] = mask[s - 1]      # interior: left/right are same for bool runs


# ----- HELPER FUNCTIONS FOR ADAPTIVE SMOOTHING -----


def mad(x):
    # Robust version of std; less affected by real transitions or spikes.
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def local_poly(rho, T, T0, h, deg=2):
    # Use a temperature window, not a point-count window.
    idx = np.abs(T - T0) <= h

    # Local polynomial needs enough points to fit stably.
    if idx.sum() < deg + 2:
        idx = np.argsort(np.abs(T - T0))[:deg + 2]

    x = T[idx] - T0
    y = rho[idx]

    # Tricube weights: closer points matter more.
    u = np.abs(x) / h
    w = (1 - u**3)**3
    w[u >= 1] = 0

    X = np.vstack([x**p for p in range(deg + 1)]).T

    # Weighted least squares.
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]

    # beta[0] is the fitted value at x0 because x = T - x0.
    return beta[0]


def moving_average(rho, T, window = None):

    window = (np.max(T) - np.min(T)) * 0.2 if window is None else window
    half = window / 2

    left  = np.searchsorted(T, T - half, side="left")
    right = np.searchsorted(T, T + half, side="right")

    crho = np.r_[0, np.cumsum(rho)]
    n = right - left

    rho_sm = (crho[right] - crho[left]) / n

    return rho_sm


def T_weights(T):
    # Each point represents the temperature interval halfway to neighbors
    w = np.empty_like(T, dtype=float)
    w[1:-1] = 0.5 * (T[2:] - T[:-2])
    w[0]    = 0.5 * (T[1] - T[0])
    w[-1]   = 0.5 * (T[-1] - T[-2])
    return w / np.sum(w)


def weighted_median(x, w):
    idx = np.argsort(x)
    xs, ws = x[idx], w[idx]
    return xs[np.searchsorted(np.cumsum(ws), 0.5)]


def weighted_mad(x, w):
    med = weighted_median(x, w)
    mad = weighted_median(np.abs(x - med), w)

    # 1.4826 makes MAD comparable to std for normally distributed noise
    return 1.4826 * mad


# 1. Performs Hampel filter
# 2. Performs Adaptive smoothing 
def adaptive_smooth(rho, T, deg=1, h_min=None, h_max=None, sensitivity=5) -> list:

    # rho = hampel(rho).filtered_data

    dT = np.median(np.diff(T))
    Tr = T[-1] - T[0]

    # h_min prevents the smoother from chasing noise.
    # h_max is the broad window used in smooth/background regions.
    h_min = 3 * dT if h_min is None else h_min
    h_max = 0.15 * Tr if h_max is None else h_max

    # First pass: broad smooth so curvature is not dominated by raw noise.
    rough = np.array([local_poly(rho, T, t, h_max, deg) for t in T])

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
    smooth = np.array([local_poly(rho, T, T[i], h[i], deg) for i in range(len(T))])

    return smooth


# def adaptive_smooth(rho, T, deg=1, h_min=None, h_max=None, sensitivity=1):
#     dT = np.median(np.diff(T))
#     Tr = T[-1] - T[0]

#     # h_min prevents the smoother from chasing noise.
#     # h_max is the broad window used in smooth/background regions.
#     h_min = 3 * dT if h_min is None else h_min
#     h_max = 0.2 * Tr if h_max is None else h_max

#     # First pass: broad smooth so curvature is not dominated by raw noise.
#     rough = np.array([local_poly(T, rho, t, h_max, deg) for t in T])

#     # Curvature estimates where the curve bends sharply.
#     d1 = np.gradient(rough, T)
#     d2 = np.gradient(d1, T)
#     sharp = np.abs(d2)
#     w = T_weights(T)

#     # MAD-score = how unusually sharp this point is compared to this linecut.
#     score = (sharp - weighted_median(sharp, w)) / weighted_mad(sharp, w)
#     score = np.clip(score, 0, 8)

#     # Large score -> shrink smoothing window to preserve transition.
#     # Small score -> use large window to smooth boring regions.
#     h = h_max / (1 + score / sensitivity)
#     h = np.clip(h, h_min, h_max)

#     # deg_adaptive = []
#     # for _ in h:
#     #     deg_adaptive.append(deg + 1) if deg > 4 else deg_adaptive.append(deg)

#     # Final pass: adaptive smoothing with local temperature window.
#     smooth = np.array([local_poly(T, rho, T[i], h[i], deg) for i in range(len(T))])

#     return smooth