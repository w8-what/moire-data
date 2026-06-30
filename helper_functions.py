
import math
import numpy as np
import pandas as pd 

from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

def load_field(E, IN):
    df = pd.read_csv(IN / f'Rxx_matrix_E-{E}mV_nm.csv')
    T = df.iloc[:, 0].astype(float).to_numpy()
    nu = np.array([float(c) for c in df.columns[1:]])
    R = df.iloc[:, 1:].astype(float).to_numpy()
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


def adaptive_smooth(rho, T, deg=1, h_min=None, h_max=None, sensitivity=5):
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
