
import numpy as np
from hampel import hampel 
import math

# ----- GENERAL SIGNAL HELPER FUNCTIONS ----- 


# Moving average based on temperaure window 
def moving_average(rho, T, window = None):

    window = (np.max(T) - np.min(T)) * 0.2 if window is None else window
    half = window / 2

    left  = np.searchsorted(T, T - half, side="left")
    right = np.searchsorted(T, T + half, side="right")

    crho = np.r_[0, np.cumsum(rho)]
    n = right - left

    rho_sm = (crho[right] - crho[left]) / n

    return rho_sm

# Produces weights for T based on T spacing
def T_weights(T):
    # Each point represents the temperature interval halfway to neighbors
    w = np.empty_like(T, dtype=float)
    w[1:-1] = 0.5 * (T[2:] - T[:-2])
    w[0]    = 0.5 * (T[1] - T[0])
    w[-1]   = 0.5 * (T[-1] - T[-2])
    return w / np.sum(w)

def weighted_median(T, w):
    idx = np.argsort(T)
    Ts, ws = T[idx], w[idx]
    return Ts[np.searchsorted(np.cumsum(ws), 0.5* np.sum(w))]


def weighted_mad(T, w):
    med = weighted_median(T, w)
    mad = weighted_median(np.abs(T - med), w)

    # 1.4826 makes MAD comparable to std for normally distributed noise
    return 1.4826 * mad




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


# ----- HELPER FUNCTIONS FOR SMOOTHING & NOISE ----- 

def local_poly(rho, T, T0, h, deg=1):
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


# 1. Performs Hampel filter
# 2. Performs Adaptive smoothing 
def adaptive_smooth(T, rho, deg=1, h_min=None, h_max=None, sensitivity=5):

    dT = np.median(np.diff(T))
    Tr = np.max(T) - np.min(T)
    rho = hampel(rho).filtered_data

    # h_min prevents the smoother from chasing noise.
    # h_max is the broad window used in smooth/background regions.
    h_min = 3 * dT if h_min is None else h_min
    h_max = 0.2 * Tr if h_max is None else h_max

    # h_max = -1 
    # First pass: broad smooth so curvature is not dominated by raw noise.
    rough = np.array([local_poly(rho, T, t, -1, deg) for t in T])

    # Curvature estimates where the curve bends sharply.
    d1 = np.gradient(rough, T)
    d2 = np.gradient(d1, T)
    sharp = np.abs(d2)

    # Weighted MAD-score = how unusually sharp this point is compared to this linecut.
    w = T_weights(T)
    score = (sharp - weighted_median(sharp, w)) / weighted_mad(sharp, w)
    score = np.clip(score, 0, 8)

    # Large score -> shrink smoothing window to preserve transition.
    # Small score -> use large window to smooth boring regions.
    h = h_max / (1 + score / sensitivity)
    h = np.clip(h, h_min, h_max)

    # Final pass: adaptive smoothing with local temperature window.
    smooth = np.array([local_poly(rho, T, T[i], h[i], deg) for i in range(len(T))])

    return smooth


def local_noise(T, rho, rho_smoothed, T_window = 0.5, fallback_points = 9):

    noise = []
    residuals = rho - rho_smoothed
    w = T_weights(T)

    for t in T:

        # find indicies of points neiboring T 
        mask = np.abs(T - t) < T_window
        local_idx = np.flatnonzero(mask)

        if len(local_idx) < fallback_points:
            local_idx = np.argsort(np.abs(T - t))[:fallback_points]
        
        noise.append(weighted_mad(residuals[local_idx], w[local_idx]))
            
    return noise