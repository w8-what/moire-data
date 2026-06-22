import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from pathlib import Path
from decimal import Decimal

from phase_config import PHASES, PHASE_COLORS, PHASE_LABELS

OUT = Path('output/individual_labeling')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]
SC_THRESHOLD = 20.0


plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


# Loading data and returns T (array of temperatures, x-axis), nu (array of fillings), and R (2D array of resistivity)
def load_field(E):
    df = pd.read_csv(IN / f'Rxx_matrix_E-{E}mV_nm.csv')
    T = df.iloc[:, 0].astype(float).to_numpy()
    nu = np.array([float(c) for c in df.columns[1:]])
    R = df.iloc[:, 1:].astype(float).to_numpy()
    return T, nu, R


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

import numpy as np

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

def smooth_rho(T, rho, window_frac=0.08, polyorder=1):
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



def find_SC(T, roh) -> list:

    # TODO: there is a gap at the botton that is not shaded by the region

    candidates = []

    below = roh < SC_THRESHOLD
    intervals = contiguous_regions(below)

    for interval in intervals:
        candidate = {}

        left, right = interval
        left_temp = T[left]; right_temp = T[right]

        candidate.update({"phase": "SC"})
        candidate.update({"fit_range": (left_temp, right_temp)})
        candidate.update({"confidence": 0.9})

        candidates.append(candidate)

    return candidates


def find_sublin_M(T, roh):
    return None

def find_SM(T, roh):
    return None

def find_FL(T, roh):
    return None


def find_AFM_M(T, roh):
    return None


def find_AFM_I(T, roh):
    return None









# Generating linecut roh v. T plots
def plot_behavior_fits(params, T, rho, candidates) -> None:

    filling = Decimal(params[1]).quantize(Decimal("0.000"))
    param_string = str(params[0]) + " = " + str(filling)

    print(str(params[0]) + " = " + str(params[1]))
    print(candidates)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.12), dpi=180)
    fig.suptitle(param_string)

    for ax in (ax1, ax2):
        ax.set_ylabel("Resistivity (Ω·cm)") 
        ax.set_xlabel("Temperature (K)")
        ax.set_xlim(0, int(max(T)+0.5))
        ax.set_ylim(0, int(max(rho)+0.5))
        

    ax1.set_title("Raw Data")
    ax2.set_title("Smoothed Data + Behavior Fits")

    ax1.plot(T, rho, marker='o', markerfacecolor='none', markeredgecolor='navy', linestyle='none')

    smoothed_rho = smooth_rho(T, rho)
    ax2.plot(T, smoothed_rho, marker='o', markerfacecolor='none', markeredgecolor='navy', linestyle='none')

    for candidate in candidates:

        phase = candidate.get("phase")
        fit_range = candidate.get("fit_range")
        # TODO: Shade phase with color and add label onto the graph 
        ax2.axvspan(fit_range[0], fit_range[1], 0, int(max(rho)+0.5), alpha = 0.5)

        
    # Saving plots to path
    path = OUT / Path(param_string + ".png")
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()



# Fits and plots ~ numLinecuts linecuts evenly spread 
def test_behavior_fits(E: int, numLinecuts: int) -> None:
    T, nu, R = load_field(E)

    numCol = R.shape[1]
    currCol = 0
    spacing = (numCol - 1) / (numLinecuts - 1)

    # Finding intervals for each linecut and plotting them
    for i in range(numLinecuts):
        currColInt = int(currCol + 0.5)
        linecut_roh = R[:, currColInt]

        # Finding intervals for each phase
        candidates = []
        print(PHASES)
        for phase in PHASES:
            func = globals().get(f"find_{phase}")
            result = func(T, linecut_roh)
            if result is not None:
                candidates.extend(result)

        # Plotting the intervals
        plot_behavior_fits(("Filling", nu[currColInt]), T, linecut_roh, candidates)

        currCol += spacing

test_behavior_fits(103, 75)