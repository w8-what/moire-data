import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from functions import *
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


def contiguous_regions(mask):
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


def find_SC(T, roh) -> dict:

    candidates = []

    below = roh < SC_THRESHOLD
    intervals = contiguous_regions(below)

    for interval in intervals:
        candidate = {}
        candidate.update({"phase": "SC"})

        print(interval)

        left, right = interval
        left_temp = T[left]; right_temp = T[right]

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
def plot_behavior_fits(params, T, roh, candidates) -> None:

    filling = Decimal(params[1]).quantize(Decimal("0.000"))
    param_string = str(params[0]) + " = " + str(filling)

    # Plotting (1) raw data
    plt.plot(T, roh, marker='o', markerfacecolor='none', markeredgecolor='black', linestyle='none')
    plt.ylabel("Resistivity (Ω·cm)") 
    plt.xlabel("Temperature (K)")
    plt.title(param_string)
    plt.xlim(0)
    plt.ylim(0)

    print(str(params[0]) + " = " + str(params[1]))
    print(candidates)

    for candidate in candidates:

        phase = candidate.get("phase")
        fit_range = candidate.get("fit_range")
        # TODO: Shade phase with color and add label onto the graph 
        plt.fill_betweenx(range(int(max(roh)+0.5)), fit_range[0], fit_range[1], alpha = 0.5)

        

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
        for phase in PHASES:
            func = globals().get(f"find_{phase}")
            result = func(T, linecut_roh)
            if result is not None:
                candidates.extend(result)

        # Plotting the intervals
        plot_behavior_fits(("Filling", nu[currColInt]), T, linecut_roh, candidates)

        currCol += spacing

test_behavior_fits(103, 50)