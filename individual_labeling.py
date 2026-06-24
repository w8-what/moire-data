import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from helper_functions import fmt4, contiguous_regions, smooth_rho
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

def find_SM(T, rho):
    return None

def find_FL(T, rho):
    return None


def find_AFM(T, rho):
    """
    Detect candidate AFM-like upturn/minimum features.
    """

    smoothed_rho = smooth_rho(T, rho)
    dr_dT = np.gradient(smoothed_rho, T)
    d2r_dT2 = np.gradient(dr_dT, T)

    cp_threshold = 1
    temp_window = 0.5
    squish_factor = 1000

    candidates_i = []
    candidates_score = []
    scored_i = []

    for i in range(len(dr_dT)):
        if abs(dr_dT[i]) < cp_threshold and d2r_dT2[i] > 0:
            candidates_i.append(i)

    for i in candidates_i:
        slope_sum = 0.0
        j = i - 1
        size = 0

        while j > 0 and d2r_dT2[j] > 0:
            slope_sum += dr_dT[j]
            j -= 1
            size += 1

        if j == 0:
            start = max(0, np.searchsorted(T, T[i] - temp_window))
            end = i

            if end <= start:
                continue

            average = dr_dT[start:end].mean()

        elif size > 0:
            average = slope_sum / size

        else:
            continue

        if not np.isfinite(average):
            continue

        score = (2 / np.pi) * np.arctan(-average / squish_factor)

        candidates_score.append(score)
        scored_i.append(i)

    if len(candidates_score) == 0:
        return None

    best_pos, val = max(enumerate(candidates_score), key=lambda p: p[1])
    best_i = scored_i[best_pos]

    return [{
        "phase": "AFM",
        "confidence": float(np.clip(val, 0, 1)),
        "fit_range": (0, T[best_i])
    }]




# TODO: add AFM-M and AFM-I (separate detection)

# def find_AFM_M(T, rho):
#     return None


# def find_AFM_I(T, roh):
#     return None


# Generating linecut roh v. T plots
def plot_behavior_fits(params: dict, T: np.ndarray, rho: np.ndarray, candidates: list) -> None:

    keys = params.keys()
    param_string = ""

    for key in keys:
        value = params[key]
        value = fmt4(value)
        string = str(key) + " = " + str(value) + "  "
        param_string += string

    print(param_string)
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
        ax2.axvspan(fit_range[0], fit_range[1], alpha = candidate["confidence"], 
                    color = PHASE_COLORS.get(phase), label = PHASE_LABELS.get(phase))
        

        
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

        # Finding candidate intervals for each phase
        candidates = []
        print(PHASES)
        for phase in PHASES:
            func = globals().get(f"find_{phase}")
            result = func(T, linecut_roh)
            if result is not None:
                candidates.extend(result)

        # Plotting the intervals
        plot_behavior_fits({"E" : E, "Filling" : nu[currColInt]}, T, linecut_roh, candidates)
        currCol += spacing


for field in FIELDS:
    test_behavior_fits(field, 20)
