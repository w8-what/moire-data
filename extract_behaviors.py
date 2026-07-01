# Setup Imports and Define Presets

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import norm

import os 
from pathlib import Path

from helper_functions import fmt4, clean_boolean_mask, adaptive_smooth, load_field, mad
from phase_config import PHASES, PHASE_COLORS, PHASE_LABELS
from hampel import hampel

OUT = Path('output/extract_behaviors')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]

if not os.path.exists(OUT):
    os.mkdir(OUT)

# Extract candidate transition temperatures from sharp turns 
def extract_upturns(T, rho, sensitivity = 1) -> list[dict]:

    candidate_upturns = []
    threshold = sensitivity * 100

    # rho = hampel(rho).filtered_data
    rho_smoothed = adaptive_smooth(rho, T)
    dpdT = np.gradient(rho_smoothed, T)
    d2pdT2 = np.gradient(dpdT, T)

    peaks, prop = find_peaks(-rho_smoothed)

    dpdT_neg = dpdT[np.where(dpdT < 0)] # array of all negative dpdT values
    med_dpdT_neg = np.median(dpdT_neg)
    mad_dpdT_neg = mad(dpdT_neg)
    med_d2pdT2 = np.median(d2pdT2)
    mad_d2pdT2 = mad(d2pdT2)

    for idx in peaks:

        # Percentile of second deriverative at candidate assuming normal distribution 
        curvature_score = norm.cdf((d2pdT2[idx] - med_d2pdT2) / mad_d2pdT2)

        # Finds range of cliff using by stopping when concavity changes
        # Calculates score by finding average slope in cliff range and comparing to median slope (in negative slopes)

        kernel = np.ones(3)/3
        d2pdT2_moving_average = np.convolve(d2pdT2, kernel, mode = "same")

        idx_l = idx; sum = 0
        while idx_l > 0 or d2pdT2_moving_average[idx_l] < 0:
            sum += dpdT[idx_l]
            idx_l -= 1

        avg_dpdT_cliff = 0 if idx_l == 0 else np.mean(dpdT[idx_l:idx])
        left_cliff_score = norm.cdf(-(avg_dpdT_cliff - med_dpdT_neg) / mad_dpdT_neg) # more negative z-score is desirable here (steeper slope)

        # Combines scores
        comb_score = 0.5 * curvature_score + 0.5 * left_cliff_score

        candidate = {
            "T" : T[idx],
            "confidence" : comb_score,

            "phase_left" : "AFM",
            "phase_right" : None
        }

        candidate_upturns.append(candidate)
    return candidate_upturns


# Extract candidate transition temperatures from change in curve fits (metallic transitions)
def extract_metallic_transitions(T, rho, candidates) -> list[dict]:

    return []


# Plot candidate transition temperatures, along with candidate phases (if suggested)
def plot_single_linecut(params, T, rho, candidates) -> None:
 
    param_string = "  ".join(f"{k} = {fmt4(v)}" for k, v in params.items())
 
    # Derived curves
    rho_smoothed = adaptive_smooth(hampel(rho).filtered_data, T)
    dpdT = np.gradient(rho_smoothed, T)
    d2pdT2 = np.gradient(dpdT, T)
    dlnpdlnT = np.gradient(np.log(np.clip(rho_smoothed, 0, np.inf)), np.log(np.clip(T, 0, np.inf)))
 
    # Plotting 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=150, constrained_layout=True)
    fig.suptitle(param_string)

    axes = axes.flatten()
 
    axes[0].set_title("Raw Data")
    axes[0].plot(T, rho, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='navy',linewidth=1.0, color='blue')
 
    axes[1].set_title("Hampel Smoothed Data + Candidate Transitions")
    axes[1].plot(T, rho_smoothed, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='navy',linewidth=1.0, color='blue')

    for ax in axes[:2]:
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Resistivity (Ω*cm)")
        ax.set_xlim(0)
        ax.set_ylim(0)


    # axes[2].set_title("n")
    # axes[2].plot(T, dlnpdlnT, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='navy',linewidth=1.0, color='blue')

    axes[2].set_title("Second Deriveratives")
    axes[2].plot(T, d2pdT2, marker='o', markersize = 3, markerfacecolor = 'none', markeredgecolor = 'navy', linewidth = 1.0, color = 'blue')
    axes[2].fill_between(T, d2pdT2, alpha = 0.5)
    
    axes[3].set_title("First Deriveratives")
    axes[3].plot(T, dpdT, marker='o', markersize = 3, markerfacecolor = 'none', markeredgecolor = 'navy', linewidth = 1.0, color = 'blue')
    axes[3].fill_between(T, dpdT, alpha = 0.5)



    # Plotting transition points 
    for cand in candidates:

        T_t = cand["T"]
        conf = cand["confidence"]
        phase_left = cand["phase_left"]
        phase_right = cand["phase_right"]

        
        rho_at_T_t = rho_smoothed[np.where(T == T_t)]
        axes[1].scatter(T_t, rho_at_T_t, color = "red", alpha = 0.8)
        axes[1].axvline(T_t, linewidth = 1, linestyle='--', color = "grey", zorder=3)


    # Save to path
    path = OUT / Path(param_string + ".png")
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
 

# Go through each linecut; extract and plot all candidate transition temperature 
def plot_all_linecuts(E: float, numLines: int, left = None, right = None) -> None:

    # TODO: learn np.linspace thats endpoint inclusive for clearer code

    T, nu, R = load_field(E, IN)
    row, col = R.shape

    currCol = 0
    spacing = (col - 1) / (numLines - 1)

    for _ in range(numLines):
        currColInt = int(currCol + 0.5)
        linecut_rho = R[:, currColInt]

        # Plotting the intervals
        print(f"{nu[currColInt]=}")
        candidates = extract_upturns(T, linecut_rho)
        candidates += extract_metallic_transitions(T, linecut_rho, candidates)

        print(candidates)

        plot_single_linecut({"E" : E, "Filling" : nu[currColInt]}, T, linecut_rho, candidates)

        currCol += spacing

# plot_all_linecuts(103, 100)
# plot_all_linecuts(96.2, 10)
# plot_all_linecuts(176, 10)

for field in FIELDS:
    plot_all_linecuts(field, 30)

