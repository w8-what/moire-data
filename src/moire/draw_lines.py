
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import os

from moire.extraction_helpers import adaptive_smooth, moving_average
from moire.io import fmt4, load_field
from moire.extract_behaviors import extract_metallic_transitions, extract_upturns, extract_upturns_new
from hampel import hampel


# Plot candidate transition temperatures, along with candidate phases (if suggested)
def plot_single_linecut(params: dict, T: list, rho: list, candidates: list[dict], OUT: Path) -> None:

    if not os.path.exists(OUT):
        os.mkdir(OUT)
 
    param_string = "  ".join(f"{k} = {fmt4(v)}" for k, v in params.items())
 
    # Derived curves
    rho_smoothed = adaptive_smooth(rho, T)
    dpdT = np.gradient(rho_smoothed, T)
    d2pdT2 = np.gradient(dpdT, T)
    # dlnpdlnT = np.gradient(np.log(np.clip(rho_smoothed, 0, np.inf)), np.log(np.clip(T, 0, np.inf)))
 
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

    axes[3].set_title("Second Deriveratives")
    axes[3].plot(T, d2pdT2, marker='o', markersize = 3, markerfacecolor = 'none', markeredgecolor = 'navy', linewidth = 1.0, color = 'blue')
    axes[3].fill_between(T, d2pdT2, alpha = 0.5)

    axes[2].set_title("First Deriveratives")
    axes[2].plot(T, dpdT, marker='o', markersize = 3, markerfacecolor = 'none', markeredgecolor = 'navy', linewidth = 1.0, color = 'blue')
    axes[2].fill_between(T, dpdT, alpha = 0.5)

    # for candidate in candidates:
    #     print(candidate)
    # print("\n")

    # Plotting transition points and fitted lines 
    for cand in candidates:

        T_t = cand["T"]
        conf = cand["confidence"] 
        phase_left = cand["phase_left"]
        phase_right = cand["phase_right"]
        t_color = "blue" if (phase_left == "Metal" or phase_left == "Insulator") else "red"

        rho_at_T_t = rho_smoothed[np.argmin(np.abs(T - T_t))]
        axes[1].scatter(T_t, rho_at_T_t, color = t_color, alpha = 0.8)
        axes[1].axvline(T_t, linewidth = 1, linestyle='--', color = "grey", zorder=3)

        max_rho = np.max(rho_smoothed)
        top_half = rho_at_T_t > (max_rho/2)
        y_text = 0.8 * max_rho if top_half else 0.2 * max_rho
        axes[1].annotate(f"{conf=}", xy=(T_t, rho_at_T_t), xytext=(T_t, y_text),
            bbox=dict(boxstyle="round", fc="0.8", alpha = 0.8),
            arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=10, connectionstyle="angle,angleA=0,angleB=90,rad=10", alpha = 0.8))

    # Save to path
    path = OUT / Path(param_string + ".png")
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
 
 

# Go through each linecut; extract and plot all candidate transition temperature 
def plot_all_linecuts(E: float, numLines: int, IN: Path, OUT: Path) -> None:

    # TODO: learn np.linspace thats endpoint inclusive for clearer code
    T, nu, R = load_field(E, IN)
    row, col = R.shape

    currCol = 0
    spacing = (col - 1) / (numLines - 1)

    for _ in range(numLines):
        currColInt = int(currCol + 0.5
        linecut_rho = R[:, currColInt]

        # Plotting the intervals
        print(f"{nu[currColInt]=}")
        print(f"{E=}\n")
        candidates = extract_upturns_new(T, linecut_rho)
        candidates = extract_metallic_transitions(T, linecut_rho, candidates)

        plot_single_linecut({"E" : E, "Filling" : nu[currColInt]}, T, linecut_rho, candidates, OUT)
        currCol += spacing
