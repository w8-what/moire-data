# Setup Imports and Define Presets

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os 
from pathlib import Path

from helper_functions import fmt4, contiguous_regions, smooth_rho, clean_boolean_mask, robust_snr
from phase_config import PHASES, PHASE_COLORS, PHASE_LABELS
from hampel import hampel

OUT = Path('output/extract_behaviors')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.xmargin' : 0.05,
    'axes.ymargin' : 0.05
})

if not os.path.exists(OUT):
    os.mkdir(OUT)



# Loading data and returns T (array of temperatures, x-axis), nu (array of fillings), and R (2D array of resistivity)
def load_field(E):
    df = pd.read_csv(IN / f'Rxx_matrix_E-{E}mV_nm.csv')
    T = df.iloc[:, 0].astype(float).to_numpy()
    nu = np.array([float(c) for c in df.columns[1:]])
    R = df.iloc[:, 1:].astype(float).to_numpy()
    return T, nu, R



# Extract candidate transition temperatures from sharp turns 


def extract_upturns(T, rho, sensitivity = 1) -> list[dict]:

    threshold = sensitivity * 100

    rho_smoothed = smooth_rho(rho, T)
    dpdT = np.gradient(rho_smoothed, T)
    d2pdT2 = np.gradient(dpdT, T)

    # create mask for abs(dpdT) < threshold (clustering? basically if theres a hole -> fill it, or if theres a single dot, ignore)
    dpdT_mask = abs(dpdT) < threshold
    dpdT_mask = clean_boolean_mask(dpdT_mask)

    d2pdT2_mask = (d2pdT2 > 0)
    d2pdT2_mask = clean_boolean_mask(d2pdT2_mask)

    mask = dpdT_mask & d2pdT2_mask
    print(mask)

    candidates = T[mask]
    candidate_upturns = []

    for cand in candidates:
        candidate = {
            "T" : cand,
            "confidence" : 0.5,

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
 
    param_string = "  ".join(
        f"{k} = {fmt4(v)}" for k, v in params.items()
    )
 
    # Derived curves
    rho_smoothed = smooth_rho(rho, T)
    dpdT_raw      = np.gradient(rho,          T)
    dpdT_smoothed = np.gradient(rho_smoothed, T)


    dlnpdlnT = np.gradient(np.log(np.clip(rho_smoothed, 0, np.inf)), np.log(np.clip(T, 0, np.inf)))
 
    # --- layout: 1 row × 2 cols ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=150, constrained_layout=True)
    fig.suptitle(param_string, fontsize=9, y=1.02)
 
    # ── shared axis labels ──────────────────────────────────────────────────
    for ax in (ax1, ax2):
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Resistivity (Ω)")
        ax.tick_params(which='both', direction='in', top=True, right=True)
 
    ax1.set_title("Raw Data")
    ax1.plot(T, rho, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='navy',linewidth=1.0, color='blue')
 
    ax2.set_title("Smoothed Data + Candidate Transitions")
    ax2.plot(T, rho_smoothed, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='navy',linewidth=1.0, color='blue')

    ax3.set_title("dln(P)/dln(T)")
    ax3.plot(T, dlnpdlnT, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='navy',linewidth=1.0, color='blue')

    
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0)
        ax.set_ylim(0)

    # --- transition temperature markers ------------------------------------
    # Sort by temperature so staggered labels alternate neatly
    sorted_candidates = sorted(candidates, key=lambda c: c["T"])
 
    y_min, y_max = ax2.get_ylim()
    y_span = y_max - y_min
 
    for idx, cand in enumerate(sorted_candidates):
        Tc    = cand["T"]
        conf  = cand.get("confidence", 1.0)
        phase = cand.get("phase_left", "")
 
        # vertical line — opacity encodes confidence
        ax2.axvline(
            Tc,
            color='#2c3e50', linewidth=1.0,
            linestyle='--', alpha=0.4 + 0.6 * conf,
            zorder=3,
        )
 
        # label: stagger vertically to avoid overlap
        # odd indices go high, even go low
        y_frac  = 0.82 if idx % 2 == 0 else 0.65
        y_label = ax2.get_ylim()[0] + y_frac * (ax2.get_ylim()[1] - ax2.get_ylim()[0])
 
        label_parts = [f"$T_c$ = {fmt4(Tc)} K"]
        if phase:
            label_parts.append(phase)
        label_parts.append(f"(conf: {conf:.2f})")
        label_text = "\n".join(label_parts)
 
        ax2.annotate(
            label_text,
            xy=(Tc, y_label),
            xytext=(4, 0),
            textcoords='offset points',
            fontsize=6.5,
            color='#2c3e50',
            va='center',
            ha='left',
            zorder=4,
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='white',
                edgecolor='#cccccc',
                linewidth=0.5,
                alpha=0.85,
            ),
        )
 
        # small dot on the smoothed curve at Tc
        # interpolate rho_smoothed at Tc
        rho_at_Tc = float(np.interp(Tc, T, rho_smoothed))
        ax2.scatter(
            [Tc], [rho_at_Tc],
            s=28, color='#c0392b', zorder=5,
            edgecolors='white', linewidths=0.6,
        )
  
    # ── save ────────────────────────────────────────────────────────────────
    safe_name = param_string.replace("/", "_").replace(" ", "_")
    path = OUT / Path(safe_name + ".png")
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
 


# Go through each linecut; extract and plot all candidate transition temperature 

def plot_all_linecuts(E: int, numLines: int) -> None:

    # TODO: learn np.linspace thats endpoint inclusive for clearer code

    T, nu, R = load_field(E)

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

        plot_single_linecut({"E" : E, "Filling" : nu[currColInt]}, T, linecut_rho, candidates)

        currCol += spacing

plot_all_linecuts(103, 50)



