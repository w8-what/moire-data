import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

from pathlib import Path
import os 
import numpy as np 

from moire.phase_diagram_helpers import centers_to_edges
from moire.extract_behaviors import extract_metallic_transitions, extract_upturns



# generate heatmaps
# draw points on the heatmap w.r.t. the transition points
    # for each point use these crude rules first:
    # AFM ON LEFT -> Neel temp (mark as brown)
    # METAL ON LEFT (metal -> insulator) -> green?
    # INSULATOR ON LEFT (insulator -> metal) -> yellow?
    # and later just make it adaptive (left phase -> right phase) legends


def draw_heatmap(col, row, data, OUT=None, name="heatmap", save=False,
    xlabel="Filling v", ylabel="Temperature T (K)", cbar_label="Resistivity"):

    # Log rounded vmin & vmax
    vmin_raw, vmax_raw = np.nanpercentile(data[data > 0], [1, 99])

    emin = int(np.floor(np.log10(vmin_raw)))
    emax = int(np.ceil(np.log10(vmax_raw)))

    vmin = 10**emin
    vmax = 10**emax

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(col, row, data, cmap="bwr", shading="nearest", norm = LogNorm(vmin = vmin, vmax = vmax))

    tick_exps = np.arange(emin, emax + 1)
    ticks = 10**tick_exps

    # Drawing colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", location="right", pad=0.03)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())    
    
    cbar.set_label(cbar_label, rotation=90)
    cbar.set_ticks(ticks)
    cbar.set_label(r"$\rho_{xx}$ ($\Omega$)")

    # Axis titles and labels 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(name)

    fig.tight_layout()

    if save:
        OUT = Path(OUT)
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}_heatmap.png", dpi=250, bbox_inches="tight")

    return fig, ax, im




def draw_heatmap_candidates(col, row, data, OUT = None, name = "heatmap_candidates", save = True):

    fig, ax, im = draw_heatmap(col, row, data, save=False, name=name)

    styles = {
        "AFM":       dict(color="red",    marker="^", label="AFM left"),
        "Metal":     dict(color="green",  marker="o", label="Metal left"),
        "Insulator": dict(color="yellow", marker="o", label="Insulator left"),
    }

    used_labels = set()

    for j, filling in enumerate(col):
        linecut = data[:, j]

        candidates = extract_upturns(row, linecut)
        candidates = extract_metallic_transitions(row, linecut, candidates)

        for cand in candidates:
            phase_left = cand.get("phase_left")
            T_transition = cand.get("T")

            if phase_left not in styles:
                continue

            style = styles[phase_left].copy()
            label = style["label"]

            if label in used_labels:
                style["label"] = None
            else:
                used_labels.add(label)

            ax.scatter(
                filling,
                T_transition,
                s=35,
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
                **style
            )

    ax.legend(frameon=True, fontsize=8)

    if save:
        if OUT is None:
            raise ValueError("OUT must be provided when save=True")

        OUT = Path(OUT)
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}_heatmap_candidates.png", dpi=250, bbox_inches="tight")

    return fig, ax, im