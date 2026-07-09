import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

from pathlib import Path
import os 
import numpy as np 

from moire.phase_diagram_helpers import centers_to_edges



# generate heatmaps
# draw points on the heatmap w.r.t. the transition points
    # for each point use these crude rules first:
    # AFM ON LEFT -> Neel temp (mark as brown)
    # METAL ON LEFT (metal -> insulator) -> green?
    # INSULATOR ON LEFT (insulator -> metal) -> yellow?
    # and later just make it adaptive (left phase -> right phase) legends


def draw_heatmap(row, col, data, OUT=None, name=None, save=False,
    xlabel="Filling v", ylabel="Temperature T (K)", cbar_label="Resistivity"):

    vmin, vmax = np.nanpercentile(data, [1, 99])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(row, col, data, cmap="bwr", shading="nearest", norm = LogNorm(vmin = vmin, vmax = vmax))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=90)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(name)

    fig.tight_layout()

    if save:
        if OUT is None:
            raise ValueError("OUT must be provided when save=True")

        OUT = Path(OUT)
        OUT.mkdir(parents=True, exist_ok=True)

        name = "heatmap" if name is None else name
        path = OUT / f"{name}.png"
        fig.savefig(path, dpi=250, bbox_inches="tight")

    return fig, ax, im



def draw_heatmap_candidates(ax, candidates, OUT, name = "heatmap_candidates", save = True):
    # plot the points 



    # save to path 

    if save:
        if not os.path.exists(OUT):
            os.mkdir(OUT)

        path = OUT / Path(name + ".png")
        fig.savefig(path, dpi=250, bbox_inches='tight')


    return None 