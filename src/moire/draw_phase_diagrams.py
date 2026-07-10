from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from moire.extract_behaviors import extract_metallic_transitions, extract_upturns


def draw_mosaic_diagrams(col, row, data, OUT=None, name="mosaic_phase_diagram", save=False):
    phase_to_id = {"Unknown": 0, "AFM": 1, "Metal": 2, "Insulator": 3}
    colors = ["silver", "maroon", "steelblue", "beige"]

    phase = np.zeros(data.shape, dtype=int)
    candidates_all = []

    for j, filling in enumerate(col):
        linecut = data[:, j]

        candidates = extract_upturns(row, linecut)
        candidates = extract_metallic_transitions(row, linecut, candidates)
        candidates = sorted(candidates, key=lambda c: c["T"])

        candidates_all.append(candidates)

        for k in range(len(candidates) - 1):
            phase_name = candidates[k].get("phase_right")
            T0 = candidates[k]["T"]
            T1 = candidates[k + 1]["T"]

            if phase_name is not None:
                mask = (row >= T0) & (row < T1)
                phase[mask, j] = phase_to_id.get(phase_name, 0)

        if candidates:
            phase[row >= candidates[-1]["T"], j] = phase_to_id.get(
                candidates[-1].get("phase_left"), 0
            )

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.pcolormesh(
        col,
        row,
        phase,
        cmap=cmap,
        norm=norm,
        shading="nearest",
    )

    ax.set_xlabel(r"Filling $\nu$")
    ax.set_ylabel(r"Temperature $T$ (K)")
    ax.set_title(name)

    labels = ["Unknown", "AFM", "Metal", "Insulator"]
    ax.legend(
        handles=[Patch(facecolor=c, label=p) for p, c in zip(labels, colors)]
    )

    fig.tight_layout()

    if save:
        OUT = Path(OUT)
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")

    return fig, ax, phase, candidates_all