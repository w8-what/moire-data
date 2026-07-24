import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

from pathlib import Path
import numpy as np 


def draw_heatmap(fig, ax, col, row, data, title="heatmap", xlabel="Filling v", ylabel="Temperature T (K)", cbar_label="Resistivity"):

    # Log rounded vmin & vmax
    vmin_raw, vmax_raw = np.nanpercentile(data[data > 0], [1, 99])

    emin = int(np.floor(np.log10(vmin_raw)))
    emax = int(np.ceil(np.log10(vmax_raw)))

    vmin = 10**emin
    vmax = 10**emax

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
    ax.set_title(title)

    fig.tight_layout()


def overlay_features_heatmap(ax, linecuts, feature_name = "features", score_name = "confidence", filter = 0):

    styles = {
        "upturn":   dict(color="yellow",    marker="^", label = "upturn"),
        "downturn": dict(color = "green", marker ="o", label = "downturn")
    }

    used_labels = set()


    for linecut in linecuts:
        features = linecut.get(feature_name) 
        nu = linecut.get("nu")

        for feat in features:

            type = feat.get("type")
            T_transition = feat.get("T")
            score = feat.get(score_name)

            if type not in styles or score < filter:
                continue

            style = styles[type].copy()
            label = style["label"]

            if label in used_labels:
                style["label"] = None
            else:
                used_labels.add(label)

            ax.scatter(
                nu,
                T_transition,
                s=35,
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
                alpha = score,
                **style
            )

    ax.legend(frameon=True, fontsize=8)



# def draw_mosaic_diagrams(col, row, data, OUT=None, name="mosaic_phase_diagram", save=False):
#     phase_to_id = {"Unknown": 0, "AFM": 1, "Metal": 2, "Insulator": 3}
#     colors = ["silver", "maroon", "steelblue", "beige"]

#     phase = np.zeros(data.shape, dtype=int)
#     candidates_all = []

#     for j, filling in enumerate(col):
#         linecut = data[:, j]

#         candidates = extract_upturns(row, linecut)
#         # candidates = extract_metallic_transitions(row, linecut, candidates)
#         candidates = sorted(candidates, key=lambda c: c["T"])

#         candidates_all.append(candidates)

#         for k in range(len(candidates) - 1):
#             phase_name = candidates[k].get("phase_right")
#             T0 = candidates[k]["T"]
#             T1 = candidates[k + 1]["T"]

#             if phase_name is not None:
#                 mask = (row >= T0) & (row < T1)
#                 phase[mask, j] = phase_to_id.get(phase_name, 0)

#         if candidates:
#             phase[row >= candidates[-1]["T"], j] = phase_to_id.get(
#                 candidates[-1].get("phase_left"), 0
#             )

#     cmap = ListedColormap(colors)
#     norm = BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)

#     fig, ax = plt.subplots(figsize=(8, 6))

#     ax.pcolormesh(
#         col,
#         row,
#         phase,
#         cmap=cmap,
#         norm=norm,
#         shading="nearest",
#     )

#     ax.set_xlabel(r"Filling $\nu$")
#     ax.set_ylabel(r"Temperature $T$ (K)")
#     ax.set_title(name)

#     labels = ["Unknown", "AFM", "Metal", "Insulator"]
#     ax.legend(
#         handles=[Patch(facecolor=c, label=p) for p, c in zip(labels, colors)]
#     )

#     fig.tight_layout()

#     if save:
#         OUT = Path(OUT)
#         OUT.mkdir(parents=True, exist_ok=True)
#         fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")

#     return fig, ax, phase, candidates_all