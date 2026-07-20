
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path

from moire.signal_helpers import adaptive_smooth
from moire.io import fmt4






def plot_grid(plots, title):
    return 



def plot_line(x, y, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, shaded=False, error=None, **plot_kwargs):
    _, ax = plt.subplots()

    ax.plot(x, y, **plot_kwargs)

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xlim=xlim,
        ylim=ylim,
    )

    if shaded:
        ax.fill_between(x, 0, y, alpha=0.2)

    if error is not None:
        ax.fill_between(x, y - error, y + error, alpha=0.2)

    return ax


def overlay_features(ax, linecut):

    T = linecut.get("T")
    rho_smoothed = linecut.get("rho_smoothed")
    features = linecut.get("features")

    for feature in features:
        
        T_feature = feature.get("T")
        rho_at_T = rho_smoothed[np.argmin(np.abs(T - T_feature))]

        conf = feature.get("confidence")
        color = "blue" if feature.get("type") == "downturn" else "red"

        ax.scatter(T_feature, rho_at_T, alpha = conf, color = color)
        ax.axvline(T_feature, linewidth = 1, linestyle='--', color = "grey", zorder=3)

        max_rho = np.max(rho_smoothed)
        top_half = rho_at_T > (max_rho/2)
        y_text = 0.8 * max_rho if top_half else 0.2 * max_rho
        ax.annotate(f"{conf=}", xy=(T_feature, rho_at_T), xytext=(T_feature, y_text),
            bbox=dict(boxstyle="round", fc="0.8", alpha = 0.8),
            arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=10, connectionstyle="angle,angleA=0,angleB=90,rad=10", alpha = 0.8))

    return ax

def overlay_phases(ax, linecut):
    return 



# # Plot candidate transition temperatures, along with candidate phases (if suggested)
# def plot_linecut(T: list, linecut, OUT):

#     param_string = "  ".join(f"{k} = {fmt4(v)}" for k, v in linecut.items() if k == "E" or k == "nu")

#     # Getting data to plot
#     rho = linecut.get("rho")
#     rho_smoothed = linecut.get("rho_smoothed")
#     dpdT = np.gradient(rho_smoothed, T)
#     d2pdT2 = np.gradient(dpdT, T)
#     # dlnpdlnT = np.gradient(np.log(np.clip(rho_smoothed, 0, np.inf)), np.log(np.clip(T, 0, np.inf)))

 
#     # Plotting 2x2 grid
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=150, constrained_layout=True)
#     fig.suptitle(param_string)
#     axes = axes.flatten()
 
#     axes[0].set_title("Raw Data")
#     axes[0].plot(T, rho, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='navy',linewidth=1.0, color='blue')
 
#     axes[1].set_title("Smoothed Data + Candidate Transitions")
#     axes[1].plot(T, rho_smoothed, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='navy',linewidth=1.0, color='blue')

#     for ax in axes[:2]:
#         ax.set_xlabel("Temperature (K)")
#         ax.set_ylabel("Resistivity (Ω*cm)")
#         ax.set_xlim(0)
#         ax.set_ylim(0)

#     axes[3].set_title("Second Deriveratives")
#     axes[3].plot(T, d2pdT2, marker='o', markersize = 3, markerfacecolor = 'none', markeredgecolor = 'navy', linewidth = 1.0, color = 'blue')
#     axes[3].fill_between(T, d2pdT2, alpha = 0.5)

#     axes[2].set_title("First Deriveratives")
#     axes[2].plot(T, dpdT, marker='o', markersize = 3, markerfacecolor = 'none', markeredgecolor = 'navy', linewidth = 1.0, color = 'blue')
#     axes[2].fill_between(T, dpdT, alpha = 0.5)


#     # Plotting transition points and fitted lines 
#     for cand in linecut.get("candidates"):

#         T_t = cand["T"]
#         conf = cand["confidence"] 
#         phase_left = cand["phase_left"]
#         phase_right = cand["phase_right"]
#         t_color = "blue" if (phase_left == "Metal" or phase_left == "Insulator") else "red"

#         rho_at_T_t = rho_smoothed[np.argmin(np.abs(T - T_t))]
#         axes[1].scatter(T_t, rho_at_T_t, color = t_color, alpha = 0.8)
#         axes[1].axvline(T_t, linewidth = 1, linestyle='--', color = "grey", zorder=3)

#         max_rho = np.max(rho_smoothed)
#         top_half = rho_at_T_t > (max_rho/2)
#         y_text = 0.8 * max_rho if top_half else 0.2 * max_rho
#         axes[1].annotate(f"{conf=}", xy=(T_t, rho_at_T_t), xytext=(T_t, y_text),
#             bbox=dict(boxstyle="round", fc="0.8", alpha = 0.8),
#             arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=10, connectionstyle="angle,angleA=0,angleB=90,rad=10", alpha = 0.8))

#     # Save to path
#     OUT.mkdir(parents = True, exist_ok = True)
#     path = OUT / Path(param_string + ".png")
#     fig.savefig(path, dpi=250, bbox_inches='tight')
#     plt.close(fig)

#     return fig, axes



# # Plot candidate transition temperatures, along with candidate phases (if suggested)
# def plot_linecut_noise(T: list, linecut, save = False, OUT = None):

#     param_string = "  ".join(f"{k} = {fmt4(v)}" for k, v in linecut.items() if k == "E" or k == "nu")
 
#     # Derived curves
#     rho = linecut.get("rho")
#     rho_smoothed = linecut.get("rho_smoothed")
#     local_noise = linecut.get("local_noise")
#     # dlnpdlnT = np.gradient(np.log(np.clip(rho_smoothed, 0, np.inf)), np.log(np.clip(T, 0, np.inf)))
 
#     # Plotting 2x2 grid
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=150, constrained_layout=True)
#     fig.suptitle(param_string)

#     axes = axes.flatten()
 
#     axes[0].set_title("Raw Data")
#     axes[0].plot(T, rho, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='navy',linewidth=1.0, color='blue')
 
#     axes[1].set_title("Smoothed Data with Noise")
#     axes[1].plot(T, rho_smoothed, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='navy',linewidth=1.0, color='blue')

#     axes[2].set_title("Candidates with Noise")
#     axes[2].plot(T, rho_smoothed, marker='o', markersize = 3, markerfacecolor = 'none', markeredgecolor = 'navy', linewidth = 1.0, color = 'blue')

#     axes[2].fill_between(T, rho_smoothed - local_noise, rho_smoothed + local_noise, alpha = 0.5)
#     axes[0].fill_between(T, rho - local_noise, rho + local_noise, alpha = 0.5)

#     for ax in axes[:3]:
#         ax.set_xlabel("Temperature (K)")
#         ax.set_ylabel("Resistivity (Ω*cm)")
#         ax.set_xlim(0)
#         ax.set_ylim(0)


#     # Plotting transition points and fitted lines 
#     for cand in linecut.get("candidates"):

#         T_t = cand["T"]
#         type = cand["type"]
#         conf = cand["confidence"] 
#         phase_left = cand["phase_left"]
#         phase_right = cand["phase_right"]
#         t_color = "blue" if (type == "downturn") else "red"

#         rho_at_T_t = rho_smoothed[np.argmin(np.abs(T - T_t))]
#         axes[2].scatter(T_t, rho_at_T_t, color = t_color, alpha = 0.8)
#         axes[2].axvline(T_t, linewidth = 1, linestyle='--', color = "grey", zorder=3)

#         max_rho = np.max(rho_smoothed)
#         top_half = rho_at_T_t > (max_rho/2)
#         y_text = 0.8 * max_rho if top_half else 0.2 * max_rho
#         axes[2].annotate(f"{conf=}", xy=(T_t, rho_at_T_t), xytext=(T_t, y_text),
#             bbox=dict(boxstyle="round", fc="0.8", alpha = 0.8),
#             arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=10, connectionstyle="angle,angleA=0,angleB=90,rad=10", alpha = 0.8))


#     # Save to path
#     OUT.mkdir(parents = True, exist_ok = True)
#     path = OUT / Path(param_string + ".png")
#     fig.savefig(path, dpi=250, bbox_inches='tight')
#     plt.close(fig)

#     return fig, axes
