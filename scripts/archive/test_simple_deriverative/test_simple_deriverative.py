import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / Path("src")))

from hampel import hampel
from moire.io import load_field, clean_sort_data, fmt4
from moire.signal_helpers import local_noise, moving_average
from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth
from moire.extract_features import extract_upturns, extract_downturns, extract_Tc, get_fit_range

from moire.draw_lines import plot_general_line, generate_layout, overlay_features, overlay_behaviors
from moire.draw_2d import draw_heatmap, overlay_features_heatmap, overlay_behaviors_heatmap
from moire.update_scoring import update_score
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / Path("output_hampel_movingavg")
IN = ROOT / Path("source_data")
FIELDS = [87, 96, 99, 103, 74, 96.2, 151, 176]
SELECT_FIELDS = [87, 96, 99, 103, 74, 96.2, 151, 176]

for field in [103, 99]:

    # ----- Data Preprocessing -----
    T, nu, R = load_field(field, IN)  # loads initial dataset
    T, nu, R = clean_sort_data(T, nu, R)  # sorts data and removes nans

    linecuts = []
    for i, v in enumerate(nu):
        linecuts.append({"E": field, "nu": v, "T": T, "rho": R[:, i]})

    # ----- Data Processing -----
    for linecut in linecuts:

        # Smoothing

        rho = linecut.get("rho")
        rho_hampel = hampel(rho).filtered_data
        rho_smoothed = adaptive_multiscale_smooth(T, rho, z_threshold=3)
        linecut.update({"rho_smoothed": rho_smoothed})

        # Noise estimates
        noise = local_noise(T, rho, rho_smoothed)
        linecut.update({"local_noise": noise})

        # Upturn & downturn feature extraction
        features = []
        features += extract_upturns(T, linecut)
        features += extract_downturns(T, linecut)
        features += extract_Tc(T, linecut, max_candidates=1)
        linecut.update({"features": features})
        linecut.update({"behaviors": []})

    # ----- New Scoring Updates -----

    linecuts = update_score(linecuts)

    # getting fit range
    for linecut in linecuts:
        get_fit_range(T, linecut)

    # ----- Plotting and creating figures -----

    numLinecuts = 60
    selectedLinecuts = np.linspace(0, len(linecuts), numLinecuts, dtype="int")
    for i, linecut in enumerate(linecuts):
        if i in selectedLinecuts:
            param_string = "  ".join(f"{k} = {fmt4(v)}" for k, v in linecut.items() if k == "E" or k == "nu")
        
            rho = linecut.get("rho")
            rho_smoothed = linecut.get("rho_smoothed")

            dpdT = hampel(np.gradient(rho_smoothed, T)).filtered_data

            dln_dpdT_dT = moving_average(np.gradient(np.log(dpdT), T), T, 1)

            
        
            fig, axes = generate_layout(4, title=param_string)
            linecut_axis_kwargs = {
                "xlabel": "Temperature (K)",
                "ylabel": "Resistivity (Ω*cm)",
                "xlim": (0, None),
                "ylim": (0, None),
            }
        
            plot_general_line(axes[0], T, rho, title="Raw Data", **linecut_axis_kwargs)
            plot_general_line(axes[1], T, rho_smoothed, title="Smoothed Data, Features, Behaviors", **linecut_axis_kwargs)
            plot_general_line(axes[2], T, dpdT, title="First Derivative", shaded=True, fill_alpha=0.5)
            plot_general_line(axes[3], T, dln_dpdT_dT, title="dln(p\')dT", shaded=True, fill_alpha=0.5)
        
            overlay_features(axes[1], linecut, score_name="score_15", feature_name="features_new")
            overlay_behaviors(axes[1], linecut)
            fig.tight_layout()
        
            OUT.mkdir(parents=True, exist_ok=True)
            path = OUT / Path(param_string + ".png")
            fig.savefig(path, dpi=250, bbox_inches="tight")
            plt.close()

    # ----- 2d Figures -----

    # name = f"{field}_Score_Comparison"
    # fig, axes = generate_layout(2, title=name)

    # draw_heatmap(fig, axes[0], nu, T, R, title="original scoring")
    # overlay_features_heatmap(axes[0], linecuts, score_name="confidence")

    # draw_heatmap(fig, axes[1], nu, T, R, title="3 passes x 5 iterations")
    # overlay_features_heatmap(axes[1], linecuts, feature_name="features_new", score_name="score_15")
    # overlay_behaviors_heatmap(axes[1], linecuts)

    # path = OUT / Path("heatmaps_comparison")
    # path.mkdir(exist_ok=True, parents=True)
    # fig.savefig(path / Path(name + ".png"))
