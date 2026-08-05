import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hampel import hampel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / Path("src")))

# Data Preprocessing Imports
from moire.io import load_field, clean_sort_data, fmt4
from moire.signal_helpers import local_noise
from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth

# Extracting Features and Behavior Imports
from moire.extract_features import extract_upturns, extract_downturns, extract_Tc, extract_Tcoh
from moire.extract_behaviors import extract_fit_range
from moire.extract_power_law import extract_local_fits

# Plotting Imports
from moire.draw_lines import generate_layout, plot_line_default, plot_line_general, overlay_behaviors, overlay_features
from moire.draw_2d import draw_heatmap, overlay_features_heatmap, overlay_behaviors_heatmap

# Score Updating
from moire.update_scoring import update_score


OUT = ROOT / Path("output")
IN = ROOT / Path("source_data")
FIELDS = [87, 96, 99, 103, 74, 96.2, 151, 176]
SELECT_FIELDS = [87, 96, 99, 103, 74, 96.2, 151, 176]

for field in SELECT_FIELDS:

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
        rho_smoothed = adaptive_multiscale_smooth(T, rho_hampel, z_threshold=3)
        linecut.update({"rho_smoothed": rho_smoothed})

        # Noise estimates
        noise = local_noise(T, rho, rho_smoothed)
        linecut.update({"local_noise": noise})

        # Upturn & downturn feature extraction
        features = []
        features += extract_upturns(T, linecut)
        features += extract_downturns(T, linecut)
        features += extract_Tc(T, linecut)
        linecut.update({"features": features})

    # ----- New Scoring Updates -----

    linecuts = update_score(linecuts)

    # getting fit range
    for linecut in linecuts:
        linecut["behaviors"] = extract_fit_range(T, linecut)
        linecut["exponent_fit"] = extract_local_fits(T, linecut)

        linecut["features"] += extract_Tcoh(T, linecut)



    # ----- Plotting and creating figures -----

    numLinecuts = 30
    selectedLinecuts = np.linspace(0, len(linecuts), numLinecuts, dtype="int")
    for i, linecut in enumerate(linecuts):
        if i in selectedLinecuts:

            param_string = "     ".join([f"{k} = {fmt4(v)}" for k, v in linecut.items() if k == "E" or k =="nu"])
            fig, axes = plot_line_default(T, linecut)

            # Creating directory
            linecut_dir = OUT / Path("linecuts")
            linecut_dir.mkdir(parents=True, exist_ok=True)
            path = str(linecut_dir / Path(f"{param_string}.png"))

            # Saving and closing figure
            fig.savefig(path, dpi=250, bbox_inches="tight")
            plt.close(fig)

    # ----- 2d Figures -----

    # name = f"{field}_Score_Comparison"
    # fig, axes = generate_layout(2, title=name)

    # draw_heatmap(fig, axes[0], nu, T, R, title="original scoring")
    # overlay_features_heatmap(axes[0], linecuts, score_name="confidence")

    # draw_heatmap(fig, axes[1], nu, T, R, title="3 passes x 5 iterations")
    # overlay_features_heatmap(axes[1], linecuts, feature_name="features_new", score_name="score_15")
    # overlay_behaviors_heatmap(axes[1], linecuts, drawn_behaviors=[])

    # path = OUT / Path("heatmaps_comparison")
    # path.mkdir(exist_ok=True, parents=True)
    # fig.savefig(path / Path(name + ".png"))
