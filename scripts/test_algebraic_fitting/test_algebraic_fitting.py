import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hampel import hampel


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / Path("src")))


from moire.io import load_field, clean_sort_data, fmt4
from moire.signal_helpers import local_noise, moving_average
from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth
from moire.extract_features import extract_upturns, extract_downturns, extract_Tc
from moire.extract_power_law import extract_local_fits
from moire.extract_behaviors import get_fit_range

from moire.draw_lines import generate_layout, plot_general_line, overlay_behaviors, overlay_features
from moire.draw_2d import draw_heatmap, overlay_features_heatmap, overlay_behaviors_heatmap
from moire.update_scoring import update_score

OUT = Path(__file__).resolve().parent
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
        rho_smoothed = adaptive_multiscale_smooth(T, rho, z_threshold=3)
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
        linecut.update({"behaviors": []})

    # ----- New Scoring Updates -----

    linecuts = update_score(linecuts)

    # getting fit range
    for linecut in linecuts:
        get_fit_range(T, linecut)

        # extracting algebraic behaviors



    # ----- Plotting and creating figures -----

    numLinecuts = 30
    selectedLinecuts = np.linspace(0, len(linecuts), numLinecuts, dtype="int")
    for i, linecut in enumerate(linecuts):
        if i in selectedLinecuts:

            param_string = "  ".join(f"{k} = {fmt4(v)}" for k, v in linecut.items() if k == "E" or k == "nu")

            fig, axes = generate_layout(4, title=param_string)
            linecut_axis_kwargs = {
                "xlabel": "Temperature (K)",
                "ylabel": "Resistivity (Ω*cm)",
                "xlim": (0, None),
                "ylim": (0, None),
            }

            plot_general_line(axes[0], T, linecut.get("rho"), title="Raw Data", **linecut_axis_kwargs)
            plot_general_line(axes[1], T, linecut.get("rho_smoothed"), error = linecut["local_noise"], title="Smoothed Data, Features, Behaviors", **linecut_axis_kwargs)

            fit_rho = extract_local_fits(T, linecut)
            n = fit_rho["n"]
            n_sigma = fit_rho["n_sigma"]

            n_avg = moving_average(n, T, 1)

            plot_general_line(axes[2], T, n, error = n_sigma, title="Raw Rho N", xlim = (0, np.max(T)), ylim = (0, 4))
            plot_general_line(axes[3], T, n_avg, title="Moving Average of N", xlim = (0, np.max(T)), ylim = (0, 4))
            
            for y in [1, 0.8, 1.2]:
                axes[2].axhline(y=y, alpha=0.5, linestyle="-", color = "grey")
                axes[3].axhline(y=y, alpha=0.5, linestyle="-", color = "grey")

            overlay_features(axes[1], linecut, score_name="score_15", feature_name="features_new")
            overlay_behaviors(axes[1], linecut, drawn_behaviors = ["extraction_range"])
            fig.tight_layout()

            linecut_dir = OUT / Path("linecuts") / Path("moving_average")
            linecut_dir.mkdir(parents=True, exist_ok=True)
            path = linecut_dir / f"{param_string}.png"
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
