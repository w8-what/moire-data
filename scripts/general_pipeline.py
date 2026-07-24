import sys
from pathlib import Path
import numpy as np 

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / Path("src")))

from hampel import hampel 
from moire.io import load_field, clean_sort_data
from moire.signal_helpers import adaptive_smooth, local_noise
from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth
from moire.extract_features import extract_upturns, extract_downturns

from moire.draw_lines import plot_linecut, plot_linecut_noise, generate_layout
from moire.draw_2d import draw_heatmap, overlay_features_heatmap
from moire.update_scoring import update_score

OUT = ROOT / Path("output")
IN = ROOT / Path("source_data")
FIELDS = [87, 96, 99, 103, 74, 96.2, 151, 176]
SELECT_FIELDS = [87, 96, 99, 103, 74, 96.2, 151, 176]

for field in SELECT_FIELDS:

    # ----- Data Preprocessing -----
    T, nu, R = load_field(field, IN) # loads initial dataset
    T, nu, R = clean_sort_data(T, nu, R) # sorts data and removes nans

    linecuts = []
    for i, v in enumerate(nu):
        linecuts.append({"E" : field, "nu" : v, "T": T, "rho" : R[:, i]}) 


    # ----- Data Processing -----
    for linecut in linecuts:

        # Smoothing

        rho = linecut.get("rho")
        rho_hampel = hampel(rho).filtered_data
        rho_smoothed = adaptive_multiscale_smooth(T, rho, z_threshold=3)
        linecut.update({"rho_smoothed" : rho_smoothed})

        # Noise estimates
        noise = local_noise(T, rho, rho_smoothed)
        linecut.update({"local_noise" : noise})

        # Upturn & downturn feature extraction  
        features = []
        features += extract_upturns(T, linecut)
        features += extract_downturns(T, linecut)
        linecut.update({"features" : features})

    # ----- New Scoring Updates -----

    linecuts = update_score(linecuts)

    # ----- Plotting and creating figures -----
    numLinecuts = 60
    selectedLinecuts = np.linspace(0, len(linecuts), numLinecuts, dtype = "int")
    for i, linecut in enumerate(linecuts):
        if i in selectedLinecuts:
            plot_linecut_noise(T, linecut, OUT = OUT / Path("linecuts"))
        

    # ----- Plotting and creating figures -----
    # numLinecuts = 60
    # selectedLinecuts = np.linspace(0, len(linecuts), numLinecuts, dtype = "int")
    # for i, linecut in enumerate(linecuts):
    #     if i in selectedLinecuts:
    #         plot_linecut_noise(T, linecut, OUT = OUT / Path("linecuts"))

    name = f"{field}_Score_Comparison"
    fig, axes = generate_layout(2, title = name)

    draw_heatmap(fig, axes[0], nu, T, R, title = "original scoring")
    overlay_features_heatmap(axes[0], linecuts, score_name = "confidence")

    draw_heatmap(fig, axes[1], nu, T, R, title = "3 passes x 5 iterations")
    overlay_features_heatmap(axes[1], linecuts, feature_name = "features_new", score_name = "score_15")

    path = OUT / Path("heatmaps_comparison")
    path.mkdir(exist_ok = True, parents = True)
    fig.savefig(path / Path(name + ".png"))


    



        

