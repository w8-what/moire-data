import sys
from pathlib import Path
import numpy as np 

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / Path("src")))

from hampel import hampel 
from moire.io import load_field, clean_sort_data
from moire.signal_helpers import adaptive_smooth, local_noise
from moire.adaptive_multiscale_smooth import adaptive_multiscale_smooth, estimate_noise_matrix
from moire.extract_features import extract_upturns, extract_downturns

from moire.draw_lines import plot_linecut, plot_linecut_noise
from moire.draw_2d import draw_heatmap_candidates

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
        sigma = estimate_noise_matrix(T, R)

        rho = linecut.get("rho")
        rho_hampel = hampel(rho).filtered_data
        rho_smoothed = adaptive_multiscale_smooth(T, rho, sigma = sigma)
        linecut.update({"rho_smoothed" : rho_smoothed})

        # Noise estimates
        noise = local_noise(T, rho, rho_smoothed)
        linecut.update({"local_noise" : noise})

        # Upturn & downturn feature extraction  
        features = []
        features += extract_upturns(T, linecut)
        features += extract_downturns(T, linecut)
        linecut.update({"features" : features})


    # ----- Plotting and creating figures -----
    numLinecuts = 30
    selectedLinecuts = np.linspace(0, len(linecuts), numLinecuts, dtype = "int")
    for i, linecut in enumerate(linecuts):
        if i in selectedLinecuts:
            plot_linecut(T, linecut, OUT = OUT / Path("linecuts"))


    fig, ax, im = draw_heatmap_candidates(nu, T, R, linecuts, filter = 0.01, OUT = OUT / Path("heatmaps"), save = True, name = f"{field}_heatmap_opqaue")
        
        

        

