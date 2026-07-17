import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moire.draw_lines import plot_linecut, plot_linecut_noise
from moire.io import load_field, clean_data
from moire.signal_helpers import adaptive_smooth, local_noise
from moire.extract_features import extract_upturns, extract_downturns
from moire.draw_2d import draw_heatmap_candidates
from hampel import hampel 

OUT = Path('output/heatmaps/opaque')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 96.2, 151, 176]
SELECT_FIELDS = [87, 96, 99, 103, 74, 96.2, 151, 176]

for field in SELECT_FIELDS:

    # ----- Data Preprocessing -----
    T, nu, R = load_field(field, IN) # loads initial dataset
    T, nu, R = clean_data(T, nu, R) # sorts data and removes nans

    linecuts = []
    for i, v in enumerate(nu):
        linecuts.append({"E" : field, "nu" : v, "rho" : R[:, i]}) 


    # ----- Data Processing -----
    for linecut in linecuts:

        # Smoothing
        rho = linecut.get("rho")
        rho_hampel = hampel(rho_hampel).filtered_data
        rho_smoothed = adaptive_smooth(T, rho)
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
    for linecut in linecuts:
        plot_linecut_noise(linecut, save = True, OUT = None)

    draw_heatmap_candidates(nu, T, R, linecuts, OUT = OUT, save = True, name = f"{field}_heatmap_opqaue")
        
        

        

