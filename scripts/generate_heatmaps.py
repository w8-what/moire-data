import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moire.draw_lines import plot_linecut, plot_linecut_noise
from moire.io import load_field, clean_data
from moire.signal_helpers import adaptive_smooth, local_noise
from moire.extract_behaviors import extract_upturns, extract_downturns
from moire.draw_2d import draw_heatmap_candidates
from hampel import hampel 

OUT = Path('output/heatmaps/opaque')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 96.2, 151, 176]
SELECT_FIELDS = [87, 96, 99, 103, 74, 96.2, 151, 176]

for field in SELECT_FIELDS:

    T, nu, R = load_field(field, IN) # loads initial dataset
    T, nu, R = clean_data(T, nu, R) # sorts data and removes nans

    # Creating list of linecuts 
    linecuts = []
    for i, v in enumerate(nu):
        linecuts.append({"E" : field, "nu" : v, "rho" : R[:, i]}) 

    # Proccessing each linecut: smoothing + extraction
    for linecut in linecuts:
        rho = linecut.get("rho")

        # TODO: iterative LOESS or hampel based on T window instead of points
        # Smoothing
        rho = hampel(rho).filtered_data
        rho_smoothed = adaptive_smooth(T, rho)
        linecut.update({"rho_smoothed" : rho_smoothed})

        # TODO: estimating local noise per linecut
        noise = local_noise(T, rho, rho_smoothed)
        linecut.update({"local_noise" : noise})

        # Upturn & downturn extraction  
        candidates = []
        candidates += extract_upturns(T, linecut)
        candidates += extract_downturns(T, linecut)
        linecut.update({"candidates" : candidates})

    # Plotting and saving single linecuts 
    draw_heatmap_candidates(nu, T, R, linecuts, OUT = OUT, save = True, name = f"{field}_heatmap_opqaue")
        
        

        

