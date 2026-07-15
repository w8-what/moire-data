import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moire.draw_lines import plot_all_linecuts, plot_single_linecut
from moire.io import load_field
from moire.extraction_helpers import adaptive_smooth
from moire.extract_behaviors import extract_upturns_new, extract_metallic_transitions
from hampel import hampel 

OUT = Path('output/extract_behaviors')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]

for field in FIELDS:

    T, nu, R = load_field(field, IN) # loads initial dataset
    T, nu, R = clean_data(T, nu, R) # sorts data and removes nans

    # Creating list of linecuts 
    linecuts = []
    for i, v in enumerate(nu):
        linecuts.append({"nu" : v, "rho" : R[:, i]}) 

    # Proccessing each linecut: smoothing + extraction
    for linecut in linecuts:
        rho = linecut.get("rho")

        # TODO: iterative LOESS or hampel based on T window instead of points
        # Smoothing
        rho = hampel(rho).filtered_data
        rho_smoothed = adaptive_smooth(rho, T)
        linecut.update({"rho_smoothed" : rho_smoothed})

        # Upturn & downturn extraction  
        candidates = [] + extract_upturns_new(T, rho)
        candidates += extract_metallic_transitions(T, rho, candidates)
        linecut.update({"candidates" : candidates})

    # Plotting and saving single linecuts 
    for linecut in linecuts:
        fig = plot_single_linecut(linecut, save = True, OUT = OUT)
        

        

