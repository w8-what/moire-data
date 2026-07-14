import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moire.draw_2d import plot_all_linecuts, plot_single_linecut
from moire.io import load_field

OUT = Path('output/extract_behaviors')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]

for field in FIELDS:

    T, nu, R = load_field(field, IN)
    T, nu, R = pre_process(T, nu, R)

    [(nu, [asdjkfalsdfj]), [nu]] # array of linecuts

    for linecut in linecuts:
        # hampel
        # smooth data
        # extract on smooth data
            # local min & max upturns, alonside scoring 
            # -> get candidates 

        # draw linecut if want to 
        
    

    # later ...
    # candidates processing and plotting heatmaps & phase_diagrams 
    # passing in T, nu, R, and candidates 

        



    # plot each linecut


    plot_all_linecuts(field, 20, IN, OUT)
    

# plot_all_linecuts(103, 50, IN, OUT)

