import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moire.draw_2d import draw_heatmap, draw_heatmap_candidates
from moire.io import load_field


FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]
IN = Path("source_data")
OUT = Path("output/heatmaps/diagnostic")



for field in FIELDS:
    T, nu, R = load_field(field, IN)
    name = f"E = {field}mV"
    draw_heatmap_candidates(nu, T, R, OUT = OUT, save = True, name = name)





