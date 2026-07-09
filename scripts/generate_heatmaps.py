import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moire.draw_heatmaps import draw_heatmap
from moire.io import load_field

IN = Path("source_data")
OUT = Path("output/heatmaps/diagnostic")

T, nu, R = load_field(103, IN)

draw_heatmap(nu, T, R, OUT, save = True)

