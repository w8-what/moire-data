import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moire.io import load_field
from moire.draw_2d import draw_mosaic_diagrams


FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]
IN = Path("source_data")
OUT = Path("output/phase_diagrams/mosaic")


for field in FIELDS:
    T, nu, R = load_field(field, IN)
    name = f"E = {field}mV"
    draw_mosaic_diagrams(nu, T, R, OUT = OUT, save = True, name = name)





