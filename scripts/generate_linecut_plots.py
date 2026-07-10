import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moire.draw_linecuts import plot_all_linecuts, plot_single_linecut

OUT = Path('output/extract_behaviors')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]

for field in FIELDS:
    plot_all_linecuts(field, 20, IN, OUT)
    

