import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from decimal import Decimal
from functions import *
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path('output/manual_fitting')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]
SC_THRESHOLD = 20.0

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})


# Loading data and returns T (array of temperatures, x-axis), nu (array of fillings), and R (2D array of resistivity)
def load_field(E):
    df = pd.read_csv(IN / f'Rxx_matrix_E-{E}mV_nm.csv')
    T = df.iloc[:, 0].astype(float).to_numpy()
    nu = np.array([float(c) for c in df.columns[1:]])
    R = df.iloc[:, 1:].astype(float).to_numpy()  # (???) how does this ignore the first row 
    return T, nu, R



# Generating linecut roh v. T plots
# Input: Dictionary of critical T's, along with 'params', the experimental conditions
# and 'roh', array of resistivity, and 'T', array of temperatures for the axis
# Output: Plots of (1) graph of regular roh vs T (2) graph of regular roh
# vs T with critical T's and different coloring for different behaviors
# Generating linecut roh v. T plots with smooth fits
def plot_behavior_fits(params, T, roh, pdf) -> None:
    
    fig, ax = plt.subplots()

    # Plotting (1) raw data
    ax.plot(T, roh, marker='o', markerfacecolor='none', markeredgecolor='blue', linestyle='none')
    ax.set_ylabel("Resistivity (Ω·cm)") 
    ax.set_xlabel("Temperature (K)")

    ax.set_ylim(0, None)
    ax.set_xlim(0, None)

    filling = Decimal(params[1])
    filling = str(filling.quantize(Decimal("0.000")))
    ax.set_title(f"{params[0]} =  {filling}")

    # Saving plots to path
    pdf.savefig(fig, dpi=250, bbox_inches='tight')    
    plt.close(fig)




# Making the linecuts, and generating tests

T, nu, R = load_field(103)

pdf_path = OUT / "all_behavior_fits.pdf"    

with PdfPages(pdf_path) as pdf:
    for i in range(R.shape[1]):
        linecut_roh = R[:, i]
        plot_behavior_fits(("filling", nu[i]), T, linecut_roh, pdf)







