import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.optimize
from functions import *
from decimal import Decimal

OUT = Path('output/individual_labeling')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]
SC_THRESHOLD = 20.0

PHASES = ["SC", "sublin_M", "SM", "FL", "AFM_M", "AFM_I"]

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
    R = df.iloc[:, 1:].astype(float).to_numpy()
    return T, nu, R


def find_SC(T, roh) -> dict:

    intervals_dict = {}

    below_threshold = roh < SC_THRESHOLD
    
    # make it so that 
    # detects a sharp drop in resistivity, and if it drops below the threshold
    # basically flag all resistivity below the threshold as that? 

    return intervals_dict


def find_sublin_M(T, roh):
    return None

def find_SM(T, roh):
    return None

def find_FL(T, roh):
    return None


def find_AFM_M(T, roh):
    return None


def find_AFM_I(T, roh):
    return None









# Generating linecut roh v. T plots
def plot_behavior_fits(params, T, roh, intervals) -> None:

    # Create a mapping from function names (strings) back to the actual callable functions
    
    # Plotting (1) raw data
    plt.plot(T, roh, marker='o', markerfacecolor='none', markeredgecolor='black', linestyle='none')
    plt.ylabel("Resistivity (Ω·cm)") 
    plt.xlabel("Temperature (K)")
    plt.title("Lines with Fits")
    plt.xlim(0)
    plt.ylim(0)

    for interval in intervals:
        if interval is None:
            continue

        # TODO: Shade phase with color and add label onto the graph 
        plt.fill_betweenx(range(int(max(roh)+0.5)), interval[0][0], interval[0][1], alpha = 0.5)

        

    # Saving plots to path
    filling = Decimal(params[1]).quantize(Decimal("0.000"))
    path = OUT / f"fit_{params[0]}_{filling}.png" 
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()



# Fits and plots ~ numLinecuts linecuts evenly spread 
def test_behavior_fits(E: int, numLinecuts: int) -> None:
    T, nu, R = load_field(E)

    numCol = R.shape[1]
    currCol = 0
    spacing = (numCol - 1) / (numLinecuts - 1)

    # Finding intervals for each linecut and plotting them
    for i in range(numLinecuts):
        currColInt = int(currCol + 0.5)
        linecut_roh = R[:, currColInt]

        # Finding intervals for each phase
        intervals = []
        for phase in PHASES:
            func = globals().get(f"find_{phase}")
            intervals.append(func(T, linecut_roh))

        plot_behavior_fits(("Filling", nu[currColInt]), T, linecut_roh, intervals)

        currCol += spacing

test_behavior_fits(103, 20)