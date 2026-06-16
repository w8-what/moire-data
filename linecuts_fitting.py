import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.optimize
from functions import *

OUT = Path('output/linecuts_fitting')
IN = Path('source_data')
FIELDS = [87, 96, 99, 103, 74, 87, 96.2, 151, 176]
SC_THRESHOLD = 20.0

FUNCTIONS = {linear, quadratic}

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


def fit_score(pcov):
    st_dev = np.sqrt(np.sqrt(np.diag(pcov)))
    return np.sum(st_dev)/st_dev.shape[0]

# Fitting linecuts
# Input: 1D array
# Output: Dictionary of critical T's {float T, (str left_behavior, str right_behavior)}
def fitting_linecuts(T, roh, min_window = 10) -> dict:

    t_curr = T[0]
    fit_left = None
    fit_right = None
    dict_T = {}

    len_T = len(T)
    i = 0

    while i < len_T:
        j = i + min_window
        max_fit_score = np.inf
        best_j = None

        # EDGE CASE WHERE TOWARDS THE END, THE WINDOW IS FORCED TO BE LESS THAN MIN WINDOW
        if i + min_window >= len_T:
            best_j = len_T

        while j < len_T:
            # Fits the range over all possible functions
            for function in FUNCTIONS:
                popt, pcov = scipy.optimize.curve_fit(function, T[i:j], roh[i:j])
                curr_fit_score = fit_score(pcov)
                if curr_fit_score < max_fit_score:
                    best_j = j
                    max_fit_score = curr_fit_score
                    fit_right = function.__name__
            j += 1

        dict_T.update({T[i] : (fit_left, fit_right)})

        i = best_j
        fit_left = fit_right
    
    dict_T.update({T[len_T-1] : (fit_left, None)})

    print(dict_T)
    return dict_T






# Generating linecut roh v. T plots
# Input: Dictionary of critical T's, along with 'params', the experimental conditions
# and 'roh', array of resistivity, and 'T', array of temperatures for the axis
# Output: Plots of (1) graph of regular roh vs T (2) graph of regular roh
# vs T with critical T's and different coloring for different behaviors
def plot_behavior_fits(params, T, roh, critical_Ts) -> None:

    plt.suptitle(params[0] + " = " + str(params[1]))
    

    # Plotting (1) raw data
    plt.subplot(1, 2, 1)
    plt.plot(T, roh)
    plt.ylabel("Resistivity (? units ?)")
    plt.xlabel("Temperature (K)")
    plt.title("Raw Data")

    # Plotting (2) colored behavior fits
    plt.subplot(1, 2, 2)
    plt.plot(T, roh)

    # Plotting critical Ts and labeling phases 
    
    # Extract critical T's (the boundaries of our windows)
    crit_t_keys = list(critical_Ts.keys())
    
    # Dynamically assign a unique color to each phase (behavior)
    unique_phases = set(val[1] for val in critical_Ts.values() if val[1] is not None)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    phase_colors = {phase: color_cycle[i % len(color_cycle)] for i, phase in enumerate(unique_phases)}

    plotted_labels = set()

    # Plot each phase segment
    for i in range(len(crit_t_keys) - 1):
        t_start = crit_t_keys[i]
        t_end = crit_t_keys[i+1]
        
        # Find indices in the T array to slice the segment
        idx_start = np.where(T == t_start)[0][0]
        idx_end = np.where(T == t_end)[0][0]
        
        # Slice T and roh (adding +1 to idx_end so segments connect visually)
        T_seg = T[idx_start:idx_end+1]
        roh_seg = roh[idx_start:idx_end+1]
        
        # Get the right-side behavior for this window
        phase = critical_Ts[t_start][1] 
        color = phase_colors.get(phase, 'black')
        
        # Ensure we only add each phase to the legend once
        label = phase if phase not in plotted_labels else ""
        if phase: 
            plotted_labels.add(phase)
            
        plt.plot(T_seg, roh_seg, color=color, label=label, linewidth=2)

    # Plotting critical Ts as distinct red dots
    for t_crit in crit_t_keys[1:-1]: # Skip the first and last bounds
        idx = np.where(T == t_crit)[0][0]
        plt.plot(t_crit, roh[idx], 'ro', markersize=6, zorder=5) 
    
    # Add critical T to the legend
    plt.plot([], [], 'ro', markersize=6, label="Critical $T$")
    plt.legend()


    plt.ylabel("Resistivity (? units ?)")
    plt.xlabel("Temperature (K)")
    plt.title("Behavior Fits")

    # Saving plots to path
    path = OUT / f"fit: {params[0]} = {params[1]}.png"
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()





# Fits and plots ~ numLinecuts linecuts evenly spread 
def test_behavior_fits(E: int, numLinecuts: int) -> None:
    T, nu, R = load_field(E)

    numCol = R.shape[1]
    currCol = 0
    spacing = (numCol - 1) / (numLinecuts - 1)

    # Making the linecuts, and generating tests
    for i in range(numLinecuts):
        currColInt = int(currCol + 0.5)
        linecut_roh = R[:, currColInt]

        print("Filling: " + str(nu[currColInt]))
        critical_Ts = fitting_linecuts(T, linecut_roh) # Fitting linecuts
        plot_behavior_fits(("filling", nu[currColInt]), T, linecut_roh, critical_Ts) # Plotting linecuts

        currCol += spacing


test_behavior_fits(99, 12)