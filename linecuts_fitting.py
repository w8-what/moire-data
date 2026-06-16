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

FUNCTIONS = {linear, quadratic, sublinear}

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
    fit_left = {None : None}
    fit_right = {None : None}
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
                    fit_right = {function.__name__ : popt}
            j += 1

        dict_T.update({T[i] : (fit_left, fit_right)})

        i = best_j
        fit_left = fit_right
    
    dict_T.update({T[len_T-1] : (fit_left, {None : None})})

    print(dict_T)
    return dict_T






# Generating linecut roh v. T plots
# Input: Dictionary of critical T's, along with 'params', the experimental conditions
# and 'roh', array of resistivity, and 'T', array of temperatures for the axis
# Output: Plots of (1) graph of regular roh vs T (2) graph of regular roh
# vs T with critical T's and different coloring for different behaviors
# Generating linecut roh v. T plots with smooth fits
def plot_behavior_fits(params, T, roh, critical_ts) -> None:

    # Create a mapping from function names (strings) back to the actual callable functions
    # This assumes your global FUNCTIONS = {linear, quadratic} is still present in the file
    func_map = {f.__name__: f for f in FUNCTIONS}

    plt.suptitle(f"{params[0]} = {params[1]}")
    
    # Plotting (1) raw data
    plt.subplot(1, 2, 1)
    plt.plot(T, roh, marker='o', markerfacecolor='none', markeredgecolor='black', linestyle='none')
    plt.ylabel("Resistivity (Ω·cm)") 
    plt.xlabel("Temperature (K)")
    plt.title("Raw Data")

    # Plotting (2) colored behavior fits
    plt.subplot(1, 2, 2)
    plt.title("Behavior Fits")
    plt.ylabel("Resistivity (Ω·cm)")
    plt.xlabel("Temperature (K)")

    # Plot the raw data points in the background (zorder=1)
    plt.plot(T, roh, marker='o', markerfacecolor='none', markeredgecolor='gray', linestyle='none', alpha=0.5, zorder=1)

    crit_t_keys = list(critical_ts.keys())
    
    # Dynamically assign a unique color to each phase by inspecting the right-side fits
    unique_phases = set()
    for val in critical_ts.values():
        right_fit = val[1]
        if right_fit:
            phase_name = list(right_fit.keys())[0]
            unique_phases.add(phase_name)
            
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    phase_colors = {phase: color_cycle[i % len(color_cycle)] for i, phase in enumerate(unique_phases)}

    plotted_labels = set()

    # Plot each smoothed phase segment ON TOP of the raw data dots (zorder=2)
    for i in range(len(crit_t_keys) - 1):
        t_start = crit_t_keys[i]
        t_end = crit_t_keys[i+1]
        
        # Extract the right-side behavior dictionary for this window
        right_info = critical_ts[t_start][1] 
        
        if not right_info:
            continue
            
        # Extract the phase name and the optimal parameters
        phase = list(right_info.keys())[0]
        popt = right_info[phase]
        
        # Generate dense T values for a mathematically smooth curve
        T_smooth = np.linspace(t_start, t_end, 200)
        
        # Evaluate the mathematical function using the unpacked *popt parameters
        roh_smooth = func_map[phase](T_smooth, *popt)
        
        color = phase_colors.get(phase, 'black')
        label = phase if phase not in plotted_labels else ""
        if phase: 
            plotted_labels.add(phase)
            
        # Plotting the smooth segment line overlay
        plt.plot(T_smooth, roh_smooth, color=color, label=label, linewidth=2.5, zorder=2)

    # Plotting critical Ts as distinct solid red dots on the very top (zorder=3)
    for t_crit in crit_t_keys[1:-1]: 
        idx = np.where(T == t_crit)[0][0]
        plt.plot(t_crit, roh[idx], marker='o', color='red', markersize=6, linestyle='none', zorder=3) 
    
    # Add critical T to the legend
    plt.plot([], [], marker='o', color='red', markersize=6, linestyle='none', label="Critical T")
    plt.legend()

    # Saving plots to path
    path = OUT / f"fit_{params[0]}_{params[1]}.png" 
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


test_behavior_fits(99, 20)