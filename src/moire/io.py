import pandas as pd
import numpy as np 
import math 

def load_field(E, IN):

    df = pd.read_csv(IN / f'Rxx_matrix_E-{E}mV_nm.csv')

    T  = df.iloc[:, 0].astype(float).to_numpy()
    nu = np.array([float(c) for c in df.columns[1:]])
    R  = df.iloc[:, 1:].astype(float).to_numpy()

    return T, nu, R


# Sorts the data in increasing T and increasing nu
def clean_sort_data(T, nu, R):

    # Remove rows with invalid T or resistivity values.
    valid_rows = np.isfinite(T) & np.all(np.isfinite(R), axis=1)
    T, R = T[valid_rows], R[valid_rows, :]

    # Remove columns with invalid filling or resistivity values.
    valid_cols = np.isfinite(nu) & np.all(np.isfinite(R), axis=0)
    nu, R = nu[valid_cols], R[:, valid_cols]

    # Sort by increasing temperature.
    idx_T = np.argsort(T)
    T, R = T[idx_T], R[idx_T, :]

    # Sort by increasing filling.
    idx_nu = np.argsort(nu)
    nu, R = nu[idx_nu], R[:, idx_nu]

    return T, nu, R


def fmt4(x):
    if x == 0:
        return "0.000"

    digits_before = 1 if abs(x) < 1 else int(math.log10(abs(x))) + 1
    decimals = max(0, 4 - digits_before)
    return f"{x:.{decimals}f}"