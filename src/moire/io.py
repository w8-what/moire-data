import pandas as pd
import numpy as np 
import math 

def load_field(E, IN):

    # TODO: remove nans for preprocessing 

    df = pd.read_csv(IN / f'Rxx_matrix_E-{E}mV_nm.csv')

    T  = df.iloc[:, 0].astype(float).to_numpy()
    nu = np.array([float(c) for c in df.columns[1:]])
    R  = df.iloc[:, 1:].astype(float).to_numpy()

    # Sort rows by increasing temperature; R rows must follow T
    idx = np.argsort(T)
    T, R = T[idx], R[idx]

    return T, nu, R


def clean_data(T, nu, R):

    idx = np.argsort(T)
    T, R = T[idx], R[idx]




def fmt4(x):
    if x == 0:
        return "0.000"

    digits_before = 1 if abs(x) < 1 else int(math.log10(abs(x))) + 1
    decimals = max(0, 4 - digits_before)
    return f"{x:.{decimals}f}"