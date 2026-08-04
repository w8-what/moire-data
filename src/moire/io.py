import math
import warnings

import numpy as np
import pandas as pd


def load_field(E, IN):
    field_token = int(E) if float(E).is_integer() else E
    df = pd.read_csv(IN / f"Rxx_matrix_E-{field_token}mV_nm.csv")

    T = df.iloc[:, 0].astype(float).to_numpy()
    nu = np.array([float(c) for c in df.columns[1:]])
    R = df.iloc[:, 1:].astype(float).to_numpy()

    return T, nu, R


def clean_sort_data(T, nu, R, *, invalid="raise"):
    """Validate and sort a resistance matrix by increasing temperature and filling.

    By default, invalid measurements raise an informative error so data loss cannot
    happen silently. Pass ``invalid="drop"`` to retain the previous behavior of
    dropping every row or column containing a non-finite value; a warning reports
    exactly how much data was removed.
    """
    T = np.asarray(T, dtype=float)
    nu = np.asarray(nu, dtype=float)
    R = np.asarray(R, dtype=float)

    if T.ndim != 1 or nu.ndim != 1 or R.ndim != 2:
        raise ValueError("T and nu must be 1D arrays and R must be a 2D array")
    if R.shape != (len(T), len(nu)):
        raise ValueError(f"R has shape {R.shape}; expected ({len(T)}, {len(nu)}) from T and nu")
    if invalid not in {"raise", "drop"}:
        raise ValueError("invalid must be either 'raise' or 'drop'")

    invalid_T = ~np.isfinite(T)
    invalid_nu = ~np.isfinite(nu)
    invalid_R = ~np.isfinite(R)
    if invalid == "raise" and (invalid_T.any() or invalid_nu.any() or invalid_R.any()):
        raise ValueError(
            "non-finite input detected: "
            f"T={int(invalid_T.sum())}, nu={int(invalid_nu.sum())}, "
            f"R={int(invalid_R.sum())}; pass invalid='drop' to remove affected rows and columns"
        )

    original_rows, original_cols = R.shape

    # Remove rows with invalid T or resistivity values.
    valid_rows = np.isfinite(T) & np.all(np.isfinite(R), axis=1)
    T, R = T[valid_rows], R[valid_rows, :]

    # Remove columns with invalid filling or resistivity values.
    valid_cols = np.isfinite(nu) & np.all(np.isfinite(R), axis=0)
    nu, R = nu[valid_cols], R[:, valid_cols]

    if invalid == "drop":
        removed_rows = original_rows - len(T)
        removed_cols = original_cols - len(nu)
        if removed_rows or removed_cols:
            warnings.warn(
                f"dropped {removed_rows} temperature rows and {removed_cols} filling columns "
                "containing non-finite values",
                RuntimeWarning,
                stacklevel=2,
            )

    if len(T) == 0 or len(nu) == 0:
        raise ValueError("no data remains after removing non-finite values")

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
