import numpy as np
import pytest

from moire.extract_features import extract_Tprime


def _linecut(T, rho, *, smooth=None, lower=None, upper=None):
    behaviors = []
    if lower is not None and upper is not None:
        behaviors.append({"type": "extraction_range", "T_lower": lower, "T_upper": upper})
    return {
        "nu": 0.94,
        "rho": np.asarray(rho, float),
        "rho_smoothed": np.asarray(rho if smooth is None else smooth, float),
        "local_noise": np.full_like(T, 0.5, dtype=float),
        "behaviors": behaviors,
    }


def _extract(T, linecut, **kwargs):
    return extract_Tprime(
        T, linecut, min_fit_points=8, min_fit_span=1.0, min_sublinear_span=0.5, **kwargs
    )


def test_extract_Tprime_finds_persistent_sublinear_departure_from_high_T_line():
    T = np.linspace(1.0, 6.0, 101)
    linear = 100 + 20 * T
    sublinear = -72.13203435596427 + 150 * np.sqrt(T)
    rho = np.where(T < 2.0, sublinear, linear)
    linecut = _linecut(T, rho, lower=1.0, upper=6.0)

    candidates = _extract(T, linecut)

    relative_difference = np.abs(rho - linear) / linear
    expected_index = np.flatnonzero(relative_difference >= 0.10)[-1]
    assert candidates
    assert candidates[0]["T"] == pytest.approx(T[expected_index])
    assert candidates[0]["type"] == "Tprime"
    assert candidates[0]["fit_T_lower"] > candidates[0]["T"]
    assert candidates[0]["fit_T_upper"] == pytest.approx(T[-1])
    assert candidates[0]["persistence_fraction"] >= 0.8
    assert candidates[0]["sublinear_n"] == pytest.approx(0.5, abs=1e-6)
    assert candidates[0]["sublinear_probability"] >= 0.8


def test_extract_Tprime_rejects_a_curve_that_remains_linear():
    T = np.linspace(1.0, 6.0, 101)
    rho = 100 + 20 * T

    assert _extract(T, _linecut(T, rho)) == []


def test_extract_Tprime_rejects_departure_into_superlinear_behavior():
    T = np.linspace(1.0, 6.0, 101)
    linear = 100 + 20 * T
    superlinear = -80 + 40 * T**1.5
    superlinear += linear[np.searchsorted(T, 2.0)] - (-80 + 40 * 2.0**1.5)
    rho = np.where(T < 2.0, superlinear, linear)

    assert _extract(T, _linecut(T, rho)) == []


def test_extract_Tprime_rejects_an_isolated_ten_percent_outlier():
    T = np.linspace(1.0, 6.0, 101)
    linear = 100 + 20 * T
    sublinear = -72.13203435596427 + 150 * np.sqrt(T)
    raw = np.where(T < 2.0, sublinear, linear)
    smooth = linear.copy()
    outlier_index = np.searchsorted(T, 3.0)
    smooth[outlier_index] *= 1.2

    assert _extract(T, _linecut(T, raw, smooth=smooth)) == []


def test_extract_Tprime_respects_the_extraction_range():
    T = np.linspace(1.0, 6.0, 101)
    linear = 100 + 20 * T
    sublinear = -72.13203435596427 + 150 * np.sqrt(T)
    rho = np.where(T < 2.0, sublinear, linear)

    linecut = _linecut(T, rho, lower=2.0, upper=6.0)
    assert _extract(T, linecut) == []


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("deviation", 0),
        ("min_pvalue", 1.1),
        ("persistence_fraction", 0),
        ("min_sublinear_probability", -0.1),
        ("exponent_bounds", (1.0, 4.0)),
    ],
)
def test_extract_Tprime_validates_thresholds(parameter, value):
    T = np.linspace(1.0, 6.0, 20)
    rho = 100 + 20 * T

    with pytest.raises(ValueError):
        extract_Tprime(T, _linecut(T, rho), **{parameter: value})
