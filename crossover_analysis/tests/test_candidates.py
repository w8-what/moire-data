from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from crossover_candidates import analyze_crossover


def temperatures():
    return np.r_[np.arange(0.05, 0.51, 0.05), np.arange(0.6, 4.01, 0.1)]


def linecut(rho, nu=0.9, sigma=0.03):
    return {
        "rho": np.asarray(rho),
        "rho_smoothed": np.asarray(rho),
        "local_noise": np.full_like(rho, sigma, dtype=float),
        "nu": nu,
    }


def test_tcoh_finds_persistent_quadratic_departure():
    T = temperatures()
    quadratic = 10 + 1.8 * T**2
    rho = quadratic.copy()
    high = T > 1.35
    rho[high] -= 4.2 * (T[high] - 1.35)

    result = analyze_crossover(T, linecut(rho, nu=0.86), "Tcoh")

    assert result["status"] == "locally_supported"
    assert result["candidates"]
    best = result["candidates"][0]
    assert 1.4 < best["T"] < 2.1
    assert 0.65 <= best["plausibility"] <= 1
    assert best["supporting_windows"] >= 2


def test_tprime_finds_departure_below_high_temperature_linear_fit():
    T = temperatures()
    linear = 12 + 3 * T
    rho = linear.copy()
    low = T < 1.55
    rho[low] += 3.5 * (1.55 - T[low]) ** 1.4

    result = analyze_crossover(T, linecut(rho, nu=0.92), "Tprime")

    assert result["status"] == "locally_supported"
    assert result["candidates"]
    best = result["candidates"][0]
    assert 0.7 < best["T"] < 1.4
    # The exponent term may be neutral when the sub-crossing span is too short.
    assert best["C"] >= 0.5


def test_exact_baselines_do_not_force_crossovers():
    T = temperatures()
    tcoh = analyze_crossover(T, linecut(8 + 2 * T**2), "Tcoh")
    tprime = analyze_crossover(T, linecut(8 + 2 * T), "Tprime")

    assert tcoh["status"] == "not_identifiable"
    assert tcoh["candidates"] == []
    assert tprime["status"] == "not_identifiable"
    assert tprime["candidates"] == []


def test_filling_metadata_cannot_change_scores():
    T = temperatures()
    linear = 12 + 3 * T
    rho = linear.copy()
    rho[T < 1.55] += 3.5 * (1.55 - T[T < 1.55]) ** 1.4

    first = analyze_crossover(T, linecut(rho, nu=0.2), "Tprime")
    second = analyze_crossover(T, linecut(rho, nu=1.8), "Tprime")

    first_scores = [(item["T"], item["plausibility"]) for item in first["candidates"]]
    second_scores = [(item["T"], item["plausibility"]) for item in second["candidates"]]
    assert first_scores == second_scores


def test_every_component_is_bounded():
    T = temperatures()
    quadratic = 10 + 1.8 * T**2
    rho = quadratic - np.where(T > 1.35, 4.2 * (T - 1.35), 0)
    candidates = analyze_crossover(T, linecut(rho), "Tcoh")["candidates"]

    assert candidates
    for candidate in candidates:
        for name in ("plausibility", "B", "D", "R", "C"):
            assert 0 <= candidate[name] <= 1
