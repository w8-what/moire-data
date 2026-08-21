from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from improved_tcoh import (
    ABSOLUTE_PLAUSIBILITY,
    MAX_RETAINED_CANDIDATES,
    RELATIVE_PLAUSIBILITY_MARGIN,
    analyze_tcoh,
    select_candidates,
    strict_clusters,
)


def test_strict_clusters_do_not_chain_across_more_than_two_indices():
    proposals = [{"index": index} for index in (10, 12, 14)]

    clusters = strict_clusters(proposals, maximum_diameter=2)

    assert [[proposal["index"] for proposal in cluster] for cluster in clusters] == [[10, 12], [14]]


def test_improved_extractor_returns_competitive_local_candidates_with_uncertainty():
    temperatures = np.r_[np.arange(0.05, 0.51, 0.05), np.arange(0.6, 4.01, 0.1)]
    baseline = 10 + 1.8 * temperatures**2
    rho = baseline - np.where(temperatures > 1.35, 4.2 * (temperatures - 1.35), 0)
    linecut = {
        "rho": rho,
        "rho_smoothed": rho,
        "local_noise": np.full_like(temperatures, 0.03),
        "nu": 0.86,
    }

    result = analyze_tcoh(temperatures, linecut)

    assert 1 <= len(result["candidates"]) <= MAX_RETAINED_CANDIDATES
    assert all(0 <= candidate["plausibility"] <= 1 for candidate in result["candidates"])
    assert all(
        candidate["plausibility"] >= ABSOLUTE_PLAUSIBILITY for candidate in result["candidates"]
    )
    assert all(candidate["T_uncertainty_K"] >= 0 for candidate in result["candidates"])
    best = result["candidates"][0]["plausibility"]
    assert all(
        best - candidate["plausibility"] <= RELATIVE_PLAUSIBILITY_MARGIN + 1e-12
        for candidate in result["candidates"]
    )


def test_competitive_selection_can_return_no_candidate():
    pool = [{"plausibility": 0.64, "T": 1.0}]

    assert select_candidates(pool) == []


def test_exact_quadratic_has_no_departure_candidate():
    temperatures = np.r_[np.arange(0.05, 0.51, 0.05), np.arange(0.6, 4.01, 0.1)]
    rho = 10 + 1.8 * temperatures**2
    linecut = {
        "rho": rho,
        "rho_smoothed": rho,
        "local_noise": np.full_like(temperatures, 0.03),
        "nu": 0.86,
    }

    result = analyze_tcoh(temperatures, linecut)

    assert result["status"] == "not_identifiable"
    assert result["candidates"] == []
