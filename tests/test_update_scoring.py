import copy

import numpy as np
import pytest

from moire.update_scoring import update_extrema


def _linecuts():
    return [
        {
            "nu": nu,
            "features": [
                {"T": 1.0, "nu": nu, "type": "upturn", "confidence": 0.9},
                {"T": 2.0, "nu": nu, "type": "downturn", "confidence": 0.02},
            ],
        }
        for nu in (0.9, 1.0, 1.1)
    ]


def test_update_extrema_returns_surviving_feature_signatures():
    temperatures = np.array([0.0, 1.0, 2.0])
    linecuts = _linecuts()
    original_features = copy.deepcopy([linecut["features"] for linecut in linecuts])

    extrema = update_extrema(temperatures, linecuts)

    assert isinstance(extrema, list)
    assert len(extrema) == 3
    assert all(isinstance(feature, dict) for feature in extrema)
    assert all(
        {"T", "nu", "type", "confidence", "score_15"} <= feature.keys() for feature in extrema
    )
    assert all(feature["type"] == "upturn" for feature in extrema)

    # The flat return and the grouped downstream representation share the
    # copied feature dictionaries; the original extracted signatures are safe.
    grouped = [feature for linecut in linecuts for feature in linecut["features_new"]]
    assert all(returned is stored for returned, stored in zip(extrema, grouped))
    assert [linecut["features"] for linecut in linecuts] == original_features


def test_update_extrema_prunes_after_each_pass_and_keeps_score_numbers_consecutive():
    temperatures = [0.0, 1.0, 2.0]
    linecuts = _linecuts()

    extrema = update_extrema(temperatures, linecuts, num_iter=2, num_passes=2, filter=0.1)

    assert all(f"score_{iteration}" in feature for feature in extrema for iteration in range(1, 5))
    assert all(feature["confidence"] == 0.9 for feature in extrema)
    assert all(len(linecut["features_new"]) == 1 for linecut in linecuts)


@pytest.mark.parametrize(
    "temperatures, options",
    [
        ([], {}),
        ([[0.0, 1.0]], {}),
        ([0.0, 1.0], {"num_iter": 0}),
        ([0.0, 1.0], {"num_passes": 0}),
        ([0.0, 1.0], {"filter": -0.01}),
        ([0.0, 1.0], {"filter": 1.01}),
    ],
)
def test_update_extrema_rejects_invalid_configuration(temperatures, options):
    with pytest.raises(ValueError):
        update_extrema(temperatures, _linecuts(), **options)
