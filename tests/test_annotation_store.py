import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "phase_visualizer"))

from annotation_store import (  # noqa: E402
    empty_annotations,
    load_annotations,
    save_annotations,
    validate_annotations,
)


def test_round_trip_annotations(tmp_path):
    path = tmp_path / "annotations" / "crossover_labels.json"
    annotations = empty_annotations()
    annotations["fields"]["103.0"] = {
        "boundaries": {"Tcoh": [{"nu": 1.02, "T": 1.4}], "Tprime": [{"nu": 0.92, "T": 2.1}]},
        "reviewed": {"Tcoh": True, "Tprime": False},
        "linecuts": {
            "7": {
                "nu": 0.92,
                "notes": "Representative sublinear crossover",
                "Tcoh": {"status": "not_identifiable"},
                "Tprime": {"status": "value", "T": 2.1, "certainty": "certain"},
            }
        },
    }

    saved = save_annotations(path, annotations)

    assert saved["updatedAt"]
    assert load_annotations(path) == json.loads(path.read_text())
    assert load_annotations(path)["fields"]["103.0"]["linecuts"]["7"]["Tcoh"] == {
        "status": "not_identifiable"
    }


def test_published_figure3_annotations_are_valid_and_segmented():
    annotations = load_annotations(ROOT / "annotations" / "crossover_labels.json")

    assert set(annotations["fields"]) == {"87", "96", "99", "103"}
    assert annotations["fields"]["103"]["provenance"]["sheet"] == "panel_e"
    assert annotations["fields"]["103"]["boundaries"]["Tprime"][0] == {
        "nu": 0.888,
        "T": 0.14,
        "segment": "electron",
    }
    assert {point["segment"] for point in annotations["fields"]["103"]["boundaries"]["Tcoh"]} == {
        "electron",
        "hole",
    }
    assert annotations["fields"]["87"]["boundaries"]["Tprime"] == []


@pytest.mark.parametrize(
    "label",
    [
        {"status": "value", "T": 1.5},
        {"status": "value", "T": float("nan"), "certainty": "certain"},
        {"status": "maybe"},
    ],
)
def test_rejects_invalid_linecut_labels(label):
    annotations = empty_annotations()
    annotations["fields"]["103.0"] = {
        "boundaries": {"Tcoh": [], "Tprime": []},
        "reviewed": {"Tcoh": False, "Tprime": False},
        "linecuts": {"0": {"nu": 0.9, "Tcoh": label}},
    }

    with pytest.raises(ValueError):
        validate_annotations(annotations)
