"""Validation and durable storage for human crossover annotations."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
FEATURES = ("Tcoh", "Tprime")
LABEL_STATUSES = {"value", "not_identifiable"}
CERTAINTIES = {"certain", "approximate", "ambiguous"}


def empty_annotations():
    return {"schemaVersion": SCHEMA_VERSION, "updatedAt": None, "fields": {}}


def load_annotations(path: Path):
    if not path.exists():
        return empty_annotations()
    value = json.loads(path.read_text())
    validate_annotations(value)
    return value


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def validate_annotations(value):
    """Reject malformed or unexpectedly large annotation documents."""
    if not isinstance(value, dict):
        raise ValueError("annotation document must be an object")
    if value.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}")

    fields = value.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("fields must be an object")
    if len(fields) > 100:
        raise ValueError("too many annotated fields")

    total_points = 0
    total_linecuts = 0
    for field_key, field in fields.items():
        if not isinstance(field_key, str) or not isinstance(field, dict):
            raise ValueError("each field annotation must be an object")

        boundaries = field.get("boundaries", {})
        reviewed = field.get("reviewed", {})
        linecuts = field.get("linecuts", {})
        if not isinstance(boundaries, dict) or not isinstance(reviewed, dict):
            raise ValueError("boundaries and reviewed must be objects")
        if not isinstance(linecuts, dict):
            raise ValueError("linecuts must be an object")

        for feature in FEATURES:
            points = boundaries.get(feature, [])
            if not isinstance(points, list):
                raise ValueError(f"{feature} boundary must be an array")
            total_points += len(points)
            for point in points:
                if not isinstance(point, dict):
                    raise ValueError("boundary points must be objects")
                _finite_number(point.get("nu"), "boundary nu")
                _finite_number(point.get("T"), "boundary T")
                if "segment" in point and (
                    not isinstance(point["segment"], str) or len(point["segment"]) > 100
                ):
                    raise ValueError("boundary segment must be short text")
            if feature in reviewed and not isinstance(reviewed[feature], bool):
                raise ValueError("reviewed flags must be booleans")

        total_linecuts += len(linecuts)
        for index, linecut in linecuts.items():
            if not isinstance(index, str) or not isinstance(linecut, dict):
                raise ValueError("linecut annotations must be objects")
            if "nu" in linecut:
                _finite_number(linecut["nu"], "linecut nu")
            if "notes" in linecut and (
                not isinstance(linecut["notes"], str) or len(linecut["notes"]) > 2000
            ):
                raise ValueError("linecut notes must be text shorter than 2000 characters")

            for feature in FEATURES:
                label = linecut.get(feature)
                if label is None:
                    continue
                if not isinstance(label, dict) or label.get("status") not in LABEL_STATUSES:
                    raise ValueError(f"invalid {feature} label")
                if label["status"] == "value":
                    _finite_number(label.get("T"), f"{feature} temperature")
                    if label.get("certainty") not in CERTAINTIES:
                        raise ValueError(f"invalid {feature} certainty")

    if total_points > 100_000 or total_linecuts > 100_000:
        raise ValueError("annotation document is too large")
    return value


def save_annotations(path: Path, value):
    validate_annotations(value)
    value = json.loads(json.dumps(value))
    value["updatedAt"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return value
