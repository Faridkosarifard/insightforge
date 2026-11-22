"""Exploratory report contents and JSON-safe numeric values."""

from __future__ import annotations

import math

import pandas as pd

from insightforge.services.eda import _json_float, build_report


def test_json_float_rejects_nan_and_inf() -> None:
    assert _json_float(1.5) == 1.5
    assert _json_float(math.nan) is None
    assert _json_float(math.inf) is None
    assert _json_float("nope") is None


def test_report_numeric_and_missing() -> None:
    frame = pd.DataFrame(
        {
            "height": [1.0, 2.0, 3.0, None],
            "width": [4.0, 5.0, 6.0, 7.0],
            "label": ["a", "b", "a", "c"],
        }
    )
    report = build_report(frame, target="label")
    assert report["n_rows"] == 4
    assert report["n_cols"] == 3
    assert report["missing"]["height"] == 1
    assert report["missing"]["label"] == 0
    assert "height" in report["descriptives"]
    assert "width" in report["correlations"]["height"]
    assert report["target"]["type"] == "categorical"
    assert report["target"]["top_counts"]["a"] == 2


def test_report_numeric_target(iris_frame: pd.DataFrame) -> None:
    report = build_report(iris_frame, target="sepal_length")
    assert report["target"]["type"] == "numeric"
    assert report["target"]["mean"] is not None
    assert report["n_rows"] == 150


def test_report_without_numeric_columns() -> None:
    frame = pd.DataFrame({"city": ["Berlin", "Hamburg", "Munich"]})
    report = build_report(frame)
    assert report["descriptives"] == {}
    assert report["correlations"] == {}
    assert report["target"] is None


def test_report_unknown_target_is_ignored() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3]})
    report = build_report(frame, target="missing")
    assert report["target"] is None
