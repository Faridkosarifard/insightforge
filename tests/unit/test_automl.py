"""Baseline training for classification and regression, plus validation errors."""

from __future__ import annotations

import pandas as pd
import pytest

from insightforge.services.automl import (
    MIN_TRAIN_ROWS,
    TrainingError,
    infer_task_type,
    train_baseline,
)


def test_infer_task_type_classification_and_regression() -> None:
    assert infer_task_type(pd.Series(["a", "b", "a"])) == "classification"
    assert infer_task_type(pd.Series([0, 1, 0, 1])) == "classification"
    assert infer_task_type(pd.Series(range(20))) == "regression"


def test_train_classifier_on_iris(iris_frame: pd.DataFrame) -> None:
    result = train_baseline(iris_frame, "species", n_estimators=20)
    assert result["task_type"] == "classification"
    assert result["estimator"] == "RandomForestClassifier"
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert "f1_macro" in result["metrics"]
    # Three iris classes should produce a one-vs-rest ROC AUC.
    assert "roc_auc_ovr" in result["metrics"]
    predictions = result["model"].predict(iris_frame.drop(columns=["species"]).head(3))
    assert len(predictions) == 3


def test_train_regressor(regression_frame: pd.DataFrame) -> None:
    result = train_baseline(regression_frame, "y", n_estimators=20)
    assert result["task_type"] == "regression"
    assert result["estimator"] == "RandomForestRegressor"
    metrics = result["metrics"]
    assert {"mae", "rmse", "r2"} <= metrics.keys()
    assert metrics["mae"] >= 0.0


def test_train_handles_missing_and_categorical() -> None:
    frame = pd.DataFrame(
        {
            "num": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "cat": ["x", "y", "x", "y", "x", "y", "x", None],
            "label": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        }
    )
    result = train_baseline(frame, "label", n_estimators=20)
    assert result["task_type"] == "classification"
    assert result["metrics"]["accuracy"] >= 0.0


def test_train_binary_roc_auc() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    result = train_baseline(frame, "y", n_estimators=20)
    assert "roc_auc" in result["metrics"]


def test_train_rejects_missing_target(iris_frame: pd.DataFrame) -> None:
    with pytest.raises(TrainingError, match="not in the dataset"):
        train_baseline(iris_frame, "not_a_column")


def test_train_rejects_too_few_rows() -> None:
    frame = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    with pytest.raises(TrainingError, match=str(MIN_TRAIN_ROWS)):
        train_baseline(frame, "y")


def test_train_rejects_no_features() -> None:
    frame = pd.DataFrame({"y": [0, 1, 0, 1, 0, 1]})
    with pytest.raises(TrainingError, match="feature column"):
        train_baseline(frame, "y")


def test_train_rejects_all_missing_target() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [None, None, None, None]})
    with pytest.raises(TrainingError, match="entirely missing"):
        train_baseline(frame, "y")


def test_stratify_fallback_on_rare_class() -> None:
    # One class appears once, so stratified splitting fails and we fall back.
    frame = pd.DataFrame(
        {
            "x": list(range(8)),
            "y": ["a", "a", "a", "a", "a", "a", "a", "b"],
        }
    )
    result = train_baseline(frame, "y", n_estimators=10)
    assert result["task_type"] == "classification"
