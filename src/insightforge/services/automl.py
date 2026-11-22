"""Train a scikit-learn baseline on a tabular DataFrame."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MIN_TRAIN_ROWS = 4
CLASSIFICATION_CARDINALITY = 10


class TrainingError(ValueError):
    """Raised when a dataset cannot be used to train a baseline model."""


def infer_task_type(y: pd.Series) -> str:
    """Classify a target as classification or regression.

    A numeric column with more than ten distinct values is treated as regression;
    every other target is classification.
    """
    if y.dtype.kind in "ifu" and y.nunique(dropna=True) > CLASSIFICATION_CARDINALITY:
        return "regression"
    return "classification"


def train_baseline(
    df: pd.DataFrame,
    target: str,
    *,
    n_estimators: int = 200,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit a Random Forest baseline with a standard numeric/categorical pipeline.

    Returns a dict with ``task_type``, ``estimator``, ``metrics``, and ``model``
    (the fitted sklearn pipeline).
    """
    if target not in df.columns:
        raise TrainingError(f"Target column {target!r} is not in the dataset")
    if len(df) < MIN_TRAIN_ROWS:
        raise TrainingError(f"Need at least {MIN_TRAIN_ROWS} rows to train a baseline model, got {len(df)}")

    features = df.drop(columns=[target])
    labels = df[target]
    if labels.isna().all():
        raise TrainingError("Target column is entirely missing")
    if features.shape[1] == 0:
        raise TrainingError("Dataset must contain at least one feature column besides the target")

    task = infer_task_type(labels)
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    categorical_cols = [column for column in features.columns if column not in numeric_cols]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    if task == "classification":
        estimator: Any = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        estimator_name = "RandomForestClassifier"
    else:
        estimator = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        estimator_name = "RandomForestRegressor"

    pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])

    stratify = labels if task == "classification" else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    pipeline.fit(x_train, y_train)
    y_hat = pipeline.predict(x_test)

    if task == "classification":
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_hat)),
            "f1_macro": float(f1_score(y_test, y_hat, average="macro", zero_division=0)),
        }
        _maybe_add_roc_auc(pipeline, x_test, y_test, metrics)
    else:
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_hat)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_hat))),
            "r2": float(r2_score(y_test, y_hat)),
        }

    return {
        "task_type": task,
        "estimator": estimator_name,
        "metrics": metrics,
        "model": pipeline,
    }


def _maybe_add_roc_auc(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: dict[str, Any],
) -> None:
    if not hasattr(pipeline, "predict_proba"):
        return
    try:
        proba = pipeline.predict_proba(x_test)
    except Exception:
        return
    try:
        if proba.ndim == 2 and proba.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
        elif proba.ndim == 2 and proba.shape[1] > 2:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_test, proba, multi_class="ovr"))
    except ValueError:
        return
