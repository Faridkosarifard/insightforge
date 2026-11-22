"""Persist and load trained model artifacts with joblib."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib


def save_model(dataset_id: int, model: Any, storage_dir: Path) -> Path:
    """Write a trained estimator (or pipeline) next to other artifacts."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = storage_dir / f"model_ds{dataset_id}_{int(time.time())}.joblib"
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> Any:
    """Load a previously saved artifact.

    Artifacts are written by this process on local disk; they are not loaded
    from untrusted sources.
    """
    return joblib.load(path)
