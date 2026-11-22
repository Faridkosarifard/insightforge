"""Build a JSON exploratory analysis report from a tabular DataFrame."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

JSONScalar = float | int | str | None


def _json_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def build_report(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Summarise columns, missingness, descriptives, correlations, and the target."""
    report: dict[str, Any] = {
        "n_rows": len(df),
        "n_cols": int(df.shape[1]),
        "columns": {str(column): str(df[column].dtype) for column in df.columns},
        "missing": {str(column): int(df[column].isna().sum()) for column in df.columns},
        "descriptives": {},
        "correlations": {},
        "target": None,
    }
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        descriptives = numeric.describe().to_dict()
        report["descriptives"] = {
            str(column): {str(stat): _json_float(value) for stat, value in stats.items()}
            for column, stats in descriptives.items()
        }
        corr = numeric.corr(numeric_only=True)
        report["correlations"] = {
            str(column): {str(other): _json_float(value) for other, value in corr[column].items()}
            for column in corr.columns
        }
    if target and target in df.columns:
        series = df[target]
        if series.dtype.kind in "ifu":
            report["target"] = {
                "name": target,
                "type": "numeric",
                "mean": _json_float(series.mean()),
                "std": _json_float(series.std()),
            }
        else:
            counts = series.astype("string").value_counts(dropna=False).head(10).to_dict()
            report["target"] = {
                "name": target,
                "type": "categorical",
                "top_counts": {str(key): int(value) for key, value in counts.items()},
            }
    return report
