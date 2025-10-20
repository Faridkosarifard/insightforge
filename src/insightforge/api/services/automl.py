# Third-party (others)
import numpy as np
import pandas as pd

# Scikit-Learn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def _task_type(y: pd.Series) -> str:
    if y.dtype.kind in "ifu":
        return "classification" if y.nunique(dropna=True) <= 10 else "regression"
    return "classification"


def train_baseline(df: pd.DataFrame, target: str) -> dict:
    y = df[target]
    X = df.drop(columns=[target])
    task = _task_type(y)
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ]
    )

    model = (
        RandomForestClassifier(n_estimators=200, random_state=42)
        if task == "classification"
        else RandomForestRegressor(n_estimators=300, random_state=42)
    )
    pipe = Pipeline([("prep", pre), ("model", model)])

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if task == "classification" else None,
    )
    pipe.fit(Xtr, ytr)

    if task == "classification":
        yhat = pipe.predict(Xte)
        metrics = {
            "accuracy": float(accuracy_score(yte, yhat)),
            "f1_macro": float(f1_score(yte, yhat, average="macro")),
        }
        try:
            proba = pipe.predict_proba(Xte)
            if proba.shape[1] > 1:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(yte, proba, multi_class="ovr")
                )
        except Exception:
            pass
        est = "RandomForestClassifier"
    else:
        yhat = pipe.predict(Xte)
        metrics = {
            "mae": float(mean_absolute_error(yte, yhat)),
            "rmse": float(np.sqrt(mean_squared_error(yte, yhat))),
            "r2": float(r2_score(yte, yhat)),
        }
        est = "RandomForestRegressor"

    return {"task_type": task, "estimator": est, "metrics": metrics, "model": pipe}
