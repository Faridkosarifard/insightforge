# Third-party (others)
import pandas as pd


def build_report(df: pd.DataFrame, target: str | None = None) -> dict:
    rep = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "descriptives": {},
        "correlations": {},
        "target": None,
    }
    num = df.select_dtypes(include="number")
    if not num.empty:
        desc = num.describe().to_dict()
        rep["descriptives"] = {
            k: {m: float(v) for m, v in d.items()} for k, d in desc.items()
        }
        corr = num.corr(numeric_only=True).fillna(0.0)
        rep["correlations"] = {
            c: {k: float(v) for k, v in corr[c].items()} for c in corr.columns
        }
    if target and target in df.columns:
        s = df[target]
        if s.dtype.kind in "ifu":
            rep["target"] = {
                "name": target,
                "type": "numeric",
                "mean": float(s.mean()),
                "std": float(s.std()),
            }
        else:
            vc = s.astype("string").value_counts().head(10).to_dict()
            rep["target"] = {
                "name": target,
                "type": "categorical",
                "top_counts": {str(k): int(v) for k, v in vc.items()},
            }
    return rep
