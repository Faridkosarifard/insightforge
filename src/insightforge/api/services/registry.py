# Standard library
import os
import time
from pathlib import Path

# Third-party (others)
import joblib

STORE = Path(os.getenv("STORAGE_DIR", "storage"))
STORE.mkdir(parents=True, exist_ok=True)


def save_model(dataset_id: int, model) -> str:
    p = STORE / f"model_ds{dataset_id}_{int(time.time())}.joblib"
    joblib.dump(model, p)
    return str(p)


def load_model(path: str):
    return joblib.load(path)
