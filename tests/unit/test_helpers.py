"""Small helpers that are awkward to reach only through the HTTP API."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from insightforge.api.main import _read_csv_or_400, _to_python
from insightforge.db import make_engine
from insightforge.services import storage
from insightforge.services.automl import _maybe_add_roc_auc


def test_to_python_numpy_and_plain() -> None:
    assert _to_python(3) == 3
    assert _to_python("setosa") == "setosa"
    assert _to_python(np.int64(5)) == 5
    assert isinstance(_to_python(np.float64(1.25)), float)


def test_read_csv_no_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("ignored", encoding="utf-8")

    def fake_read(_path: Path) -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr("insightforge.api.main.pd.read_csv", fake_read)
    with pytest.raises(HTTPException) as exc:
        _read_csv_or_400(path)
    assert exc.value.status_code == 400
    assert "no columns" in exc.value.detail


def test_make_engine_non_sqlite() -> None:
    engine = make_engine("postgresql+psycopg2://insight:insight@localhost:5432/insightdb")
    assert engine.dialect.name == "postgresql"
    engine.dispose()


class _NoProba:
    pass


class _BoomProba:
    def predict_proba(self, _x: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("cannot score")


class _OddShape:
    def predict_proba(self, _x: pd.DataFrame) -> np.ndarray:
        return np.array([0.1, 0.9])  # 1-d, neither binary nor ovr


def test_roc_auc_edge_cases() -> None:
    empty: dict[str, float] = {}
    _maybe_add_roc_auc(_NoProba(), pd.DataFrame({"x": [1]}), pd.Series([0]), empty)  # type: ignore[arg-type]
    assert empty == {}

    _maybe_add_roc_auc(_BoomProba(), pd.DataFrame({"x": [1]}), pd.Series([0]), empty)  # type: ignore[arg-type]
    assert empty == {}

    _maybe_add_roc_auc(_OddShape(), pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1]), empty)  # type: ignore[arg-type]
    assert empty == {}


def test_save_upload_without_seek(tmp_path: Path) -> None:
    class _NoSeek:
        def read(self, _size: int) -> bytes:
            if getattr(self, "_done", False):
                return b""
            self._done = True
            return b"a,b\n1,2\n"

    uploaded = SimpleNamespace(filename="plain.csv", file=_NoSeek())
    path = storage.save_upload(uploaded, tmp_path)
    assert path.read_bytes() == b"a,b\n1,2\n"
