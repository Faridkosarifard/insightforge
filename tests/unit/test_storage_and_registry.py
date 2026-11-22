"""Upload storage and model artifact registry."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

from sklearn.dummy import DummyClassifier

from insightforge.services import registry, storage


def test_save_upload_writes_unique_csv(tmp_path: Path) -> None:
    payload = b"a,b\n1,2\n"
    uploaded = SimpleNamespace(filename="nested/data.CSV", file=io.BytesIO(payload))
    path = storage.save_upload(uploaded, tmp_path)
    assert path.parent == tmp_path / "uploads"
    assert path.suffix == ".csv"
    assert path.read_bytes() == payload
    again = storage.save_upload(
        SimpleNamespace(filename="data.CSV", file=io.BytesIO(payload)),
        tmp_path,
    )
    assert again != path


def test_save_upload_default_name(tmp_path: Path) -> None:
    uploaded = SimpleNamespace(filename=None, file=io.BytesIO(b"x,y\n0,1\n"))
    path = storage.save_upload(uploaded, tmp_path)
    assert path.suffix == ".csv"
    assert path.is_file()


def test_registry_roundtrip(tmp_path: Path) -> None:
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0], [1]], [0, 1])
    path = registry.save_model(dataset_id=7, model=model, storage_dir=tmp_path)
    assert path.parent == tmp_path
    assert "model_ds7_" in path.name
    loaded = registry.load_model(path)
    assert loaded.predict([[0]])[0] in (0, 1)
