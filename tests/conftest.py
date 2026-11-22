"""Shared fixtures for unit and API tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from insightforge.api.main import create_app
from insightforge.core.settings import Settings

IRIS_PATH = Path(__file__).resolve().parents[1] / "data" / "samples" / "iris.csv"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        storage_dir=tmp_path / "storage",
        cors_origins="http://localhost:8501",
        log_level="WARNING",
        max_upload_mb=2,
        api_base="http://testserver",
    )


@pytest.fixture()
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture()
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def iris_bytes() -> bytes:
    return IRIS_PATH.read_bytes()


@pytest.fixture()
def iris_frame() -> pd.DataFrame:
    return pd.read_csv(IRIS_PATH)


@pytest.fixture()
def regression_frame() -> pd.DataFrame:
    rng = pd.Series(range(40), name="x")
    frame = pd.DataFrame(
        {
            "x": rng,
            "group": ["a" if i % 2 == 0 else "b" for i in rng],
            "y": [2.5 * i + 0.1 * (i % 5) for i in rng],
        }
    )
    return frame
