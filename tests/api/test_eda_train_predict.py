"""EDA, training, and prediction endpoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient


def upload_iris(client: TestClient, iris_bytes: bytes, target: str | None = "species") -> int:
    params: dict[str, str] = {"name": "iris"}
    if target:
        params["target"] = target
    response = client.post(
        "/datasets",
        params=params,
        files={"file": ("iris.csv", iris_bytes, "text/csv")},
    )
    assert response.status_code == 200, response.text
    return int(response.json()["id"])


def upload_frame(client: TestClient, frame: pd.DataFrame, name: str, target: str) -> int:
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/datasets",
        params={"name": name, "target": target},
        files={"file": (f"{name}.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200, response.text
    return int(response.json()["id"])


def test_eda_and_missing_dataset(client: TestClient, iris_bytes: bytes) -> None:
    assert client.post("/eda/99").status_code == 404
    dataset_id = upload_iris(client, iris_bytes)
    response = client.post(f"/eda/{dataset_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dataset_id"] == dataset_id
    report = body["report_json"]
    assert report["n_rows"] == 150
    assert report["target"]["type"] == "categorical"
    assert "setosa" in report["target"]["top_counts"]


def test_train_requires_target(client: TestClient, iris_bytes: bytes) -> None:
    dataset_id = upload_iris(client, iris_bytes, target=None)
    response = client.post(f"/train/{dataset_id}")
    assert response.status_code == 400
    assert "Target column" in response.json()["detail"]


def test_train_and_predict_iris(client: TestClient, iris_bytes: bytes) -> None:
    dataset_id = upload_iris(client, iris_bytes)
    missing = client.post("/train/99")
    assert missing.status_code == 404

    trained = client.post(f"/train/{dataset_id}", params={"n_estimators": 20})
    assert trained.status_code == 200, trained.text
    body = trained.json()
    assert body["task_type"] == "classification"
    assert body["estimator"] == "RandomForestClassifier"
    assert "accuracy" in body["metrics"]
    assert body["artifact_path"]
    assert Path(body["artifact_path"]).is_file()

    rows = [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
    ]
    predicted = client.post(f"/predict/{dataset_id}", json=rows)
    assert predicted.status_code == 200, predicted.text
    preds = predicted.json()["predictions"]
    assert len(preds) == 1
    assert preds[0] in {"setosa", "versicolor", "virginica"}


def test_predict_errors(client: TestClient, iris_bytes: bytes) -> None:
    dataset_id = upload_iris(client, iris_bytes)
    assert client.post(f"/predict/{dataset_id}", json=[]).status_code == 400
    assert client.post(f"/predict/{dataset_id}", json=[{"x": 1}]).status_code == 404

    client.post(f"/train/{dataset_id}", params={"n_estimators": 20})
    bad = client.post(f"/predict/{dataset_id}", json=[{"not": "features"}])
    assert bad.status_code == 400
    assert "Prediction failed" in bad.json()["detail"]


def test_predict_rejects_too_many_rows(client: TestClient, iris_bytes: bytes) -> None:
    dataset_id = upload_iris(client, iris_bytes)
    client.post(f"/train/{dataset_id}", params={"n_estimators": 20})
    rows = [{"sepal_length": 5.0, "sepal_width": 3.0, "petal_length": 1.0, "petal_width": 0.2}] * 10_001
    response = client.post(f"/predict/{dataset_id}", json=rows)
    assert response.status_code == 400
    assert "10" in response.json()["detail"]


def test_train_and_predict_regression(client: TestClient, regression_frame: pd.DataFrame) -> None:
    dataset_id = upload_frame(client, regression_frame, "synth", "y")
    trained = client.post(f"/train/{dataset_id}", params={"n_estimators": 20})
    assert trained.status_code == 200, trained.text
    assert trained.json()["task_type"] == "regression"
    rows = [{"x": 10, "group": "a"}]
    predicted = client.post(f"/predict/{dataset_id}", json=rows)
    assert predicted.status_code == 200
    assert isinstance(predicted.json()["predictions"][0], float)


def test_eda_missing_file_returns_404(client: TestClient, iris_bytes: bytes, app) -> None:
    dataset_id = upload_iris(client, iris_bytes)
    # Remove the stored file to simulate a missing artifact.
    from sqlalchemy.orm import Session

    from insightforge.models import Dataset

    factory = app.state.session_factory
    session: Session = factory()
    try:
        row = session.get(Dataset, dataset_id)
        assert row is not None
        Path(row.filename).unlink()
    finally:
        session.close()
    response = client.post(f"/eda/{dataset_id}")
    assert response.status_code == 404
    assert "missing" in response.json()["detail"].lower()


def test_train_invalid_dataset_file(client: TestClient, tmp_path: Path, app, iris_bytes: bytes) -> None:
    # Not used; kept as a hook if we add more file-corruption cases.
    dataset_id = upload_iris(client, iris_bytes)
    from sqlalchemy.orm import Session

    from insightforge.models import Dataset

    session: Session = app.state.session_factory()
    try:
        row = session.get(Dataset, dataset_id)
        assert row is not None
        Path(row.filename).write_text("", encoding="utf-8")
    finally:
        session.close()
    # empty csv with no columns/content
    response = client.post(f"/train/{dataset_id}", params={"n_estimators": 20})
    assert response.status_code == 400


def test_train_rejects_too_few_rows(client: TestClient) -> None:
    frame = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "a"]})
    dataset_id = upload_frame(client, frame, "tiny", "y")
    response = client.post(f"/train/{dataset_id}", params={"n_estimators": 20})
    assert response.status_code == 400
    assert "rows" in response.json()["detail"].lower()


def test_predict_missing_artifact_returns_404(client: TestClient, iris_bytes: bytes, app) -> None:
    dataset_id = upload_iris(client, iris_bytes)
    trained = client.post(f"/train/{dataset_id}", params={"n_estimators": 20})
    assert trained.status_code == 200
    Path(trained.json()["artifact_path"]).unlink()
    response = client.post(
        f"/predict/{dataset_id}",
        json=[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}],
    )
    assert response.status_code == 404
    assert "artifact" in response.json()["detail"].lower()
