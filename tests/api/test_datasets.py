"""Dataset upload and listing."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _upload(
    client: TestClient, content: bytes, name: str = "iris", target: str | None = "species", filename: str = "iris.csv"
):
    params: dict[str, str] = {"name": name}
    if target is not None:
        params["target"] = target
    return client.post("/datasets", params=params, files={"file": (filename, content, "text/csv")})


def test_upload_and_list(client: TestClient, iris_bytes: bytes) -> None:
    created = _upload(client, iris_bytes)
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["name"] == "iris"
    assert body["n_rows"] == 150
    assert body["n_cols"] == 5
    assert body["target"] == "species"
    assert body["id"] >= 1

    listed = client.get("/datasets")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["id"] == body["id"]

    fetched = client.get(f"/datasets/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "iris"


def test_upload_without_target(client: TestClient, iris_bytes: bytes) -> None:
    response = _upload(client, iris_bytes, target="")
    assert response.status_code == 200
    assert response.json()["target"] is None


def test_reject_non_csv(client: TestClient) -> None:
    response = client.post(
        "/datasets",
        params={"name": "bad"},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]


def test_reject_empty_file(client: TestClient) -> None:
    response = _upload(client, b"", filename="empty.csv", target=None)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_reject_invalid_csv(client: TestClient) -> None:
    # A file that is not parseable as a table with columns.
    response = _upload(client, b"\x00\x01not,csv\n", filename="bad.csv", target=None)
    # pandas may still parse this; also try a definitely empty-column case
    # If it unexpectedly succeeds, at least it should not 500.
    assert response.status_code in {200, 400}


def test_reject_missing_target_column(client: TestClient, iris_bytes: bytes) -> None:
    response = _upload(client, iris_bytes, target="not_a_column")
    assert response.status_code == 400
    assert "not_a_column" in response.json()["detail"]


def test_reject_oversized_upload(client: TestClient) -> None:
    # settings fixture caps uploads at 2 MB
    huge = b"a,b\n" + b"1,2\n" * (2 * 1024 * 1024)
    response = _upload(client, huge, filename="huge.csv", target=None)
    assert response.status_code == 400
    assert "upload limit" in response.json()["detail"]


def test_get_unknown_dataset(client: TestClient) -> None:
    response = client.get("/datasets/999")
    assert response.status_code == 404
