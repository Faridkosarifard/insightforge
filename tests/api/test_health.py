"""Root and health endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from insightforge import __version__


def test_root_and_health(client: TestClient) -> None:
    root = client.get("/")
    assert root.status_code == 200
    body = root.json()
    assert body["name"] == "InsightForge"
    assert body["version"] == __version__
    assert body["docs"] == "/docs"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "database": "ok"}


def test_health_degraded_when_database_fails(app, client: TestClient, monkeypatch) -> None:
    from insightforge.api import main as main_mod

    def boom(_engine) -> bool:
        raise RuntimeError("db down")

    monkeypatch.setattr(main_mod, "healthcheck", boom)
    # The health function in the already-built app closed over the original healthcheck
    # imported at decoration time... actually it looks up healthcheck from the module
    # at call time because it's referenced as healthcheck(engine) inside the nested
    # function, and healthcheck is a global of main. The nested function uses the
    # name healthcheck from the enclosing module scope. monkeypatch.setattr on
    # main.healthcheck should work for the next call through the same app.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["database"] == "error"
