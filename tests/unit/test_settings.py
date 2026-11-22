"""Settings load from kwargs, aliases, and the environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from insightforge.core.settings import Settings, clear_settings_cache, get_settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("sqlite:")
    assert settings.storage_dir == Path("storage")
    assert "http://localhost:8501" in settings.cors_origin_list
    assert settings.max_upload_mb == 25


def test_database_url_alias(tmp_path: Path) -> None:
    settings = Settings(
        DB_URL="sqlite:////tmp/custom.db",
        STORAGE_DIR=str(tmp_path),
        _env_file=None,
    )
    assert settings.database_url == "sqlite:////tmp/custom.db"
    assert settings.storage_dir == tmp_path


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://insight:secret@db:5432/insightdb")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8501, http://example.test")
    settings = Settings(_env_file=None)
    assert settings.database_url.endswith("/insightdb")
    assert settings.cors_origin_list == ["http://localhost:8501", "http://example.test"]


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_URL", raising=False)
    clear_settings_cache()
    first = get_settings()
    second = get_settings()
    assert first is second
    clear_settings_cache()
    third = get_settings()
    assert third is not first
    clear_settings_cache()
