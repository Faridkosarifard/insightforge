"""Runtime configuration loaded from environment variables or a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Environment variables (see ``.env.example``):

    * ``DATABASE_URL`` or ``DB_URL``
    * ``STORAGE_DIR``
    * ``API_BASE``
    * ``LOG_LEVEL``
    * ``CORS_ORIGINS``
    * ``MAX_UPLOAD_MB``
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = Field(
        default="sqlite:///./insight.db",
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL"),
    )
    storage_dir: Path = Field(
        default=Path("storage"),
        validation_alias=AliasChoices("STORAGE_DIR"),
    )
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        validation_alias=AliasChoices("CORS_ORIGINS"),
    )
    api_base: str = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("API_BASE"),
    )
    max_upload_mb: int = Field(default=25, ge=1, validation_alias=AliasChoices("MAX_UPLOAD_MB"))

    @field_validator("storage_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        return Path(value)

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object, loaded once."""
    return Settings()


def clear_settings_cache() -> None:
    """Drop the cached settings (used in tests)."""
    get_settings.cache_clear()
