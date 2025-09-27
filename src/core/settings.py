from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    DB_URL: str = "sqlite:///./insight.db"
    STORAGE_DIR: str = "storage"

    class Config:
        extra = "ignore"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
