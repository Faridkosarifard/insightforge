from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

from core.settings import get_settings


class Base(DeclarativeBase):
    """Global declarative base for all ORM models."""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
    pass


_settings = get_settings()
engine = create_engine(
    _settings.DB_URL,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = scoped_session(
    sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
)


def init_db() -> None:
    """Create tables for all mapped models."""
    from src.api import models as _  # noqa: F401

    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope():
    """
    Context manager for scripts/CLI:
        with session_scope() as s:
            s.add(obj); ...
    Commits on success, rollbacks on error.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator:
    """
    FastAPI dependency:
        def endpoint(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def healthcheck() -> bool:
    """Lightweight DB ping for /health endpoints."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
