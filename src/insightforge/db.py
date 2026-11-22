"""SQLAlchemy engine, session factory, and schema bootstrap."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def make_engine(url: str) -> Engine:
    """Create an engine. SQLite gets ``check_same_thread=False`` for FastAPI."""
    kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


def import_models() -> None:
    """Import model modules so they register on ``Base.metadata``."""
    from insightforge.models import dataset as _dataset  # noqa: F401
    from insightforge.models import edareport as _edareport  # noqa: F401
    from insightforge.models import model as _model  # noqa: F401


def init_db(engine: Engine) -> None:
    """Create tables for all mapped models."""
    import_models()
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Commit on success, roll back on error. Useful for scripts and jobs."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def healthcheck(engine: Engine) -> bool:
    """Lightweight database ping for the ``/health`` endpoint."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
