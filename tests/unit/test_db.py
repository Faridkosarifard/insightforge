"""Engine helpers, schema bootstrap, and session_scope behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from insightforge.db import (
    healthcheck,
    init_db,
    make_engine,
    make_session_factory,
    session_scope,
)
from insightforge.models import Dataset


def test_healthcheck_and_init(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    init_db(engine)
    assert healthcheck(engine) is True
    engine.dispose()


def test_session_scope_commits(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    init_db(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        session.add(Dataset(name="demo", filename="a.csv", n_rows=1, n_cols=1, target=None))
    with factory() as session:
        assert session.query(Dataset).count() == 1
    engine.dispose()


def test_session_scope_rolls_back(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    init_db(engine)
    factory = make_session_factory(engine)
    with pytest.raises(RuntimeError, match="boom"), session_scope(factory) as session:
        session.add(Dataset(name="demo", filename="a.csv", n_rows=1, n_cols=1, target=None))
        raise RuntimeError("boom")
    with factory() as session:
        assert session.query(Dataset).count() == 0
    # healthcheck still works after a rollback
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    engine.dispose()
