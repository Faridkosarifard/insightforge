"""Dataset registry row: an uploaded CSV and optional target column."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from insightforge.db import Base

if TYPE_CHECKING:
    from insightforge.models.edareport import EDAReport
    from insightforge.models.model import Model


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    filename: Mapped[str] = mapped_column(Text)
    n_rows: Mapped[int] = mapped_column(Integer)
    n_cols: Mapped[int] = mapped_column(Integer)
    target: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow_iso)

    eda_reports: Mapped[list[EDAReport]] = relationship(back_populates="dataset")
    models: Mapped[list[Model]] = relationship(back_populates="dataset")
