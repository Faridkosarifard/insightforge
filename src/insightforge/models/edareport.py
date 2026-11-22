"""Stored exploratory analysis report for a dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from insightforge.db import Base
from insightforge.models.dataset import utcnow_iso

if TYPE_CHECKING:
    from insightforge.models.dataset import Dataset


class EDAReport(Base):
    __tablename__ = "eda_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow_iso)

    dataset: Mapped[Dataset] = relationship(back_populates="eda_reports")
