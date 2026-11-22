"""Trained baseline model metadata and artifact path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from insightforge.db import Base
from insightforge.models.dataset import utcnow_iso

if TYPE_CHECKING:
    from insightforge.models.dataset import Dataset


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    task_type: Mapped[str] = mapped_column(String(40))
    estimator: Mapped[str] = mapped_column(String(80))
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(40), default=utcnow_iso)

    dataset: Mapped[Dataset] = relationship(back_populates="models")
