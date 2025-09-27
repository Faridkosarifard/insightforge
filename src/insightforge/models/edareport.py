# Standard library
from datetime import datetime

# SQLAlchemy
from sqlalchemy import JSON
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

# Insightforge project
from insightforge.models import Dataset
from insightforge.utils.db_manager import Base


class EDAReport(Base):
    __tablename__ = "eda_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    report_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(
        String(32), default=lambda: datetime.utcnow().isoformat()
    )
    dataset: Mapped[Dataset] = relationship(back_populates="eda_reports")
