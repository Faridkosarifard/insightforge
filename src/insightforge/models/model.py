# Standard library
from datetime import datetime

# SQLAlchemy
from sqlalchemy import JSON
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

# Insightforge project
from insightforge.models import Dataset
from insightforge.utils.db_manager import Base


class Model(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    task_type: Mapped[str] = mapped_column(String(40))
    estimator: Mapped[str] = mapped_column(String(80))
    metrics_json: Mapped[dict] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(
        String(32), default=lambda: datetime.utcnow().isoformat()
    )
    dataset: Mapped[Dataset] = relationship(back_populates="models")
