# Standard library
from datetime import datetime

# SQLAlchemy
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

# Insightforge project
from insightforge.models import EDAReport
from insightforge.models import Model
from insightforge.utils.db_manager import Base


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    filename: Mapped[str] = mapped_column(Text)
    n_rows: Mapped[int] = mapped_column(Integer)
    n_cols: Mapped[int] = mapped_column(Integer)
    target: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[str] = mapped_column(
        String(32), default=lambda: datetime.utcnow().isoformat()
    )

    eda_reports: Mapped[list["EDAReport"]] = relationship(back_populates="dataset")
    models: Mapped[list["Model"]] = relationship(back_populates="dataset")
