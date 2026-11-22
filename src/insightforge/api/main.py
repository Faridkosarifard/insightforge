"""FastAPI application factory and HTTP routes."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from insightforge import __version__
from insightforge.core.logging import setup_logging
from insightforge.core.settings import Settings, get_settings
from insightforge.db import healthcheck, init_db, make_engine, make_session_factory
from insightforge.models.dataset import Dataset
from insightforge.models.edareport import EDAReport
from insightforge.models.model import Model
from insightforge.models.schemas import (
    DatasetOut,
    EDAReportOut,
    HealthOut,
    PredictOut,
    RootOut,
    TrainResultOut,
)
from insightforge.services import automl, eda, registry, storage
from insightforge.services.automl import TrainingError

logger = logging.getLogger(__name__)

MAX_PREDICT_ROWS = 10_000


def _to_python(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _clean_target(target: str | None) -> str | None:
    if target is None:
        return None
    stripped = target.strip()
    return stripped or None


def _read_csv_or_400(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    if frame.shape[1] == 0:
        raise HTTPException(status_code=400, detail="CSV file has no columns")
    return frame


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API. Pass ``settings`` from tests to use a temporary database."""
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    storage_dir = Path(settings.storage_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        storage_dir.mkdir(parents=True, exist_ok=True)
        init_db(engine)
        yield
        engine.dispose()

    application = FastAPI(
        title="InsightForge API",
        version=__version__,
        description=("Upload a CSV, run exploratory analysis, train a baseline model, and run predictions."),
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.engine = engine
    application.state.session_factory = session_factory
    application.state.storage_dir = storage_dir

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list or ["http://localhost:8501"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_db() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    @application.get("/", response_model=RootOut, include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": "InsightForge", "version": __version__, "docs": "/docs"}

    @application.get("/health", response_model=HealthOut, tags=["ops"])
    def health() -> dict[str, str]:
        try:
            healthcheck(engine)
        except Exception:
            logger.exception("Database health check failed")
            return {"status": "degraded", "database": "error"}
        return {"status": "ok", "database": "ok"}

    @application.post("/datasets", response_model=DatasetOut, tags=["datasets"])
    async def upload_dataset(
        name: str = Query(..., min_length=1, max_length=200),
        target: str | None = Query(default=None),
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
    ) -> Dataset:
        filename = file.filename or ""
        if not filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        raw = file.file
        if hasattr(raw, "seek") and hasattr(raw, "tell"):
            raw.seek(0, 2)
            size = raw.tell()
            raw.seek(0)
        else:
            size = 0
        limit = settings.max_upload_mb * 1024 * 1024
        if size > limit:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds the {settings.max_upload_mb} MB upload limit",
            )
        if size == 0:
            # Still try to save if tell() was unavailable; empty content is rejected after parse.
            pass

        path = storage.save_upload(file, storage_dir)
        try:
            if path.stat().st_size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            frame = _read_csv_or_400(path)
        except HTTPException:
            path.unlink(missing_ok=True)
            raise

        cleaned_target = _clean_target(target)
        if cleaned_target and cleaned_target not in frame.columns:
            path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Target column {cleaned_target!r} was not found in the CSV",
            )

        dataset = Dataset(
            name=name.strip(),
            filename=str(path),
            n_rows=len(frame),
            n_cols=int(frame.shape[1]),
            target=cleaned_target,
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        return dataset

    @application.get("/datasets", response_model=list[DatasetOut], tags=["datasets"])
    def list_datasets(db: Session = Depends(get_db)) -> list[Dataset]:
        return list(db.query(Dataset).order_by(Dataset.id.asc()).all())

    @application.get("/datasets/{dataset_id}", response_model=DatasetOut, tags=["datasets"])
    def get_dataset(dataset_id: int, db: Session = Depends(get_db)) -> Dataset:
        dataset = db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset

    def _load_dataset_frame(dataset: Dataset) -> pd.DataFrame:
        path = Path(dataset.filename)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Dataset file is missing on disk")
        return _read_csv_or_400(path)

    @application.post("/eda/{dataset_id}", response_model=EDAReportOut, tags=["eda"])
    def run_eda(dataset_id: int, db: Session = Depends(get_db)) -> EDAReport:
        dataset = db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        frame = _load_dataset_frame(dataset)
        report = eda.build_report(frame, target=dataset.target)
        row = EDAReport(dataset_id=dataset.id, report_json=report)
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    @application.post("/train/{dataset_id}", response_model=TrainResultOut, tags=["train"])
    def train_model(
        dataset_id: int,
        n_estimators: int = Query(default=200, ge=10, le=1000),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        dataset = db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if not dataset.target:
            raise HTTPException(status_code=400, detail="Target column was not set when the dataset was uploaded")
        frame = _load_dataset_frame(dataset)
        try:
            result = automl.train_baseline(frame, target=dataset.target, n_estimators=n_estimators)
        except TrainingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        artifact = registry.save_model(dataset.id, result["model"], storage_dir)
        model_row = Model(
            dataset_id=dataset.id,
            task_type=result["task_type"],
            estimator=result["estimator"],
            metrics_json=result["metrics"],
            artifact_path=str(artifact),
        )
        db.add(model_row)
        db.commit()
        db.refresh(model_row)
        return {
            "dataset_id": dataset.id,
            "task_type": model_row.task_type,
            "estimator": model_row.estimator,
            "metrics": model_row.metrics_json,
            "artifact_path": model_row.artifact_path,
        }

    @application.post("/predict/{dataset_id}", response_model=PredictOut, tags=["predict"])
    def predict(
        dataset_id: int,
        rows: list[dict[str, Any]],
        db: Session = Depends(get_db),
    ) -> dict[str, list[Any]]:
        if not rows:
            raise HTTPException(status_code=400, detail="Provide at least one row to predict")
        if len(rows) > MAX_PREDICT_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"At most {MAX_PREDICT_ROWS} rows can be scored in one request",
            )
        model_row = db.query(Model).filter_by(dataset_id=dataset_id).order_by(Model.id.desc()).first()
        if not model_row:
            raise HTTPException(status_code=404, detail="Trained model not found")
        try:
            fitted = registry.load_model(model_row.artifact_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Model artifact is missing on disk") from exc
        features = pd.DataFrame(rows)
        try:
            predictions = fitted.predict(features)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
        return {"predictions": [_to_python(value) for value in predictions.tolist()]}

    return application
