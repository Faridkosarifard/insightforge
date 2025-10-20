# Standard library
from typing import Optional

# Third-party (others)
import pandas as pd
from core.logging import setup_logging
from fastapi import Depends
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware

# SQLAlchemy
from sqlalchemy.orm import Session

# Insightforge project
from insightforge import models
from insightforge.api.services import automl
from insightforge.api.services import eda
from insightforge.api.services import registry
from insightforge.api.services import storage
from insightforge.models import schemas
from insightforge.utils.db_manager import SessionLocal
from insightforge.utils.db_manager import init_db

app = FastAPI(title="InsightForge API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    setup_logging()
    init_db()


@app.post("/datasets", response_model=schemas.DatasetOut)
async def upload_dataset(
    name: str,
    target: Optional[str] = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV supported for MVP")
    path = storage.save_upload(file)
    df = pd.read_csv(path)
    ds = models.Dataset(
        name=name, filename=str(path), n_rows=len(df), n_cols=df.shape[1], target=target
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return schemas.DatasetOut.model_validate(ds.__dict__)


@app.get("/datasets", response_model=list[schemas.DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    ds = db.query(models.Dataset).all()
    return [schemas.DatasetOut.model_validate(d.__dict__) for d in ds]


@app.post("/eda/{dataset_id}", response_model=schemas.EDAReportOut)
def run_eda(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.get(models.Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    df = pd.read_csv(ds.filename)
    rep = eda.build_report(df, target=ds.target)
    row = models.EDAReport(dataset_id=ds.id, report_json=rep)
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "dataset_id": ds.id, "report_json": rep}


@app.post("/train/{dataset_id}", response_model=schemas.TrainResultOut)
def train_model(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.get(models.Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    if not ds.target:
        raise HTTPException(400, "Target column not set on dataset upload")
    df = pd.read_csv(ds.filename)
    result = automl.train_baseline(df, target=ds.target)
    artifact = registry.save_model(dataset_id=ds.id, model=result["model"])
    m = models.Model(
        dataset_id=ds.id,
        task_type=result["task_type"],
        estimator=result["estimator"],
        metrics_json=result["metrics"],
        artifact_path=artifact,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return {
        "dataset_id": ds.id,
        "task_type": m.task_type,
        "estimator": m.estimator,
        "metrics": m.metrics_json,
        "artifact_path": m.artifact_path,
    }


@app.post("/predict/{dataset_id}", response_model=schemas.PredictOut)
def predict(dataset_id: int, rows: list[dict], db: Session = Depends(get_db)):
    m = (
        db.query(models.Model)
        .filter_by(dataset_id=dataset_id)
        .order_by(models.Model.id.desc())
        .first()
    )
    if not m:
        raise HTTPException(404, "Trained model not found")
    model = registry.load_model(m.artifact_path)
    X = pd.DataFrame(rows)
    preds = model.predict(X).tolist()
    return {"predictions": preds}
