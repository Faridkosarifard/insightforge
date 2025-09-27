# Third-party (others)
from pydantic import BaseModel
from pydantic import Field


class DatasetOut(BaseModel):
    id: int
    name: str
    filename: str
    n_rows: int
    n_cols: int
    target: str | None


class EDAReportOut(BaseModel):
    id: int
    dataset_id: int
    report: dict = Field(..., alias="report_json")


class TrainResultOut(BaseModel):
    dataset_id: int
    task_type: str
    estimator: str
    metrics: dict
    artifact_path: str


class PredictOut(BaseModel):
    predictions: list
