"""Pydantic response and request models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    filename: str
    n_rows: int
    n_cols: int
    target: str | None
    created_at: str | None = None


class EDAReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    report_json: dict[str, Any]


class TrainResultOut(BaseModel):
    dataset_id: int
    task_type: str
    estimator: str
    metrics: dict[str, Any]
    artifact_path: str


class PredictOut(BaseModel):
    predictions: list[Any]


class HealthOut(BaseModel):
    status: str
    database: str


class RootOut(BaseModel):
    name: str
    version: str
    docs: str = Field(description="OpenAPI documentation path")
