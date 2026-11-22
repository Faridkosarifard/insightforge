"""Save uploaded CSV files under the configured storage directory."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import BinaryIO, Protocol


class UploadedFile(Protocol):
    """Minimal file-like upload (compatible with FastAPI ``UploadFile``)."""

    filename: str | None
    file: BinaryIO


def save_upload(file: UploadedFile, storage_dir: Path) -> Path:
    """Persist an uploaded file under ``storage_dir/uploads`` and return its path."""
    uploads = storage_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    original = Path(file.filename or "upload.csv").name
    suffix = Path(original).suffix.lower() or ".csv"
    path = uploads / f"{uuid.uuid4().hex}{suffix}"
    src = file.file
    if hasattr(src, "seek"):
        src.seek(0)
    with path.open("wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return path
