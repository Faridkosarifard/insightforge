# Standard library
import os
import shutil
import uuid
from pathlib import Path

BASE = Path(os.getenv("STORAGE_DIR", "storage"))
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)


def save_upload(file) -> Path:
    suffix = Path(file.filename).suffix
    path = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return path
