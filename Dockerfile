FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 \
    STORAGE_DIR=/app/storage \
    DATABASE_URL=sqlite:////app/insight.db

RUN mkdir -p /app/storage

EXPOSE 8000 8501

CMD ["uvicorn", "insightforge.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
