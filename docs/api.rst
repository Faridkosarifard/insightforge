HTTP API
========

The OpenAPI schema is served at ``/docs`` and ``/openapi.json`` when the API is
running. All routes below are relative to the API base URL (default
``http://localhost:8000``).

``GET /health``
    Liveness and database ping. Returns ``{"status": "ok", "database": "ok"}``
    or ``{"status": "degraded", "database": "error"}``.

``POST /datasets``
    Multipart upload. Query parameters: ``name`` (required), ``target``
    (optional column name). File field: ``file``, a ``.csv``. Returns the
    stored dataset (id, row/column counts, target).

``GET /datasets``
    List uploaded datasets, oldest first.

``GET /datasets/{dataset_id}``
    Fetch one dataset, or ``404`` if it does not exist.

``POST /eda/{dataset_id}``
    Build and persist an exploratory report for the dataset. The report is
    returned as ``report_json``.

``POST /train/{dataset_id}``
    Train a baseline model. The dataset must have been uploaded with a
    ``target``. Optional query parameter: ``n_estimators`` (10–1000, default
    200). Returns ``task_type``, ``estimator``, ``metrics``, and
    ``artifact_path``.

``POST /predict/{dataset_id}``
    Request body: a JSON array of row objects (feature names as keys). Scores
    the most recently trained model for that dataset. Returns
    ``{"predictions": [...]}``.

Errors use FastAPI's standard ``{"detail": "..."}`` body. Typical status
codes are ``400`` (validation) and ``404`` (unknown dataset or model).
