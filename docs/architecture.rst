Architecture
============

InsightForge is a small three-part application:

* A **FastAPI** service that stores dataset metadata, runs exploratory analysis,
  trains a baseline model, and scores new rows.
* A **Streamlit** dashboard that calls that API. It has no persistence of its
  own.
* A **database** (SQLite by default, PostgreSQL in Docker Compose) that holds
  dataset records, EDA reports, and trained-model metadata. Uploaded CSVs and
  joblib artifacts live on the local filesystem under ``STORAGE_DIR``.

Request flow
------------

1. ``POST /datasets`` stores the CSV on disk and inserts a ``datasets`` row.
2. ``POST /eda/{id}`` reads the file, builds a JSON report (column types,
   missingness, descriptives, correlations, target summary), and stores it.
3. ``POST /train/{id}`` loads the file, infers classification vs. regression
   from the target, fits a scikit-learn pipeline (imputation, scaling or
   one-hot encoding, Random Forest), writes a joblib artifact, and stores
   metrics.
4. ``POST /predict/{id}`` loads the latest artifact for that dataset and
   scores the JSON rows in the request body.

Task inference
--------------

A numeric target with more than ten distinct values is treated as regression.
Every other target is classification. This is a heuristic, not a contract:
set the target carefully and inspect the returned ``task_type``.

Layout
------

::

    src/insightforge/
        api/main.py          Application factory and HTTP routes
        core/settings.py     Environment-backed configuration
        db.py                Engine, sessions, and schema bootstrap
        models/              SQLAlchemy models and Pydantic schemas
        services/            EDA, AutoML, storage, and the model registry
        dashboard/app.py     Streamlit UI
