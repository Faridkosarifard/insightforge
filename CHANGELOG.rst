Changelog
=========

All notable changes to this project are documented in this file.

Unreleased
----------

* Configuration is loaded from environment variables (see ``.env.example``).
  ``DATABASE_URL`` and the older alias ``DB_URL`` are both accepted.
* Application factory, ``/health`` endpoint, ``GET /datasets/{id}``, and stricter
  CSV validation (file type, empty files, missing target column, upload size).
* Unit and API tests, GitHub Actions CI (Python 3.11–3.13), Dockerfile, and
  working Compose entrypoints.
* Apache License 2.0, contributor notes, and a complete project README.

0.1.0 (2025-10-20)
------------------

* Initial application: CSV upload, exploratory analysis, a Random Forest
  baseline, predictions, and a Streamlit dashboard.
