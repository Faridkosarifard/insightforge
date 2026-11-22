PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
COMPOSE := docker compose

.PHONY: help install lint format typecheck test cov check up down run-api run-dashboard clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## Create a virtualenv and install the package with dev dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint:  ## Run ruff lint and format checks
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:  ## Auto-format and fix lint issues
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

typecheck:  ## Run mypy in strict mode
	$(BIN)/mypy

test:  ## Run the test suite
	$(BIN)/pytest

cov:  ## Run tests with coverage
	$(BIN)/pytest --cov --cov-report=term-missing --cov-report=xml

check: lint typecheck cov  ## Everything CI runs

run-api:  ## Start the API locally (SQLite by default)
	$(BIN)/uvicorn insightforge.api.main:create_app --factory --reload --host 127.0.0.1 --port 8000

run-dashboard:  ## Start the Streamlit dashboard (expects the API on port 8000)
	$(BIN)/streamlit run src/insightforge/dashboard/app.py --server.port 8501 --server.address 127.0.0.1

up:  ## Build and start the API, dashboard, and Postgres with Docker Compose
	$(COMPOSE) up --build

down:  ## Stop Compose services and remove volumes
	$(COMPOSE) down -v

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build
