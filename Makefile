
.PHONY: 
format: _isort_apply _black_apply

fix: format

lint: _isort_check _black_check flake8

check: lint
	@echo ">>> Import check"
	@$(RUN) python -c "import importlib; import pkgutil; print('OK: python import smoke test')"

flake8:
	flake8 .

_isort_apply:
	isort .

_black_apply:
	black .

_isort_check:
	isort --check-only .

_black_check:
	black --check .
