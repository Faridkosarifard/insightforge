Contributing to InsightForge
============================

Thanks for considering a contribution.

Getting started
---------------

.. code-block:: bash

   git clone https://github.com/Faridkosarifard/insightforge
   cd insightforge
   make install      # creates .venv and installs the package with dev dependencies
   make check        # ruff, mypy --strict, and the test suite with coverage

Requirements: Python 3.11 or newer. Docker is only needed for ``make up``.

Workflow
--------

1. Open an issue first for anything larger than a small fix, so the approach can be agreed on.
2. Create a branch from ``main``.
3. Add or update tests for the behaviour you change.
4. Run ``make check`` and make sure it passes.
5. Update ``CHANGELOG.rst`` under *Unreleased*.
6. Open a pull request that explains the change.

Style
-----

* ``ruff format`` and ``ruff check`` (configuration in ``pyproject.toml``).
* ``mypy --strict`` for ``src/``.
* Commit messages: imperative mood, a short summary line, and a body that explains the reason for the change.

License
-------

By contributing you agree that your contributions are licensed under the Apache License 2.0.
