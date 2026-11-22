"""Logging bootstrap replaces existing root handlers."""

from __future__ import annotations

import logging

from insightforge.core.logging import setup_logging


def test_setup_logging_is_idempotent() -> None:
    setup_logging("INFO")
    setup_logging("DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    setup_logging("WARNING")
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
