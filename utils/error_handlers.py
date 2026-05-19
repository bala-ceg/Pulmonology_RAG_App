"""
Centralized error handling utilities.

Provides:
- get_logger(name)  — module-level logger factory
- handle_route_errors — decorator that catches exceptions in Flask route
  handlers, logs them with traceback, and returns a consistent JSON error
  response so each route doesn't need its own try/except boilerplate.
"""

from __future__ import annotations

import logging
import traceback
from functools import wraps
from typing import Any, Callable

from flask import jsonify


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger.

    Relies on the root logger (configured by setup_logging()) for all
    output — no per-logger handlers are added here.  This ensures every
    log record reaches both the terminal (human-readable) and the ECS
    JSON file handler (Kibana) without duplication.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # propagate=True (default) so records reach root's handlers
    return logger


_default_logger = get_logger("app")


def handle_route_errors(func: Callable) -> Callable:
    """
    Decorator for Flask route handlers.

    Wraps the handler in a try/except block. On success the handler's
    return value is passed through unchanged.  On any unhandled exception
    the decorator:
      1. Logs the full traceback at ERROR level.
      2. Returns a JSON response ``{"error": "<message>", "status": 500}``
         with HTTP status code 500.

    Usage::

        @bp.route("/some-endpoint", methods=["POST"])
        @handle_route_errors
        def some_endpoint():
            ...
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            _default_logger.error(
                "Unhandled error in %s: %s\n%s",
                func.__qualname__,
                exc,
                traceback.format_exc(),
            )
            return jsonify({"error": str(exc), "status": 500}), 500

    return wrapper
