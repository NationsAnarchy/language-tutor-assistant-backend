"""Structured, request-correlated logging for the backend application."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_CONFIGURED = False


def current_request_id() -> str | None:
    """Return the request ID for the current async task, if one exists."""
    return _request_id.get()


def log_level() -> int:
    """Resolve ``LOG_LEVEL`` safely, falling back to INFO for invalid values."""
    configured = os.getenv("LOG_LEVEL", "INFO").upper()
    return logging._nameToLevel.get(configured, logging.INFO)


class RequestContextFilter(logging.Filter):
    """Attach the request-scoped context variable to records handled on stdout."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = current_request_id()
        if request_id and not getattr(record, "request_id", None):
            record.request_id = request_id  # type: ignore[attr-defined]
        return True

class StructuredFormatter(logging.Formatter):
    """Emit one JSON object per stdout log line."""

    RESERVED = frozenset(logging.LogRecord.__dict__.keys())

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "lineno": record.lineno,
            "message": record.getMessage(),
            "service": os.getenv("SERVICE_NAME", "language-tutor-assistant-backend"),
            "environment": os.getenv("APP_ENV", "development"),
        }
        service_version = os.getenv("SERVICE_VERSION")
        if service_version:
            payload["service_version"] = service_version

        # Pull request_id from the record if present
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id

        # Merge any extra fields the caller passed via `extra=`
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and key != "request_id":
                payload[key] = value

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: int | None = None) -> None:
    """Configure idempotent JSON stdout logging using ``LOG_LEVEL`` by default."""
    global _CONFIGURED
    resolved_level = log_level() if level is None else level
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(resolved_level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(StructuredFormatter())
    root.setLevel(resolved_level)
    # Remove any existing handlers to avoid duplicate output
    root.handlers.clear()
    root.addHandler(handler)

    # Request completions replace noisy unstructured Uvicorn access logs.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named standard-library logger."""
    return logging.getLogger(name)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Set a task-local request ID and expose it in every HTTP response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        token = _request_id.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            _request_id.reset(token)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log one safe, structured completion event for every HTTP request."""

    def __init__(self, app: Any, *, logger: logging.Logger | None = None) -> None:
        super().__init__(app)
        self.logger = logger or get_logger("app.http")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            self.logger.exception(
                "HTTP request failed before a response was created",
                extra={
                    "event": "http_request_completed",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                },
            )
            raise

        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        log_method = self.logger.error if response.status_code >= 500 else (
            self.logger.warning if response.status_code >= 400 else self.logger.info
        )
        log_method(
            "HTTP request completed",
            extra={
                "event": "http_request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response