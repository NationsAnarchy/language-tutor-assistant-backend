"""
Structured logging configuration for the Language Tutor Agent backend.

Uses stdlib `logging` with a JSON-ish formatter so logs are parseable by
log aggregators (CloudWatch, Datadog, Loki, etc.) without adding a new
dependency. Each request gets a unique request ID via RequestIdMiddleware.

Usage:
    from app.logging_config import get_logger, configure_logging, RequestIdMiddleware

    configure_logging()  # call once at startup
    logger = get_logger(__name__)
    logger.info("something happened", extra={"request_id": "abc-123"})
"""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class StructuredFormatter(logging.Formatter):
    """
    Emit one JSON object per log line.

    Fields: timestamp, level, logger, message, module, lineno, request_id (if set).
    Extra kwargs passed via `extra=` are merged into the top-level object.
    """

    RESERVED = frozenset(logging.LogRecord.__dict__.keys())

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "lineno": record.lineno,
            "message": record.getMessage(),
        }

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with the structured formatter.

    Idempotent — safe to call multiple times.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any existing handlers to avoid duplicate output
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Call `configure_logging()` once at startup."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------

class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Generate a unique request ID for each incoming request, store it on
    `request.state.request_id`, and return it in the `X-Request-ID` response
    header. Also injects it into the log context for the duration of the request.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        # Inject request_id into the root logger's context for this request.
        # We use a LoggerAdapter-style approach: add a filter that attaches
        # request_id to every LogRecord emitted during this request.
        request_id_filter = _RequestIdFilter(request_id)
        root = logging.getLogger()
        root.addFilter(request_id_filter)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            root.removeFilter(request_id_filter)


class _RequestIdFilter(logging.Filter):
    """Attach request_id to every log record while active."""

    def __init__(self, request_id: str):
        super().__init__()
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self.request_id  # type: ignore[attr-defined]
        return True