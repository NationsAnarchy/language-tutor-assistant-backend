"""
Typed exception hierarchy for the Language Tutor Agent backend.

Each exception carries:
  - `code`: a stable machine-readable error code (returned in JSON responses)
  - `message`: a user-friendly message (returned in JSON responses)
  - `status_code`: the HTTP status code to return

The global exception handler in `main.py` maps these to JSON responses of the form:
    { "detail": "...", "code": "...", "request_id": "..." }
"""

from __future__ import annotations


class TutorError(Exception):
    """Base class for all application-level errors."""

    code: str = "tutor_error"
    message: str = "Something went wrong."
    status_code: int = 500

    def __init__(self, message: str | None = None, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message or self.message)
        if message:
            self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


# ---------------------------------------------------------------------------
# Session errors
# ---------------------------------------------------------------------------

class SessionNotFoundError(TutorError):
    code = "session_not_found"
    message = "We couldn't find that conversation."
    status_code = 404


class SessionAccessDeniedError(TutorError):
    code = "session_access_denied"
    message = "You don't have access to that conversation."
    status_code = 403


# ---------------------------------------------------------------------------
# Graph / agent errors
# ---------------------------------------------------------------------------

class GraphExecutionError(TutorError):
    code = "graph_execution_error"
    message = "The tutor hit a snag processing your message. Please try again."
    status_code = 500


class ToolExecutionError(TutorError):
    code = "tool_execution_error"
    message = "A tutor tool failed. Please try again."
    status_code = 500


# ---------------------------------------------------------------------------
# TTS errors
# ---------------------------------------------------------------------------

class TTSError(TutorError):
    code = "tts_error"
    message = "Audio generation failed. The text reply is still available."
    status_code = 502


# ---------------------------------------------------------------------------
# Database errors
# ---------------------------------------------------------------------------

class DatabaseError(TutorError):
    code = "database_error"
    message = "A database error occurred. Please try again."
    status_code = 500


# ---------------------------------------------------------------------------
# Auth errors
# ---------------------------------------------------------------------------

class AuthenticationError(TutorError):
    code = "authentication_error"
    message = "Your session has expired. Please sign in again."
    status_code = 401