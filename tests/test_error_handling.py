"""
Tests for the error handling layer.

Covers:
  - Global exception handlers return consistent JSON shape with request_id
  - Typed TutorError subclasses map to correct HTTP status codes
  - X-Request-ID header is present on all responses
  - /health/deps returns dependency status
  - Input validation rejects empty/oversized messages
  - Session not-found / access-denied errors
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.exceptions import (
    AuthenticationError,
    DatabaseError,
    GraphExecutionError,
    SessionAccessDeniedError,
    SessionNotFoundError,
    TTSError,
    TutorError,
)
from app.main import app


@pytest.fixture
def client():
    """Create a test client with dev auth bypass."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Headers for dev auth bypass."""
    return {"X-Dev-User-Id": "test-user"}


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_tutor_error_defaults(self):
        err = TutorError()
        assert err.status_code == 500
        assert err.code == "tutor_error"
        assert err.message == "Something went wrong."

    def test_session_not_found(self):
        err = SessionNotFoundError()
        assert err.status_code == 404
        assert err.code == "session_not_found"

    def test_session_access_denied(self):
        err = SessionAccessDeniedError()
        assert err.status_code == 403
        assert err.code == "session_access_denied"

    def test_graph_execution_error(self):
        err = GraphExecutionError()
        assert err.status_code == 500
        assert err.code == "graph_execution_error"

    def test_tts_error(self):
        err = TTSError()
        assert err.status_code == 502
        assert err.code == "tts_error"

    def test_database_error(self):
        err = DatabaseError()
        assert err.status_code == 500
        assert err.code == "database_error"

    def test_authentication_error(self):
        err = AuthenticationError()
        assert err.status_code == 401
        assert err.code == "authentication_error"

    def test_custom_message(self):
        err = TutorError("Custom message")
        assert err.message == "Custom message"
        assert str(err) == "Custom message"


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_deps(self, client):
        response = client.get("/health/deps")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "dependencies" in data
        assert "gemini_api_key" in data["dependencies"]
        assert "google_embedding_api_key" in data["dependencies"]
        assert "pinecone" in data["dependencies"]

    def test_health_deps_has_request_id(self, client):
        response = client.get("/health/deps")
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------

class TestRequestIdMiddleware:
    def test_request_id_in_response(self, client):
        response = client.get("/health")
        assert "x-request-id" in response.headers
        request_id = response.headers["x-request-id"]
        assert len(request_id) == 16  # hex[:16]

    def test_request_id_unique_per_request(self, client):
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]

    def test_request_id_preserved_from_header(self, client):
        custom_id = "my-custom-id-1234"
        response = client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.headers["x-request-id"] == custom_id


# ---------------------------------------------------------------------------
# Auth errors
# ---------------------------------------------------------------------------

class TestAuthErrors:
    def test_missing_auth(self, client):
        response = client.get("/sessions")
        assert response.status_code == 401
        data = response.json()
        assert data["code"] == "authentication_error"
        assert "request_id" in data

    def test_invalid_token(self, client):
        response = client.get(
            "/sessions",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401
        data = response.json()
        assert data["code"] == "authentication_error"


# ---------------------------------------------------------------------------
# Session errors
# ---------------------------------------------------------------------------

class TestSessionErrors:
    def test_session_not_found(self, client, auth_headers):
        response = client.get("/session/nonexistent-id", headers=auth_headers)
        assert response.status_code == 404
        data = response.json()
        assert data["code"] == "session_not_found"
        assert "request_id" in data

    def test_session_not_found_on_chat(self, client, auth_headers):
        response = client.post(
            "/chat",
            headers=auth_headers,
            json={"session_id": "nonexistent-id", "message": "hello"},
        )
        assert response.status_code == 404
        data = response.json()
        assert data["code"] == "session_not_found"

    def test_session_not_found_on_tts(self, client, auth_headers):
        response = client.post("/session/nonexistent-id/tts", headers=auth_headers)
        assert response.status_code == 404
        data = response.json()
        assert data["code"] == "session_not_found"

    def test_session_not_found_on_delete(self, client, auth_headers):
        response = client.delete("/session/nonexistent-id", headers=auth_headers)
        assert response.status_code == 404
        data = response.json()
        assert data["code"] == "session_not_found"

    def test_session_not_found_on_rename(self, client, auth_headers):
        response = client.patch(
            "/session/nonexistent-id",
            headers=auth_headers,
            json={"title": "new title"},
        )
        assert response.status_code == 404
        data = response.json()
        assert data["code"] == "session_not_found"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_message_rejected(self, client, auth_headers):
        # First create a session
        session_resp = client.post(
            "/session",
            headers=auth_headers,
            json={"language": "en", "level": "beginner"},
        )
        assert session_resp.status_code == 200
        session_id = session_resp.json()["session_id"]

        # Try to send an empty message
        response = client.post(
            "/chat",
            headers=auth_headers,
            json={"session_id": session_id, "message": ""},
        )
        assert response.status_code == 422
        data = response.json()
        assert data["code"] == "validation_error"

    def test_whitespace_only_message_rejected(self, client, auth_headers):
        session_resp = client.post(
            "/session",
            headers=auth_headers,
            json={"language": "en", "level": "beginner"},
        )
        session_id = session_resp.json()["session_id"]

        response = client.post(
            "/chat",
            headers=auth_headers,
            json={"session_id": session_id, "message": "   "},
        )
        assert response.status_code == 422

    def test_oversized_message_rejected(self, client, auth_headers):
        session_resp = client.post(
            "/session",
            headers=auth_headers,
            json={"language": "en", "level": "beginner"},
        )
        session_id = session_resp.json()["session_id"]

        response = client.post(
            "/chat",
            headers=auth_headers,
            json={"session_id": session_id, "message": "x" * 5000},
        )
        assert response.status_code == 422

    def test_invalid_language_rejected(self, client, auth_headers):
        response = client.post(
            "/session",
            headers=auth_headers,
            json={"language": "fr", "level": "beginner"},
        )
        assert response.status_code == 400

    def test_invalid_level_rejected(self, client, auth_headers):
        response = client.post(
            "/session",
            headers=auth_headers,
            json={"language": "en", "level": "expert"},
        )
        assert response.status_code == 400

    def test_empty_rename_rejected(self, client, auth_headers):
        session_resp = client.post(
            "/session",
            headers=auth_headers,
            json={"language": "en", "level": "beginner"},
        )
        session_id = session_resp.json()["session_id"]

        response = client.patch(
            f"/session/{session_id}",
            headers=auth_headers,
            json={"title": ""},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Error response shape consistency
# ---------------------------------------------------------------------------

class TestErrorResponseShape:
    """All error responses should have the same JSON shape."""

    def _assert_error_shape(self, response):
        data = response.json()
        assert "detail" in data
        assert "code" in data
        assert "request_id" in data
        assert isinstance(data["detail"], str)
        assert isinstance(data["code"], str)
        assert isinstance(data["request_id"], str)

    def test_401_shape(self, client):
        response = client.get("/sessions")
        assert response.status_code == 401
        self._assert_error_shape(response)

    def test_404_shape(self, client, auth_headers):
        response = client.get("/session/fake", headers=auth_headers)
        assert response.status_code == 404
        self._assert_error_shape(response)

    def test_422_shape(self, client, auth_headers):
        response = client.post(
            "/session",
            headers=auth_headers,
            json={"language": "invalid"},
        )
        # This will be a 400 from the route, not 422 — but still has the shape
        data = response.json()
        assert "detail" in data
        assert "request_id" in data

    def test_400_shape(self, client, auth_headers):
        response = client.post(
            "/session",
            headers=auth_headers,
            json={"language": "fr", "level": "beginner"},
        )
        assert response.status_code == 400
        self._assert_error_shape(response)


# ---------------------------------------------------------------------------
# Session access control
# ---------------------------------------------------------------------------

class TestSessionAccessControl:
    def test_access_denied_different_user(self, client):
        # Create a session as user-a
        session_resp = client.post(
            "/session",
            headers={"X-Dev-User-Id": "user-a"},
            json={"language": "en", "level": "beginner"},
        )
        session_id = session_resp.json()["session_id"]

        # Try to access it as user-b
        response = client.get(
            f"/session/{session_id}",
            headers={"X-Dev-User-Id": "user-b"},
        )
        assert response.status_code == 403
        data = response.json()
        assert data["code"] == "session_access_denied"

    def test_access_denied_on_chat(self, client):
        session_resp = client.post(
            "/session",
            headers={"X-Dev-User-Id": "user-a"},
            json={"language": "en", "level": "beginner"},
        )
        session_id = session_resp.json()["session_id"]

        response = client.post(
            "/chat",
            headers={"X-Dev-User-Id": "user-b"},
            json={"session_id": session_id, "message": "hello"},
        )
        assert response.status_code == 403
        data = response.json()
        assert data["code"] == "session_access_denied"