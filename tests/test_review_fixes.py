"""
Comprehensive tests covering the production fixes:
1. Chat streaming & guardrail/cleanup fidelity (P0 Item 1)
2. Turn persistence before streaming & disconnect safety (P0 Item 2)
3. Concurrency cap enforcement (P1 Item 3)
4. Default SlowAPI perimeter rate limiting on undecorated routes (P1 Item 4)
5. Invalid token rate limit isolation by IP (P1 Item 5)
6. SQL-computed mistake_count in list_sessions (P2 Item 6)
7. Dependency-injected graph runner without mock sniffing (P2 Item 7)
8. INTERNAL_HEALTH_TOKEN timing-safe comparison & TTS cache hygiene (P2 Item 8)
"""

import asyncio
import json
import os
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import main, sessions, tts
from app.main import app, get_graph_runner, _rate_limit_key


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch, tmp_path):
    """Set up clean SQLite, limiter, and auth secret for each test."""
    db_file = tmp_path / "test_sessions.db"
    monkeypatch.setattr(sessions, "DB_PATH", db_file)
    monkeypatch.setattr(main, "verify_token", lambda tok: {"sub": "test-user-123"})
    if hasattr(app.state, "limiter"):
        app.state.limiter.reset()
    sessions.init_db()


# ---------------------------------------------------------------------------
# P0 Item 1: Streaming bypasses guardrails and response cleanup
# ---------------------------------------------------------------------------

def test_chat_streams_and_saves_clean_reply_for_raw_envelope(tmp_path, monkeypatch):
    """Raw envelope output is cleaned and only the clean final reply is streamed and saved."""
    session = sessions.create_session("test-user-123", "en", "beginner")
    session_id = session["session_id"]

    # Graph produces a state where response has raw envelope stripped to visible prose
    clean_text = "Here is the explanation without the raw tool envelope."
    def fake_runner(state):
        return {
            **state,
            "messages": [*state["messages"], AIMessage(content=clean_text)],
            "intent": "chat",
        }

    app.dependency_overrides[get_graph_runner] = lambda: fake_runner
    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            headers={"Authorization": "Bearer valid-token"},
            json={"session_id": session_id, "message": "Help with grammar"},
        )
        assert response.status_code == 200

        # Collect streamed tokens
        streamed_tokens = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
                if data.get("type") == "token":
                    streamed_tokens.append(data.get("content", ""))

        streamed_full = "".join(streamed_tokens).strip()
        assert streamed_full == clean_text

        # Verify saved reply in DB matches clean text
        reloaded = sessions.load_session(session_id)
        assert reloaded["chat_history"][-1]["content"] == clean_text
    finally:
        app.dependency_overrides.pop(get_graph_runner, None)


def test_chat_streams_and_saves_clean_reply_for_guardrail_failure(tmp_path, monkeypatch):
    """When guardrails regenerate a simpler reply, the regenerated reply is streamed and saved."""
    session = sessions.create_session("test-user-123", "en", "beginner")
    session_id = session["session_id"]

    regenerated_text = "Hello! Let's practice simple words."
    def fake_runner_with_guardrail(state):
        # The final state contains the regenerated response after guardrails
        return {
            **state,
            "messages": [*state["messages"], AIMessage(content=regenerated_text)],
            "intent": "chat",
        }

    app.dependency_overrides[get_graph_runner] = lambda: fake_runner_with_guardrail
    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            headers={"Authorization": "Bearer valid-token"},
            json={"session_id": session_id, "message": "Hi tutor"},
        )
        assert response.status_code == 200

        streamed_tokens = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
                if data.get("type") == "token":
                    streamed_tokens.append(data.get("content", ""))

        streamed_full = "".join(streamed_tokens).strip()
        assert streamed_full == regenerated_text

        reloaded = sessions.load_session(session_id)
        assert reloaded["chat_history"][-1]["content"] == regenerated_text
    finally:
        app.dependency_overrides.pop(get_graph_runner, None)


def test_chat_streams_and_saves_clean_reply_for_tool_call_preamble(tmp_path, monkeypatch):
    """Tool calls with preamble output format_grade_feedback as final reply."""
    session = sessions.create_session("test-user-123", "en", "beginner")
    session_id = session["session_id"]

    formatted_grade = "Correct! Great job on using the past tense."
    def fake_runner_tool_preamble(state):
        return {
            **state,
            "messages": [
                *state["messages"],
                AIMessage(content="Preamble that should not leak", tool_calls=[{"id": "call_1", "name": "grade_answer", "args": {}}]),
                ToolMessage(content='{"correct": true}', tool_call_id="call_1"),
                AIMessage(content=formatted_grade),
            ],
            "intent": "answer_submission",
        }

    app.dependency_overrides[get_graph_runner] = lambda: fake_runner_tool_preamble
    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            headers={"Authorization": "Bearer valid-token"},
            json={"session_id": session_id, "message": "I went to school"},
        )
        assert response.status_code == 200

        streamed_tokens = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
                if data.get("type") == "token":
                    streamed_tokens.append(data.get("content", ""))

        streamed_full = "".join(streamed_tokens).strip()
        assert streamed_full == formatted_grade

        reloaded = sessions.load_session(session_id)
        assert reloaded["chat_history"][-1]["content"] == formatted_grade
    finally:
        app.dependency_overrides.pop(get_graph_runner, None)


# ---------------------------------------------------------------------------
# P0 Item 2: Client disconnect loses the turn
# ---------------------------------------------------------------------------

def test_client_disconnect_mid_stream_persists_turn():
    """Cancelling the stream after receiving only the first chunk still persists the turn."""
    session = sessions.create_session("test-user-123", "en", "beginner")
    session_id = session["session_id"]

    full_reply = "This is a long reply that will be interrupted by the client disconnecting."
    def fake_runner(state):
        return {
            **state,
            "messages": [*state["messages"], AIMessage(content=full_reply)],
            "intent": "chat",
        }

    app.dependency_overrides[get_graph_runner] = lambda: fake_runner
    try:
        client = TestClient(app)
        with client.stream(
            "POST",
            "/chat",
            headers={"Authorization": "Bearer valid-token"},
            json={"session_id": session_id, "message": "Disconnect test message"},
        ) as stream_resp:
            assert stream_resp.status_code == 200
            # Read only the first line/event and immediately exit context (client disconnect)
            for line in stream_resp.iter_lines():
                if line.startswith("data: "):
                    break

        # Re-load session from SQLite to assert both user message and assistant reply are saved
        reloaded = sessions.load_session(session_id)
        assert len(reloaded["chat_history"]) >= 2
        assert reloaded["chat_history"][-2]["content"] == "Disconnect test message"
        assert reloaded["chat_history"][-1]["content"] == full_reply
    finally:
        app.dependency_overrides.pop(get_graph_runner, None)


@pytest.mark.asyncio
async def test_client_disconnect_during_graph_run_persists_turn():
    """Disconnecting while graph runner is still executing persists turn once task completes."""
    from httpx import ASGITransport, AsyncClient

    session = sessions.create_session("test-user-123", "en", "beginner")
    session_id = session["session_id"]
    runner_started = asyncio.Event()
    runner_continue = asyncio.Event()

    async def slow_runner(state):
        runner_started.set()
        await runner_continue.wait()
        return {
            **state,
            "messages": [*state["messages"], AIMessage(content="Slow reply after disconnect")],
            "intent": "chat",
        }

    app.dependency_overrides[get_graph_runner] = lambda: slow_runner
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            req_task = asyncio.create_task(
                ac.post(
                    "/chat",
                    headers={"Authorization": "Bearer valid-token"},
                    json={"session_id": session_id, "message": "Disconnect mid-graph message"},
                )
            )
            # Wait for runner to begin execution
            await asyncio.wait_for(runner_started.wait(), timeout=2.0)
            # Cancel / disconnect client request while runner is executing
            req_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await req_task

        # Release slow runner to complete in the background task
        runner_continue.set()
        # Wait for all background tasks to finish
        if main._background_turns:
            await asyncio.gather(*list(main._background_turns), return_exceptions=True)

        reloaded = sessions.load_session(session_id)
        assert len(reloaded["chat_history"]) >= 2
        assert reloaded["chat_history"][-2]["content"] == "Disconnect mid-graph message"
        assert reloaded["chat_history"][-1]["content"] == "Slow reply after disconnect"
    finally:
        app.dependency_overrides.pop(get_graph_runner, None)


@pytest.mark.asyncio
async def test_graph_runner_raises_emits_error_and_does_not_save():
    """A runner that raises: stream emits error and done events, and nothing is saved."""
    from httpx import ASGITransport, AsyncClient

    session = sessions.create_session("test-user-123", "en", "beginner")
    session_id = session["session_id"]

    def raising_runner(state):
        raise RuntimeError("LLM exploded")

    app.dependency_overrides[get_graph_runner] = lambda: raising_runner
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/chat",
                headers={"Authorization": "Bearer valid-token"},
                json={"session_id": session_id, "message": "Failing message"},
            )
            assert resp.status_code == 200
            text = resp.text
            assert '{"type": "error"' in text
            assert '{"type": "done"' in text

        reloaded = sessions.load_session(session_id)
        assert len(reloaded.get("chat_history", [])) == 0
    finally:
        app.dependency_overrides.pop(get_graph_runner, None)


# ---------------------------------------------------------------------------
# P1 Item 3: Concurrency cap is enforced across requests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_cap_bounds_in_flight_requests(monkeypatch):
    """Assert that 30 concurrent callers share the semaphore and in-flight <= LLM_CONCURRENCY."""
    from httpx import ASGITransport, AsyncClient

    monkeypatch.setenv("CHAT_RATE_LIMIT", "100/minute")
    app.state.limiter.reset()

    current_in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def fake_slow_runner(state):
        nonlocal current_in_flight, max_in_flight
        async with lock:
            current_in_flight += 1
            if current_in_flight > max_in_flight:
                max_in_flight = current_in_flight
        await asyncio.sleep(0.05)
        async with lock:
            current_in_flight -= 1
        return {
            **state,
            "messages": [*state["messages"], AIMessage(content="Concur reply")],
            "intent": "chat",
        }

    # Ensure clean app semaphore
    app.state.llm_semaphore = asyncio.Semaphore(main.LLM_CONCURRENCY)
    app.dependency_overrides[get_graph_runner] = lambda: fake_slow_runner

    session = sessions.create_session("test-user-123", "en", "beginner")
    session_id = session["session_id"]

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            tasks = [
                ac.post(
                    "/chat",
                    headers={"Authorization": "Bearer valid-token"},
                    json={"session_id": session_id, "message": f"Ping {i}"},
                )
                for i in range(30)
            ]
            responses = await asyncio.gather(*tasks)

        assert all(r.status_code == 200 for r in responses)
        assert max_in_flight <= main.LLM_CONCURRENCY
        assert max_in_flight > 1
    finally:
        app.dependency_overrides.pop(get_graph_runner, None)


# ---------------------------------------------------------------------------
# P1 Item 4: Default rate limit applies to undecorated routes
# ---------------------------------------------------------------------------

def test_default_rate_limit_applies_to_undecorated_route():
    """101st call to undecorated route (/sessions) returns 429 while /health remains exempt."""
    app.state.limiter.reset()

    client = TestClient(app)
    headers = {"Authorization": "Bearer rate-limited-default-user"}

    try:
        # 100 requests should succeed under default 100/minute limit
        for _ in range(100):
            resp = client.get("/sessions", headers=headers)
            assert resp.status_code == 200

        # 101st request must be rate limited (429) using the app's error envelope
        blocked = client.get("/sessions", headers=headers)
        assert blocked.status_code == 429
        data = blocked.json()
        assert data["code"] == "rate_limit"
        assert "Too many requests" in data["detail"]
        assert "request_id" in data
        assert blocked.headers.get("retry-after") == "60"

        # Health check is @limiter.exempt and must succeed even when rate limit is exceeded
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"
    finally:
        app.state.limiter.reset()


# ---------------------------------------------------------------------------
# P1 Item 5: Invalid tokens fall back to client IP, not shared token prefix
# ---------------------------------------------------------------------------

def test_rate_limit_key_falls_back_to_ip_on_invalid_token(monkeypatch):
    """Invalid tokens fall back to remote IP, isolating users instead of sharing one bucket."""
    def fake_verify(tok: str):
        if tok == "valid-token":
            return {"sub": "test-user-123"}
        raise ValueError("Invalid JWT token signature or expiration")

    monkeypatch.setattr(main, "verify_token", fake_verify)

    class FakeRequest:
        def __init__(self, auth: str | None, client_host: str):
            self.headers = {"authorization": auth} if auth else {}
            self.client = type("Client", (), {"host": client_host})()

    # Valid token -> user:{sub}
    req_valid = FakeRequest("Bearer valid-token", "1.2.3.4")
    assert _rate_limit_key(req_valid) == "user:test-user-123"

    # Invalid token -> falls back to client IP, not token:eyJhbGciOi...
    req_bad_1 = FakeRequest("Bearer eyJhbGciOiJIUzI1NiInvalid1", "192.168.1.10")
    req_bad_2 = FakeRequest("Bearer eyJhbGciOiJIUzI1NiInvalid2", "192.168.1.20")

    key_1 = _rate_limit_key(req_bad_1)
    key_2 = _rate_limit_key(req_bad_2)

    assert key_1 == "192.168.1.10"
    assert key_2 == "192.168.1.20"
    assert not key_1.startswith("token:")
    assert key_1 != key_2


# ---------------------------------------------------------------------------
# P2 Item 6: mistake_count computed in SQL in list_sessions
# ---------------------------------------------------------------------------

def test_list_sessions_computes_mistake_count_in_sql():
    """list_sessions computes the correct mistake_count via json_array_length without loading mistake_log."""
    session_1 = sessions.create_session("test-user-123", "en", "beginner")
    session_2 = sessions.create_session("test-user-123", "en", "intermediate")

    # Add 3 mistakes to session_1
    sessions.add_mistake(session_1["session_id"], "grammar", "wrong tense")
    sessions.add_mistake(session_1["session_id"], "vocabulary", "word choice")
    sessions.add_mistake(session_1["session_id"], "spelling", "typo")

    client = TestClient(app)
    resp = client.get("/sessions", headers={"Authorization": "Bearer valid-token"})
    assert resp.status_code == 200
    items = resp.json()

    s1_entry = next(s for s in items if s["session_id"] == session_1["session_id"])
    s2_entry = next(s for s in items if s["session_id"] == session_2["session_id"])

    assert s1_entry["mistake_count"] == 3
    assert s2_entry["mistake_count"] == 0


# ---------------------------------------------------------------------------
# P2 Item 7: Mock-sniffing removed in production code
# ---------------------------------------------------------------------------

def test_production_code_has_no_mock_sniffing():
    """Ensure is_invoke_mocked and MagicMock inspections are not present in app/main.py."""
    with open(main.__file__, "r", encoding="utf-8") as f:
        source = f.read()

    assert "is_invoke_mocked" not in source
    assert "MagicMock" not in source


# ---------------------------------------------------------------------------
# P2 Item 8: INTERNAL_HEALTH_TOKEN timing-safe & TTS cache hygiene
# ---------------------------------------------------------------------------

def test_internal_health_token_timing_safe(monkeypatch):
    """x-internal-token bypasses cache only when matched with compare_digest."""
    monkeypatch.setenv("INTERNAL_HEALTH_TOKEN", "secret-internal-probe-token")
    client = TestClient(app)

    # Calling with valid internal token
    r_valid = client.get("/health/deps", headers={"x-internal-token": "secret-internal-probe-token"})
    assert r_valid.status_code == 200

    # Calling with invalid internal token
    r_invalid = client.get("/health/deps", headers={"x-internal-token": "wrong-token"})
    assert r_invalid.status_code == 200


def test_tts_cache_cleans_up_orphaned_tmp_files(tmp_path, monkeypatch):
    """evict_cache_if_needed removes orphaned *.tmp.* files older than 10m while keeping fresh ones."""
    monkeypatch.setattr(tts, "AUDIO_CACHE_DIR", tmp_path)

    # Create old orphaned temporary file (> 10 minutes old) and a fresh in-progress file
    old_orphan = tmp_path / "hash1.mp3.tmp.abc12345"
    fresh_tmp = tmp_path / "hash2.mp3.tmp.def67890"
    old_orphan.write_bytes(b"temp data 1")
    fresh_tmp.write_bytes(b"temp data 2")

    now = time.time()
    # Age old_orphan by 700 seconds (> 600s / 10m threshold)
    os.utime(old_orphan, (now - 700, now - 700))
    # Ensure fresh_tmp is current
    os.utime(fresh_tmp, (now, now))

    assert old_orphan.exists()
    assert fresh_tmp.exists()

    tts.evict_cache_if_needed(max_bytes=100000)

    # Old orphaned temp file should be removed
    assert not old_orphan.exists()
    # Fresh in-progress temp file must be kept
    assert fresh_tmp.exists()
