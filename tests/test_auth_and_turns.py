"""Focused tests for JWT verification, agent context, and turn persistence."""

from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

import jwt
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from jwt.exceptions import InvalidTokenError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import verify_token
from app.agent_execution import messages_for_response
from app.sessions import create_session, load_session, save_turn, set_audio_hash
from app.text_utils import strip_leading_raw_tool_call
from app.tools import format_grade_feedback


def _token(secret: str, **claims: object) -> str:
    payload = {"sub": "user-1", "email": "user-1@example.test", **claims}
    return jwt.encode(payload, secret, algorithm="HS256")


def test_verify_token_accepts_valid_signed_jwt(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET", "test-secret")
    monkeypatch.delenv("NEXTAUTH_SECRET", raising=False)
    payload = verify_token(_token("test-secret", exp=datetime.now(timezone.utc) + timedelta(minutes=5)))
    assert payload["sub"] == "user-1"


def test_verify_token_rejects_expired_or_wrongly_signed_jwt(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET", "test-secret")
    expired = _token("test-secret", exp=datetime.now(timezone.utc) - timedelta(minutes=1))
    wrong_secret = _token("other-secret", exp=datetime.now(timezone.utc) + timedelta(minutes=5))
    for token in (expired, wrong_secret):
        with pytest.raises(InvalidTokenError):
            verify_token(token)


def test_verify_token_accepts_legacy_secret_during_migration(monkeypatch):
    monkeypatch.delenv("AUTH_SECRET", raising=False)
    monkeypatch.setenv("NEXTAUTH_SECRET", "legacy-secret")
    assert verify_token(_token("legacy-secret", exp=datetime.now(timezone.utc) + timedelta(minutes=5)))["sub"] == "user-1"


def test_verify_token_requires_secret_even_in_development(monkeypatch):
    monkeypatch.delenv("AUTH_SECRET", raising=False)
    monkeypatch.delenv("NEXTAUTH_SECRET", raising=False)
    monkeypatch.setenv("ENV", "development")
    with pytest.raises(InvalidTokenError, match="AUTH_SECRET not configured"):
        verify_token(_token("dev-secret-change-in-production", exp=datetime.now(timezone.utc) + timedelta(minutes=5)))


def test_response_prompt_retains_tool_results_as_private_context():
    messages = [
        HumanMessage(content="Give me an exercise"),
        AIMessage(content="", tool_calls=[{"name": "generate_exercise", "args": {}, "id": "call-1"}]),
        ToolMessage(content="Retrieved exercise context", tool_call_id="call-1"),
    ]
    prompt_messages = messages_for_response("Tutor prompt", messages)
    contents = [message.content for message in prompt_messages]
    assert any("Retrieved exercise context" in content for content in contents)
    assert all(not isinstance(message, ToolMessage) for message in prompt_messages)
    assert all(not (isinstance(message, AIMessage) and message.tool_calls) for message in prompt_messages)


def test_save_turn_commits_state_and_preserves_audio_hash(tmp_path, monkeypatch):
    import app.sessions as sessions

    monkeypatch.setattr(sessions, "DB_PATH", tmp_path / "sessions.db")
    sessions.init_db()
    session = create_session("user-1", "en")
    session_id = session["session_id"]
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    assert save_turn(session_id, history, {"active": True}, [{"type": "grammar", "detail": "x"}])
    assert set_audio_hash(session_id, "a" * 16, "Hi there")

    updated_history = history + [{"role": "user", "content": "Thanks"}]
    assert save_turn(session_id, updated_history, {"active": False}, [{"type": "grammar", "detail": "x"}])
    saved = load_session(session_id)
    assert saved["last_exercise"] == {"active": False}
    assert saved["mistake_log"] == [{"type": "grammar", "detail": "x"}]
    assert saved["chat_history"][1]["audio_hash"] == "a" * 16


@pytest.mark.parametrize(
    ("grade", "expected"),
    [
        ('{"correct": true, "explanation": "Your tense choice is accurate.", "correct_answer": null}', "Correct! Your tense choice is accurate."),
        ('{"correct": false, "explanation": "Use the past tense here.", "correct_answer": "They only focused."}', "Not quite. Use the past tense here.\n\n**Suggested answer**\nThey only focused."),
        ('{"correct": null, "explanation": "Please try again shortly.", "correct_answer": null}', "Please try again shortly."),
    ],
)
def test_format_grade_feedback_hides_internal_grade_schema(grade, expected):
    assert format_grade_feedback(grade) == expected


def test_format_grade_feedback_handles_malformed_tool_output():
    assert format_grade_feedback("not valid JSON") == "I couldn't grade that answer reliably. Please try submitting it again."


def test_strip_leading_raw_tool_call_keeps_only_tutor_prose():
    content = '''{ "action": "grade_answer", "action_input": {"user_answer": "on"} }

Perfect! You got every answer right.'''
    assert strip_leading_raw_tool_call(content) == "Perfect! You got every answer right."
