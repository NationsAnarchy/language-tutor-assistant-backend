"""Unit tests for behavior-preserving internal refactors."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import _sse_event
from app import tools


def test_sse_event_serializes_content_safely():
    event = _sse_event("token", content='She said "hello"')

    assert event.startswith("data: ")
    assert event.endswith("\n\n")
    assert json.loads(event.removeprefix("data: ").strip()) == {
        "type": "token",
        "content": 'She said "hello"',
    }


def test_retrieve_notes_preserves_grammar_and_vocabulary_responses(monkeypatch):
    docs = [SimpleNamespace(metadata={"topic": "verbs"}, page_content="Use the past tense.")]
    monkeypatch.setattr(tools, "_get_retriever", lambda *_args, **_kwargs: SimpleNamespace(invoke=lambda _: docs))

    assert tools.retrieve_grammar.invoke({"language": "en", "topic": "verbs"}) == (
        "**verbs**\nUse the past tense."
    )
    assert tools.retrieve_vocab.invoke({"language": "en", "topic_or_word": "verbs"}) == (
        "**verbs**\nUse the past tense."
    )


def test_retrieve_notes_preserves_empty_result_messages(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_get_retriever",
        lambda *_args, **_kwargs: SimpleNamespace(invoke=lambda _: []),
    )

    assert tools.retrieve_grammar.invoke({"language": "en", "topic": "verbs"}) == (
        "(no retrieved grammar notes available for this topic)"
    )
    assert tools.retrieve_vocab.invoke({"language": "en", "topic_or_word": "verbs"}) == (
        "(no retrieved vocabulary available for this topic)"
    )


def test_list_sessions_returns_only_metadata_columns(tmp_path, monkeypatch):
    import app.sessions as sessions

    monkeypatch.setattr(sessions, "DB_PATH", tmp_path / "sessions.db")
    sessions.init_db()
    session = sessions.create_session("user-prune", "ko", "intermediate")
    session_id = session["session_id"]
    sessions.save_turn(
        session_id,
        [{"role": "user", "content": "Large chat history payload"}],
        {"prompt": "exercise payload"},
        [{"type": "grammar", "detail": "mistake payload"}],
    )

    results = sessions.list_sessions("user-prune")
    assert len(results) == 1
    row = results[0]
    assert row["session_id"] == session_id
    assert row["user_id"] == "user-prune"
    assert row["language"] == "ko"
    assert row["level"] == "intermediate"
    # Blobs must be pruned from list_sessions query
    assert "chat_history" not in row
    assert "last_exercise" not in row
    assert "mistake_log" not in row
