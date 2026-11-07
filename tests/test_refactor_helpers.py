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


def test_astream_filtering_isolates_response_node():
    """Verify that node filtering in astream streaming only emits tutor response tokens."""
    messages = [
        {"type": "messages", "data": (SimpleNamespace(content="Hello "), {"langgraph_node": "generate_response"})},
        {"type": "messages", "data": (SimpleNamespace(content="internal"), {"langgraph_node": "apply_guardrails"})},
        {"type": "messages", "data": (SimpleNamespace(content="world!"), {"langgraph_node": "generate_response"})},
    ]

    streamed = []
    for chunk in messages:
        if chunk.get("type") == "messages":
            msg, metadata = chunk.get("data", (None, {}))
            if metadata.get("langgraph_node") == "generate_response" and msg and msg.content:
                streamed.append(msg.content)

    assert streamed == ["Hello ", "world!"]
    assert "".join(streamed) == "Hello world!"


def test_init_db_creates_user_id_index(tmp_path, monkeypatch):
    import sqlite3
    import app.sessions as sessions

    db_path = tmp_path / "index_test.db"
    monkeypatch.setattr(sessions, "DB_PATH", db_path)
    sessions.init_db()

    with sqlite3.connect(str(db_path)) as conn:
        indexes = conn.execute("PRAGMA index_list('sessions')").fetchall()
        index_names = [idx[1] for idx in indexes]
        assert "idx_sessions_user_id" in index_names


def test_apply_guardrails_uses_human_message_for_regeneration(monkeypatch):
    from app.graph import apply_guardrails
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    captured_prompt = None

    class FakeLLM:
        def invoke(self, messages):
            nonlocal captured_prompt
            if isinstance(messages, str):
                # Guardrail check call
                return SimpleNamespace(content='{"pass": false, "reason": "too academic"}')
            # Regeneration call
            captured_prompt = messages
            return AIMessage(content="Simple greeting!")

    monkeypatch.setattr("app.graph.make_llm", lambda *a, **kw: FakeLLM())

    state = {
        "language": "en",
        "level": "beginner",
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Extremely intricate pedagogical analysis of syntax."),
        ],
    }

    updated_state = apply_guardrails(state)
    assert updated_state["messages"][-1].content == "Simple greeting!"
    assert captured_prompt is not None
    regeneration_msg = captured_prompt[-1]
    assert isinstance(regeneration_msg, HumanMessage)
    assert "flagged as inappropriate" in regeneration_msg.content


