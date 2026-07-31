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
