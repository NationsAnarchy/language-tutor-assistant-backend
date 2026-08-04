"""Hermetic unit tests for typed practice-mode routing and prompt mapping."""

from pathlib import Path
import sys

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent_execution import execute_tool_calls
from app.graph import route_intent
from app.tools import PRACTICE_MODE_CONFIG, practice_mode_instruction


def _state(**overrides):
    state = {
        "user_id": "user", "session_id": "session", "language": "en", "level": "beginner",
        "messages": [HumanMessage(content="Please practice something")], "last_exercise": {},
        "intent": "chat", "mistake_log": [], "speed": "normal", "practice_type": None,
    }
    state.update(overrides)
    return state


def test_every_practice_mode_has_a_retrieval_query_and_instruction():
    assert set(PRACTICE_MODE_CONFIG) == {
        "grammar", "vocabulary", "reading", "writing", "translation", "mistake_review",
    }
    for mode in PRACTICE_MODE_CONFIG:
        query, instruction = practice_mode_instruction(mode)
        assert query
        assert instruction


def test_explicit_mode_wins_over_ambiguous_chat_text():
    routed = route_intent(_state(messages=[HumanMessage(content="Can you explain this grammar rule?")], practice_type="translation"))
    assert routed["intent"] == "exercise_request"


def test_active_exercise_keeps_answer_submission_semantics_without_new_practice_request():
    routed = route_intent(_state(last_exercise={"active": True}))
    assert routed["intent"] == "answer_submission"


def test_explicit_mode_replaces_an_active_exercise():
    routed = route_intent(_state(practice_type="grammar", last_exercise={"active": True}))
    assert routed["intent"] == "exercise_request"


def test_mistake_review_injects_only_the_latest_five_records():
    captured = {}

    class Tool:
        def invoke(self, args):
            captured.update(args)
            return "exercise context"

    mistakes = [{"type": "grammar", "detail": str(index)} for index in range(7)]
    state = _state(mistake_log=mistakes)
    execute_tool_calls(
        type("Response", (), {"tool_calls": [{"name": "generate_exercise", "args": {"skill": "mistake_review"}, "id": "call"}]})(),
        state,
        {"generate_exercise": Tool()},
    )
    assert "[grammar] 1" not in captured["recent_mistakes"]
    assert "[grammar] 2" in captured["recent_mistakes"]
    assert "[grammar] 6" in captured["recent_mistakes"]
    assert state["last_exercise"]["active"] is True


def test_mistake_review_instruction_includes_recent_context():
    _, instruction = practice_mode_instruction("mistake_review", "[grammar] particles")
    assert "particles" in instruction
