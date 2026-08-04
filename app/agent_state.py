"""State shared by the language-tutor LangGraph nodes."""

from typing import Any, Literal

from typing_extensions import TypedDict


class TutorState(TypedDict):
    """Persisted and transient state for a single tutor turn."""

    user_id: str
    session_id: str
    language: str
    level: str
    messages: list[Any]
    last_exercise: dict[str, Any]
    intent: str
    audio_url: str
    mistake_log: list[dict[str, Any]]
    speed: str
    # Request-scoped only: persisted sessions remain backwards compatible.
    practice_type: Literal["grammar", "vocabulary", "reading", "writing", "translation", "mistake_review"] | None
