"""Pydantic v2 schemas for the public Language Tutor HTTP API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

LanguageCode = Literal["en", "ko", "ja"]
Level = Literal["beginner", "intermediate", "advanced"]
PracticeType = Literal["grammar", "vocabulary", "reading", "writing", "translation", "mistake_review"]

_MAX_SESSION_ID_LENGTH = 128
_MAX_TITLE_LENGTH = 200
_MAX_MESSAGE_LENGTH = 4_000
_MAX_TTS_CONTENT_LENGTH = 20_000


class RequestModel(BaseModel):
    """Shared external-input behavior for API request models."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class SessionRequest(RequestModel):
    language: LanguageCode
    level: Level = "beginner"


class RenameRequest(RequestModel):
    title: str

    @field_validator("title")
    @classmethod
    def title_is_valid(cls, value: str) -> str:
        if not value:
            raise ValueError("Title must not be empty")
        if len(value) > _MAX_TITLE_LENGTH:
            raise ValueError(f"Title must be {_MAX_TITLE_LENGTH} characters or fewer")
        return value


class ChatRequest(RequestModel):
    session_id: str
    message: str
    practice_type: PracticeType | None = None

    @field_validator("session_id")
    @classmethod
    def session_id_is_valid(cls, value: str) -> str:
        if not value:
            raise ValueError("Session ID must not be empty")
        if len(value) > _MAX_SESSION_ID_LENGTH:
            raise ValueError(f"Session ID must be {_MAX_SESSION_ID_LENGTH} characters or fewer")
        return value

    @field_validator("message")
    @classmethod
    def message_is_valid(cls, value: str) -> str:
        if not value:
            raise ValueError("Message must not be empty")
        if len(value) > _MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message must be {_MAX_MESSAGE_LENGTH} characters or fewer")
        return value


class TTSRequest(RequestModel):
    content: str

    @field_validator("content")
    @classmethod
    def content_is_valid(cls, value: str) -> str:
        if not value:
            raise ValueError("Content must not be empty")
        if len(value) > _MAX_TTS_CONTENT_LENGTH:
            raise ValueError(f"Content must be {_MAX_TTS_CONTENT_LENGTH} characters or fewer")
        return value


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    language: LanguageCode
    level: Level
    created_at: str


class SessionDetailResponse(SessionResponse):
    title: str = ""
    chat_history: list[dict[str, Any]]
    updated_at: str


class SessionListItemResponse(SessionResponse):
    title: str = ""
    mistake_count: int
    updated_at: str


class MutationResponse(BaseModel):
    ok: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]


class DependencyHealthResponse(HealthResponse):
    dependencies: dict[str, Any]


class ChatResponse(BaseModel):
    reply: str
    intent: str
    audio_url: str | None = None
    practice_type: PracticeType | None = None
