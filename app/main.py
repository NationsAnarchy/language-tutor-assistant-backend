"""
FastAPI application for the Language Tutor Agent.

Routes:
- POST /session   — create or load a session
- POST /chat      — send a message, run the LangGraph agent, return reply
- GET  /sessions  — list sessions for the authenticated user
- GET  /health    — basic health check
- GET  /health/deps — dependency health (Pinecone, API keys)
"""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pinecone import Pinecone
from pydantic import BaseModel, field_validator

from .auth import verify_token
from .exceptions import (
    AuthenticationError,
    DatabaseError,
    GraphExecutionError,
    SessionAccessDeniedError,
    SessionNotFoundError,
    TTSError,
    TutorError,
)
from .graph import graph_no_tts
from .text_utils import extract_text
from .logging_config import RequestIdMiddleware, configure_logging, get_logger
from .sessions import (
    create_session,
    delete_session,
    init_db,
    load_session,
    list_sessions,
    rename_session,
    save_chat_history_merge,
    save_session,
    set_audio_hash,
)
from .tools import clear_session_context, init_vector_store, set_session_context
from .tts import AUDIO_CACHE_DIR, synthesize_speech

load_dotenv()
configure_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Middleware — order matters: outermost first
# ---------------------------------------------------------------------------

# Request ID middleware (adds X-Request-ID header + injects into logs)
app = None  # placeholder — we set it below after defining lifespan


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

_pinecone_index = None


async def _startup_pinecone() -> None:
    global _pinecone_index
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not pinecone_api_key or not gemini_api_key:
        logger.warning(
            "Missing PINECONE_API_KEY or GEMINI_API_KEY — "
            "vector retrieval will fail until configured."
        )
        return

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = os.getenv("PINECONE_INDEX", "language-tutor")
    embedding_api_key = os.getenv("GOOGLE_EMBEDDING_API_KEY") or gemini_api_key

    try:
        _pinecone_index = pc.Index(index_name)
        init_vector_store(_pinecone_index, embedding_api_key)
        logger.info("Connected to Pinecone index '%s'", index_name)
    except Exception as exc:
        logger.warning("Could not connect to Pinecone: %s", exc)
        logger.warning("Run 'python -m app.pinecone_setup' to create the index.")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup: initialize DB schema and Pinecone connection."""
    init_db()
    await _startup_pinecone()
    yield
    # Shutdown: no cleanup needed currently


app = FastAPI(title="Language Tutor Agent", version="0.1.0", lifespan=lifespan)

# Now attach middleware
app.add_middleware(RequestIdMiddleware)

# CORS — allow frontend during development
_CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_CORS_ORIGINS = [origin.strip() for origin in _CORS_ORIGINS_ENV.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

def _get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


@app.exception_handler(TutorError)
async def tutor_error_handler(request: Request, exc: TutorError) -> JSONResponse:
    """Handle typed application errors with a consistent JSON shape."""
    logger.warning(
        "TutorError: %s (code=%s, status=%d)",
        exc.message, exc.code, exc.status_code,
        exc_info=(exc.status_code >= 500),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "code": exc.code,
            "request_id": _get_request_id(request),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPExceptions with request_id included."""
    code_map = {
        400: "bad_request",
        401: "authentication_error",
        403: "session_access_denied",
        404: "not_found",
        422: "validation_error",
        429: "rate_limit",
    }
    code = code_map.get(exc.status_code, "http_error")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "code": code,
            "request_id": _get_request_id(request),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors with field details."""
    logger.info("Validation error: %s", exc.errors())
    safe_errors = []
    for err in exc.errors():
        safe_err = {}
        for k, v in err.items():
            try:
                json.dumps(v)
                safe_err[k] = v
            except (TypeError, ValueError):
                safe_err[k] = str(v)
        safe_errors.append(safe_err)
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Something in the request wasn't quite right.",
            "code": "validation_error",
            "errors": safe_errors,
            "request_id": _get_request_id(request),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected errors — log the full traceback, return a friendly message."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Our tutor is having a moment. Please try again.",
            "code": "internal_error",
            "request_id": _get_request_id(request),
        },
    )


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def get_current_user(request: Request) -> dict[str, Any]:
    """Extract and verify JWT from Authorization header. Returns user info."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        dev_user = request.headers.get("X-Dev-User-Id")
        if dev_user:
            return {"sub": dev_user, "email": f"{dev_user}@dev.local"}
        raise AuthenticationError("Missing or invalid Authorization header")

    token = auth_header.removeprefix("Bearer ").strip()
    try:
        payload = verify_token(token)
        return payload
    except Exception as exc:
        dev_user = request.headers.get("X-Dev-User-Id")
        if dev_user:
            return {"sub": dev_user, "email": f"{dev_user}@dev.local"}
        logger.info("Token verification failed: %s", exc)
        raise AuthenticationError()


def _user_id(user: dict[str, Any]) -> str:
    """Extract the stable user identifier from the auth payload."""
    return user.get("sub") or user.get("email")


def _load_owned_session(session_id: str, user: dict[str, Any]) -> dict[str, Any]:
    """Load a session by ID and verify the authenticated user owns it.

    Raises:
        DatabaseError: if the DB operation fails.
        SessionNotFoundError: if the session doesn't exist.
        SessionAccessDeniedError: if the session belongs to a different user.

    Returns:
        The deserialized session dict.
    """
    try:
        session = load_session(session_id)
    except Exception as exc:
        logger.exception("Failed to load session %s", session_id)
        raise DatabaseError() from exc

    if session is None:
        raise SessionNotFoundError()

    if session["user_id"] != _user_id(user):
        raise SessionAccessDeniedError()

    return session


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------

def _dicts_to_messages(history: list[dict[str, Any]]) -> list:
    """Convert chat_history dicts to LangChain message objects."""
    messages = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def _messages_to_dicts(messages: list) -> list[dict[str, Any]]:
    """Convert LangChain message objects to chat_history dicts."""
    chat_history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            chat_history.append({"role": "user", "content": extract_text(msg.content)})
        elif isinstance(msg, AIMessage) and msg.content:
            chat_history.append({"role": "assistant", "content": extract_text(msg.content)})
    return chat_history


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SessionRequest(BaseModel):
    language: Literal["en", "ko", "ja"]
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"


class RenameRequest(BaseModel):
    title: str

    @field_validator("title")
    @classmethod
    def title_must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Title must not be empty")
        return v.strip()


class ChatRequest(BaseModel):
    session_id: str
    message: str

    @field_validator("message")
    @classmethod
    def message_must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message must not be empty")
        if len(v) > 4000:
            raise ValueError("Message must be 4000 characters or fewer")
        return v.strip()


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    language: str
    level: str
    created_at: str


class ChatResponse(BaseModel):
    reply: str
    intent: str
    audio_url: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/session", response_model=SessionResponse)
async def create_or_load_session(
    body: SessionRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a new session — always starts fresh (Issue #2)."""
    user_id = _user_id(user)
    try:
        session = create_session(user_id, body.language, body.level)
    except Exception as exc:
        logger.exception("Failed to create session for user %s", user_id)
        raise DatabaseError() from exc

    return {
        "session_id": session["session_id"],
        "user_id": session["user_id"],
        "language": session["language"],
        "level": session["level"],
        "created_at": session["created_at"],
    }


@app.get("/session/{session_id}")
async def get_session(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Get a session by ID, including full chat history."""
    session = _load_owned_session(session_id, user)

    return {
        "session_id": session["session_id"],
        "user_id": session["user_id"],
        "language": session["language"],
        "level": session["level"],
        "title": session.get("title", ""),
        "chat_history": session.get("chat_history", []),
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
    }


@app.get("/sessions")
async def list_user_sessions(
    user: dict = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """List all sessions for the authenticated user."""
    try:
        user_id = _user_id(user)
        sessions = list_sessions(user_id)
    except Exception as exc:
        logger.exception("Failed to list sessions for user %s", user_id)
        raise DatabaseError() from exc

    return [
        {
            "session_id": s["session_id"],
            "user_id": s["user_id"],
            "language": s["language"],
            "level": s["level"],
            "title": s.get("title", ""),
            "mistake_count": len(json.loads(s.get("mistake_log", "[]"))),
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
        }
        for s in sessions
    ]


@app.post("/chat")
async def chat(
    body: ChatRequest,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Process a user message through the LangGraph agent and stream the reply.

    Returns a Server-Sent Events stream with the following event types:
      data: {"type": "token", "content": "..."}   — incremental token
      data: {"type": "done", "intent": "..."}       — stream complete
      data: {"type": "error", "message": "..."}     — error occurred

    The frontend appends tokens to the chat bubble in real time.
    TTS is still synthesized separately via the /tts endpoint.
    """
    # Load session and verify ownership
    session = _load_owned_session(body.session_id, user)
    user_id = _user_id(user)

    # Set session context for tool access (Week 2: grade_answer, log_mistake)
    set_session_context(user_id, body.session_id)

    async def _event_generator() -> AsyncGenerator[str, None]:
        nonlocal session
        try:
            # Build initial state for LangGraph
            messages = _dicts_to_messages(session.get("chat_history", []))
            messages.append(HumanMessage(content=body.message))

            state = {
                "user_id": user_id,
                "session_id": body.session_id,
                "language": session["language"],
                "level": session.get("level", "beginner"),
                "messages": messages,
                "last_exercise": session.get("last_exercise", {}),
                "intent": "chat",
                "mistake_log": session.get("mistake_log", []),
                "speed": "normal",
            }

            # Run the graph WITHOUT TTS — text returns immediately, audio synthesized later (Issue #13)
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(graph_no_tts.invoke, state),
                    timeout=50.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Graph execution timed out for session %s (>=50s)", body.session_id)
                yield f'data: {{"type": "token", "content": {json.dumps("I\'m sorry, I took too long to respond. Please try sending your message again!")}}}\n\n'
                yield f'data: {{"type": "done", "intent": "chat"}}\n\n'
                return
            except Exception as exc:
                logger.exception("Graph execution failed for session %s", body.session_id)
                yield f'data: {{"type": "error", "message": {json.dumps("Our tutor is having a moment. Please try again.")}}}\n\n'
                yield f'data: {{"type": "done", "intent": "chat"}}\n\n'
                return

            # Extract final reply and intent
            final_reply = ""
            intent = result.get("intent", "chat")
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    final_reply = extract_text(msg.content)
                    break

            # Stream the reply word-by-word for a typing-effect UX
            words = final_reply.split(" ")
            for i, word in enumerate(words):
                prefix = "" if i == 0 else " "
                token = prefix + word
                yield f'data: {{"type": "token", "content": {json.dumps(token)}}}\n\n'
                if i % 3 == 0:
                    await asyncio.sleep(0.01)

            # Send completion event
            yield f'data: {{"type": "done", "intent": {json.dumps(intent)}}}\n\n'

            # Atomically save chat_history
            try:
                chat_history = _messages_to_dicts(result["messages"])
                save_chat_history_merge(body.session_id, chat_history)
                save_session(
                    body.session_id,
                    last_exercise=result.get("last_exercise", {}),
                    mistake_log=result.get("mistake_log"),
                )
            except Exception as exc:
                logger.exception("Failed to save session %s", body.session_id)
                raise DatabaseError() from exc
        finally:
            clear_session_context()

    gen = _event_generator()
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


class TTSRequest(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def content_must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Content must not be empty")
        if len(v) > 5000:
            raise ValueError("Content must be 5000 characters or fewer")
        return v.strip()


@app.post("/session/{session_id}/tts")
async def synthesize_session_audio(
    session_id: str,
    body: TTSRequest,
    user: dict = Depends(get_current_user),
) -> Response:
    """Synthesize speech for a specific assistant message (Issue #13).

    The frontend passes the message content in the request body so the backend
    can match the exact message and set audio_hash correctly, even if multiple
    concurrent TTS requests exist for the same session.

    Returns MP3 audio bytes directly. The audio is cached on disk so subsequent
    requests for the same text return instantly without calling the Gemini API.
    """
    session = _load_owned_session(session_id, user)

    # Use the content passed by the frontend directly — this eliminates the
    # race condition where a stale TTS request loads an updated session and
    # picks the wrong "last assistant" message.
    content = body.content

    try:
        result = synthesize_speech(
            content,
            session["language"],
        )
    except Exception as exc:
        logger.exception("TTS synthesis failed for session %s", session_id)
        raise TTSError() from exc

    if result is None:
        raise TTSError("Speech synthesis returned no audio data")

    audio_bytes, media_type = result

    # Compute the audio hash from the content (matches cache key)
    from .tts import _build_tts_text, _get_cache_path
    tts_text = _build_tts_text(content, session["language"], "normal")
    cache_path = _get_cache_path(tts_text)
    audio_hash = cache_path.stem  # e.g. "a1b2c3d4e5f6g7h8"

    # Atomically set audio_hash via a single SQLite transaction to prevent
    # the read-then-write race that happens when two concurrent TTS requests
    # try to update chat_history simultaneously. (Issue #13 race condition)
    if media_type == "audio/mpeg":
        try:
            set_audio_hash(session_id, audio_hash, content)
        except Exception as exc:
            logger.warning("TTS: Failed to persist audio_hash for session %s: %s", session_id, exc)
            # Non-fatal — audio still returned successfully

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/audio/{audio_hash}.mp3")
async def get_cached_audio(
    audio_hash: str,
) -> Response:
    """Serve a cached MP3 file by its hash.

    No auth required — the SHA-256 hash acts as an unguessable token.
    The frontend's <audio> element cannot send Authorization headers,
    so this endpoint must be publicly accessible.

    The frontend uses this to replay audio from previous responses without
    calling the TTS endpoint again. The audio_hash is stored in the session's
    chat_history (see /session/{id}/tts).

    Returns 404 if the audio file is not in cache (e.g. cache was cleared).
    """
    # Validate format: SHA-256 hex prefix (16 chars), no path separators
    if not audio_hash.isalnum() or len(audio_hash) != 16:
        raise HTTPException(status_code=404, detail="Audio not found in cache")

    cache_path = (AUDIO_CACHE_DIR / f"{audio_hash}.mp3").resolve()
    if not str(cache_path).startswith(str(AUDIO_CACHE_DIR)):
        raise HTTPException(status_code=404, detail="Audio not found in cache")

    if not cache_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found in cache")

    audio_bytes = cache_path.read_bytes()
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/session/{session_id}/mistakes")
async def get_mistakes(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Return the mistake log for a session (Issue #11)."""
    session = _load_owned_session(session_id, user)
    return session.get("mistake_log", [])


@app.patch("/session/{session_id}")
async def rename(
    session_id: str,
    body: RenameRequest,
    user: dict = Depends(get_current_user),
):
    """Rename a session (Issue #24)."""
    _load_owned_session(session_id, user)
    try:
        rename_session(session_id, body.title)
    except Exception as exc:
        logger.exception("Failed to rename session %s", session_id)
        raise DatabaseError() from exc
    return {"ok": True}


@app.delete("/session/{session_id}")
async def remove_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a session (Issue #24)."""
    _load_owned_session(session_id, user)
    try:
        delete_session(session_id)
    except Exception as exc:
        logger.exception("Failed to delete session %s", session_id)
        raise DatabaseError() from exc
    return {"ok": True}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/deps")
async def health_deps() -> dict[str, Any]:
    """Dependency health check — reports status of external services.

    Returns a JSON object with the status of each dependency.
    Used by monitoring / load balancers to determine if the service is ready.
    """
    status: dict[str, Any] = {"status": "ok", "dependencies": {}}

    # Check API keys
    status["dependencies"]["gemini_api_key"] = "configured" if os.getenv("GEMINI_API_KEY") else "missing"
    status["dependencies"]["google_embedding_api_key"] = "configured" if os.getenv("GOOGLE_EMBEDDING_API_KEY") else "not_set"

    # Check Pinecone connectivity
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if pinecone_key and _pinecone_index is not None:
        try:
            stats = _pinecone_index.describe_index_stats()
            status["dependencies"]["pinecone"] = "ok"
            status["dependencies"]["pinecone_vector_count"] = stats.get("total_vector_count", 0)
        except Exception as exc:
            status["dependencies"]["pinecone"] = "error"
            status["dependencies"]["pinecone_error"] = str(exc)
            status["status"] = "degraded"
    elif pinecone_key:
        status["dependencies"]["pinecone"] = "not_initialized"
        status["status"] = "degraded"
    else:
        status["dependencies"]["pinecone"] = "not_configured"

    return status