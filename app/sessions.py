"""
SQLite session store for the Language Tutor Agent.

Each session is keyed by (user_id, language, level). A user can have up to 9 sessions
(one per language per difficulty level), each with independent chat history and progress.

Error handling:
  All database operations are wrapped in try/except. OperationalError (locked DB,
  corrupt file, etc.) is caught and re-raised as DatabaseError with a user-friendly
  message. Connections are always closed via context managers.
"""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from .config import data_dir
from .exceptions import DatabaseError
from .logging_config import get_logger
from .text_utils import strip_for_matching

logger = get_logger(__name__)

_DB_DIR = data_dir()
DB_PATH = _DB_DIR / "sessions.db"

# Ensure the data directory exists
_DB_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a SQLite connection and ensures it's closed.

    Raises DatabaseError if the connection fails.
    """
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
    except sqlite3.Error as exc:
        logger.exception("Database connection error: %s", exc)
        raise DatabaseError() from exc
    finally:
        if conn:
            conn.close()


def init_db() -> None:
    """Initialize the database schema if not already created."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                language TEXT NOT NULL CHECK(language IN ('en', 'ko', 'ja')),
                level TEXT NOT NULL DEFAULT 'beginner' CHECK(level IN ('beginner', 'intermediate', 'advanced')),
                title TEXT DEFAULT '',
                chat_history TEXT NOT NULL DEFAULT '[]',
                last_exercise TEXT DEFAULT '{}',
                mistake_log TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # Week 2 migration: add mistake_log column if missing (idempotent)
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN mistake_log TEXT NOT NULL DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass

        # Issue #2 migration: add title column if missing (idempotent)
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass

        conn.commit()


def create_session(user_id: str, language: str, level: str = "beginner") -> dict[str, Any]:
    """Always create a new session — no automatic resume (Issue #2).

    The frontend explicitly lists existing sessions and lets the user choose
    whether to resume an old session or start fresh.

    Raises:
        DatabaseError: If the database operation fails.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")
    if language not in ("en", "ko", "ja"):
        raise ValueError(f"Invalid language: {language}")
    if level not in ("beginner", "intermediate", "advanced"):
        raise ValueError(f"Invalid level: {level}")

    with _get_connection() as conn:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Generate a human-readable title: "ko-beginner-18072026"
        date_str = datetime.now(timezone.utc).strftime("%d%m%Y")
        title = f"{language}-{level}-{date_str}"

        try:
            conn.execute(
                """INSERT INTO sessions (session_id, user_id, language, level, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, user_id, language, level, title, now, now),
            )
            conn.commit()

            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            return dict(row)
        except sqlite3.Error as exc:
            logger.exception("Failed to create session for user %s: %s", user_id, exc)
            raise DatabaseError() from exc


def load_session(session_id: str) -> dict[str, Any] | None:
    """Load a session by its ID.

    Returns None if the session doesn't exist.
    Raises DatabaseError if the database operation fails.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    with _get_connection() as conn:
        try:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            return _deserialize_row(dict(row))
        except sqlite3.Error as exc:
            logger.exception("Failed to load session %s: %s", session_id, exc)
            raise DatabaseError() from exc


def _deserialize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Deserialize JSON fields in a session row."""
    row["chat_history"] = json.loads(row["chat_history"])
    row["last_exercise"] = json.loads(row["last_exercise"])
    row["mistake_log"] = json.loads(row.get("mistake_log", "[]"))
    return row


def list_sessions(user_id: str) -> list[dict[str, Any]]:
    """List all sessions for a user.

    Raises DatabaseError if the database operation fails.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")

    with _get_connection() as conn:
        try:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY language, level",
                (user_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            logger.exception("Failed to list sessions for user %s: %s", user_id, exc)
            raise DatabaseError() from exc


def save_turn(
    session_id: str,
    chat_history: list[dict[str, Any]],
    last_exercise: dict,
    mistake_log: list,
) -> bool:
    """Persist one completed chat turn in a single transaction.

    Uses BEGIN IMMEDIATE to serialize with concurrent set_audio_hash() calls.
    Reads the current chat_history from DB, copies any audio_hash values to the
    corresponding messages in the new chat_history, then writes the merged history,
    exercise state, and mistakes together.

    Args:
        session_id: The session ID.
        chat_history: The new chat history, without transient tool messages.
        last_exercise: Exercise state after the turn.
        mistake_log: Complete mistake log after the turn.

    Returns:
        True on success, raises DatabaseError on failure.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    with _get_connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")

            # Read the current DB state — may have audio_hash from concurrent TTS
            row = conn.execute(
                "SELECT chat_history FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if row is None:
                conn.rollback()
                return False

            db_history: list = json.loads(row["chat_history"])

            # Build index: normalized_content → audio_hash
            db_hashes: dict[str, str] = {}
            for msg in db_history:
                if msg.get("audio_hash"):
                    key = " ".join(msg.get("content", "").split())
                    if key:
                        db_hashes[key] = msg["audio_hash"]

            # Merge: copy audio_hash from DB to the new history
            for entry in chat_history:
                if entry.get("role") == "assistant":
                    key = " ".join(entry.get("content", "").split())
                    if key in db_hashes:
                        entry["audio_hash"] = db_hashes[key]

            conn.execute(
                """UPDATE sessions
                   SET chat_history = ?, last_exercise = ?, mistake_log = ?, updated_at = ?
                   WHERE session_id = ?""",
                (json.dumps(chat_history, ensure_ascii=False),
                 json.dumps(last_exercise, ensure_ascii=False),
                 json.dumps(mistake_log, ensure_ascii=False),
                 datetime.now(timezone.utc).isoformat(),
                 session_id),
            )
            conn.commit()
            return True
        except sqlite3.Error as exc:
            conn.rollback()
            logger.exception("Failed to save chat turn for session %s: %s", session_id, exc)
            raise DatabaseError() from exc


def save_chat_history_merge(session_id: str, chat_history: list[dict[str, Any]]) -> bool:
    """Backward-compatible history-only persistence helper.

    New chat requests should use :func:`save_turn` so all turn state commits
    atomically. This helper remains for callers that only update history.
    """
    session = load_session(session_id)
    if session is None:
        return False
    return save_turn(
        session_id,
        chat_history,
        session.get("last_exercise", {}),
        session.get("mistake_log", []),
    )


def save_session(
    session_id: str,
    chat_history: list | None = None,
    last_exercise: dict | None = None,
    level: str | None = None,
    mistake_log: list | None = None,
) -> bool:
    """Update session state. Only provided fields are updated.

    Returns True if the session was updated, False if no fields were provided.
    Raises DatabaseError if the database operation fails.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    with _get_connection() as conn:
        updates = []
        params = []

        if chat_history is not None:
            updates.append("chat_history = ?")
            params.append(json.dumps(chat_history, ensure_ascii=False))

        if last_exercise is not None:
            updates.append("last_exercise = ?")
            params.append(json.dumps(last_exercise, ensure_ascii=False))

        if level is not None:
            if level not in ("beginner", "intermediate", "advanced"):
                raise ValueError(f"Invalid level: {level}")
            updates.append("level = ?")
            params.append(level)

        if mistake_log is not None:
            updates.append("mistake_log = ?")
            params.append(json.dumps(mistake_log, ensure_ascii=False))

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(session_id)

        try:
            conn.execute(
                f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?",
                params,
            )
            conn.commit()
            return True
        except sqlite3.Error as exc:
            logger.exception("Failed to save session %s: %s", session_id, exc)
            raise DatabaseError() from exc


def set_audio_hash(session_id: str, audio_hash: str, content_to_match: str) -> bool:
    """Atomically set audio_hash on the assistant message whose content matches.

    Uses a single SQLite transaction (load + modify + save) to prevent
    the read-then-write race that happens when two concurrent TTS requests
    try to update chat_history simultaneously.

    Matches by stripped content, NOT by position. This ensures TTS A tags
    agent_A and TTS B tags agent_B even if they both loaded different snapshots
    of the session.

    Args:
        session_id: The session ID.
        audio_hash: The SHA-256 hash to store (e.g. "a1b2c3d4e5f6g7h8").
        content_to_match: The raw content text to match against assistant
                         messages (will be stripped for comparison).

    Returns:
        True if the hash was set, False if no matching message was found
        or the hash was already set.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")
    if not audio_hash:
        raise ValueError("audio_hash must not be empty")
    if not content_to_match:
        raise ValueError("content_to_match must not be empty")

    # Normalize the content_to_match once for matching
    target = strip_for_matching(content_to_match)

    with _get_connection() as conn:
        try:
            # Use IMMEDIATE transaction to serialize concurrent writes.
            # This prevents the read-then-write race: if two TTS requests
            # arrive simultaneously, one waits for the other to complete.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT chat_history FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if row is None:
                conn.rollback()
                return False

            chat_history = json.loads(row["chat_history"])

            # If this hash is already stored on any message, no-op
            for msg in reversed(chat_history):
                if msg.get("role") == "assistant" and msg.get("audio_hash") == audio_hash:
                    conn.rollback()
                    return True

            # Find the assistant message whose stripped content matches,
            # and which doesn't already have an audio_hash.
            # Match by content, NOT by position — a stale TTS request that
            # loaded an old session snapshot could see a different "last"
            # assistant message.
            found = False
            for i in range(len(chat_history) - 1, -1, -1):
                msg = chat_history[i]
                if msg.get("role") != "assistant":
                    continue
                if msg.get("audio_hash"):
                    continue  # Already has audio — skip
                msg_stripped = strip_for_matching(msg.get("content", ""))
                if msg_stripped == target:
                    chat_history[i]["audio_hash"] = audio_hash
                    found = True
                    break

            if not found:
                conn.rollback()
                return False

            conn.execute(
                "UPDATE sessions SET chat_history = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(chat_history, ensure_ascii=False),
                 datetime.now(timezone.utc).isoformat(),
                 session_id),
            )
            conn.commit()
            return True
        except sqlite3.Error as exc:
            conn.rollback()
            logger.exception("Failed to set audio_hash for session %s: %s", session_id, exc)
            raise DatabaseError() from exc


def add_mistake(session_id: str, mistake_type: str, detail: str) -> None:
    """Append a mistake entry to the session's mistake_log.

    Args:
        session_id: The session ID.
        mistake_type: e.g. 'grammar', 'vocabulary', 'pronunciation', 'spelling'.
        detail: The specific mistake description (what the user said vs. correct form).

    Raises DatabaseError if the database operation fails.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")
    if mistake_type not in ("grammar", "vocabulary", "pronunciation", "spelling"):
        raise ValueError(f"Invalid mistake_type: {mistake_type}")

    with _get_connection() as conn:
        try:
            row = conn.execute(
                "SELECT mistake_log FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if row is None:
                return

            mistake_log = json.loads(row["mistake_log"] or "[]")
            mistake_log.append({
                "type": mistake_type,
                "detail": detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            conn.execute(
                "UPDATE sessions SET mistake_log = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(mistake_log, ensure_ascii=False), datetime.now(timezone.utc).isoformat(), session_id),
            )
            conn.commit()
        except sqlite3.Error as exc:
            logger.exception("Failed to add mistake for session %s: %s", session_id, exc)
            raise DatabaseError() from exc


def delete_session(session_id: str) -> bool:
    """Delete a session by ID.

    Returns True if the session was deleted, False if it didn't exist.
    Raises DatabaseError if the database operation fails.

    Note: Audio files are no longer stored on disk (Issue #43), so no
    audio cleanup is needed here.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    with _get_connection() as conn:
        try:
            row = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                return False

            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        except sqlite3.Error as exc:
            logger.exception("Failed to delete session %s: %s", session_id, exc)
            raise DatabaseError() from exc

    return True


def rename_session(session_id: str, title: str) -> bool:
    """Rename a session's title.

    Returns True if the session was renamed, False if it didn't exist.
    Raises DatabaseError if the database operation fails.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")
    if not title or not title.strip():
        raise ValueError("title must not be empty")

    with _get_connection() as conn:
        try:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title.strip(), datetime.now(timezone.utc).isoformat(), session_id),
            )
            conn.commit()
            changes = conn.total_changes
            return changes > 0
        except sqlite3.Error as exc:
            logger.exception("Failed to rename session %s: %s", session_id, exc)
            raise DatabaseError() from exc

