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
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from .exceptions import DatabaseError
from .logging_config import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "sessions.db"

# Ensure the data directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


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
            result = dict(row)
            # Deserialize JSON fields
            result["chat_history"] = json.loads(result["chat_history"])
            result["last_exercise"] = json.loads(result["last_exercise"])
            result["mistake_log"] = json.loads(result.get("mistake_log", "[]"))
            return result
        except sqlite3.Error as exc:
            logger.exception("Failed to load session %s: %s", session_id, exc)
            raise DatabaseError() from exc


def load_session_by_user_language(user_id: str, language: str, level: str | None = None) -> dict[str, Any] | None:
    """Load a session by user_id, language, and optionally level.

    Returns None if no matching session exists.
    Raises DatabaseError if the database operation fails.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")
    if language not in ("en", "ko", "ja"):
        raise ValueError(f"Invalid language: {language}")

    with _get_connection() as conn:
        try:
            if level:
                if level not in ("beginner", "intermediate", "advanced"):
                    raise ValueError(f"Invalid level: {level}")
                row = conn.execute(
                    "SELECT * FROM sessions WHERE user_id = ? AND language = ? AND level = ?",
                    (user_id, language, level),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE user_id = ? AND language = ?",
                    (user_id, language),
                ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["chat_history"] = json.loads(result["chat_history"])
            result["last_exercise"] = json.loads(result["last_exercise"])
            result["mistake_log"] = json.loads(result.get("mistake_log", "[]"))
            return result
        except sqlite3.Error as exc:
            logger.exception("Failed to load session for user %s, language %s: %s", user_id, language, exc)
            raise DatabaseError() from exc


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
    """Delete a session by ID and its associated audio files (Issue #28).

    Returns True if the session was deleted, False if it didn't exist.
    Raises DatabaseError if the database operation fails.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    with _get_connection() as conn:
        try:
            # Look up user_id first so we can clean up audio
            row = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                return False
            user_id = row["user_id"]

            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        except sqlite3.Error as exc:
            logger.exception("Failed to delete session %s: %s", session_id, exc)
            raise DatabaseError() from exc

    # Remove audio directory for this session (Issue #28)
    # Sanitize user_id to match TTS directory naming (Issue #29)
    try:
        from .tts import _user_dir
        user_dir = _user_dir(user_id)
        audio_dir = BASE_DIR / "audio" / user_dir / session_id
        if audio_dir.exists() and audio_dir.is_dir():
            shutil.rmtree(audio_dir)
    except Exception as exc:
        # Audio cleanup failure is non-critical — log but don't fail the deletion
        logger.warning("Failed to clean up audio for session %s: %s", session_id, exc)

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


# Initialize DB on import
init_db()