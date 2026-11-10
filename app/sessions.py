"""
Dual-mode (PostgreSQL / SQLite) session store for the Language Tutor Agent.

When DATABASE_URL is set (e.g. Neon PostgreSQL in production / Cloud Run),
sessions and chat history are persisted in PostgreSQL.
When DATABASE_URL is absent (local development & testing), it automatically
falls back to local SQLite (data/sessions.db).

Each session is keyed by (user_id, language, level). A user can have up to 9 sessions
(one per language per difficulty level), each with independent chat history and progress.
"""

import json
import os
import sqlite3
import threading
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
_DB_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PostgreSQL Connection Pool (Neon / Cloud Run)
# ---------------------------------------------------------------------------

_pg_pool: Any = None
_pg_pool_lock = threading.Lock()


def is_postgres() -> bool:
    """Return True if DATABASE_URL is configured for PostgreSQL."""
    db_url = os.getenv("DATABASE_URL", "").strip()
    return bool(db_url and (db_url.startswith("postgres://") or db_url.startswith("postgresql://")))


def _normalize_db_url(url: str) -> str:
    """Ensure standard postgresql:// prefix for psycopg."""
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


def _get_pg_pool() -> Any:
    """Lazy-initialize thread-safe psycopg ConnectionPool."""
    global _pg_pool
    if _pg_pool is None:
        with _pg_pool_lock:
            if _pg_pool is None:
                try:
                    from psycopg_pool import ConnectionPool
                except ImportError as exc:
                    raise DatabaseError("psycopg and psycopg-pool must be installed to use PostgreSQL") from exc
                db_url = _normalize_db_url(os.getenv("DATABASE_URL", "").strip())
                _pg_pool = ConnectionPool(
                    conninfo=db_url,
                    min_size=1,
                    max_size=int(os.getenv("PG_POOL_MAX_SIZE", "10")),
                    timeout=10.0,
                    open=True,
                )
    return _pg_pool


@contextmanager
def _get_pg_connection() -> Generator[Any, None, None]:
    """Yield a connection from the Postgres pool with automatic commit/rollback."""
    pool = _get_pg_pool()
    conn = None
    try:
        with pool.connection() as conn:
            yield conn
    except Exception as exc:
        logger.exception("PostgreSQL connection error: %s", exc)
        raise DatabaseError() from exc


# ---------------------------------------------------------------------------
# SQLite Connection (Local Development / Testing)
# ---------------------------------------------------------------------------


@contextmanager
def _get_sqlite_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection and ensure it is properly closed."""
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
    except sqlite3.Error as exc:
        logger.exception("SQLite connection error: %s", exc)
        raise DatabaseError() from exc
    finally:
        if conn:
            conn.close()


def _get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Backward compatibility helper for SQLite connection."""
    return _get_sqlite_connection()


# ---------------------------------------------------------------------------
# Schema Initialization
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Initialize the database schema in either PostgreSQL or SQLite."""
    if is_postgres():
        with _get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        language VARCHAR(10) NOT NULL,
                        level VARCHAR(20) NOT NULL DEFAULT 'beginner',
                        title TEXT DEFAULT '',
                        chat_history TEXT NOT NULL DEFAULT '[]',
                        last_exercise TEXT DEFAULT '{}',
                        mistake_log TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
                """)
            conn.commit()
            logger.info("Initialized PostgreSQL schema in Neon")
    else:
        with _get_sqlite_connection() as conn:
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
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN mistake_log TEXT NOT NULL DEFAULT '[]'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
            conn.commit()
            logger.info("Initialized SQLite schema at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Session CRUD Operations
# ---------------------------------------------------------------------------


def _deserialize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Deserialize JSON fields in a session row."""
    row["chat_history"] = json.loads(row["chat_history"])
    row["last_exercise"] = json.loads(row["last_exercise"])
    row["mistake_log"] = json.loads(row.get("mistake_log", "[]"))
    return row


def create_session(user_id: str, language: str, level: str = "beginner") -> dict[str, Any]:
    """Create a new session."""
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")
    if language not in ("en", "ko", "ja"):
        raise ValueError(f"Invalid language: {language}")
    if level not in ("beginner", "intermediate", "advanced"):
        raise ValueError(f"Invalid level: {level}")

    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    date_str = datetime.now(timezone.utc).strftime("%d%m%Y")
    title = f"{language}-{level}-{date_str}"

    if is_postgres():
        from psycopg.rows import dict_row
        with _get_pg_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                try:
                    cur.execute(
                        """INSERT INTO sessions (session_id, user_id, language, level, title, created_at, updated_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (session_id, user_id, language, level, title, now, now),
                    )
                    conn.commit()
                    cur.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
                    row = cur.fetchone()
                    return dict(row)
                except Exception as exc:
                    conn.rollback()
                    logger.exception("Failed to create session for user %s in Postgres: %s", user_id, exc)
                    raise DatabaseError() from exc
    else:
        with _get_sqlite_connection() as conn:
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
                logger.exception("Failed to create session for user %s in SQLite: %s", user_id, exc)
                raise DatabaseError() from exc


def load_session(session_id: str) -> dict[str, Any] | None:
    """Load a session by its ID."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    if is_postgres():
        from psycopg.rows import dict_row
        with _get_pg_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                try:
                    cur.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
                    row = cur.fetchone()
                    if row is None:
                        return None
                    return _deserialize_row(dict(row))
                except Exception as exc:
                    logger.exception("Failed to load session %s from Postgres: %s", session_id, exc)
                    raise DatabaseError() from exc
    else:
        with _get_sqlite_connection() as conn:
            try:
                row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
                if row is None:
                    return None
                return _deserialize_row(dict(row))
            except sqlite3.Error as exc:
                logger.exception("Failed to load session %s from SQLite: %s", session_id, exc)
                raise DatabaseError() from exc


def list_sessions(user_id: str) -> list[dict[str, Any]]:
    """List all sessions for a user."""
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")

    if is_postgres():
        from psycopg.rows import dict_row
        with _get_pg_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                try:
                    cur.execute(
                        "SELECT session_id, user_id, language, level, title, mistake_log, created_at, updated_at "
                        "FROM sessions WHERE user_id = %s ORDER BY language, level",
                        (user_id,),
                    )
                    rows = cur.fetchall()
                    result = []
                    for row in rows:
                        r = dict(row)
                        try:
                            m_list = json.loads(r.get("mistake_log") or "[]")
                            m_count = len(m_list) if isinstance(m_list, list) else 0
                        except Exception:
                            m_count = 0
                        r["mistake_count"] = m_count
                        r.pop("mistake_log", None)
                        result.append(r)
                    return result
                except Exception as exc:
                    logger.exception("Failed to list sessions for user %s from Postgres: %s", user_id, exc)
                    raise DatabaseError() from exc
    else:
        with _get_sqlite_connection() as conn:
            try:
                rows = conn.execute(
                    "SELECT session_id, user_id, language, level, title, "
                    "COALESCE(CASE WHEN json_valid(mistake_log) THEN json_array_length(mistake_log) ELSE 0 END, 0) AS mistake_count, "
                    "created_at, updated_at "
                    "FROM sessions WHERE user_id = ? ORDER BY language, level",
                    (user_id,),
                ).fetchall()
                return [dict(row) for row in rows]
            except sqlite3.Error as exc:
                logger.exception("Failed to list sessions for user %s from SQLite: %s", user_id, exc)
                raise DatabaseError() from exc


def save_turn(
    session_id: str,
    chat_history: list[dict[str, Any]],
    last_exercise: dict,
    mistake_log: list,
) -> bool:
    """Persist one completed chat turn in a single transaction."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    if is_postgres():
        from psycopg.rows import dict_row
        with _get_pg_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                try:
                    # Row-level lock protects against concurrent updates
                    cur.execute("SELECT chat_history FROM sessions WHERE session_id = %s FOR UPDATE", (session_id,))
                    row = cur.fetchone()
                    if row is None:
                        conn.rollback()
                        return False

                    db_history: list = json.loads(row["chat_history"])
                    db_hashes: dict[str, str] = {
                        " ".join(msg.get("content", "").split()): msg["audio_hash"]
                        for msg in db_history
                        if msg.get("audio_hash") and " ".join(msg.get("content", "").split())
                    }

                    for entry in chat_history:
                        if entry.get("role") == "assistant":
                            key = " ".join(entry.get("content", "").split())
                            if key in db_hashes:
                                entry["audio_hash"] = db_hashes[key]

                    cur.execute(
                        """UPDATE sessions
                           SET chat_history = %s, last_exercise = %s, mistake_log = %s, updated_at = %s
                           WHERE session_id = %s""",
                        (json.dumps(chat_history, ensure_ascii=False),
                         json.dumps(last_exercise, ensure_ascii=False),
                         json.dumps(mistake_log, ensure_ascii=False),
                         datetime.now(timezone.utc).isoformat(),
                         session_id),
                    )
                    conn.commit()
                    return True
                except Exception as exc:
                    conn.rollback()
                    logger.exception("Failed to save chat turn for session %s in Postgres: %s", session_id, exc)
                    raise DatabaseError() from exc
    else:
        with _get_sqlite_connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT chat_history FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

                if row is None:
                    conn.rollback()
                    return False

                db_history = json.loads(row["chat_history"])
                db_hashes = {
                    " ".join(msg.get("content", "").split()): msg["audio_hash"]
                    for msg in db_history
                    if msg.get("audio_hash") and " ".join(msg.get("content", "").split())
                }

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
                logger.exception("Failed to save chat turn for session %s in SQLite: %s", session_id, exc)
                raise DatabaseError() from exc


def save_chat_history_merge(session_id: str, chat_history: list[dict[str, Any]]) -> bool:
    """Backward-compatible history-only persistence helper."""
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
    """Update session state. Only provided fields are updated."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    if is_postgres():
        updates = []
        params = []
        if chat_history is not None:
            updates.append("chat_history = %s")
            params.append(json.dumps(chat_history, ensure_ascii=False))
        if last_exercise is not None:
            updates.append("last_exercise = %s")
            params.append(json.dumps(last_exercise, ensure_ascii=False))
        if level is not None:
            if level not in ("beginner", "intermediate", "advanced"):
                raise ValueError(f"Invalid level: {level}")
            updates.append("level = %s")
            params.append(level)
        if mistake_log is not None:
            updates.append("mistake_log = %s")
            params.append(json.dumps(mistake_log, ensure_ascii=False))

        if not updates:
            return False

        updates.append("updated_at = %s")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(session_id)

        with _get_pg_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = %s", params)
                    conn.commit()
                    return True
                except Exception as exc:
                    conn.rollback()
                    logger.exception("Failed to save session %s in Postgres: %s", session_id, exc)
                    raise DatabaseError() from exc
    else:
        with _get_sqlite_connection() as conn:
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
                conn.execute(f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?", params)
                conn.commit()
                return True
            except sqlite3.Error as exc:
                logger.exception("Failed to save session %s in SQLite: %s", session_id, exc)
                raise DatabaseError() from exc


def set_audio_hash(session_id: str, audio_hash: str, content_to_match: str) -> bool:
    """Atomically set audio_hash on the matching assistant message."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")
    if not audio_hash:
        raise ValueError("audio_hash must not be empty")
    if not content_to_match:
        raise ValueError("content_to_match must not be empty")

    target = strip_for_matching(content_to_match)

    if is_postgres():
        from psycopg.rows import dict_row
        with _get_pg_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                try:
                    cur.execute("SELECT chat_history FROM sessions WHERE session_id = %s FOR UPDATE", (session_id,))
                    row = cur.fetchone()
                    if row is None:
                        conn.rollback()
                        return False

                    chat_history = json.loads(row["chat_history"])
                    for msg in reversed(chat_history):
                        if msg.get("role") == "assistant" and msg.get("audio_hash") == audio_hash:
                            conn.rollback()
                            return True

                    found = False
                    for i in range(len(chat_history) - 1, -1, -1):
                        msg = chat_history[i]
                        if msg.get("role") != "assistant" or msg.get("audio_hash"):
                            continue
                        if strip_for_matching(msg.get("content", "")) == target:
                            chat_history[i]["audio_hash"] = audio_hash
                            found = True
                            break

                    if not found:
                        conn.rollback()
                        return False

                    cur.execute(
                        "UPDATE sessions SET chat_history = %s, updated_at = %s WHERE session_id = %s",
                        (json.dumps(chat_history, ensure_ascii=False),
                         datetime.now(timezone.utc).isoformat(),
                         session_id),
                    )
                    conn.commit()
                    return True
                except Exception as exc:
                    conn.rollback()
                    logger.exception("Failed to set audio_hash for session %s in Postgres: %s", session_id, exc)
                    raise DatabaseError() from exc
    else:
        with _get_sqlite_connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT chat_history FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

                if row is None:
                    conn.rollback()
                    return False

                chat_history = json.loads(row["chat_history"])
                for msg in reversed(chat_history):
                    if msg.get("role") == "assistant" and msg.get("audio_hash") == audio_hash:
                        conn.rollback()
                        return True

                found = False
                for i in range(len(chat_history) - 1, -1, -1):
                    msg = chat_history[i]
                    if msg.get("role") != "assistant" or msg.get("audio_hash"):
                        continue
                    if strip_for_matching(msg.get("content", "")) == target:
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
                logger.exception("Failed to set audio_hash for session %s in SQLite: %s", session_id, exc)
                raise DatabaseError() from exc


def add_mistake(session_id: str, mistake_type: str, detail: str) -> None:
    """Append a mistake entry to the session's mistake_log."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")
    if mistake_type not in ("grammar", "vocabulary", "pronunciation", "spelling"):
        raise ValueError(f"Invalid mistake_type: {mistake_type}")

    if is_postgres():
        from psycopg.rows import dict_row
        with _get_pg_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                try:
                    cur.execute("SELECT mistake_log FROM sessions WHERE session_id = %s FOR UPDATE", (session_id,))
                    row = cur.fetchone()
                    if row is None:
                        conn.rollback()
                        return
                    mistake_log = json.loads(row["mistake_log"] or "[]")
                    mistake_log.append({
                        "type": mistake_type,
                        "detail": detail,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    cur.execute(
                        "UPDATE sessions SET mistake_log = %s, updated_at = %s WHERE session_id = %s",
                        (json.dumps(mistake_log, ensure_ascii=False), datetime.now(timezone.utc).isoformat(), session_id),
                    )
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    logger.exception("Failed to add mistake for session %s in Postgres: %s", session_id, exc)
                    raise DatabaseError() from exc
    else:
        with _get_sqlite_connection() as conn:
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
                logger.exception("Failed to add mistake for session %s in SQLite: %s", session_id, exc)
                raise DatabaseError() from exc


def delete_session(session_id: str) -> bool:
    """Delete a session by ID."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    if is_postgres():
        with _get_pg_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
                    deleted = cur.rowcount > 0
                    conn.commit()
                    return deleted
                except Exception as exc:
                    conn.rollback()
                    logger.exception("Failed to delete session %s in Postgres: %s", session_id, exc)
                    raise DatabaseError() from exc
    else:
        with _get_sqlite_connection() as conn:
            try:
                row = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
                if row is None:
                    return False
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                return True
            except sqlite3.Error as exc:
                logger.exception("Failed to delete session %s in SQLite: %s", session_id, exc)
                raise DatabaseError() from exc


def rename_session(session_id: str, title: str) -> bool:
    """Rename a session's title."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")
    if not title or not title.strip():
        raise ValueError("title must not be empty")

    if is_postgres():
        with _get_pg_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "UPDATE sessions SET title = %s, updated_at = %s WHERE session_id = %s",
                        (title.strip(), datetime.now(timezone.utc).isoformat(), session_id),
                    )
                    renamed = cur.rowcount > 0
                    conn.commit()
                    return renamed
                except Exception as exc:
                    conn.rollback()
                    logger.exception("Failed to rename session %s in Postgres: %s", session_id, exc)
                    raise DatabaseError() from exc
    else:
        with _get_sqlite_connection() as conn:
            try:
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title.strip(), datetime.now(timezone.utc).isoformat(), session_id),
                )
                conn.commit()
                return conn.total_changes > 0
            except sqlite3.Error as exc:
                logger.exception("Failed to rename session %s in SQLite: %s", session_id, exc)
                raise DatabaseError() from exc
