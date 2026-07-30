"""
Shared configuration helpers for the Language Tutor Agent backend.

Avoids duplicating env-var resolution across modules.
"""
import os
from pathlib import Path


def data_dir() -> Path:
    """Return the persistent data directory.

    Uses RAILWAY_VOLUME_PATH on Railway (mounted volume), falls back to
    the local data/ directory relative to the project root.
    """
    volume_path = os.getenv("RAILWAY_VOLUME_PATH", "")
    if volume_path:
        return Path(volume_path)
    return Path(__file__).resolve().parent.parent / "data"