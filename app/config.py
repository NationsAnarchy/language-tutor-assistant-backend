"""
Shared configuration helpers for the Language Tutor Agent backend.

Avoids duplicating env-var resolution across modules.
"""
import os
from pathlib import Path


def auth_secret() -> str:
    """Return the shared JWT signing secret.

    ``AUTH_SECRET`` is the cross-service contract. ``NEXTAUTH_SECRET`` remains
    a temporary compatibility fallback for deployments that have not migrated.
    """
    return os.getenv("AUTH_SECRET", "") or os.getenv("NEXTAUTH_SECRET", "")


def cors_origins() -> list[str]:
    """Parse the explicitly allowed browser origins from the environment."""
    configured = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def data_dir() -> Path:
    """Return the persistent data directory.

    Uses RAILWAY_VOLUME_PATH on Railway (mounted volume), falls back to
    the local data/ directory relative to the project root.
    """
    volume_path = os.getenv("RAILWAY_VOLUME_PATH", "")
    if volume_path:
        return Path(volume_path)
    return Path(__file__).resolve().parent.parent / "data"
