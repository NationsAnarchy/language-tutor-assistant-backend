"""
JWT verification for NextAuth-issued tokens.

Supports both HS256 (shared secret) and RS256 (public key) verification.
For development, also accepts a X-Dev-User-Id header as a bypass
(handled in main.py, not here).
"""

import os
from typing import Any

import jwt
from jwt.exceptions import InvalidTokenError

from .logging_config import get_logger

logger = get_logger(__name__)

_DEV_MODE = os.getenv("NEXTAUTH_SECRET", "").startswith("dev-") or os.getenv("ENV") == "development"


def verify_token(token: str) -> dict[str, Any]:
    """Verify a JWT issued by NextAuth.

    Returns the decoded payload dict with at least 'sub' and 'email' keys.

    Raises InvalidTokenError if verification fails.
    """
    # In development mode with a simple shared secret, use HS256
    secret = os.getenv("NEXTAUTH_SECRET", "")
    if not secret:
        if _DEV_MODE:
            # For local development without NextAuth configured,
            # accept a self-signed dev token
            secret = "dev-secret-change-in-production"
        else:
            logger.error("NEXTAUTH_SECRET not configured — cannot verify tokens")
            raise InvalidTokenError("NEXTAUTH_SECRET not configured")

    try:
        # Try HS256 first (most common NextAuth config)
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_exp": True},
        )
        return payload
    except InvalidTokenError:
        pass

    # Try RS256 (if using public/private key pair)
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["RS256"],
            options={"verify_exp": True},
        )
        return payload
    except InvalidTokenError:
        pass

    # Log the failure (without leaking the token itself)
    logger.info("Token verification failed — invalid signature or algorithm (token length: %d)", len(token))
    raise InvalidTokenError("Token verification failed — invalid signature or algorithm")
