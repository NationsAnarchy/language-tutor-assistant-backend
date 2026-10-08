"""
JWT verification for NextAuth-issued tokens.

Verifies HS256 tokens using the NextAuth shared secret.
"""

import os
from typing import Any

import jwt
from jwt.exceptions import InvalidTokenError

from .logging_config import get_logger
from .config import auth_secret

logger = get_logger(__name__)


def verify_token(token: str) -> dict[str, Any]:
    """Verify a JWT issued by NextAuth.

    Returns the decoded payload dict with at least 'sub' and 'email' keys.

    Raises InvalidTokenError if verification fails.
    """
    if not os.getenv("AUTH_SECRET") and os.getenv("NEXTAUTH_SECRET"):
        logger.warning(
            "NEXTAUTH_SECRET is deprecated; please configure AUTH_SECRET instead.",
            extra={"event": "deprecated_auth_secret_used"},
        )

    secret = auth_secret()
    if not secret:
        logger.error("AUTH_SECRET not configured — cannot verify tokens")
        raise InvalidTokenError("AUTH_SECRET not configured")

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_exp": True, "require": ["exp", "sub"]},
        )
        sub = payload.get("sub")
        if not sub or not isinstance(sub, str) or not sub.strip():
            logger.info("Token verification failed — missing or empty 'sub' claim")
            raise InvalidTokenError("Token missing 'sub' claim")
        return payload
    except InvalidTokenError as exc:
        if "sub" in str(exc).lower():
            raise
        pass

    # Log the failure (without leaking the token itself)
    logger.info("Token verification failed — invalid signature or algorithm (token length: %d)", len(token))
    raise InvalidTokenError("Token verification failed — invalid signature or algorithm")

