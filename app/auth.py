"""
JWT verification for NextAuth-issued tokens.

Verifies HS256 tokens using the NextAuth shared secret.
"""

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
    secret = auth_secret()
    if not secret:
        logger.error("AUTH_SECRET not configured — cannot verify tokens")
        raise InvalidTokenError("AUTH_SECRET not configured")

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_exp": True},
        )
        return payload
    except InvalidTokenError:
        pass

    # Log the failure (without leaking the token itself)
    logger.info("Token verification failed — invalid signature or algorithm (token length: %d)", len(token))
    raise InvalidTokenError("Token verification failed — invalid signature or algorithm")
