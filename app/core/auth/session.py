from __future__ import annotations

import hashlib
import secrets
from typing import Final

_SESSION_TOKEN_BYTES: Final[int] = 32


class SessionTokenService:
    """Generate and hash cryptographically secure session tokens."""

    def generate_token(self) -> str:
        """Generate a cryptographically secure opaque session token."""

        return secrets.token_urlsafe(_SESSION_TOKEN_BYTES)

    @staticmethod
    def hash_token(token: str) -> str:
        """Return the deterministic hash used for session lookup."""

        return hashlib.sha256(
            token.encode("utf-8"),
        ).hexdigest()
