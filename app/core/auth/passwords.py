from __future__ import annotations

from typing import Final

from argon2 import PasswordHasher
from argon2.exceptions import HashingError, InvalidHashError, VerificationError
from argon2.low_level import Type

_MAX_PASSWORD_BYTES: Final[int] = 1024


class PasswordHasherService:
    """Hash and verify passwords using Argon2id."""

    def __init__(
        self,
        *,
        time_cost: int = 3,
        memory_cost: int = 65536,
        parallelism: int = 4,
        hash_len: int = 32,
        salt_len: int = 16,
    ) -> None:
        self._hasher = PasswordHasher(
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=hash_len,
            salt_len=salt_len,
            type=Type.ID,
        )

    def hash(self, password: str) -> str:
        """Return an Argon2id password hash."""

        if not password:
            raise ValueError("Password cannot be empty")

        if len(password.encode("utf-8")) > _MAX_PASSWORD_BYTES:
            raise ValueError(
                f"Password exceeds maximum length of {_MAX_PASSWORD_BYTES} bytes"
            )

        try:
            return self._hasher.hash(password)
        except HashingError as exc:
            raise RuntimeError("Password hashing failed") from exc

    def verify(self, *, password: str, password_hash: str) -> bool:
        """Verify a plaintext password against a stored hash."""

        if not password or not password_hash:
            return False

        if len(password.encode("utf-8")) > _MAX_PASSWORD_BYTES:
            return False

        try:
            return self._hasher.verify(password_hash, password)
        except (VerificationError, InvalidHashError):
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        """Return whether the stored hash uses outdated parameters."""

        if not password_hash:
            return True

        try:
            return self._hasher.check_needs_rehash(password_hash)
        except (InvalidHashError, VerificationError):
            return True
