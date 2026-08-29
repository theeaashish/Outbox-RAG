from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.auth.passwords import PasswordHasherService
from app.core.exceptions import ConflictException
from app.db.models.password_credential import PasswordCredential
from app.db.models.user import User
from app.repositories.password_credential import PasswordCredentialRepository
from app.repositories.user import UserRepository

logger = logging.getLogger(__name__)


class AuthService:
    """Application service for authentication use cases."""

    def __init__(
        self,
        *,
        db: Session,
        user_repository: UserRepository,
        password_credential_repository: PasswordCredentialRepository,
        password_hasher: PasswordHasherService,
    ) -> None:
        self._db = db
        self._user_repository = user_repository
        self._password_credential_repository = password_credential_repository
        self._password_hasher = password_hasher

    @staticmethod
    def _normalize_email(email: str) -> str:
        """Return the canonical representation used for email lookups."""

        return email.strip().casefold()

    def register(
        self,
        *,
        email: str,
        password: str,
        name: str | None = None,
    ) -> User:
        """Register a local user with a password credential."""

        email_normalized = self._normalize_email(email=email)

        existing_user = self._user_repository.get_by_email_normalized(
            email_normalized=email_normalized
        )

        if existing_user is not None:
            raise ConflictException("An account with this email already exists.")

        password_hash = self._password_hasher.hash(password=password)

        user = User(
            email=email.strip(),
            email_normalized=email_normalized,
            name=name.strip() if name else None,
        )

        password_credential = PasswordCredential(
            user=user,
            password_hash=password_hash,
            password_changed_at=datetime.now(UTC),
        )

        self._db.add(user)
        self._db.add(password_credential)

        try:
            self._db.commit()
        except IntegrityError as exc:
            self._db.rollback()

            logger.info(
                "User registration conflicted with an existing account",
                extra={"email_normalized": email_normalized},
            )

            raise ConflictException(
                "An account with this email already exists."
            ) from exc

        except Exception:
            self._db.rollback()
            raise

        self._db.refresh(user)

        logger.info(
            "User registered successfully",
            extra={"user_id": str(user.id)},
        )

        return user
