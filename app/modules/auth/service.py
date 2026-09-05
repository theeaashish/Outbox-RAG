from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.auth.passwords import PasswordHasherService
from app.core.auth.session import SessionTokenService
from app.core.exceptions import ConflictException, UnauthorizedException
from app.db.models.enums import UserStatus
from app.db.models.password_credential import PasswordCredential
from app.db.models.session import Session as SessionModel
from app.db.models.user import User
from app.repositories.password_credential import PasswordCredentialRepository
from app.repositories.session import SessionRepository
from app.repositories.user import UserRepository


@dataclass(frozen=True, slots=True)
class AuthenticatedSession:
    user: User
    token: str


logger = logging.getLogger(__name__)


_AUTHENTICATION_ERROR = "Invalid email or password"


class AuthService:
    """Application service for authentication use cases."""

    def __init__(
        self,
        *,
        db: Session,
        user_repository: UserRepository,
        password_credential_repository: PasswordCredentialRepository,
        session_repository: SessionRepository,
        password_hasher: PasswordHasherService,
        session_token_service: SessionTokenService,
    ) -> None:
        self._db = db
        self._user_repository = user_repository
        self._password_credential_repository = password_credential_repository
        self._session_repository = session_repository
        self._password_hasher = password_hasher
        self._session_token_service = session_token_service

    @staticmethod
    def _normalize_email(email: str) -> str:
        """Return the canonical representation used for email lookups."""

        return email.strip().casefold()

    def authenticate_session(self, *, token: str) -> User:
        """Authenticate a user from a raw session token."""

        token_hash = self._session_token_service.hash_token(token)

        session = self._session_repository.get_active_with_user_by_token_hash(
            token_hash=token_hash
        )

        if session is None:
            raise UnauthorizedException("Authentication required")

        return session.user

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

    def login(
        self,
        *,
        email: str,
        password: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
        session_lifetime: timedelta,
    ) -> AuthenticatedSession:
        """Authenticate a user and create a server-side session."""

        email_normalized = self._normalize_email(email=email)

        user = self._user_repository.get_by_email_normalized(
            email_normalized=email_normalized
        )

        credential = (
            self._password_credential_repository.get_by_user_id(
                user_id=user.id,
            )
            if user is not None
            else None
        )

        if user is None or credential is None:
            raise UnauthorizedException(_AUTHENTICATION_ERROR)

        password_valid = self._password_hasher.verify(
            password=password,
            password_hash=credential.password_hash,
        )

        if not password_valid:
            raise UnauthorizedException(_AUTHENTICATION_ERROR)

        if user.status != UserStatus.ACTIVE:
            raise UnauthorizedException(_AUTHENTICATION_ERROR)

        raw_token = self._session_token_service.generate_token()
        token_hash = self._session_token_service.hash_token(raw_token)

        now = datetime.now(UTC)

        session = SessionModel(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=now + session_lifetime,
            revoked_at=None,
            last_seen_at=now,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        self._session_repository.create(session)

        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        logger.info(
            "User logged in successfully",
            extra={"user_id": str(user.id)},
        )

        return AuthenticatedSession(
            user=user,
            token=raw_token,
        )

    def logout(self, *, session_id: UUID) -> None:
        """Revoke an active session by its ID.

        Revocation is intentionally idempotent: concurrent requests logging out the same
        session complete successfully even if the session was already revoked.
        """

        revoked = self._session_repository.revoke(session_id=session_id)

        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        logger.info(
            "Session logout processed",
            extra={
                "session_id": str(session_id),
                "revoked": revoked,
            },
        )

    def logout_all(self, *, user_id: UUID) -> int:
        """Revoke every active session belonging to a user.

        Revocation is intentionally idempotent: completes successfully even if no
        active sessions were found or revoked.
        """

        revoked_count = self._session_repository.revoke_all_for_user(user_id=user_id)

        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        logger.info(
            "All user sessions revoked successfully",
            extra={
                "user_id": str(user_id),
                "revoked_count": revoked_count,
            },
        )

        return revoked_count
