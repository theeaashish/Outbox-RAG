from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, joinedload

from app.db.models.session import Session as SessionModel
from app.repositories.base import BaseRepository


class SessionRepository(BaseRepository[SessionModel]):
    """Repository for authenticated browser session persistence."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=SessionModel,
        )

    def get_active_by_token_hash(self, *, token_hash: str) -> SessionModel | None:
        """Return an active, non-expired session for a hashed token."""

        now = datetime.now(UTC)

        statement = select(SessionModel).where(
            SessionModel.token_hash == token_hash,
            SessionModel.revoked_at.is_(None),
            SessionModel.expires_at > now,
        )

        return self.db.scalar(statement)

    def get_active_with_user_by_token_hash(
        self,
        *,
        token_hash: str,
    ) -> SessionModel | None:
        """Return an active, non-expired session with its user eagerly loaded."""

        now = datetime.now(UTC)

        statement = (
            select(SessionModel)
            .options(joinedload(SessionModel.user))
            .where(
                SessionModel.token_hash == token_hash,
                SessionModel.revoked_at.is_(None),
                SessionModel.expires_at > now,
            )
        )

        return self.db.scalar(statement)

    def revoke(self, *, session_id: UUID) -> bool:
        """Revoke a single session."""

        statement = (
            update(SessionModel)
            .where(
                SessionModel.id == session_id,
                SessionModel.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(UTC))
        )

        result = cast(CursorResult[None], self.db.execute(statement))

        return result.rowcount == 1

    def revoke_all_for_user(
        self,
        *,
        user_id: UUID,
    ) -> int:
        """Revoke every active session belonging to a user."""

        statement = (
            update(SessionModel)
            .where(
                SessionModel.user_id == user_id,
                SessionModel.revoked_at.is_(None),
            )
            .values(
                revoked_at=datetime.now(UTC),
            )
        )

        result = cast(CursorResult[None], self.db.execute(statement))

        return result.rowcount
