from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, joinedload

from app.db.models.session import Session as SessionModel
from app.repositories.base import BaseRepository


class SessionRepository(BaseRepository[SessionModel]):
    """Repository for authenticated browser session persistence."""

    def __init__(self, *, db: Session, idle_timeout: timedelta) -> None:
        super().__init__(
            db=db,
            model=SessionModel,
        )
        self._idle_timeout = idle_timeout

    def _active_conditions(self, *, token_hash: str, now: datetime):
        """Return the shared conditions defining an active session."""

        idle_threshold = now - self._idle_timeout

        return (
            SessionModel.token_hash == token_hash,
            SessionModel.revoked_at.is_(None),
            SessionModel.expires_at > now,
            func.coalesce(SessionModel.last_seen_at, SessionModel.created_at)
            > idle_threshold,
        )

    def get_active_by_token_hash(self, *, token_hash: str) -> SessionModel | None:
        """Return an active, non-expired session for a hashed token."""

        now = datetime.now(UTC)

        statement = select(SessionModel).where(
            *self._active_conditions(token_hash=token_hash, now=now)
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
            .where(*self._active_conditions(token_hash=token_hash, now=now))
        )

        return self.db.scalar(statement)

    @property
    def idle_timeout(self) -> timedelta:
        """Return the configured idle timeout for sessions."""
        return self._idle_timeout

    def touch_if_stale(
        self,
        *,
        session_id: UUID,
        threshold: datetime,
        now: datetime,
    ) -> bool:
        """Advance activity for a non-revoked session when its timestamp is stale."""

        statement = (
            update(SessionModel)
            .where(
                SessionModel.id == session_id,
                SessionModel.revoked_at.is_(None),
                SessionModel.expires_at > now,
                or_(
                    SessionModel.last_seen_at.is_(None),
                    SessionModel.last_seen_at < threshold,
                ),
            )
            .values(last_seen_at=now)
        )

        result = cast(CursorResult[None], self.db.execute(statement))

        return result.rowcount == 1

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

    def delete_cleanup_batch(
        self, *, now: datetime, retention: timedelta, limit: int
    ) -> int:
        """Delete a bounded batch of sessions that have been dead beyond retention."""

        cutoff = now - retention
        idle_expired_cutoff = cutoff - self._idle_timeout

        eligible_ids = (
            select(SessionModel.id)
            .where(
                or_(
                    # Explicitly revoked long enough ago.
                    and_(
                        SessionModel.revoked_at.is_not(None),
                        SessionModel.revoked_at <= cutoff,
                    ),
                    # Naturally expired long enough ago.
                    and_(
                        SessionModel.revoked_at.is_(None),
                        SessionModel.expires_at <= cutoff,
                    ),
                    # Idle-expired long enough ago.
                    and_(
                        SessionModel.revoked_at.is_(None),
                        func.coalesce(
                            SessionModel.last_seen_at,
                            SessionModel.created_at,
                        )
                        <= idle_expired_cutoff,
                    ),
                )
            )
            .order_by(SessionModel.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
            .cte("cleanup_candidates")
        )

        statement = (
            delete(SessionModel)
            .where(SessionModel.id.in_(select(eligible_ids.c.id)))
            .execution_options(synchronize_session=False)
        )

        result = cast(CursorResult[None], self.db.execute(statement))

        return result.rowcount
