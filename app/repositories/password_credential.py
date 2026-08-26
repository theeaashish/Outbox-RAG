from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.password_credential import PasswordCredential
from app.repositories.base import BaseRepository


class PasswordCredentialRepository(BaseRepository[PasswordCredential]):
    """Repository for password credential persistence."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=PasswordCredential,
        )

    def get_by_user_id(
        self,
        *,
        user_id: UUID,
    ) -> PasswordCredential | None:
        """Retrieve the password credential belonging to a user."""

        statement = select(PasswordCredential).where(
            PasswordCredential.user_id == user_id,
        )

        return self.db.scalar(statement)
