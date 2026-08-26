from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """Repository for User persistence and lookup operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(db=db, model=User)

    def get_by_email_normalized(self, *, email_normalized: str) -> User | None:
        """Retrieve a user by their normalized email address."""

        statement = select(User).where(
            User.email_normalized == email_normalized,
        )

        return self.db.scalar(statement)
