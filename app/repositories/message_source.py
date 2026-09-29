from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session

from app.db.models import MessageSource
from app.repositories.base import BaseRepository


class MessageSourceRepository(BaseRepository[MessageSource]):
    """Repository for MessageSource model operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=MessageSource,
        )

    def bulk_create(
        self,
        sources: Sequence[MessageSource],
    ) -> Sequence[MessageSource]:
        """Add multiple MessageSource records to the current session."""
        self.db.add_all(sources)
        return sources
