from __future__ import annotations

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeBase
from app.repositories.base import BaseRepository


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    """Repository for KnowledgeBase model operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=KnowledgeBase,
        )

    def get_by_name(self, name: str) -> KnowledgeBase | None:
        """Get knowledge base by its unique name."""
        statement = select(KnowledgeBase).where(KnowledgeBase.name == name)
        return self.db.scalar(statement)

    def exists_by_name(self, name: str) -> bool:
        """Check if a knowledge base with the given name exists."""
        statement = select(exists().where(KnowledgeBase.name == name))
        return bool(self.db.scalar(statement))
