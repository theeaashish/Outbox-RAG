from __future__ import annotations

from uuid import UUID

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

    def get_by_user_and_id(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
    ) -> KnowledgeBase | None:
        """Retrieve a knowledge base by user and knowledge base ID."""
        statement = select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == user_id,
        )
        return self.db.scalar(statement)

    def get_by_user_and_project_id(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
    ) -> KnowledgeBase | None:
        """Retrieve the knowledge base associated with a user and project."""
        statement = select(KnowledgeBase).where(
            KnowledgeBase.project_id == project_id,
            KnowledgeBase.user_id == user_id,
        )
        return self.db.scalar(statement)

    def get_by_user_and_name(
        self,
        *,
        user_id: UUID,
        name: str,
    ) -> KnowledgeBase | None:
        """Get knowledge base by user and unique name."""
        statement = select(KnowledgeBase).where(
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.name == name,
        )
        return self.db.scalar(statement)

    def exists_by_user_and_name(
        self,
        *,
        user_id: UUID,
        name: str,
    ) -> bool:
        """Check if a knowledge base exists for a user by name."""
        statement = select(
            exists().where(
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.name == name,
            )
        )
        return bool(self.db.scalar(statement))
