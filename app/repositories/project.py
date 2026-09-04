from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Project
from app.repositories.base import BaseRepository


class ProjectRepository(BaseRepository[Project]):
    """Repository for project persistence and user-scoped lookups."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(db=db, model=Project)

    def get_by_user_and_id(self, *, user_id: UUID, project_id: UUID) -> Project | None:
        statement = select(Project).where(
            Project.id == project_id,
            Project.user_id == user_id,
        )
        return self.db.scalar(statement)

    def list_by_user(
        self,
        *,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Project]:
        statement = (
            select(Project)
            .where(Project.user_id == user_id)
            .order_by(Project.created_at.desc(), Project.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return self.db.scalars(statement).all()
