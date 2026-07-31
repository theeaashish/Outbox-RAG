from __future__ import annotations

from uuid import UUID

from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from app.db.base import UUIDMixin


class BaseRepository[ModelT: UUIDMixin]:
    """Base repository class for generic database operations."""

    def __init__(
        self,
        *,
        db: Session,
        model: type[ModelT],
    ) -> None:
        self.db = db
        self.model = model

    def create(self, instance: ModelT) -> ModelT:
        """Add an entity to the current session transaction."""
        self.db.add(instance)
        return instance

    def get_by_id(self, entity_id: UUID) -> ModelT | None:
        """Retrieve an entity by its primary key."""
        statement = select(self.model).where(self.model.id == entity_id)
        return self.db.scalar(statement)

    def delete(self, instance: ModelT) -> None:
        """Remove an entity from the current session transaction."""
        self.db.delete(instance)

    def exists(self, entity_id: UUID) -> bool:
        """Check whether an entity exists by primary key."""
        statement = select(exists().where(self.model.id == entity_id))
        return bool(self.db.scalar(statement))

    def count(self) -> int:
        """Return the total number of persisted entities."""
        statement = select(func.count()).select_from(self.model)
        return int(self.db.scalar(statement) or 0)

    def flush(self) -> None:
        """Flush pending changes in the session to the database."""
        self.db.flush()

    def refresh(self, instance: ModelT) -> None:
        """Refresh the attributes of an entity from the database."""
        self.db.refresh(instance)
