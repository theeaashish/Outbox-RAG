from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.conversation import Conversation
    from app.db.models.knowledge_base import KnowledgeBase
    from app.db.models.user import User


class Project(UUIDMixin, TimestampMixin, Base):
    """Product-level container for a user's knowledge base and conversations."""

    __tablename__ = "projects"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(
        back_populates="projects",
        lazy="raise",
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
        uselist=False,
        foreign_keys="[KnowledgeBase.project_id]",
    )

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
        foreign_keys="[Conversation.project_id]",
    )

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_projects_user_name"),
        # Supporting unique constraint for composite foreign keys (id is already primary key)
        UniqueConstraint("id", "user_id", name="uq_projects_id_user_id"),
        Index("ix_projects_user_id", "user_id"),
    )
