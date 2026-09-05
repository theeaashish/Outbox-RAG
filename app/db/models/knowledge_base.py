from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.conversation import Conversation
    from app.db.models.document import Document
    from app.db.models.project import Project
    from app.db.models.user import User


class KnowledgeBase(UUIDMixin, TimestampMixin, Base):
    """Knowledge Base model."""

    __tablename__ = "knowledge_bases"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    project_id: Mapped[UUID] = mapped_column(
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped[User] = relationship(
        back_populates="knowledge_bases",
        lazy="raise",
    )

    project: Mapped[Project] = relationship(
        back_populates="knowledge_base",
        foreign_keys=[project_id],
        lazy="raise",
    )

    documents: Mapped[list[Document]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        lazy="select",
    )

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        lazy="select",
        foreign_keys="[Conversation.knowledge_base_id]",
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "user_id"],
            ["projects.id", "projects.user_id"],
            name="fk_knowledge_bases_project_id_user_id_projects",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "user_id",
            "name",
            name="uq_knowledge_bases_user_name",
        ),
        UniqueConstraint(
            "project_id",
            name="uq_knowledge_bases_project_id",
        ),
        # Supporting unique constraint for Conversation composite foreign key (id is already primary key)
        UniqueConstraint(
            "id",
            "project_id",
            name="uq_knowledge_bases_id_project_id",
        ),
        Index(
            "ix_knowledge_bases_user_id",
            "user_id",
        ),
    )
