from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, ForeignKeyConstraint, Index, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.knowledge_base import KnowledgeBase
    from app.db.models.message import Message
    from app.db.models.project import Project
    from app.db.models.user import User


class Conversation(UUIDMixin, TimestampMixin, Base):
    """Represents a chat conversation."""

    __tablename__ = "conversations"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    project_id: Mapped[UUID] = mapped_column(
        nullable=False,
    )

    knowledge_base_id: Mapped[UUID] = mapped_column(
        nullable=False,
    )

    user: Mapped[User] = relationship(
        back_populates="conversations",
        lazy="raise",
    )

    project: Mapped[Project] = relationship(
        back_populates="conversations",
        foreign_keys=[project_id],
        lazy="raise",
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(
        back_populates="conversations",
        foreign_keys=[knowledge_base_id],
        lazy="select",
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="Message.created_at",
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "user_id"],
            ["projects.id", "projects.user_id"],
            name="fk_conversations_project_id_user_id_projects",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["knowledge_base_id", "project_id"],
            ["knowledge_bases.id", "knowledge_bases.project_id"],
            name="fk_conversations_kb_id_project_id_knowledge_bases",
            ondelete="CASCADE",
        ),
        Index(
            "ix_conversations_user_id",
            "user_id",
        ),
        Index(
            "ix_conversations_knowledge_base_created_at_id_desc",
            "knowledge_base_id",
            text("created_at DESC"),
            text("id DESC"),
        ),
    )
