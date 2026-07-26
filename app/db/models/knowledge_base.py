from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.conversation import Conversation
    from app.db.models.document import Document


class KnowledgeBase(UUIDMixin, TimestampMixin, Base):
    """Knowledge Base model."""

    __tablename__ = "knowledge_bases"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    documents: Mapped[list[Document]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
