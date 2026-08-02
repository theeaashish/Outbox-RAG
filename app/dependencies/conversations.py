from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.dependencies.repositories import (
    DBSession,
    get_conversation_repository,
    get_message_repository,
)
from app.dependencies.services import KnowledgeBaseRepositoryDep
from app.modules.conversations.controller import ConversationController
from app.modules.conversations.service import ConversationService
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository

ConversationRepositoryDep = Annotated[
    ConversationRepository,
    Depends(get_conversation_repository),
]

MessageRepositoryDep = Annotated[
    MessageRepository,
    Depends(get_message_repository),
]


def get_conversation_service(
    db: DBSession,
    conversation_repository: ConversationRepositoryDep,
    knowledge_base_repository: KnowledgeBaseRepositoryDep,
    message_repository: MessageRepositoryDep,
) -> ConversationService:
    """Return a configured ConversationService."""

    return ConversationService(
        db=db,
        conversation_repository=conversation_repository,
        knowledge_base_repository=knowledge_base_repository,
        message_repository=message_repository,
    )


ConversationServiceDep = Annotated[
    ConversationService,
    Depends(get_conversation_service),
]


def get_conversation_controller(
    conversation_service: ConversationServiceDep,
) -> ConversationController:
    """Return a configured ConversationController."""

    return ConversationController(
        conversation_service=conversation_service,
    )


ConversationControllerDep = Annotated[
    ConversationController,
    Depends(get_conversation_controller),
]
