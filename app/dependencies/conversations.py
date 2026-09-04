from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.core.pagination import CursorCodec
from app.dependencies.core import get_cursor_codec
from app.dependencies.repositories import (
    DBSession,
    get_conversation_repository,
    get_message_repository,
    get_project_repository,
)
from app.dependencies.services import KnowledgeBaseRepositoryDep
from app.modules.conversations.controller import ConversationController
from app.modules.conversations.service import ConversationService
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository
from app.repositories.project import ProjectRepository

ConversationRepositoryDep = Annotated[
    ConversationRepository,
    Depends(get_conversation_repository),
]

MessageRepositoryDep = Annotated[
    MessageRepository,
    Depends(get_message_repository),
]

ProjectRepositoryDep = Annotated[
    ProjectRepository,
    Depends(get_project_repository),
]

CursorCodecDep = Annotated[CursorCodec, Depends(get_cursor_codec)]


def get_conversation_service(
    db: DBSession,
    conversation_repository: ConversationRepositoryDep,
    knowledge_base_repository: KnowledgeBaseRepositoryDep,
    project_repository: ProjectRepositoryDep,
    message_repository: MessageRepositoryDep,
    cursor_codec: CursorCodecDep,
) -> ConversationService:
    """Return a configured ConversationService."""

    return ConversationService(
        db=db,
        conversation_repository=conversation_repository,
        knowledge_base_repository=knowledge_base_repository,
        project_repository=project_repository,
        message_repository=message_repository,
        cursor_codec=cursor_codec,
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
