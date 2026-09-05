from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query, Response, status

from app.dependencies.auth import CurrentUser
from app.dependencies.conversations import ConversationControllerDep
from app.modules.conversations.schemas import (
    ConversationListResponse,
    ConversationResponse,
    MessageListResponse,
)

kb_router = APIRouter(
    prefix="/knowledge-bases",
    tags=["Conversations"],
)

project_router = APIRouter(
    prefix="/projects",
    tags=["Conversations"],
)

conversation_router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)

router = APIRouter()
router.include_router(kb_router)
router.include_router(project_router)
router.include_router(conversation_router)


@kb_router.post(
    "/{knowledge_base_id}/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    *,
    knowledge_base_id: UUID,
    controller: ConversationControllerDep,
) -> ConversationResponse:
    """Create a new conversation for a knowledge base."""

    return controller.create_conversation(
        knowledge_base_id=knowledge_base_id,
    )


@project_router.post(
    "/{project_id}/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_project_conversation(
    *,
    project_id: UUID,
    controller: ConversationControllerDep,
    current_user: CurrentUser,
) -> ConversationResponse:
    """Create a conversation for a project and its knowledge base."""
    return controller.create_project_conversation(
        user_id=current_user.id,
        project_id=project_id,
    )


@kb_router.get(
    "/{knowledge_base_id}/conversations",
    response_model=ConversationListResponse,
    deprecated=True,
)
def list_conversations(
    *,
    knowledge_base_id: UUID,
    controller: ConversationControllerDep,
    response: Response,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ConversationListResponse:
    """List conversations for a knowledge base (deprecated: use /api/v2)."""

    response.headers["Deprecation"] = "true"
    return controller.list_conversations(
        knowledge_base_id=knowledge_base_id,
        limit=limit,
        offset=offset,
    )


@conversation_router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def get_conversation(
    *,
    conversation_id: UUID,
    controller: ConversationControllerDep,
) -> ConversationResponse:
    """Retrieve a conversation."""

    return controller.get_conversation(
        conversation_id=conversation_id,
    )


@conversation_router.get(
    "/{conversation_id}/messages",
    response_model=MessageListResponse,
    deprecated=True,
)
def list_messages(
    *,
    conversation_id: UUID,
    controller: ConversationControllerDep,
    response: Response,
    limit: int = Query(100, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> MessageListResponse:
    """List messages for a conversation (deprecated: use /api/v2)."""

    response.headers["Deprecation"] = "true"
    return controller.list_messages(
        conversation_id=conversation_id,
        limit=limit,
        offset=offset,
    )


@conversation_router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_conversation(
    *,
    conversation_id: UUID,
    controller: ConversationControllerDep,
) -> Response:
    """Delete a conversation."""

    controller.delete_conversation(
        conversation_id=conversation_id,
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
