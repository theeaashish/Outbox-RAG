from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query

from app.dependencies.auth import CurrentUser
from app.dependencies.conversations import ConversationControllerDep
from app.modules.conversations.schemas import (
    ConversationCursorPageResponse,
    MessageCursorPageResponse,
)

api_v2_router = APIRouter()


@api_v2_router.get(
    "/knowledge-bases/{knowledge_base_id}/conversations",
    response_model=ConversationCursorPageResponse,
    tags=["Conversations"],
)
def list_conversations_cursor(
    *,
    knowledge_base_id: UUID,
    controller: ConversationControllerDep,
    current_user: CurrentUser,
    page_size: int = Query(50, ge=1, le=100),
    after: str | None = Query(default=None),
    before: str | None = Query(default=None),
) -> ConversationCursorPageResponse:
    """List conversations using v2 signed cursor pagination."""

    return controller.list_conversations_cursor(
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
        page_size=page_size,
        after=after,
        before=before,
    )


@api_v2_router.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessageCursorPageResponse,
    tags=["Conversations"],
)
def list_messages_cursor(
    *,
    conversation_id: UUID,
    controller: ConversationControllerDep,
    current_user: CurrentUser,
    page_size: int = Query(50, ge=1, le=100),
    after: str | None = Query(default=None),
    before: str | None = Query(default=None),
) -> MessageCursorPageResponse:
    """List conversation messages using v2 signed cursor pagination."""

    return controller.list_messages_cursor(
        user_id=current_user.id,
        conversation_id=conversation_id,
        page_size=page_size,
        after=after,
        before=before,
    )
