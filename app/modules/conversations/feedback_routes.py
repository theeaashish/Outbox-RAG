from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, status

from app.dependencies.auth import CurrentUser
from app.dependencies.conversations import ConversationControllerDep
from app.modules.conversations.schemas import (
    FeedbackRequest,
    MessageFeedbackResponse,
)

router = APIRouter(prefix="/messages", tags=["Message Feedback"])


@router.put("/{message_id}/feedback", response_model=MessageFeedbackResponse)
def set_message_feedback(
    *,
    message_id: UUID,
    payload: FeedbackRequest,
    controller: ConversationControllerDep,
    current_user: CurrentUser,
) -> MessageFeedbackResponse:
    """Set or update feedback rating for an assistant message."""
    return controller.set_feedback(
        user_id=current_user.id,
        message_id=message_id,
        rating=payload.rating,
    )


@router.delete("/{message_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
def delete_message_feedback(
    *,
    message_id: UUID,
    controller: ConversationControllerDep,
    current_user: CurrentUser,
) -> None:
    """Delete feedback rating for an assistant message."""
    controller.delete_feedback(
        user_id=current_user.id,
        message_id=message_id,
    )
