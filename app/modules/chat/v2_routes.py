from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.core.exceptions import ValidationException
from app.dependencies.auth import CurrentUser
from app.dependencies.chat import ChatControllerDep
from app.modules.chat.protocol.schemas import RunAgentInput

router = APIRouter(
    prefix="/conversations",
    tags=["Chat V2"],
)


@router.post("/{conversation_id}/messages/stream")
async def stream_ag_ui_message(
    *,
    conversation_id: UUID,
    request: RunAgentInput,
    http_request: Request,
    current_user: CurrentUser,
    controller: ChatControllerDep,
) -> EventSourceResponse:
    """Submit an AG-UI run turn and return an AG-UI SSE event stream."""

    # 1. Validate threadId if provided
    if request.thread_id:
        try:
            thread_uuid = UUID(request.thread_id)
        except ValueError as exc:
            raise ValidationException("threadId must be a valid UUID") from exc
        if thread_uuid != conversation_id:
            raise ValidationException("threadId does not match the URL conversation_id")

    # 2. Extract latest user message
    if not request.messages:
        raise ValidationException("RunAgentInput must contain at least one message")

    last_message = request.messages[-1]
    if last_message.role.lower() != "user":
        raise ValidationException(
            "The latest message in RunAgentInput must have role 'user'"
        )

    content = last_message.content.strip()
    if not content:
        raise ValidationException("User message content cannot be blank")

    if len(content) > settings.chat_max_message_characters:
        raise ValidationException(
            f"User message exceeds maximum length of {settings.chat_max_message_characters} characters"
        )

    return await controller.stream_ag_ui_message(
        user_id=current_user.id,
        conversation_id=conversation_id,
        content=content,
        request=http_request,
        client_run_id=request.run_id,
    )
