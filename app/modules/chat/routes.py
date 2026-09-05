from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.dependencies.auth import CurrentUser
from app.dependencies.chat import ChatControllerDep
from app.modules.chat.schemas import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/conversations",
    tags=["Chat"],
)


@router.post(
    "/{conversation_id}/messages",
    response_model=ChatResponse,
)
async def send_message(
    *,
    conversation_id: UUID,
    request: ChatRequest,
    current_user: CurrentUser,
    controller: ChatControllerDep,
) -> ChatResponse:
    """Submit a synchronous retrieval-augmented chat turn."""

    return await controller.send_message(
        user_id=current_user.id,
        conversation_id=conversation_id,
        content=request.content,
    )


@router.post("/{conversation_id}/messages/stream")
async def stream_message(
    *,
    conversation_id: UUID,
    request: ChatRequest,
    http_request: Request,
    current_user: CurrentUser,
    controller: ChatControllerDep,
) -> EventSourceResponse:
    """Submit a synchronous preparation phase followed by an SSE chat stream."""

    return await controller.stream_message(
        user_id=current_user.id,
        conversation_id=conversation_id,
        content=request.content,
        request=http_request,
    )
