from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

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
    controller: ChatControllerDep,
) -> ChatResponse:
    """Submit a synchronous retrieval-augmented chat turn."""

    return await controller.send_message(
        conversation_id=conversation_id,
        content=request.content,
    )
