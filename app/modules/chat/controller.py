from __future__ import annotations

from uuid import UUID

from starlette.concurrency import run_in_threadpool

from app.modules.chat import mapper
from app.modules.chat.schemas import ChatResponse
from app.modules.chat.service import ChatService


class ChatController:
    """Thin controller for synchronous chat requests."""

    def __init__(self, *, chat_service: ChatService) -> None:
        self._chat_service = chat_service

    async def send_message(
        self,
        *,
        conversation_id: UUID,
        content: str,
    ) -> ChatResponse:
        """Generate one assistant response and map it to the API contract."""

        result = await run_in_threadpool(
            self._chat_service.send_message,
            conversation_id=conversation_id,
            content=content,
        )

        return mapper.to_chat_response(
            assistant_message=result.assistant_message,
            context=result.context,
        )
