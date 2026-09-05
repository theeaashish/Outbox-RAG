from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast
from uuid import UUID

from fastapi import Request
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.exceptions import (
    AIServiceException,
    DatabaseException,
    ResourceNotFoundException,
)
from app.modules.chat import mapper
from app.modules.chat.schemas import ChatResponse
from app.modules.chat.service import (
    ChatEventStream,
    ChatService,
    ChatStreamEvent,
    ChatStreamEventType,
)

_STREAM_END = object()


def _next_stream_event(stream: ChatEventStream) -> ChatStreamEvent | object:
    """Advance a synchronous stream without leaking StopIteration across await."""

    return next(stream, _STREAM_END)


class ChatController:
    """Thin controller for synchronous and streaming chat requests."""

    def __init__(self, *, chat_service: ChatService) -> None:
        self._chat_service = chat_service

    async def send_message(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        content: str,
    ) -> ChatResponse:
        """Generate one assistant response and map it to the API contract."""

        result = await run_in_threadpool(
            self._chat_service.send_message,
            user_id=user_id,
            conversation_id=conversation_id,
            content=content,
        )

        return mapper.to_chat_response(
            assistant_message=result.assistant_message,
            context=result.context,
        )

    async def stream_message(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        content: str,
        request: Request,
    ) -> EventSourceResponse:
        """Prepare a turn, then expose its provider output as SSE events."""

        prepared = await run_in_threadpool(
            self._chat_service.prepare_turn,
            user_id=user_id,
            conversation_id=conversation_id,
            content=content,
        )
        stream = self._chat_service.stream_prepared_turn(prepared=prepared)

        async def events() -> AsyncIterator[dict[str, Any]]:
            started_at = asyncio.get_running_loop().time()
            awaiting_first_token = True
            try:
                while True:
                    if await request.is_disconnected():
                        return

                    elapsed = asyncio.get_running_loop().time() - started_at
                    remaining = settings.chat_stream_total_timeout_seconds - elapsed
                    if remaining <= 0:
                        yield mapper.to_sse_error(
                            code="stream_timeout",
                            message="Assistant response timed out",
                        )
                        return

                    timeout = min(
                        remaining,
                        (
                            settings.chat_stream_first_token_timeout_seconds
                            if awaiting_first_token
                            else settings.chat_stream_idle_timeout_seconds
                        ),
                    )
                    try:
                        event = await asyncio.wait_for(
                            run_in_threadpool(_next_stream_event, stream),
                            timeout=timeout,
                        )
                    except TimeoutError:
                        yield mapper.to_sse_error(
                            code="stream_timeout",
                            message="Assistant response timed out",
                        )
                        return

                    if event is _STREAM_END:
                        return

                    domain_event = cast(ChatStreamEvent, event)
                    if domain_event.type == ChatStreamEventType.TOKEN:
                        awaiting_first_token = False
                    yield mapper.to_sse_event(domain_event)
            except asyncio.CancelledError:
                raise
            except ResourceNotFoundException:
                yield mapper.to_sse_error(
                    code="conversation_not_found",
                    message="Conversation not found",
                )
            except DatabaseException:
                yield mapper.to_sse_error(
                    code="persistence_failed",
                    message="Failed to save assistant response",
                )
            except AIServiceException:
                yield mapper.to_sse_error(
                    code="generation_failed",
                    message="Assistant generation failed",
                )
            finally:
                await run_in_threadpool(stream.close)

        return EventSourceResponse(
            events(),
            ping=settings.chat_stream_ping_interval_seconds,
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
