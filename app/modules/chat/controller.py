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
    ChatStreamToken,
)
from app.modules.chat.streaming import ChatStreamCancelReason

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
            loop = asyncio.get_running_loop()
            generation_start = loop.time()
            total_deadline = (
                generation_start + settings.chat_stream_total_timeout_seconds
            )
            ttft_deadline = (
                generation_start + settings.chat_stream_first_token_timeout_seconds
            )

            first_token_received = False
            last_token_time: float | None = None
            terminal_emitted = False
            pending_task: asyncio.Task[Any] | None = None

            def _consume_task_exception(task: asyncio.Task[Any]) -> None:
                if not task.cancelled():
                    task.exception()

            try:
                while True:
                    if await request.is_disconnected():
                        stream.request_cancel(ChatStreamCancelReason.CLIENT_DISCONNECT)
                        return

                    now = loop.time()

                    if not first_token_received:
                        token_deadline = ttft_deadline
                    else:
                        assert last_token_time is not None
                        token_deadline = (
                            last_token_time + settings.chat_stream_idle_timeout_seconds
                        )

                    active_deadline = min(total_deadline, token_deadline)
                    remaining_timeout = active_deadline - now

                    if total_deadline <= token_deadline:
                        timeout_reason = ChatStreamCancelReason.TOTAL_TIMEOUT
                        timeout_code = "TOTAL_TIMEOUT"
                        timeout_message = (
                            "Assistant response exceeded the generation time limit"
                        )
                    elif not first_token_received:
                        timeout_reason = ChatStreamCancelReason.TTFT_TIMEOUT
                        timeout_code = "TTFT_TIMEOUT"
                        timeout_message = (
                            "Assistant response timed out waiting for the first token"
                        )
                    else:
                        timeout_reason = ChatStreamCancelReason.IDLE_TIMEOUT
                        timeout_code = "IDLE_TIMEOUT"
                        timeout_message = "Assistant response stalled while generating"

                    if remaining_timeout <= 0:
                        cancelled = stream.request_cancel(timeout_reason)
                        if cancelled and not terminal_emitted:
                            terminal_emitted = True
                            yield mapper.to_sse_error(
                                code=timeout_code,
                                message=timeout_message,
                            )
                        return

                    pending_task = asyncio.create_task(
                        run_in_threadpool(_next_stream_event, stream)
                    )
                    pending_task.add_done_callback(_consume_task_exception)
                    try:
                        event = await asyncio.wait_for(
                            asyncio.shield(pending_task),
                            timeout=remaining_timeout,
                        )
                    except TimeoutError:
                        cancelled = stream.request_cancel(timeout_reason)
                        if cancelled:
                            if not terminal_emitted:
                                terminal_emitted = True
                                yield mapper.to_sse_error(
                                    code=timeout_code,
                                    message=timeout_message,
                                )
                            return
                        event = await pending_task

                    if event is _STREAM_END:
                        return

                    domain_event = cast(ChatStreamEvent, event)
                    if (
                        domain_event.type == ChatStreamEventType.TOKEN
                        and isinstance(domain_event.payload, ChatStreamToken)
                        and domain_event.payload.delta
                    ):
                        first_token_received = True
                        last_token_time = loop.time()

                    yield mapper.to_sse_event(domain_event)

                    if domain_event.type == ChatStreamEventType.COMPLETE:
                        terminal_emitted = True
                        return
            except asyncio.CancelledError:
                stream.request_cancel(ChatStreamCancelReason.CLIENT_DISCONNECT)
                raise
            except ResourceNotFoundException:
                if not terminal_emitted:
                    terminal_emitted = True
                    yield mapper.to_sse_error(
                        code="conversation_not_found",
                        message="Conversation not found",
                    )
            except DatabaseException:
                if not terminal_emitted:
                    terminal_emitted = True
                    yield mapper.to_sse_error(
                        code="PERSISTENCE_ERROR",
                        message="Failed to save assistant response",
                    )
            except AIServiceException:
                if not terminal_emitted:
                    terminal_emitted = True
                    if (
                        stream.snapshot.cancel_reason
                        == ChatStreamCancelReason.OUTPUT_LIMIT
                    ):
                        yield mapper.to_sse_error(
                            code="OUTPUT_LIMIT",
                            message="Assistant response exceeded the output limit",
                        )
                    else:
                        yield mapper.to_sse_error(
                            code="PROVIDER_ERROR",
                            message="Assistant generation failed",
                        )
            except Exception:  # noqa: BLE001
                if not terminal_emitted:
                    terminal_emitted = True
                    yield mapper.to_sse_error(
                        code="INTERNAL_ERROR",
                        message="Internal streaming error",
                    )
            finally:
                await run_in_threadpool(stream.close)
                if pending_task is not None and not pending_task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(pending_task), timeout=1.0
                        )
                    except (TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001, S110
                        pass
                if (
                    pending_task is not None
                    and pending_task.done()
                    and not pending_task.cancelled()
                ):
                    pending_task.exception()

        return EventSourceResponse(
            events(),
            ping=settings.chat_stream_ping_interval_seconds,
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
