from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, TypeVar, cast
from uuid import UUID, uuid4

from fastapi import Request
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

from app.core.config import settings
from app.core.exceptions import (
    AIServiceException,
    DatabaseException,
    ResourceNotFoundException,
)
from app.modules.chat import mapper
from app.modules.chat.protocol.adapter import AGUIProtocolAdapter
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
_T_co = TypeVar("_T_co", covariant=True)


def _next_stream_event(stream: ChatEventStream) -> ChatStreamEvent | object:
    """Advance a synchronous stream without leaking StopIteration across await."""

    return next(stream, _STREAM_END)


class StreamSink(Protocol[_T_co]):
    """Protocol for adapting internal chat stream events to wire formats."""

    def on_start(self) -> Sequence[_T_co]: ...

    def on_event(self, event: ChatStreamEvent) -> Sequence[_T_co]: ...

    def on_error(self, *, code: str, message: str) -> Sequence[_T_co]: ...


class V1ProtocolSink:
    """Sink for v1 SSE events consumed by EventSourceResponse."""

    def on_start(self) -> Sequence[dict[str, Any]]:
        return []

    def on_event(self, event: ChatStreamEvent) -> Sequence[dict[str, Any]]:
        return [mapper.to_sse_event(event)]

    def on_error(self, *, code: str, message: str) -> Sequence[dict[str, Any]]:
        return [mapper.to_sse_error(code=code, message=message)]


async def _stream_loop[T](
    stream: ChatEventStream,
    request: Request,
    sink: StreamSink[T],
) -> AsyncIterator[T]:
    loop = asyncio.get_running_loop()
    generation_start = loop.time()
    total_deadline = generation_start + settings.chat_stream_total_timeout_seconds
    ttft_deadline = generation_start + settings.chat_stream_first_token_timeout_seconds

    first_token_received = False
    last_token_time: float | None = None
    terminal_emitted = False
    pending_task: asyncio.Task[Any] | None = None

    def _consume_task_exception(task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()

    try:
        for item in sink.on_start():
            yield item

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
                    for item in sink.on_error(
                        code=timeout_code,
                        message=timeout_message,
                    ):
                        yield item
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
                        for item in sink.on_error(
                            code=timeout_code,
                            message=timeout_message,
                        ):
                            yield item
                    return
                event = await pending_task

            if event is _STREAM_END:
                if not terminal_emitted:
                    terminal_emitted = True
                    for item in sink.on_error(
                        code="INTERNAL_ERROR",
                        message="Streaming ended before completion",
                    ):
                        yield item
                return

            domain_event = cast(ChatStreamEvent, event)
            if (
                domain_event.type == ChatStreamEventType.TOKEN
                and isinstance(domain_event.payload, ChatStreamToken)
                and domain_event.payload.delta
            ):
                first_token_received = True
                last_token_time = loop.time()

            for item in sink.on_event(domain_event):
                yield item

            if domain_event.type == ChatStreamEventType.COMPLETE:
                terminal_emitted = True
                return
    except asyncio.CancelledError:
        stream.request_cancel(ChatStreamCancelReason.CLIENT_DISCONNECT)
        raise
    except ResourceNotFoundException:
        if not terminal_emitted:
            terminal_emitted = True
            for item in sink.on_error(
                code="conversation_not_found",
                message="Conversation not found",
            ):
                yield item
    except DatabaseException:
        if not terminal_emitted:
            terminal_emitted = True
            for item in sink.on_error(
                code="PERSISTENCE_ERROR",
                message="Failed to save assistant response",
            ):
                yield item
    except AIServiceException:
        if not terminal_emitted:
            terminal_emitted = True
            if stream.snapshot.cancel_reason == ChatStreamCancelReason.OUTPUT_LIMIT:
                for item in sink.on_error(
                    code="OUTPUT_LIMIT",
                    message="Assistant response exceeded the output limit",
                ):
                    yield item
            else:
                for item in sink.on_error(
                    code="PROVIDER_ERROR",
                    message="Assistant generation failed",
                ):
                    yield item
    except Exception:  # noqa: BLE001
        if not terminal_emitted:
            terminal_emitted = True
            for item in sink.on_error(
                code="INTERNAL_ERROR",
                message="Internal streaming error",
            ):
                yield item
    finally:
        await run_in_threadpool(stream.close)
        if pending_task is not None and not pending_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(pending_task), timeout=1.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
            except Exception:
                logger.debug(
                    "Unexpected error during stream task cleanup",
                    exc_info=True,
                )
        if (
            pending_task is not None
            and pending_task.done()
            and not pending_task.cancelled()
        ):
            pending_task.exception()


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
        sink = V1ProtocolSink()

        return EventSourceResponse(
            _stream_loop(stream, request, sink),
            ping=settings.chat_stream_ping_interval_seconds,
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    async def stream_ag_ui_message(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        content: str,
        request: Request,
        client_run_id: str | None = None,
    ) -> EventSourceResponse:
        """Prepare a turn, then expose its provider output as AG-UI SSE frames."""

        prepared = await run_in_threadpool(
            self._chat_service.prepare_turn,
            user_id=user_id,
            conversation_id=conversation_id,
            content=content,
        )
        stream = self._chat_service.stream_prepared_turn(prepared=prepared)
        run_id = client_run_id or str(uuid4())
        adapter = AGUIProtocolAdapter(
            run_id=run_id,
            thread_id=str(conversation_id),
            assistant_message_id=prepared.assistant_message_id,
        )

        return EventSourceResponse(
            _stream_loop(stream, request, adapter),
            ping=settings.chat_stream_ping_interval_seconds,
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
