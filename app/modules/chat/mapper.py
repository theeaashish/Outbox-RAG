from __future__ import annotations

import json
from typing import assert_never

from app.core.ai.context.models import AssembledContext, ContextChunk
from app.db.models import Message
from app.modules.chat.schemas import (
    ChatResponse,
    ChatSourceResponse,
    ChatStreamCompleteResponse,
    ChatStreamErrorResponse,
    ChatStreamMetadataResponse,
    ChatStreamTokenResponse,
)
from app.modules.chat.service import (
    ChatStreamCitations,
    ChatStreamComplete,
    ChatStreamEvent,
    ChatStreamEventType,
    ChatStreamMetadata,
    ChatStreamToken,
)
from app.modules.conversations.mapper import to_message_response


def to_chat_source_response(context_chunk: ContextChunk) -> ChatSourceResponse:
    """Map an assembled context chunk to citation metadata for the API."""

    return ChatSourceResponse(
        citation=context_chunk.citation,
        document_id=context_chunk.document_id,
        document_name=context_chunk.document_name,
        chunk_index=context_chunk.chunk_index,
        score=context_chunk.similarity,
    )


def to_chat_response(
    *,
    assistant_message: Message,
    context: AssembledContext,
) -> ChatResponse:
    """Map a completed chat turn to its API response."""

    return ChatResponse(
        assistant_message=to_message_response(assistant_message),
        sources=[
            to_chat_source_response(context_chunk) for context_chunk in context.chunks
        ],
    )


def to_sse_event(event: ChatStreamEvent) -> dict[str, str]:
    """Map a domain streaming event to an SSE event payload."""

    if event.type == ChatStreamEventType.METADATA:
        assert isinstance(event.payload, ChatStreamMetadata)
        metadata_payload = ChatStreamMetadataResponse(
            conversation_id=event.payload.conversation_id,
            source_count=event.payload.source_count,
        )
        return {
            "event": event.type.value,
            "data": json.dumps(metadata_payload.model_dump(mode="json")),
        }

    if event.type == ChatStreamEventType.CITATIONS:
        assert isinstance(event.payload, ChatStreamCitations)
        sources = [
            to_chat_source_response(context_chunk).model_dump(mode="json")
            for context_chunk in event.payload.context.chunks
        ]
        return {"event": event.type.value, "data": json.dumps(sources)}

    if event.type == ChatStreamEventType.TOKEN:
        assert isinstance(event.payload, ChatStreamToken)
        token_payload = ChatStreamTokenResponse(delta=event.payload.delta)
        return {
            "event": event.type.value,
            "data": json.dumps(token_payload.model_dump(mode="json")),
        }

    if event.type == ChatStreamEventType.COMPLETE:
        assert isinstance(event.payload, ChatStreamComplete)
        complete_payload = ChatStreamCompleteResponse(
            assistant_message=to_message_response(event.payload.assistant_message),
            model=event.payload.model,
            finish_reason=event.payload.finish_reason,
            usage=event.payload.usage,
        )
        return {
            "event": event.type.value,
            "data": json.dumps(complete_payload.model_dump(mode="json")),
        }

    assert_never(event.type)


def to_sse_error(*, code: str, message: str) -> dict[str, str]:
    """Build a safe protocol-level streaming error event."""

    payload = ChatStreamErrorResponse(code=code, message=message)
    return {"event": "error", "data": json.dumps(payload.model_dump())}
