from __future__ import annotations

from app.core.ai.context.models import AssembledContext, ContextChunk
from app.db.models import Message
from app.modules.chat.schemas import ChatResponse, ChatSourceResponse
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
