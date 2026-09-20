from __future__ import annotations

from ag_ui.core.events import (
    BaseEvent,
    CustomEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageRole,
    TextMessageStartEvent,
    TokenUsage,
)
from ag_ui.encoder import EventEncoder

from app.core.ai.context.models import AssembledContext
from app.core.ai.llm.models import LLMUsage

_encoder = EventEncoder()


def encode_event(event: BaseEvent) -> str:
    """Encode an AG-UI event into an SSE formatted string data frame."""
    return _encoder.encode(event)


def encode_event_bytes(event: BaseEvent) -> bytes:
    """Encode an AG-UI event into UTF-8 bytes for EventSourceResponse transport."""
    return _encoder.encode(event).encode("utf-8")


def to_run_started(*, run_id: str, thread_id: str) -> RunStartedEvent:
    """Create RUN_STARTED AG-UI event."""
    return RunStartedEvent(run_id=run_id, thread_id=thread_id)


def to_text_message_start(
    *, message_id: str, role: TextMessageRole = "assistant"
) -> TextMessageStartEvent:
    """Create TEXT_MESSAGE_START AG-UI event."""
    return TextMessageStartEvent(message_id=message_id, role=role)


def to_text_message_content(*, message_id: str, delta: str) -> TextMessageContentEvent:
    """Create TEXT_MESSAGE_CONTENT AG-UI event."""
    return TextMessageContentEvent(message_id=message_id, delta=delta)


def to_text_message_end(*, message_id: str) -> TextMessageEndEvent:
    """Create TEXT_MESSAGE_END AG-UI event."""
    return TextMessageEndEvent(message_id=message_id)


def to_run_finished(
    *, run_id: str, thread_id: str, usage: LLMUsage | None = None
) -> RunFinishedEvent:
    """Create RUN_FINISHED AG-UI event mapping internal LLMUsage to TokenUsage[]."""
    token_usage: list[TokenUsage] | None = None
    if usage is not None:
        token_usage = [
            TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
        ]
    return RunFinishedEvent(
        run_id=run_id,
        thread_id=thread_id,
        usage=token_usage,
    )


def to_run_error(*, message: str, code: str | None = None) -> RunErrorEvent:
    """Create RUN_ERROR AG-UI event."""
    return RunErrorEvent(message=message, code=code)


def to_metadata_event(*, conversation_id: str, source_count: int) -> CustomEvent:
    """Create CUSTOM metadata event."""
    return CustomEvent(
        name="metadata",
        value={
            "conversationId": conversation_id,
            "sourceCount": source_count,
        },
    )


def to_citations_event(*, context: AssembledContext | None) -> CustomEvent:
    """Create CUSTOM citations event."""
    sources = (
        [
            {
                "citation": chunk.citation,
                "document_id": str(chunk.document_id),
                "document_name": chunk.document_name,
                "chunk_index": chunk.chunk_index,
                "score": chunk.similarity,
            }
            for chunk in context.chunks
        ]
        if context is not None
        else []
    )
    return CustomEvent(
        name="citations",
        value={"sources": sources},
    )
