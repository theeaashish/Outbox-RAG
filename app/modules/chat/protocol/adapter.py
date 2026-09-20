from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from app.modules.chat.protocol import mapper
from app.modules.chat.service import (
    ChatStreamCitations,
    ChatStreamComplete,
    ChatStreamEvent,
    ChatStreamEventType,
    ChatStreamMetadata,
    ChatStreamToken,
)


class AGUIProtocolAdapter:
    """Adapts domain chat streaming events into AG-UI SSE frames."""

    def __init__(
        self,
        *,
        run_id: str,
        thread_id: str,
        assistant_message_id: UUID,
    ) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_message_id = str(assistant_message_id)
        self._text_message_started = False
        self._terminal_emitted = False

    def on_start(self) -> Sequence[bytes]:
        """Emit RUN_STARTED when the stream connection initiates."""
        event = mapper.to_run_started(
            run_id=self.run_id,
            thread_id=self.thread_id,
        )
        return [mapper.encode_event_bytes(event)]

    def on_event(self, event: ChatStreamEvent) -> Sequence[bytes]:
        """Map a domain event to AG-UI SSE frame(s)."""
        if self._terminal_emitted:
            return []

        if event.type == ChatStreamEventType.METADATA:
            assert isinstance(event.payload, ChatStreamMetadata)
            custom_event = mapper.to_metadata_event(
                conversation_id=str(event.payload.conversation_id),
                source_count=event.payload.source_count,
            )
            return [mapper.encode_event_bytes(custom_event)]

        if event.type == ChatStreamEventType.CITATIONS:
            assert isinstance(event.payload, ChatStreamCitations)
            custom_event = mapper.to_citations_event(
                context=event.payload.context,
            )
            return [mapper.encode_event_bytes(custom_event)]

        if event.type == ChatStreamEventType.TOKEN:
            assert isinstance(event.payload, ChatStreamToken)
            if not event.payload.delta:
                return []

            token_frames: list[bytes] = []
            if not self._text_message_started:
                self._text_message_started = True
                token_frames.append(
                    mapper.encode_event_bytes(
                        mapper.to_text_message_start(
                            message_id=self.assistant_message_id,
                            role="assistant",
                        )
                    )
                )

            token_frames.append(
                mapper.encode_event_bytes(
                    mapper.to_text_message_content(
                        message_id=self.assistant_message_id,
                        delta=event.payload.delta,
                    )
                )
            )
            return token_frames

        if event.type == ChatStreamEventType.COMPLETE:
            assert isinstance(event.payload, ChatStreamComplete)
            complete_frames: list[bytes] = []
            if not self._text_message_started:
                self._text_message_started = True
                complete_frames.append(
                    mapper.encode_event_bytes(
                        mapper.to_text_message_start(
                            message_id=self.assistant_message_id,
                            role="assistant",
                        )
                    )
                )
            complete_frames.append(
                mapper.encode_event_bytes(
                    mapper.to_text_message_end(
                        message_id=self.assistant_message_id,
                    )
                )
            )
            complete_frames.append(
                mapper.encode_event_bytes(
                    mapper.to_run_finished(
                        run_id=self.run_id,
                        thread_id=self.thread_id,
                        usage=event.payload.usage,
                    )
                )
            )
            self._terminal_emitted = True
            return complete_frames

        return []

    def on_error(self, *, code: str, message: str) -> Sequence[bytes]:
        """Emit RUN_ERROR as the terminal error event if not already finished."""
        if self._terminal_emitted:
            return []
        self._terminal_emitted = True
        return [
            mapper.encode_event_bytes(
                mapper.to_run_error(
                    message=message,
                    code=code,
                )
            )
        ]
