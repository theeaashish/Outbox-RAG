from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from app.core.ai.llm.models import LLMUsage
from app.db.models import Message, MessageSource
from app.db.models.enums import MessageRole
from app.modules.chat.protocol.adapter import AGUIProtocolAdapter
from app.modules.chat.service import (
    ChatStreamComplete,
    ChatStreamEvent,
    ChatStreamEventType,
    ChatStreamToken,
)


def _decode_sse_frame(frame: bytes) -> dict[str, object]:
    text = frame.decode("utf-8").strip()
    for line in text.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"No data line found in frame: {text!r}")


def test_ag_ui_adapter_emits_sources_final_once_before_text_message_end() -> None:
    msg_id = uuid4()
    doc_id = uuid4()
    conv_id = uuid4()

    persisted_message = Message(
        id=msg_id,
        role=MessageRole.ASSISTANT,
        content="Here is the answer.",
        conversation_id=conv_id,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    persisted_message.sources = [
        MessageSource(
            id=uuid4(),
            message_id=msg_id,
            citation_number=1,
            is_cited=False,
            source_order=0,
            document_id=doc_id,
            document_chunk_id=uuid4(),
            document_name_snapshot="Handbook.pdf",
            chunk_index_snapshot=0,
            retrieval_score=0.81,
        ),
        MessageSource(
            id=uuid4(),
            message_id=msg_id,
            citation_number=2,
            is_cited=True,
            source_order=1,
            document_id=doc_id,
            document_chunk_id=uuid4(),
            document_name_snapshot="Handbook.pdf",
            chunk_index_snapshot=3,
            retrieval_score=0.76,
        ),
    ]

    adapter = AGUIProtocolAdapter(
        run_id="run-1",
        thread_id=str(conv_id),
        assistant_message_id=msg_id,
    )

    frames: list[bytes] = []
    frames.extend(adapter.on_start())
    frames.extend(
        adapter.on_event(
            ChatStreamEvent(
                type=ChatStreamEventType.TOKEN,
                payload=ChatStreamToken(delta="Here is the answer [2]."),
            )
        )
    )
    frames.extend(
        adapter.on_event(
            ChatStreamEvent(
                type=ChatStreamEventType.COMPLETE,
                payload=ChatStreamComplete(
                    assistant_message=persisted_message,
                    context=None,
                    model="gemini-3.5-flash-lite",
                    finish_reason="stop",
                    usage=LLMUsage(
                        prompt_tokens=20,
                        completion_tokens=6,
                        total_tokens=26,
                    ),
                ),
            )
        )
    )

    events = [_decode_sse_frame(f) for f in frames]
    event_types = [
        f"{e['type']}:{e.get('name')}" if e["type"] == "CUSTOM" else str(e["type"])
        for e in events
    ]

    assert event_types == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "CUSTOM:sources.final",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]

    sources_final = events[3]
    assert sources_final["value"] == {
        "sources": [
            {
                "citation": 1,
                "documentId": str(doc_id),
                "documentName": "Handbook.pdf",
                "chunkIndex": 0,
                "score": 0.81,
                "isCited": False,
            },
            {
                "citation": 2,
                "documentId": str(doc_id),
                "documentName": "Handbook.pdf",
                "chunkIndex": 3,
                "score": 0.76,
                "isCited": True,
            },
        ]
    }
