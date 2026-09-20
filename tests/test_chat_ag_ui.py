from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.core.ai.context.models import AssembledContext, ContextChunk
from app.core.ai.llm.models import (
    ChatMessage,
    LLMUsage,
)
from app.core.ai.routing.models import (
    ConversationMode,
    QueryRoutingDecision,
    RoutingReason,
)
from app.core.config import settings
from app.core.exceptions import (
    AIServiceException,
    DatabaseException,
)
from app.db.models import Message, User
from app.db.models.enums import MessageRole, UserStatus
from app.dependencies.auth import get_current_user
from app.dependencies.chat import get_chat_controller
from app.main import app
from app.modules.chat.controller import ChatController
from app.modules.chat.protocol.adapter import AGUIProtocolAdapter
from app.modules.chat.protocol.mapper import (
    encode_event,
    to_citations_event,
    to_metadata_event,
    to_run_error,
    to_run_finished,
    to_run_started,
    to_text_message_content,
    to_text_message_end,
    to_text_message_start,
)
from app.modules.chat.protocol.schemas import AGUIMessageInput, RunAgentInput
from app.modules.chat.service import (
    ChatEventStream,
    ChatService,
    ChatStreamCitations,
    ChatStreamComplete,
    ChatStreamEvent,
    ChatStreamEventType,
    ChatStreamMetadata,
    ChatStreamToken,
    PreparedChatTurn,
)
from app.modules.chat.streaming import ChatStreamCancelReason, ChatStreamLifecycle


def _make_fake_user() -> User:
    return User(
        id=uuid4(),
        email="user@example.com",
        email_normalized="user@example.com",
        status=UserStatus.ACTIVE,
    )


def _to_text(frame: Any) -> str:
    if isinstance(frame, str):
        return frame
    if isinstance(frame, bytes):
        return frame.decode("utf-8")
    return bytes(frame).decode("utf-8")


# =====================================================================
# 1. Schemas & Wire Contract Tests
# =====================================================================


def test_ag_ui_message_input_schema():
    msg = AGUIMessageInput(role="user", content="Hello, world!", id="m-1")
    assert msg.role == "user"
    assert msg.content == "Hello, world!"
    assert msg.id == "m-1"

    # Forward compatible with extra fields
    msg2 = AGUIMessageInput.model_validate(
        {"role": "user", "content": "Hi", "extraField": 123}
    )
    assert msg2.content == "Hi"


def test_run_agent_input_wire_contract_aliases():
    payload = {
        "threadId": "11111111-1111-1111-1111-111111111111",
        "runId": "run-xyz",
        "parentRunId": "parent-abc",
        "state": {"step": 1},
        "messages": [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {"model": "gemini"},
    }
    parsed = RunAgentInput.model_validate(payload)
    assert parsed.thread_id == "11111111-1111-1111-1111-111111111111"
    assert parsed.run_id == "run-xyz"
    assert parsed.parent_run_id == "parent-abc"
    assert parsed.state == {"step": 1}
    assert len(parsed.messages) == 3
    assert parsed.messages[-1].content == "Third"
    assert parsed.forwarded_props == {"model": "gemini"}


# =====================================================================
# 2. Protocol Mapper Unit Tests
# =====================================================================


def test_mapper_events_and_encoding():
    # RUN_STARTED
    run_started = to_run_started(run_id="run-1", thread_id="thread-1")
    encoded_started = encode_event(run_started)
    assert encoded_started.startswith("data: ")
    assert encoded_started.endswith("\n\n")
    data_started = json.loads(encoded_started[6:].strip())
    assert data_started["type"] == "RUN_STARTED"
    assert data_started["runId"] == "run-1"
    assert data_started["threadId"] == "thread-1"

    # TEXT_MESSAGE_START, CONTENT, END
    msg_start = to_text_message_start(message_id="msg-1", role="assistant")
    data_start = json.loads(encode_event(msg_start)[6:].strip())
    assert data_start["type"] == "TEXT_MESSAGE_START"
    assert data_start["messageId"] == "msg-1"
    assert data_start["role"] == "assistant"

    msg_content = to_text_message_content(message_id="msg-1", delta="Hello ")
    data_content = json.loads(encode_event(msg_content)[6:].strip())
    assert data_content["type"] == "TEXT_MESSAGE_CONTENT"
    assert data_content["messageId"] == "msg-1"
    assert data_content["delta"] == "Hello "

    msg_end = to_text_message_end(message_id="msg-1")
    data_end = json.loads(encode_event(msg_end)[6:].strip())
    assert data_end["type"] == "TEXT_MESSAGE_END"
    assert data_end["messageId"] == "msg-1"

    # RUN_FINISHED with usage mapping
    usage = LLMUsage(prompt_tokens=15, completion_tokens=42, total_tokens=57)
    run_finished = to_run_finished(run_id="run-1", thread_id="thread-1", usage=usage)
    data_finished = json.loads(encode_event(run_finished)[6:].strip())
    assert data_finished["type"] == "RUN_FINISHED"
    assert data_finished["runId"] == "run-1"
    assert data_finished["threadId"] == "thread-1"
    assert "usage" in data_finished
    assert isinstance(data_finished["usage"], list)
    assert len(data_finished["usage"]) == 1
    assert data_finished["usage"][0] == {
        "inputTokens": 15,
        "outputTokens": 42,
        "totalTokens": 57,
    }

    # RUN_ERROR
    run_error = to_run_error(message="Network glitch", code="PROVIDER_ERROR")
    data_error = json.loads(encode_event(run_error)[6:].strip())
    assert data_error["type"] == "RUN_ERROR"
    assert data_error["message"] == "Network glitch"
    assert data_error["code"] == "PROVIDER_ERROR"

    # CUSTOM metadata & citations
    meta = to_metadata_event(conversation_id="conv-1", source_count=2)
    data_meta = json.loads(encode_event(meta)[6:].strip())
    assert data_meta["type"] == "CUSTOM"
    assert data_meta["name"] == "metadata"
    assert data_meta["value"] == {"conversationId": "conv-1", "sourceCount": 2}

    doc_id = uuid4()
    context = AssembledContext(
        query="test query",
        block="[1] test snippet",
        chunks=[
            ContextChunk(
                citation=1,
                document_id=doc_id,
                document_name="doc.pdf",
                chunk_index=0,
                content="test snippet",
                similarity=0.95,
            )
        ],
    )
    citations = to_citations_event(context=context)
    data_citations = json.loads(encode_event(citations)[6:].strip())
    assert data_citations["type"] == "CUSTOM"
    assert data_citations["name"] == "citations"
    assert len(data_citations["value"]["sources"]) == 1
    assert data_citations["value"]["sources"][0]["citation"] == 1
    assert data_citations["value"]["sources"][0]["document_id"] == str(doc_id)
    assert data_citations["value"]["sources"][0]["score"] == 0.95


# =====================================================================
# 3. Protocol Adapter Timing & Invariant Tests
# =====================================================================


def test_adapter_defers_text_message_start_until_first_token():
    asst_id = uuid4()
    adapter = AGUIProtocolAdapter(
        run_id="run-1",
        thread_id="thread-1",
        assistant_message_id=asst_id,
    )

    # on_start
    start_frames = adapter.on_start()
    assert len(start_frames) == 1
    assert json.loads(start_frames[0][6:].strip())["type"] == "RUN_STARTED"

    # metadata event: no TEXT_MESSAGE_START yet
    meta_frames = adapter.on_event(
        ChatStreamEvent(
            type=ChatStreamEventType.METADATA,
            payload=ChatStreamMetadata(conversation_id=uuid4(), source_count=1),
        )
    )
    assert len(meta_frames) == 1
    assert json.loads(meta_frames[0][6:].strip())["type"] == "CUSTOM"

    # citations event: no TEXT_MESSAGE_START yet
    cite_frames = adapter.on_event(
        ChatStreamEvent(
            type=ChatStreamEventType.CITATIONS,
            payload=ChatStreamCitations(context=None),
        )
    )
    assert len(cite_frames) == 1
    assert json.loads(cite_frames[0][6:].strip())["type"] == "CUSTOM"

    # First token event: TEXT_MESSAGE_START followed immediately by TEXT_MESSAGE_CONTENT
    token_frames = adapter.on_event(
        ChatStreamEvent(
            type=ChatStreamEventType.TOKEN,
            payload=ChatStreamToken(delta="Hello"),
        )
    )
    assert len(token_frames) == 2
    ev1 = json.loads(token_frames[0][6:].strip())
    ev2 = json.loads(token_frames[1][6:].strip())
    assert ev1["type"] == "TEXT_MESSAGE_START"
    assert ev1["messageId"] == str(asst_id)
    assert ev1["role"] == "assistant"
    assert ev2["type"] == "TEXT_MESSAGE_CONTENT"
    assert ev2["messageId"] == str(asst_id)
    assert ev2["delta"] == "Hello"

    # Second token event: ONLY TEXT_MESSAGE_CONTENT
    token2_frames = adapter.on_event(
        ChatStreamEvent(
            type=ChatStreamEventType.TOKEN,
            payload=ChatStreamToken(delta=" world!"),
        )
    )
    assert len(token2_frames) == 1
    ev3 = json.loads(token2_frames[0][6:].strip())
    assert ev3["type"] == "TEXT_MESSAGE_CONTENT"
    assert ev3["delta"] == " world!"

    # Complete event: TEXT_MESSAGE_END followed by RUN_FINISHED
    asst_msg = Message(
        id=asst_id,
        conversation_id=uuid4(),
        role=MessageRole.ASSISTANT,
        content="Hello world!",
    )
    complete_frames = adapter.on_event(
        ChatStreamEvent(
            type=ChatStreamEventType.COMPLETE,
            payload=ChatStreamComplete(
                assistant_message=asst_msg,
                context=None,
                model="test-model",
                finish_reason="stop",
                usage=LLMUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
        )
    )
    assert len(complete_frames) == 2
    ev4 = json.loads(complete_frames[0][6:].strip())
    ev5 = json.loads(complete_frames[1][6:].strip())
    assert ev4["type"] == "TEXT_MESSAGE_END"
    assert ev4["messageId"] == str(asst_id)
    assert ev5["type"] == "RUN_FINISHED"
    assert ev5["usage"][0]["totalTokens"] == 7


def test_adapter_provider_error_before_first_token_emits_no_message_start():
    asst_id = uuid4()
    adapter = AGUIProtocolAdapter(
        run_id="run-1",
        thread_id="thread-1",
        assistant_message_id=asst_id,
    )

    adapter.on_start()
    adapter.on_event(
        ChatStreamEvent(
            type=ChatStreamEventType.METADATA,
            payload=ChatStreamMetadata(conversation_id=uuid4(), source_count=0),
        )
    )

    # Provider dies before first token
    err_frames = adapter.on_error(code="PROVIDER_ERROR", message="API failure")
    assert len(err_frames) == 1
    err = json.loads(err_frames[0][6:].strip())
    assert err["type"] == "RUN_ERROR"
    assert err["code"] == "PROVIDER_ERROR"
    assert err["message"] == "API failure"

    # Subsequent error calls emit nothing (terminal idempotency)
    assert adapter.on_error(code="PROVIDER_ERROR", message="Second error") == []


# =====================================================================
# 4. Controller & Streaming Loop Tests
# =====================================================================


def _make_dummy_prepared(assistant_id: UUID | None = None) -> PreparedChatTurn:
    conv_id = uuid4()
    return PreparedChatTurn(
        conversation_id=conv_id,
        knowledge_base_id=uuid4(),
        user_id=uuid4(),
        user_message_id=uuid4(),
        user_content="Hello",
        context=None,
        prompt=[ChatMessage(role=MessageRole.USER, content="Hello")],
        routing=QueryRoutingDecision(
            mode=ConversationMode.CASUAL,
            reason=RoutingReason.EXPLICIT_CASUAL,
        ),
        assistant_message_id=assistant_id or uuid4(),
    )


def test_controller_stream_ag_ui_message_success():
    async def _test():
        asst_id = uuid4()
        prepared = _make_dummy_prepared(asst_id)

        events = [
            ChatStreamEvent(
                type=ChatStreamEventType.METADATA,
                payload=ChatStreamMetadata(
                    conversation_id=prepared.conversation_id, source_count=0
                ),
            ),
            ChatStreamEvent(
                type=ChatStreamEventType.CITATIONS,
                payload=ChatStreamCitations(context=None),
            ),
            ChatStreamEvent(
                type=ChatStreamEventType.TOKEN,
                payload=ChatStreamToken(delta="Hi!"),
            ),
            ChatStreamEvent(
                type=ChatStreamEventType.COMPLETE,
                payload=ChatStreamComplete(
                    assistant_message=Message(
                        id=asst_id,
                        conversation_id=prepared.conversation_id,
                        role=MessageRole.ASSISTANT,
                        content="Hi!",
                    ),
                    context=None,
                    model="test-model",
                    finish_reason="stop",
                    usage=LLMUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                ),
            ),
        ]
        lifecycle = ChatStreamLifecycle()
        chat_stream = ChatEventStream(events=iter(events), lifecycle=lifecycle)

        mock_service = MagicMock(spec=ChatService)
        mock_service.prepare_turn.return_value = prepared
        mock_service.stream_prepared_turn.return_value = chat_stream

        controller = ChatController(chat_service=mock_service)
        mock_request = AsyncMock()
        mock_request.is_disconnected.return_value = False

        response = await controller.stream_ag_ui_message(
            user_id=prepared.user_id,
            conversation_id=prepared.conversation_id,
            content="Hello",
            request=mock_request,
            client_run_id="client-run-123",
        )

        assert response.media_type == "text/event-stream"
        collected_frames: list[dict[str, Any]] = []
        async for frame in response.body_iterator:
            for line in _to_text(frame).strip().split("\n\n"):
                if line.startswith("data: "):
                    collected_frames.append(json.loads(line[6:]))

        types = [f["type"] for f in collected_frames]
        assert types == [
            "RUN_STARTED",
            "CUSTOM",  # metadata
            "CUSTOM",  # citations
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "RUN_FINISHED",
        ]
        # Check client run ID preserved
        assert collected_frames[0]["runId"] == "client-run-123"
        assert collected_frames[-1]["runId"] == "client-run-123"
        # Check assistant message ID matches prepared.assistant_message_id
        assert collected_frames[3]["messageId"] == str(asst_id)
        assert collected_frames[4]["messageId"] == str(asst_id)
        assert collected_frames[5]["messageId"] == str(asst_id)

    asyncio.run(_test())


def test_controller_stream_ag_ui_message_ttft_timeout():
    async def _test():
        prepared = _make_dummy_prepared()

        class _HangingStream:
            def __init__(self):
                self.snapshot = MagicMock()
                self.snapshot.cancel_reason = None

            def __iter__(self):
                return self

            def __next__(self):
                import time

                time.sleep(0.5)
                raise StopIteration

            def request_cancel(self, reason):
                self.snapshot.cancel_reason = reason
                return True

            def close(self):
                pass

        mock_service = MagicMock(spec=ChatService)
        mock_service.prepare_turn.return_value = prepared
        mock_service.stream_prepared_turn.return_value = _HangingStream()

        controller = ChatController(chat_service=mock_service)
        mock_request = AsyncMock()
        mock_request.is_disconnected.return_value = False

        with patch.object(settings, "chat_stream_first_token_timeout_seconds", 0.05):  # noqa: SIM117
            with patch.object(settings, "chat_stream_total_timeout_seconds", 10.0):
                response = await controller.stream_ag_ui_message(
                    user_id=prepared.user_id,
                    conversation_id=prepared.conversation_id,
                    content="Hello",
                    request=mock_request,
                )

                collected_frames = []
                async for frame in response.body_iterator:
                    for line in _to_text(frame).strip().split("\n\n"):
                        if line.startswith("data: "):
                            collected_frames.append(json.loads(line[6:]))

                assert len(collected_frames) == 2
                assert collected_frames[0]["type"] == "RUN_STARTED"
                assert collected_frames[1]["type"] == "RUN_ERROR"
                assert collected_frames[1]["code"] == "TTFT_TIMEOUT"

    asyncio.run(_test())


def test_controller_stream_ag_ui_message_idle_timeout():
    async def _test():
        prepared = _make_dummy_prepared()

        class _StalledAfterFirstTokenStream:
            def __init__(self):
                self._first = True
                self.snapshot = MagicMock()
                self.snapshot.cancel_reason = None

            def __iter__(self):
                return self

            def __next__(self):
                import time

                if self._first:
                    self._first = False
                    return ChatStreamEvent(
                        type=ChatStreamEventType.TOKEN,
                        payload=ChatStreamToken(delta="First token"),
                    )
                time.sleep(0.5)
                raise StopIteration

            def request_cancel(self, reason):
                self.snapshot.cancel_reason = reason
                return True

            def close(self):
                pass

        mock_service = MagicMock(spec=ChatService)
        mock_service.prepare_turn.return_value = prepared
        mock_service.stream_prepared_turn.return_value = _StalledAfterFirstTokenStream()

        controller = ChatController(chat_service=mock_service)
        mock_request = AsyncMock()
        mock_request.is_disconnected.return_value = False

        with patch.object(settings, "chat_stream_idle_timeout_seconds", 0.05):  # noqa: SIM117
            with patch.object(settings, "chat_stream_total_timeout_seconds", 10.0):
                response = await controller.stream_ag_ui_message(
                    user_id=prepared.user_id,
                    conversation_id=prepared.conversation_id,
                    content="Hello",
                    request=mock_request,
                )

                collected_frames = []
                async for frame in response.body_iterator:
                    for line in _to_text(frame).strip().split("\n\n"):
                        if line.startswith("data: "):
                            collected_frames.append(json.loads(line[6:]))

                types = [f["type"] for f in collected_frames]
                assert "RUN_STARTED" in types
                assert "TEXT_MESSAGE_START" in types
                assert "TEXT_MESSAGE_CONTENT" in types
                assert types[-1] == "RUN_ERROR"
                assert collected_frames[-1]["code"] == "IDLE_TIMEOUT"

    asyncio.run(_test())


def test_controller_stream_ag_ui_message_output_limit():
    async def _test():
        prepared = _make_dummy_prepared()

        class _OutputLimitStream:
            def __init__(self):
                self.snapshot = MagicMock()
                self.snapshot.cancel_reason = ChatStreamCancelReason.OUTPUT_LIMIT

            def __iter__(self):
                return self

            def __next__(self):
                raise AIServiceException("LLM response exceeded output limit")

            def request_cancel(self, reason):
                return True

            def close(self):
                pass

        mock_service = MagicMock(spec=ChatService)
        mock_service.prepare_turn.return_value = prepared
        mock_service.stream_prepared_turn.return_value = _OutputLimitStream()

        controller = ChatController(chat_service=mock_service)
        mock_request = AsyncMock()
        mock_request.is_disconnected.return_value = False

        response = await controller.stream_ag_ui_message(
            user_id=prepared.user_id,
            conversation_id=prepared.conversation_id,
            content="Hello",
            request=mock_request,
        )

        collected_frames = []
        async for frame in response.body_iterator:
            for line in _to_text(frame).strip().split("\n\n"):
                if line.startswith("data: "):
                    collected_frames.append(json.loads(line[6:]))

        assert len(collected_frames) == 2
        assert collected_frames[0]["type"] == "RUN_STARTED"
        assert collected_frames[1]["type"] == "RUN_ERROR"
        assert collected_frames[1]["code"] == "OUTPUT_LIMIT"

    asyncio.run(_test())


def test_controller_stream_ag_ui_message_persistence_error():
    async def _test():
        prepared = _make_dummy_prepared()

        class _PersistenceErrorStream:
            def __init__(self):
                self.snapshot = MagicMock()
                self.snapshot.cancel_reason = None

            def __iter__(self):
                return self

            def __next__(self):
                raise DatabaseException("DB connection dropped")

            def request_cancel(self, reason):
                return True

            def close(self):
                pass

        mock_service = MagicMock(spec=ChatService)
        mock_service.prepare_turn.return_value = prepared
        mock_service.stream_prepared_turn.return_value = _PersistenceErrorStream()

        controller = ChatController(chat_service=mock_service)
        mock_request = AsyncMock()
        mock_request.is_disconnected.return_value = False

        response = await controller.stream_ag_ui_message(
            user_id=prepared.user_id,
            conversation_id=prepared.conversation_id,
            content="Hello",
            request=mock_request,
        )

        collected_frames = []
        async for frame in response.body_iterator:
            for line in _to_text(frame).strip().split("\n\n"):
                if line.startswith("data: "):
                    collected_frames.append(json.loads(line[6:]))

        assert len(collected_frames) == 2
        assert collected_frames[0]["type"] == "RUN_STARTED"
        assert collected_frames[1]["type"] == "RUN_ERROR"
        assert collected_frames[1]["code"] == "PERSISTENCE_ERROR"

    asyncio.run(_test())


def test_controller_stream_ag_ui_message_unexpected_stream_end():
    async def _test():
        prepared = _make_dummy_prepared()

        # Stream yields one token, but ends prematurely without COMPLETE
        events = [
            ChatStreamEvent(
                type=ChatStreamEventType.TOKEN,
                payload=ChatStreamToken(delta="Partial text..."),
            ),
        ]
        lifecycle = ChatStreamLifecycle()
        chat_stream = ChatEventStream(events=iter(events), lifecycle=lifecycle)

        mock_service = MagicMock(spec=ChatService)
        mock_service.prepare_turn.return_value = prepared
        mock_service.stream_prepared_turn.return_value = chat_stream

        controller = ChatController(chat_service=mock_service)
        mock_request = AsyncMock()
        mock_request.is_disconnected.return_value = False

        response = await controller.stream_ag_ui_message(
            user_id=prepared.user_id,
            conversation_id=prepared.conversation_id,
            content="Hello",
            request=mock_request,
        )

        collected_frames = []
        async for frame in response.body_iterator:
            for line in _to_text(frame).strip().split("\n\n"):
                if line.startswith("data: "):
                    collected_frames.append(json.loads(line[6:]))

        types = [f["type"] for f in collected_frames]
        assert "RUN_STARTED" in types
        assert "TEXT_MESSAGE_START" in types
        assert "TEXT_MESSAGE_CONTENT" in types
        assert types[-1] == "RUN_ERROR"
        assert collected_frames[-1]["code"] == "INTERNAL_ERROR"
        assert "before completion" in collected_frames[-1]["message"]

    asyncio.run(_test())


def test_controller_stream_ag_ui_message_client_disconnect():
    async def _test():
        prepared = _make_dummy_prepared()
        mock_stream = MagicMock()

        mock_service = MagicMock(spec=ChatService)
        mock_service.prepare_turn.return_value = prepared
        mock_service.stream_prepared_turn.return_value = mock_stream

        controller = ChatController(chat_service=mock_service)
        mock_request = AsyncMock()
        mock_request.is_disconnected.return_value = True

        response = await controller.stream_ag_ui_message(
            user_id=prepared.user_id,
            conversation_id=prepared.conversation_id,
            content="Hello",
            request=mock_request,
        )

        collected_frames = []
        async for frame in response.body_iterator:
            collected_frames.append(frame)

        # Disconnected before events: stream cancelled and aborts
        mock_stream.request_cancel.assert_called_with(
            ChatStreamCancelReason.CLIENT_DISCONNECT
        )

    asyncio.run(_test())


# =====================================================================
# 5. Route Integration Tests via TestClient
# =====================================================================


def test_v2_stream_route_validation_thread_id_mismatch():
    conv_id = uuid4()
    other_conv_id = uuid4()
    client = TestClient(app)

    fake_user = _make_fake_user()
    app.dependency_overrides[get_current_user] = lambda: fake_user

    try:
        payload = {
            "threadId": str(other_conv_id),
            "messages": [{"role": "user", "content": "Valid query"}],
        }
        res = client.post(
            f"/api/v2/conversations/{conv_id}/messages/stream",
            json=payload,
        )
        assert res.status_code == 400
        assert "threadId does not match" in res.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_v2_stream_route_validation_thread_id_invalid_uuid():
    conv_id = uuid4()
    client = TestClient(app)

    fake_user = _make_fake_user()
    app.dependency_overrides[get_current_user] = lambda: fake_user

    try:
        payload = {
            "threadId": "not-a-uuid",
            "messages": [{"role": "user", "content": "Valid query"}],
        }
        res = client.post(
            f"/api/v2/conversations/{conv_id}/messages/stream",
            json=payload,
        )
        assert res.status_code == 400
        assert "threadId must be a valid UUID" in res.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_v2_stream_route_validation_empty_messages():
    conv_id = uuid4()
    client = TestClient(app)

    fake_user = _make_fake_user()
    app.dependency_overrides[get_current_user] = lambda: fake_user

    try:
        payload = {
            "threadId": str(conv_id),
            "messages": [],
        }
        res = client.post(
            f"/api/v2/conversations/{conv_id}/messages/stream",
            json=payload,
        )
        assert res.status_code == 400
        assert "at least one message" in res.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_v2_stream_route_validation_last_message_not_user():
    conv_id = uuid4()
    client = TestClient(app)

    fake_user = _make_fake_user()
    app.dependency_overrides[get_current_user] = lambda: fake_user

    try:
        payload = {
            "threadId": str(conv_id),
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
            ],
        }
        res = client.post(
            f"/api/v2/conversations/{conv_id}/messages/stream",
            json=payload,
        )
        assert res.status_code == 400
        assert "must have role 'user'" in res.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_v2_stream_route_validation_blank_content():
    conv_id = uuid4()
    client = TestClient(app)

    fake_user = _make_fake_user()
    app.dependency_overrides[get_current_user] = lambda: fake_user

    try:
        payload = {
            "threadId": str(conv_id),
            "messages": [{"role": "user", "content": "   \n\t  "}],
        }
        res = client.post(
            f"/api/v2/conversations/{conv_id}/messages/stream",
            json=payload,
        )
        assert res.status_code == 400
        assert "content cannot be blank" in res.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_v2_stream_route_validation_content_too_long():
    conv_id = uuid4()
    client = TestClient(app)

    fake_user = _make_fake_user()
    app.dependency_overrides[get_current_user] = lambda: fake_user

    try:
        long_content = "a" * (settings.chat_max_message_characters + 1)
        payload = {
            "threadId": str(conv_id),
            "messages": [{"role": "user", "content": long_content}],
        }
        res = client.post(
            f"/api/v2/conversations/{conv_id}/messages/stream",
            json=payload,
        )
        assert res.status_code == 400
        assert "maximum length" in res.json()["error"]["message"]
    finally:
        app.dependency_overrides.clear()


def test_v2_stream_route_extracts_latest_user_message_and_streams():
    conv_id = uuid4()
    asst_id = uuid4()
    client = TestClient(app)

    fake_user = _make_fake_user()
    app.dependency_overrides[get_current_user] = lambda: fake_user

    mock_controller = MagicMock()

    async def _mock_stream_ag_ui(*args, **kwargs):
        from sse_starlette.sse import EventSourceResponse

        async def _gen():
            yield (
                'data: {"type":"RUN_STARTED","threadId":"'
                + str(conv_id)
                + '","runId":"test-run"}\n\n'
            ).encode("utf-8")
            yield (
                'data: {"type":"TEXT_MESSAGE_START","messageId":"'
                + str(asst_id)
                + '","role":"assistant"}\n\n'
            ).encode("utf-8")
            yield (
                'data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"'
                + str(asst_id)
                + '","delta":"Answer"}\n\n'
            ).encode("utf-8")
            yield (
                'data: {"type":"TEXT_MESSAGE_END","messageId":"'
                + str(asst_id)
                + '"}\n\n'
            ).encode("utf-8")
            yield (
                'data: {"type":"RUN_FINISHED","threadId":"'
                + str(conv_id)
                + '","runId":"test-run"}\n\n'
            ).encode("utf-8")

        return EventSourceResponse(_gen())

    mock_controller.stream_ag_ui_message = AsyncMock(side_effect=_mock_stream_ag_ui)
    app.dependency_overrides[get_chat_controller] = lambda: mock_controller

    try:
        # Pass multi-message history: only the last message should be extracted and passed to controller
        payload = {
            "threadId": str(conv_id),
            "runId": "custom-run-id",
            "messages": [
                {"role": "user", "content": "Ignore this old query"},
                {"role": "assistant", "content": "Ignore this old response"},
                {"role": "user", "content": "What is the new policy?"},
            ],
        }
        res = client.post(
            f"/api/v2/conversations/{conv_id}/messages/stream",
            json=payload,
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        # Verify controller was called with ONLY the latest user message
        mock_controller.stream_ag_ui_message.assert_called_once()
        call_kwargs = mock_controller.stream_ag_ui_message.call_args.kwargs
        assert call_kwargs["content"] == "What is the new policy?"
        assert call_kwargs["conversation_id"] == conv_id
        assert call_kwargs["client_run_id"] == "custom-run-id"

        lines = [
            line for line in res.text.strip().split("\n\n") if line.startswith("data: ")
        ]
        assert len(lines) == 5
    finally:
        app.dependency_overrides.clear()


def test_controller_stream_ag_ui_emits_ping_heartbeat():
    async def _test():
        prepared = _make_dummy_prepared()

        class _SlowStream:
            def __init__(self):
                self._yielded = False
                self.snapshot = MagicMock()
                self.snapshot.cancel_reason = None

            def __iter__(self):
                return self

            def __next__(self):
                import time

                if not self._yielded:
                    self._yielded = True
                    return ChatStreamEvent(
                        type=ChatStreamEventType.METADATA,
                        payload=ChatStreamMetadata(
                            conversation_id=prepared.conversation_id,
                            source_count=0,
                        ),
                    )
                time.sleep(0.15)
                raise StopIteration

            def request_cancel(self, reason):
                return True

            def close(self):
                pass

        mock_service = MagicMock(spec=ChatService)
        mock_service.prepare_turn.return_value = prepared
        mock_service.stream_prepared_turn.return_value = _SlowStream()

        controller = ChatController(chat_service=mock_service)
        mock_request = AsyncMock()
        mock_request.is_disconnected.return_value = False

        with patch.object(settings, "chat_stream_ping_interval_seconds", 0.05):
            response = await controller.stream_ag_ui_message(
                user_id=prepared.user_id,
                conversation_id=prepared.conversation_id,
                content="Hello",
                request=mock_request,
            )

            chunks: list[bytes] = []

            async def fake_send(msg: Any) -> None:
                if msg["type"] == "http.response.body" and msg.get("body"):
                    chunks.append(msg["body"])

            async def fake_receive() -> Any:
                await asyncio.sleep(10)
                return {"type": "http.disconnect"}

            await response({"type": "http", "method": "POST"}, fake_receive, fake_send)

            text_chunks = [c.decode("utf-8") for c in chunks]
            pings = [t for t in text_chunks if t.startswith(": ping - ")]
            assert len(pings) >= 1
            ag_ui_events = [t for t in text_chunks if t.startswith("data: ")]
            assert len(ag_ui_events) >= 1

    asyncio.run(_test())
