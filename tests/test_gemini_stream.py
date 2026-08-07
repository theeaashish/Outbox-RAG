from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.ai.llm.gemini import GeminiLLMProvider, _GeminiLLMStream
from app.core.ai.llm.models import ChatMessage, LLMStreamCompletion, LLMStreamDelta
from app.core.exceptions import AIServiceException
from app.db.models.enums import MessageRole


def test_extract_text_from_string_and_content_blocks():
    assert GeminiLLMProvider._extract_text("plain") == "plain"
    assert (
        GeminiLLMProvider._extract_text(["a", {"text": "b"}, {"type": "image"}, "c"])
        == "abc"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("STOP", "stop"),
        ("stop", "stop"),
        ("MAX_TOKENS", "length"),
        ("length", "length"),
        ("SAFETY", "safety"),
        ("RECITATION", "safety"),
        ("OTHER", "error"),
        ("weird", "unknown"),
    ],
)
def test_normalize_finish_reason(raw: str, expected: str):
    assert GeminiLLMProvider._normalize_finish_reason(raw) == expected


def test_gemini_stream_emits_deltas_and_single_completion():
    chunks = [
        SimpleNamespace(content="Hel", response_metadata={}, usage_metadata=None),
        SimpleNamespace(
            content=[{"text": "lo"}],
            response_metadata={"finish_reason": "STOP"},
            usage_metadata={
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            },
        ),
    ]
    stream = _GeminiLLMStream(
        source=iter(chunks),
        model_name="gemini-test",
        message_count=1,
    )

    events = list(stream)

    assert events == [
        LLMStreamDelta(content="Hel"),
        LLMStreamDelta(content="lo"),
        LLMStreamCompletion(
            model="gemini-test",
            finish_reason="stop",
            usage=events[-1].usage,  # type: ignore[union-attr]
        ),
    ]
    assert events[-1].usage is not None  # type: ignore[union-attr]
    assert events[-1].usage.total_tokens == 6  # type: ignore[union-attr]


def test_gemini_stream_close_is_idempotent_and_closes_source():
    source = MagicMock()
    source_iter = iter(
        [
            SimpleNamespace(
                content="x",
                response_metadata={"finish_reason": "STOP"},
                usage_metadata=None,
            )
        ]
    )
    source.__iter__ = MagicMock(return_value=source_iter)
    # Use a custom iterator with close
    closed = {"value": False}

    class ClosableSource:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            closed["value"] = True

    stream = _GeminiLLMStream(
        source=ClosableSource(),
        model_name="gemini-test",
        message_count=1,
    )
    list(stream)
    stream.close()
    stream.close()
    assert closed["value"] is True


def test_gemini_stream_wraps_provider_errors():
    class FailingSource:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("network down")

        def close(self):
            return None

    stream = _GeminiLLMStream(
        source=FailingSource(),
        model_name="gemini-test",
        message_count=1,
    )

    with pytest.raises(AIServiceException, match="LLM stream failed"):
        list(stream)


def test_gemini_provider_stream_initializes_client_stream():
    provider = object.__new__(GeminiLLMProvider)
    provider._model_name = "gemini-test"
    provider._client = MagicMock()
    provider._client.stream.return_value = iter(
        [
            SimpleNamespace(
                content="ok",
                response_metadata={"finish_reason": "STOP"},
                usage_metadata=None,
            )
        ]
    )

    stream = GeminiLLMProvider.stream(
        provider,
        [ChatMessage(role=MessageRole.USER, content="hi")],
    )
    events = list(stream)
    assert isinstance(events[0], LLMStreamDelta)
    assert isinstance(events[1], LLMStreamCompletion)
