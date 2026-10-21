from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.core.ai.llm.groq import GroqLLMProvider, _GroqLLMStream
from app.core.ai.llm.models import ChatMessage, LLMStreamCompletion, LLMStreamDelta
from app.core.exceptions import AIServiceException, TransientAIServiceException
from app.db.models.enums import MessageRole


def _make_provider_with_mock_client() -> tuple[GroqLLMProvider, Any]:
    provider = object.__new__(GroqLLMProvider)
    provider._model_name = "openai/gpt-oss-120b"
    mock_client = MagicMock()
    provider._client = mock_client
    return provider, mock_client


def test_groq_extract_text():
    assert GroqLLMProvider._extract_text("plain text") == "plain text"
    assert (
        GroqLLMProvider._extract_text(
            ["part1", {"text": "part2"}, {"type": "image"}, "part3"]
        )
        == "part1part2part3"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", "stop"),
        ("STOP", "stop"),
        ("length", "length"),
        ("MAX_TOKENS", "length"),
        ("max_tokens", "length"),
        ("content_filter", "safety"),
        ("safety", "safety"),
        ("error", "error"),
        ("OTHER", "error"),
        ("unexpected_reason", "unknown"),
    ],
)
def test_groq_normalize_finish_reason(raw: str, expected: str):
    assert GroqLLMProvider._normalize_finish_reason(raw) == expected


def test_groq_finish_reason_stop_and_length_succeed():
    provider, mock_client = _make_provider_with_mock_client()
    for finish in ("stop", "STOP", "length", "max_tokens"):
        mock_client.invoke.return_value = AIMessage(
            content="Groq response text",
            response_metadata={"finish_reason": finish},
        )
        response = provider.generate(
            [ChatMessage(role=MessageRole.USER, content="Hello Groq")]
        )
        assert response.content == "Groq response text"
        assert response.finish_reason in ("stop", "length")
        assert response.model == "openai/gpt-oss-120b"


def test_groq_finish_reason_safety_raises_ai_service_exception():
    provider, mock_client = _make_provider_with_mock_client()
    for finish in ("content_filter", "safety"):
        mock_client.invoke.return_value = AIMessage(
            content="",
            response_metadata={"finish_reason": finish},
        )
        with pytest.raises(AIServiceException, match="safety filters"):
            provider.generate(
                [ChatMessage(role=MessageRole.USER, content="unsafe query")]
            )


def test_groq_finish_reason_error_raises_ai_service_exception():
    provider, mock_client = _make_provider_with_mock_client()
    for finish in ("error", "OTHER", "other"):
        mock_client.invoke.return_value = AIMessage(
            content="",
            response_metadata={"finish_reason": finish},
        )
        with pytest.raises(AIServiceException, match="LLM generation failed"):
            provider.generate(
                [ChatMessage(role=MessageRole.USER, content="trigger error")]
            )


def test_groq_finish_reason_unexpected_raises_ai_service_exception():
    provider, mock_client = _make_provider_with_mock_client()
    mock_client.invoke.return_value = AIMessage(
        content="Unexpected",
        response_metadata={"finish_reason": "bizarre_finish_reason"},
    )
    with pytest.raises(AIServiceException, match="unaccepted finish reason"):
        provider.generate([ChatMessage(role=MessageRole.USER, content="hello")])


def test_groq_provider_timeout_translates_to_transient_exception():
    provider, mock_client = _make_provider_with_mock_client()
    mock_client.invoke.side_effect = httpx.TimeoutException("connection timed out")

    with pytest.raises(TransientAIServiceException):
        provider.generate([ChatMessage(role=MessageRole.USER, content="hello")])


def test_groq_stream_emits_deltas_and_completion():
    chunks = [
        SimpleNamespace(content="Groq ", response_metadata={}, usage_metadata=None),
        SimpleNamespace(
            content="rocks!",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={
                "input_tokens": 8,
                "output_tokens": 4,
                "total_tokens": 12,
            },
        ),
    ]
    stream = _GroqLLMStream(
        source=iter(chunks),
        model_name="openai/gpt-oss-120b",
        message_count=1,
    )
    events = list(stream)

    assert len(events) == 3
    assert isinstance(events[0], LLMStreamDelta)
    assert events[0].content == "Groq "
    assert isinstance(events[1], LLMStreamDelta)
    assert events[1].content == "rocks!"
    assert isinstance(events[2], LLMStreamCompletion)
    assert events[2].model == "openai/gpt-oss-120b"
    assert events[2].finish_reason == "stop"
    assert events[2].usage is not None
    assert events[2].usage.total_tokens == 12
