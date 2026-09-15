from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.core.ai.llm.gemini import GeminiLLMProvider
from app.core.ai.llm.models import ChatMessage
from app.core.exceptions import AIServiceException, TransientAIServiceException
from app.db.models.enums import MessageRole


def _make_provider_with_mock_client() -> tuple[GeminiLLMProvider, Any]:
    provider = object.__new__(GeminiLLMProvider)
    provider._model_name = "gemini-test"
    mock_client = MagicMock()
    provider._client = mock_client
    return provider, mock_client


def test_finish_reason_stop_and_length_succeed():
    provider, mock_client = _make_provider_with_mock_client()
    for finish in ("STOP", "MAX_TOKENS", "stop", "length"):
        mock_client.invoke.return_value = AIMessage(
            content="Answer text",
            response_metadata={"finish_reason": finish},
        )
        response = provider.generate([ChatMessage(role=MessageRole.USER, content="hi")])
        assert response.content == "Answer text"
        assert response.finish_reason in ("stop", "length")


def test_finish_reason_safety_raises_ai_service_exception():
    provider, mock_client = _make_provider_with_mock_client()
    for finish in ("SAFETY", "RECITATION", "safety", "recitation"):
        mock_client.invoke.return_value = AIMessage(
            content="",
            response_metadata={"finish_reason": finish},
        )
        with pytest.raises(AIServiceException, match="safety filters"):
            provider.generate([ChatMessage(role=MessageRole.USER, content="unsafe")])


def test_finish_reason_error_raises_ai_service_exception():
    provider, mock_client = _make_provider_with_mock_client()
    for finish in ("OTHER", "other"):
        mock_client.invoke.return_value = AIMessage(
            content="",
            response_metadata={"finish_reason": finish},
        )
        with pytest.raises(AIServiceException, match="LLM generation failed"):
            provider.generate([ChatMessage(role=MessageRole.USER, content="error")])


def test_finish_reason_unknown_or_unaccepted_raises_ai_service_exception():
    provider, mock_client = _make_provider_with_mock_client()
    for finish in ("FINISH_REASON_UNSPECIFIED", "unknown", "SOMETHING_UNEXPECTED"):
        mock_client.invoke.return_value = AIMessage(
            content="Some text",
            response_metadata={"finish_reason": finish},
        )
        with pytest.raises(AIServiceException, match="unaccepted finish reason"):
            provider.generate([ChatMessage(role=MessageRole.USER, content="hi")])


def test_provider_timeout_translates_to_transient_exception():
    provider, mock_client = _make_provider_with_mock_client()
    mock_client.invoke.side_effect = httpx.TimeoutException("connection timed out")

    with pytest.raises(TransientAIServiceException):
        provider.generate([ChatMessage(role=MessageRole.USER, content="hi")])
