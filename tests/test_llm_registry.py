from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from starlette.testclient import TestClient

from app.core.ai.llm.gemini import GeminiLLMProvider
from app.core.ai.llm.groq import GroqLLMProvider
from app.core.ai.llm.registry import LLMProviderRegistry
from app.core.config import settings
from app.core.exceptions import ValidationException
from app.db.models.user import User
from app.dependencies.auth import get_current_user
from app.main import app
from app.modules.chat.service import ChatService


def test_registry_default_model():
    mock_groq = MagicMock(spec=GroqLLMProvider)
    mock_gemini = MagicMock(spec=GeminiLLMProvider)
    registry = LLMProviderRegistry(
        groq_provider=mock_groq,
        gemini_provider=mock_gemini,
    )

    assert registry.default_model_id == settings.groq_chat_model
    assert registry.default_provider is mock_groq

    models = registry.get_available_models()
    assert len(models) == 2

    groq_info = next(m for m in models if m.provider == "groq")
    assert groq_info.id == settings.groq_chat_model
    assert groq_info.is_default is True

    gemini_info = next(m for m in models if m.provider == "gemini")
    assert gemini_info.id == settings.gemini_chat_model
    assert gemini_info.is_default is False


def test_registry_respects_gemini_as_default_setting(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "default_chat_model", "gemini")
    mock_groq = MagicMock(spec=GroqLLMProvider)
    mock_gemini = MagicMock(spec=GeminiLLMProvider)
    registry = LLMProviderRegistry(
        groq_provider=mock_groq,
        gemini_provider=mock_gemini,
    )

    assert registry.default_model_id == settings.gemini_chat_model
    assert registry.default_provider is mock_gemini

    models = registry.get_available_models()
    gemini_info = next(m for m in models if m.provider == "gemini")
    assert gemini_info.is_default is True
    groq_info = next(m for m in models if m.provider == "groq")
    assert groq_info.is_default is False


def test_registry_resolve_models():
    mock_groq = MagicMock(spec=GroqLLMProvider)
    mock_gemini = MagicMock(spec=GeminiLLMProvider)
    registry = LLMProviderRegistry(
        groq_provider=mock_groq,
        gemini_provider=mock_gemini,
    )

    # Resolves default when None or empty
    assert registry.resolve(None) is mock_groq
    assert registry.resolve("") is mock_groq
    assert registry.resolve("  ") is mock_groq

    # Resolves Groq by provider name or default model id
    assert registry.resolve("groq") is mock_groq
    assert registry.resolve("GROQ") is mock_groq
    assert registry.resolve(settings.groq_chat_model) is mock_groq

    # Resolves Gemini by provider name or default model id
    assert registry.resolve("gemini") is mock_gemini
    assert registry.resolve("Gemini") is mock_gemini
    assert registry.resolve("google") is mock_gemini
    assert registry.resolve(settings.gemini_chat_model) is mock_gemini


def test_registry_rejects_unconfigured_and_arbitrary_models():
    mock_groq = MagicMock(spec=GroqLLMProvider)
    mock_gemini = MagicMock(spec=GeminiLLMProvider)
    registry = LLMProviderRegistry(
        groq_provider=mock_groq,
        gemini_provider=mock_gemini,
    )

    # Unconfigured models must fail fast
    for invalid in (
        "gemini-2.5-flash",
        "gemini-9999",
        "llama-3.3-70b-versatile",
        "whisper-large-v3",
        "openai/gpt-4o",
        "unsupported-model",
    ):
        with pytest.raises(ValidationException, match="Unsupported chat model"):
            registry.resolve(invalid)


def test_api_list_available_models_requires_auth():
    client = TestClient(app)
    response = client.get("/api/v1/conversations/models")
    assert response.status_code == 401

    alt_response = client.get("/api/v1/chat/models")
    assert alt_response.status_code == 401


def test_api_list_available_models_authenticated():
    mock_user = User(
        id=uuid4(),
        email="test@example.com",
        name="Test User",
    )
    app.dependency_overrides[get_current_user] = lambda: mock_user
    try:
        client = TestClient(app)
        response = client.get("/api/v1/conversations/models")
        assert response.status_code == 200

        data = response.json()
        assert "default_model" in data
        assert data["default_model"] == settings.groq_chat_model
        assert "models" in data
        assert len(data["models"]) == 2

        groq_entry = next(m for m in data["models"] if m["provider"] == "groq")
        assert groq_entry["is_default"] is True
        assert groq_entry["id"] == settings.groq_chat_model

        gemini_entry = next(m for m in data["models"] if m["provider"] == "gemini")
        assert gemini_entry["is_default"] is False
        assert gemini_entry["id"] == settings.gemini_chat_model

        # Also test /api/v1/chat/models
        alt_response = client.get("/api/v1/chat/models")
        assert alt_response.status_code == 200
        assert alt_response.json() == data
    finally:
        app.dependency_overrides.clear()


def test_chat_service_dynamic_provider_selection():
    mock_groq = MagicMock(spec=GroqLLMProvider)
    mock_gemini = MagicMock(spec=GeminiLLMProvider)

    registry = LLMProviderRegistry(
        groq_provider=mock_groq,
        gemini_provider=mock_gemini,
    )

    service = ChatService(
        db=MagicMock(),
        conversation_repository=MagicMock(),
        message_repository=MagicMock(),
        retrieval_service=MagicMock(),
        context_assembler=MagicMock(),
        prompt_builder=MagicMock(),
        conversational_prompt_builder=MagicMock(),
        query_router=MagicMock(),
        llm_provider=mock_groq,
        llm_provider_factory=registry.resolve,
        history_message_limit=10,
        retrieval_limit=5,
        similarity_threshold=0.5,
        stream_max_buffered_characters=65536,
    )

    # When model is None: uses default (Groq)
    assert service._get_provider(None) is mock_groq

    # When model is 'groq': uses Groq
    assert service._get_provider("groq") is mock_groq

    # When model is 'gemini': uses Gemini
    assert service._get_provider("gemini") is mock_gemini

    # When model is Gemini model id
    assert service._get_provider(settings.gemini_chat_model) is mock_gemini


def test_prepare_turn_fails_fast_on_invalid_model_without_persistence():
    mock_groq = MagicMock(spec=GroqLLMProvider)
    mock_gemini = MagicMock(spec=GeminiLLMProvider)

    registry = LLMProviderRegistry(
        groq_provider=mock_groq,
        gemini_provider=mock_gemini,
    )

    mock_db = MagicMock()
    mock_msg_repo = MagicMock()
    mock_retrieval = MagicMock()

    service = ChatService(
        db=mock_db,
        conversation_repository=MagicMock(),
        message_repository=mock_msg_repo,
        retrieval_service=mock_retrieval,
        context_assembler=MagicMock(),
        prompt_builder=MagicMock(),
        conversational_prompt_builder=MagicMock(),
        query_router=MagicMock(),
        llm_provider=mock_groq,
        llm_provider_factory=registry.resolve,
        history_message_limit=10,
        retrieval_limit=5,
        similarity_threshold=0.5,
        stream_max_buffered_characters=65536,
    )

    with pytest.raises(
        ValidationException, match="Unsupported chat model 'unsupported-xyz'"
    ):
        service.prepare_turn(
            user_id=uuid4(),
            conversation_id=uuid4(),
            content="Hello world",
            model="unsupported-xyz",
        )

    # Verify no database persistence or retrieval was executed
    mock_msg_repo.create.assert_not_called()
    mock_retrieval.retrieve.assert_not_called()
