from __future__ import annotations

from dataclasses import dataclass

from app.core.ai.llm.base import LLMProvider
from app.core.ai.llm.gemini import GeminiLLMProvider
from app.core.ai.llm.groq import GroqLLMProvider
from app.core.config import settings
from app.core.exceptions import ValidationException


@dataclass(frozen=True, slots=True)
class ChatModelInfo:
    """Descriptor for an available chat model."""

    id: str
    name: str
    provider: str
    is_default: bool = False


class LLMProviderRegistry:
    """Registry for discovering, selecting, and resolving configured LLM providers."""

    def __init__(
        self,
        *,
        groq_provider: GroqLLMProvider | None = None,
        gemini_provider: GeminiLLMProvider | None = None,
    ) -> None:
        self._groq_provider = groq_provider or GroqLLMProvider()
        self._gemini_provider = gemini_provider or GeminiLLMProvider()

    @property
    def _is_gemini_default(self) -> bool:
        default_cfg = settings.default_chat_model.strip().lower()
        return default_cfg in {
            "gemini",
            "google",
            settings.gemini_chat_model.lower(),
        }

    @property
    def default_model_id(self) -> str:
        """The identifier of the default chat model."""
        if self._is_gemini_default:
            return settings.gemini_chat_model
        return settings.groq_chat_model

    @property
    def default_provider(self) -> LLMProvider:
        """Return the default LLMProvider instance."""
        if self._is_gemini_default:
            return self._gemini_provider
        return self._groq_provider

    def get_available_models(self) -> list[ChatModelInfo]:
        """Return the list of models the frontend can present to the user."""
        gemini_default = self._is_gemini_default
        return [
            ChatModelInfo(
                id=settings.groq_chat_model,
                name=f"Groq ({settings.groq_chat_model})",
                provider="groq",
                is_default=not gemini_default,
            ),
            ChatModelInfo(
                id=settings.gemini_chat_model,
                name=f"Gemini ({settings.gemini_chat_model})",
                provider="gemini",
                is_default=gemini_default,
            ),
        ]

    def resolve(self, model: str | None = None) -> LLMProvider:
        """Resolve a requested model identifier or provider name to an LLMProvider."""
        if not model or not model.strip():
            return self.default_provider

        model_stripped = model.strip()
        model_lower = model_stripped.lower()

        if model_lower == "groq" or model_stripped == settings.groq_chat_model:
            return self._groq_provider

        if (
            model_lower in {"gemini", "google"}
            or model_stripped == settings.gemini_chat_model
        ):
            return self._gemini_provider

        supported = [m.id for m in self.get_available_models()]
        raise ValidationException(
            f"Unsupported chat model '{model_stripped}'. Supported models: {', '.join(supported)}"
        )
