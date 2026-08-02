from __future__ import annotations

import logging
from typing import Any, ClassVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from app.core.ai.llm.base import LLMProvider
from app.core.ai.llm.models import ChatMessage, LLMResponse, LLMUsage
from app.core.config import settings
from app.core.exceptions import AIServiceException
from app.db.models.enums import MessageRole

logger = logging.getLogger(__name__)


class GeminiLLMProvider(LLMProvider):
    """Gemini implementation of the LLMProvider interface."""

    _ROLE_MAPPING: ClassVar[dict[MessageRole, type[BaseMessage]]] = {
        MessageRole.SYSTEM: SystemMessage,
        MessageRole.USER: HumanMessage,
        MessageRole.ASSISTANT: AIMessage,
    }

    def __init__(self) -> None:
        self._model_name = settings.gemini_chat_model

        self._client = ChatGoogleGenerativeAI(
            model=self._model_name,
            api_key=SecretStr(settings.google_api_key),
            temperature=settings.gemini_chat_temperature,
        )

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        """Generate an assistant response."""

        try:
            langchain_messages = self._to_langchain_messages(messages)

            response = self._client.invoke(langchain_messages)

        except AIServiceException:
            raise

        except Exception as exc:
            logger.exception(
                "LLM generation failed.",
                extra={"model": self._model_name, "message_count": len(messages)},
            )
            raise AIServiceException("Failed to generate LLM response") from exc

        llm_response = self._to_response(response)

        logger.info(
            "LLM response generated",
            extra=self._response_log_extra(llm_response, len(messages)),
        )

        return llm_response

    def _to_langchain_messages(self, messages: list[ChatMessage]) -> list[BaseMessage]:
        """Convert domain messages into LangChain messages."""

        converted: list[BaseMessage] = []

        for message in messages:
            message_cls = self._ROLE_MAPPING.get(message.role)

            if message_cls is None:
                raise AIServiceException(f"Unsupported message role: {message.role}")

            converted.append(message_cls(content=message.content))

        return converted

    @staticmethod
    def _to_usage(usage_metadata: dict[str, Any] | None) -> LLMUsage | None:
        """Convert Gemini usage metadata into the domain model."""

        if usage_metadata is None:
            return None

        return LLMUsage(
            prompt_tokens=usage_metadata.get("input_tokens"),
            completion_tokens=usage_metadata.get("output_tokens"),
            total_tokens=usage_metadata.get("total_tokens"),
        )

    def _to_response(self, response: AIMessage) -> LLMResponse:
        """Convert a LangChain response into the domain response."""

        return LLMResponse(
            content=self._extract_text(response.content),
            model=self._model_name,
            finish_reason=self._extract_finish_reason(
                getattr(response, "response_metadata", None)
            ),
            usage=self._to_usage(getattr(response, "usage_metadata", None)),
        )

    def _response_log_extra(
        self, response: LLMResponse, message_count: int
    ) -> dict[str, Any]:
        """Build structured log context for a generated response."""

        extra: dict[str, Any] = {
            "model": self._model_name,
            "message_count": message_count,
            "finish_reason": response.finish_reason,
        }

        if response.usage:
            extra.update(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return extra

    @staticmethod
    def _extract_finish_reason(metadata: dict[str, Any] | None) -> str:
        """Extract the finish reason from LangChain response metadata."""

        return (metadata or {}).get("finish_reason", "unknown")

    @staticmethod
    def _extract_text(content: str | list[str | dict[str, Any]]) -> str:
        """Normalize LangChain content blocks into plain text."""

        if isinstance(content, str):
            return content

        text_parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")

                if text is not None:
                    text_parts.append(str(text))

        return "".join(text_parts)
