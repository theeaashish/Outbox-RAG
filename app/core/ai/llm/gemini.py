from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, ClassVar, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from app.core.ai.classification import classify_ai_exception
from app.core.ai.llm.base import LLMProvider
from app.core.ai.llm.models import (
    ChatMessage,
    LLMResponse,
    LLMStream,
    LLMStreamCompletion,
    LLMStreamDelta,
    LLMStreamEvent,
    LLMUsage,
)
from app.core.config import settings
from app.core.exceptions import AIServiceException
from app.db.models.enums import MessageRole

logger = logging.getLogger(__name__)


class _GeminiLLMStream:
    """Closable adapter from LangChain/Gemini chunks to provider-neutral events."""

    def __init__(
        self,
        *,
        source: Iterator[Any],
        model_name: str,
        message_count: int,
    ) -> None:
        self._source = source
        self._model_name = model_name
        self._message_count = message_count
        self._closed = False
        self._iterator = self._events()

    def __iter__(self) -> Iterator[LLMStreamEvent]:
        return self

    def __next__(self) -> LLMStreamEvent:
        return next(self._iterator)

    def close(self) -> None:
        """Release the underlying vendor iterator exactly once."""

        if self._closed:
            return
        self._closed = True
        close = getattr(self._source, "close", None)
        if callable(close):
            close()

    def _events(self) -> Iterator[LLMStreamEvent]:
        last_chunk: Any | None = None
        try:
            for chunk in self._source:
                if self._closed:
                    return
                last_chunk = chunk
                content = GeminiLLMProvider._extract_text(cast(Any, chunk.content))
                if content:
                    yield LLMStreamDelta(content=content)

            metadata = getattr(last_chunk, "response_metadata", None)
            usage_metadata = getattr(last_chunk, "usage_metadata", None)
            yield LLMStreamCompletion(
                model=self._model_name,
                finish_reason=GeminiLLMProvider._normalize_finish_reason(
                    GeminiLLMProvider._extract_finish_reason(metadata)
                ),
                usage=GeminiLLMProvider._to_usage(usage_metadata),
            )
        except AIServiceException:
            raise
        except Exception as exc:
            logger.exception(
                "LLM stream failed",
                extra={
                    "model": self._model_name,
                    "message_count": self._message_count,
                },
            )
            raise classify_ai_exception(
                exc,
                transient_message="LLM stream temporarily unavailable",
                permanent_message="LLM stream failed",
            ) from exc
        finally:
            self.close()


class GeminiLLMProvider(LLMProvider):
    """Gemini implementation of the LLMProvider interface."""

    _ROLE_MAPPING: ClassVar[dict[MessageRole, type[BaseMessage]]] = {
        MessageRole.SYSTEM: SystemMessage,
        MessageRole.USER: HumanMessage,
        MessageRole.ASSISTANT: AIMessage,
    }

    _FINISH_REASON_MAP: ClassVar[dict[str, str]] = {
        "stop": "stop",
        "STOP": "stop",
        "length": "length",
        "MAX_TOKENS": "length",
        "max_tokens": "length",
        "safety": "safety",
        "SAFETY": "safety",
        "RECITATION": "safety",
        "recitation": "safety",
        "OTHER": "error",
        "other": "error",
        "FINISH_REASON_UNSPECIFIED": "unknown",
    }

    def __init__(self) -> None:
        self._model_name = settings.gemini_chat_model

        self._client = ChatGoogleGenerativeAI(
            model=self._model_name,
            api_key=SecretStr(settings.google_api_key),
            temperature=settings.gemini_chat_temperature,
            max_output_tokens=settings.gemini_chat_max_output_tokens,
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
            raise classify_ai_exception(
                exc,
                transient_message="LLM generation temporarily unavailable",
                permanent_message="Failed to generate LLM response",
            ) from exc

        llm_response = self._to_response(response)

        logger.info(
            "LLM response generated",
            extra=self._response_log_extra(llm_response, len(messages)),
        )

        return llm_response

    def stream(self, messages: list[ChatMessage]) -> LLMStream:
        """Stream Gemini output as normalized provider-neutral events."""

        try:
            langchain_messages = self._to_langchain_messages(messages)
            source = self._client.stream(langchain_messages)
        except AIServiceException:
            raise
        except Exception as exc:
            logger.exception(
                "LLM stream initialization failed",
                extra={"model": self._model_name, "message_count": len(messages)},
            )
            raise classify_ai_exception(
                exc,
                transient_message="LLM stream temporarily unavailable",
                permanent_message="Failed to start LLM stream",
            ) from exc

        return _GeminiLLMStream(
            source=cast(Iterator[Any], source),
            model_name=self._model_name,
            message_count=len(messages),
        )

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
            finish_reason=self._normalize_finish_reason(
                self._extract_finish_reason(
                    getattr(response, "response_metadata", None)
                )
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

        return str((metadata or {}).get("finish_reason", "unknown"))

    @classmethod
    def _normalize_finish_reason(cls, finish_reason: str) -> str:
        """Map vendor finish reasons onto the controlled stream vocabulary."""

        return cls._FINISH_REASON_MAP.get(finish_reason, "unknown")

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
