from __future__ import annotations

import httpx
from google.genai.errors import APIError, ClientError, ServerError

from app.core.constants import PERMANENT_AI_STATUS_CODES, TRANSIENT_AI_STATUS_CODES
from app.core.exceptions import (
    AIServiceException,
    TransientAIServiceException,
)


def _extract_status_code(exc: BaseException) -> int | None:
    """Extract integer HTTP status code from supported client/server/httpx exceptions."""
    if isinstance(exc, (ClientError, ServerError, APIError)):
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code

    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int):
            return status

    return None


def is_transient_ai_error(exc: BaseException | None) -> bool:
    """
    Determine whether an exception from an AI provider represents a transient,
    retryable failure based on structured exception types, HTTP status codes,
    and error attributes.
    """
    if exc is None:
        return False

    # Already classified domain exceptions
    if isinstance(exc, TransientAIServiceException):
        return True
    if isinstance(exc, AIServiceException):
        return False

    # Network and timeout exceptions
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            ConnectionResetError,
            ConnectionRefusedError,
            BrokenPipeError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.TransportError,
        ),
    ):
        return True

    # Google GenAI Server Error (5xx)
    if isinstance(exc, ServerError):
        return True

    # Inspect extracted HTTP status codes
    code = _extract_status_code(exc)
    if code is not None:
        if code in TRANSIENT_AI_STATUS_CODES or 500 <= code <= 599:
            return True
        if code in PERMANENT_AI_STATUS_CODES:
            return False

    # Recursively unwrap nested causes or contexts
    cause = getattr(exc, "__cause__", None)
    if cause is not None and is_transient_ai_error(cause):
        return True

    context = getattr(exc, "__context__", None)
    return bool(context is not None and is_transient_ai_error(context))


def classify_ai_exception(
    exc: Exception,
    *,
    transient_message: str = "AI service temporarily unavailable",
    permanent_message: str = "AI service operation failed",
) -> AIServiceException:
    """
    Wrap an underlying AI provider exception into either TransientAIServiceException
    (if transient) or AIServiceException (if permanent).
    """
    if isinstance(exc, (TransientAIServiceException, AIServiceException)):
        return exc

    if is_transient_ai_error(exc):
        return TransientAIServiceException(transient_message)

    return AIServiceException(permanent_message)
