from typing import Any, Final

EMBEDDING_DIMENSION: Final[int] = 768
"""Standard embedding dimension for the project.

All embedding providers must generate vectors with this dimensionality.
Changing this value requires a database migration and re-embedding all
stored document chunks.
"""

JSONDict = dict[str, Any]

TRANSIENT_AI_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {
        408,  # Request Timeout
        429,  # Too Many Requests / Resource Exhausted / Rate Limit
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    }
)
"""HTTP status codes representing transient, retryable AI service failures."""

PERMANENT_AI_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {
        400,  # Bad Request / Invalid Argument
        401,  # Unauthorized / Invalid API Key
        403,  # Forbidden / Permission Denied
        404,  # Not Found / Model Not Found
        422,  # Unprocessable Entity
    }
)
"""HTTP status codes representing permanent, non-retryable AI client errors."""


CELERY_VISIBILITY_TIMEOUT_SECONDS: Final[int] = 3600
