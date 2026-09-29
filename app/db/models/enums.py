from enum import StrEnum


class DocumentStatus(StrEnum):
    """Document processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class MessageRole(StrEnum):
    """Message role"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class OutboxEventType(StrEnum):
    """Outbox event type for asynchronous task dispatch"""

    DOCUMENT_PROCESS = "document.process"


class UserStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DISABLED = "disabled"


class AuthProvider(StrEnum):
    PASSWORD = "password"
    GOOGLE = "google"
    GITHUB = "github"


class FeedbackRating(StrEnum):
    """User feedback rating for an assistant message"""

    UP = "up"
    DOWN = "down"
