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
