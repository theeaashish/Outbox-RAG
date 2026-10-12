from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StoredObjectMetadata:
    """Metadata reported by the configured object storage backend."""

    size: int
    content_type: str | None
    etag: str | None


class StorageService(ABC):
    """Abstract interface for application object storage."""

    @abstractmethod
    def save(self, key: str, content: bytes) -> None:
        """Persist content at the given storage key."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the object at the given storage key if it exists."""
        raise NotImplementedError

    @abstractmethod
    def head(self, key: str) -> StoredObjectMetadata:
        """Return metadata for the object at the given storage key."""
        raise NotImplementedError

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Read object content from the given storage key."""
        raise NotImplementedError
