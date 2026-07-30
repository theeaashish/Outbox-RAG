from __future__ import annotations

from abc import ABC, abstractmethod


class StorageService(ABC):
    """Abstract interface for binary object storage."""

    @abstractmethod
    def save(self, path: str, content: bytes) -> None:
        """Persist content at the given relative storage path."""

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete content at the given relative storage path if it exists."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Return whether content exists at the given relative storage path."""
