from __future__ import annotations

from abc import ABC, abstractmethod


class StorageService(ABC):
    """Abstract interface for file storage."""

    @abstractmethod
    def save(self, path: str, content: bytes) -> None:
        """Store file content at the given storage path."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a stored file."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Return whether a stored file exists."""
        raise NotImplementedError

    @abstractmethod
    def read(self, path: str) -> bytes:
        """Read file content from the given storage path."""
        raise NotImplementedError
