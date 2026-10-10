from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PresignedUrl:
    """Temporary authorization for a storage operation."""

    url: str
    method: str
    expires_in: int
    headers: dict[str, str]


class PresignedUrlService(ABC):
    """Capability for generating temporary storage authorizations."""

    @abstractmethod
    def create_upload_url(
        self,
        *,
        key: str,
        content_type: str,
        expires_in: int,
    ) -> PresignedUrl:
        """Create a temporary URL for uploading an object."""
        raise NotImplementedError

    @abstractmethod
    def create_download_url(
        self,
        *,
        key: str,
        expires_in: int,
    ) -> PresignedUrl:
        """Create a temporary URL for downloading an object."""
        raise NotImplementedError
