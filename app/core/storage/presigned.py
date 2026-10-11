from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PresignedUrl:
    """Temporary authorization details for a storage operation."""

    url: str
    method: str
    expires_in: int
    headers: dict[str, str] = field(default_factory=dict)
    form_fields: dict[str, str] = field(default_factory=dict)


class PresignedUrlService(ABC):
    """Capability for generating temporary storage authorizations."""

    @abstractmethod
    def create_upload_url(
        self,
        *,
        key: str,
        content_type: str,
        expires_in: int,
        max_size_bytes: int,
    ) -> PresignedUrl:
        """Create temporary authorization for uploading an object."""
        raise NotImplementedError

    @abstractmethod
    def create_download_url(
        self,
        *,
        key: str,
        expires_in: int,
    ) -> PresignedUrl:
        """Create a temporary URL that allows a client to download an object."""
        raise NotImplementedError
