from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IncomingFile:
    """Framework-agnostic representation of an uploaded document."""

    filename: str
    content_type: str | None
    size: int
    content: bytes
