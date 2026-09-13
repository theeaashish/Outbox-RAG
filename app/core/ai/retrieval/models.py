from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.exceptions import DatabaseException
from app.db.models import DocumentChunk


@dataclass(slots=True, frozen=True)
class ChunkProvenance:
    """Immutable typed provenance extracted from DocumentChunk.chunk_metadata."""

    source_block_indexes: tuple[int, ...] = ()
    page_start: int | None = None
    page_end: int | None = None

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> ChunkProvenance:
        """Extract typed provenance from chunk metadata with strict validation.

        Treats malformed persisted metadata as invalid domain state, raising
        DatabaseException rather than silently coercing or repairing data.
        """
        if metadata is None:
            return cls()

        if not isinstance(metadata, dict):
            raise DatabaseException("Malformed metadata: expected dictionary or None")

        if not metadata:
            return cls()

        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")

        if (page_start is None) != (page_end is None):
            raise DatabaseException(
                "Malformed metadata: page_start and page_end must either both be present or both be absent"
            )

        if page_start is not None and (
            isinstance(page_start, bool)
            or not isinstance(page_start, int)
            or page_start < 0
        ):
            raise DatabaseException(
                "Malformed metadata: page_start must be a non-negative integer"
            )

        if page_end is not None and (
            isinstance(page_end, bool) or not isinstance(page_end, int) or page_end < 0
        ):
            raise DatabaseException(
                "Malformed metadata: page_end must be a non-negative integer"
            )

        if page_start is not None and page_end is not None and page_start > page_end:
            raise DatabaseException(
                "Malformed metadata: page_start cannot exceed page_end"
            )

        block_indexes: tuple[int, ...] = ()
        if "source_block_indexes" in metadata:
            raw_block_indexes = metadata["source_block_indexes"]
            if isinstance(raw_block_indexes, (str, bytes)) or not isinstance(
                raw_block_indexes, (list, tuple)
            ):
                raise DatabaseException(
                    "Malformed metadata: source_block_indexes must be a sequence"
                )
            if len(raw_block_indexes) == 0:
                raise DatabaseException(
                    "Malformed metadata: source_block_indexes must be non-empty when present"
                )
            for idx in raw_block_indexes:
                if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
                    raise DatabaseException(
                        "Malformed metadata: source_block_indexes must contain non-negative integers"
                    )
            for i in range(len(raw_block_indexes) - 1):
                if raw_block_indexes[i] > raw_block_indexes[i + 1]:
                    raise DatabaseException(
                        "Malformed metadata: source_block_indexes must be non-decreasing"
                    )
            block_indexes = tuple(raw_block_indexes)

        return cls(
            source_block_indexes=block_indexes,
            page_start=page_start,
            page_end=page_end,
        )


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """Represents a retrieved document chunk, canonical similarity, and typed provenance view."""

    chunk: DocumentChunk
    similarity: float

    @property
    def provenance(self) -> ChunkProvenance:
        """Typed provenance view parsed defensively from canonical chunk metadata."""
        return ChunkProvenance.from_metadata(self.chunk.chunk_metadata)
