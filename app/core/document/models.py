from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BlockType(StrEnum):
    """Canonical semantic types supported by the document pipeline."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Source metadata that is independent of application persistence."""

    title: str | None
    page_count: int | None


@dataclass(frozen=True, slots=True)
class Block:
    """A canonical source block with stable provenance information."""

    type: BlockType
    text: str
    page_number: int | None
    block_index: int


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Canonical parser output shared by parsers and normalization."""

    metadata: DocumentMetadata
    blocks: tuple[Block, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    """Canonical domain chunk with stable provenance information."""

    content: str
    chunk_index: int
    source_block_indexes: tuple[int, ...]
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("Chunk content must not be empty or whitespace-only.")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative.")
        if not isinstance(self.source_block_indexes, tuple):
            object.__setattr__(
                self, "source_block_indexes", tuple(self.source_block_indexes)
            )
        if not self.source_block_indexes:
            raise ValueError("source_block_indexes must be non-empty.")
        if any(idx < 0 for idx in self.source_block_indexes):
            raise ValueError("Every source_block_index must be non-negative.")
        if any(
            self.source_block_indexes[i] > self.source_block_indexes[i + 1]
            for i in range(len(self.source_block_indexes) - 1)
        ):
            raise ValueError(
                "source_block_indexes must preserve non-decreasing source order."
            )
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError(
                "page_start and page_end must either both be None or both be present."
            )
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_start > self.page_end
        ):
            raise ValueError("page_start must be less than or equal to page_end.")
        if self.char_start is not None or self.char_end is not None:
            raise ValueError("char_start and char_end must remain None in V1.")
