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
