from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(slots=True, frozen=True)
class ContextChunk:
    """
    Provider-agnostic representation of a retrieved chunk.

    Contains only the information required for prompt construction
    and citation generation.
    """

    citation: int
    document_id: UUID
    document_name: str
    chunk_index: int
    similarity: float
    content: str


@dataclass(slots=True, frozen=True)
class AssembledContext:
    """
    Final context package produced by the context assembler.
    """

    query: str
    block: str
    chunks: list[ContextChunk]
