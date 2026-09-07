from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.ai.chunking.base import DocumentChunker
from app.core.config import settings
from app.core.document.models import Block, Chunk, ParsedDocument


class DocumentAwareChunker(DocumentChunker):
    """
    V1 document-aware chunking strategy.

    V1 is block-aware, not type-aware. The algorithm preserves source block
    boundaries and provenance, but does not yet apply special chunking rules
    based on HEADING/LIST/TABLE types. This leaves room for future structure-aware
    policies without implementing them now.
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )

        if self._chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self._chunk_size}")
        if not (0 <= self._chunk_overlap < self._chunk_size):
            raise ValueError(
                f"chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size, "
                f"got chunk_overlap={self._chunk_overlap}, chunk_size={self._chunk_size}"
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def split(self, document: ParsedDocument) -> tuple[Chunk, ...]:
        """
        Split a parsed document into canonical domain chunks.

        Walks document.blocks in source order, accumulating complete blocks
        until adding the next block would exceed chunk_size. Oversized blocks
        are split with RecursiveCharacterTextSplitter, validating that every
        emitted piece satisfies chunk_size.
        """
        chunks: list[Chunk] = []
        accumulated_blocks: list[Block] = []
        current_len = 0
        chunk_index = 0

        def flush_accumulated() -> None:
            nonlocal accumulated_blocks, current_len, chunk_index
            if not accumulated_blocks:
                return

            content = "\n\n".join(b.text for b in accumulated_blocks)
            if not content.strip():
                accumulated_blocks = []
                current_len = 0
                return

            source_block_indexes = tuple(b.block_index for b in accumulated_blocks)
            pages = [
                b.page_number for b in accumulated_blocks if b.page_number is not None
            ]
            page_start = min(pages) if pages else None
            page_end = max(pages) if pages else None

            chunks.append(
                Chunk(
                    content=content,
                    chunk_index=chunk_index,
                    source_block_indexes=source_block_indexes,
                    page_start=page_start,
                    page_end=page_end,
                    char_start=None,
                    char_end=None,
                )
            )
            chunk_index += 1
            accumulated_blocks = []
            current_len = 0

        for block in document.blocks:
            text = block.text
            if not text or not text.strip():
                continue

            block_len = len(text)

            # Check if block itself exceeds chunk_size
            if block_len > self._chunk_size:
                # 1. Flush the current accumulated chunk
                flush_accumulated()

                # 2. Split only that block with RecursiveCharacterTextSplitter
                pieces = self._splitter.split_text(text)

                # 3. Validate pieces and emit one Chunk per split piece
                for piece in pieces:
                    if not piece or not piece.strip():
                        continue
                    if len(piece) > self._chunk_size:
                        raise ValueError(
                            f"Oversized-block piece length ({len(piece)}) exceeds "
                            f"configured chunk_size ({self._chunk_size}) for block {block.block_index}."
                        )
                    chunks.append(
                        Chunk(
                            content=piece,
                            chunk_index=chunk_index,
                            source_block_indexes=(block.block_index,),
                            page_start=block.page_number,
                            page_end=block.page_number,
                            char_start=None,
                            char_end=None,
                        )
                    )
                    chunk_index += 1
                continue

            # Normal block: check if it fits into current accumulated chunk
            projected_len = (
                current_len + 2 + block_len if accumulated_blocks else block_len
            )

            if projected_len <= self._chunk_size:
                accumulated_blocks.append(block)
                current_len = projected_len
            else:
                # Next block exceeds target, flush and start new chunk
                flush_accumulated()
                accumulated_blocks.append(block)
                current_len = block_len

        # Flush any remaining accumulated blocks
        flush_accumulated()

        return tuple(chunks)


# Compatibility-only alias
RecursiveTextChunker = DocumentAwareChunker
