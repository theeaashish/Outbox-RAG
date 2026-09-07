from __future__ import annotations

from app.core.document.models import Block, ParsedDocument


class DocumentNormalizer:
    """Apply deterministic, representation-only cleanup to parsed documents."""

    def normalize(self, document: ParsedDocument) -> ParsedDocument:
        """Return a normalized document without changing semantic provenance."""

        normalized_blocks: list[Block] = []

        for block in document.blocks:
            text = block.text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                continue

            normalized_blocks.append(
                Block(
                    type=block.type,
                    text=text,
                    page_number=block.page_number,
                    block_index=block.block_index,
                )
            )

        return ParsedDocument(
            metadata=document.metadata,
            blocks=tuple(normalized_blocks),
        )
