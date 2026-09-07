from __future__ import annotations

import logging
import re
from io import BytesIO

from pypdf import PdfReader

from app.core.document.models import Block, BlockType, DocumentMetadata, ParsedDocument
from app.core.document.parsers.base import DocumentParser
from app.core.exceptions import DocumentParsingException

logger = logging.getLogger(__name__)


class PDFParser(DocumentParser):
    """Parser for converting PDF text and page structure to canonical blocks."""

    _LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+\u2022\u2023\u25e6\u25aa]\s+|\d+[.)]\s+)")

    @staticmethod
    def _title(reader: PdfReader) -> str | None:
        metadata = reader.metadata
        title = getattr(metadata, "title", None) if metadata is not None else None
        if isinstance(title, str) and title.strip():
            return title
        return None

    @classmethod
    def _page_blocks(
        cls,
        *,
        text: str,
        page_number: int,
        block_index: int,
    ) -> tuple[list[Block], int]:
        lines = text.splitlines()
        blocks: list[Block] = []
        current_paragraph: list[str] = []
        list_lines: list[str] = []

        def emit_paragraph() -> None:
            nonlocal current_paragraph
            if current_paragraph:
                blocks.append(
                    Block(
                        type=BlockType.PARAGRAPH,
                        text="\n".join(current_paragraph),
                        page_number=page_number,
                        block_index=block_index + len(blocks),
                    )
                )
                current_paragraph = []

        def emit_list() -> None:
            nonlocal list_lines
            if len(list_lines) >= 2:
                blocks.append(
                    Block(
                        type=BlockType.LIST,
                        text="\n".join(list_lines),
                        page_number=page_number,
                        block_index=block_index + len(blocks),
                    )
                )
            else:
                current_paragraph.extend(list_lines)
            list_lines = []

        for line in lines:
            if not line.strip():
                emit_list()
                emit_paragraph()
                continue

            if cls._LIST_MARKER_RE.match(line):
                if current_paragraph:
                    emit_paragraph()
                list_lines.append(line)
            else:
                emit_list()
                current_paragraph.append(line)

        emit_list()
        emit_paragraph()
        return blocks, block_index + len(blocks)

    def parse(self, content: bytes) -> ParsedDocument:
        """Parse a PDF into ordered, page-aware canonical blocks."""

        try:
            reader = PdfReader(BytesIO(content))
            blocks: list[Block] = []
            block_index = 0

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                page_blocks, block_index = self._page_blocks(
                    text=text,
                    page_number=page_number,
                    block_index=block_index,
                )
                blocks.extend(page_blocks)

            return ParsedDocument(
                metadata=DocumentMetadata(
                    title=self._title(reader),
                    page_count=len(reader.pages),
                ),
                blocks=tuple(blocks),
            )
        except DocumentParsingException:
            raise
        except Exception as exc:
            logger.exception(
                "PDF parsing failed",
                extra={"content_size": len(content)},
            )
            raise DocumentParsingException("Failed to parse PDF document") from exc

    def extract_text(self, content: bytes) -> str:
        """Flatten canonical PDF blocks for legacy callers."""

        document = self.parse(content=content)
        return "\n".join(block.text for block in document.blocks)
