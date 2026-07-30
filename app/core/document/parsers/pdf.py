from __future__ import annotations

import logging
from io import BytesIO

from pypdf import PdfReader

from app.core.document.parsers.base import DocumentParser
from app.core.exceptions import DocumentParsingException

logger = logging.getLogger(__name__)


class PDFParser(DocumentParser):
    """Parser for extracting text from pdf document"""

    def extract_text(self, content: bytes) -> str:
        """Extract text from pdf document"""

        try:
            reader = PdfReader(BytesIO(content))

            pages: list[str] = []

            for page in reader.pages:
                text = page.extract_text()

                if text:
                    pages.append(text)

            return "\n".join(pages)
        except DocumentParsingException:
            raise
        except Exception as exc:
            logger.exception(
                "PDF parsing failed",
                extra={"content_size": len(content)},
            )
            raise DocumentParsingException("Failed to parse PDF document") from exc
