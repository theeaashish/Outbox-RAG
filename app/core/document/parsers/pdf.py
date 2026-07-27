from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from app.core.document.parsers.base import DocumentParser


class PDFParser(DocumentParser):
    """Parser for extracting text from pdf document"""

    def extract_text(self, content: bytes) -> str:
        """Extract text from pdf document"""
        reader = PdfReader(BytesIO(content))

        pages: list[str] = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)
