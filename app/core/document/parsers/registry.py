from __future__ import annotations

from app.core.document.parsers.base import DocumentParser
from app.core.document.parsers.pdf import PDFParser
from app.core.exceptions import UnsupportedDocumentTypeError


class DocumentParserRegistry:
    """Registry mapping supported document file extensions to parser instances."""

    def __init__(self, parsers: dict[str, DocumentParser] | None = None) -> None:
        if parsers is not None:
            self._parsers: dict[str, DocumentParser] = {
                self._normalize_extension(ext): parser
                for ext, parser in parsers.items()
            }
        else:
            self._parsers: dict[str, DocumentParser] = {
                ".pdf": PDFParser(),
            }

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return ext

    def register(self, extension: str, parser: DocumentParser) -> None:
        """Register a parser instance for a given file extension."""
        normalized_ext = self._normalize_extension(extension)
        self._parsers[normalized_ext] = parser

    def get_parser(self, extension: str) -> DocumentParser:
        """Get parser instance for given file extension."""
        normalized_ext = self._normalize_extension(extension)
        try:
            return self._parsers[normalized_ext]
        except KeyError as exc:
            raise UnsupportedDocumentTypeError(extension) from exc
