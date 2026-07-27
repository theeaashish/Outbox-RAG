from __future__ import annotations

from app.core.document.parsers.base import DocumentParser
from app.core.document.parsers.pdf import PDFParser


class DocumentParserFactory:
    """Factory for creating document parsers"""

    _parsers: dict[str, type[DocumentParser]] = {  # noqa: RUF012
        ".pdf": PDFParser,
    }

    @classmethod
    def get_parser(cls, extension: str) -> DocumentParser:
        """Get parser for given file extension"""
        try:
            parser_class = cls._parsers[extension.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported document type: {extension}") from exc

        return parser_class()
