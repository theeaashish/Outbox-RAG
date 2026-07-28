from __future__ import annotations

import pytest

from app.core.document.parsers.base import DocumentParser
from app.core.document.parsers.pdf import PDFParser
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.exceptions import UnsupportedDocumentTypeError


class DummyTextParser(DocumentParser):
    def extract_text(self, content: bytes) -> str:
        return content.decode("utf-8")


def test_registry_default_parsers():
    registry = DocumentParserRegistry()
    pdf_parser = registry.get_parser(".pdf")
    assert isinstance(pdf_parser, PDFParser)


def test_registry_supports():
    registry = DocumentParserRegistry()
    assert registry.supports(".pdf") is True
    assert registry.supports("pdf") is True
    assert registry.supports(".PDF") is True
    assert registry.supports(".docx") is False


def test_registry_case_insensitive_and_leading_dot():
    registry = DocumentParserRegistry()
    assert registry.get_parser(".pdf") is registry.get_parser("pdf")
    assert registry.get_parser(".PDF") is registry.get_parser("PDF")


def test_registry_register_custom_parser():
    registry = DocumentParserRegistry()
    dummy = DummyTextParser()
    assert registry.supports("txt") is False
    registry.register("txt", dummy)
    assert registry.supports("txt") is True
    assert registry.get_parser(".txt") is dummy
    assert dummy.extract_text(b"hello world") == "hello world"


def test_registry_unsupported_type_raises_custom_exception():
    registry = DocumentParserRegistry()
    with pytest.raises(
        UnsupportedDocumentTypeError, match="Unsupported document type: .docx"
    ):
        registry.get_parser(".docx")
