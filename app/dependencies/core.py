from __future__ import annotations

from functools import lru_cache

from app.core.ai.chunking.base import TextChunker
from app.core.ai.chunking.recursive import RecursiveTextChunker
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.ai.embeddings.gemini import GeminiEmbeddingGenerator
from app.core.document.hasher import FileHasher
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.document.validator import UploadValidator


@lru_cache
def get_text_chunker() -> TextChunker:
    """Return the application's text chunker"""
    return RecursiveTextChunker()


@lru_cache
def get_embedding_generator() -> EmbeddingGenerator:
    """Return the application's embedding generator"""
    return GeminiEmbeddingGenerator()


@lru_cache
def get_document_parser_registry() -> DocumentParserRegistry:
    """Return the application's document parser registry"""
    return DocumentParserRegistry()


@lru_cache
def get_file_hasher() -> FileHasher:
    """Return the application's file hasher"""
    return FileHasher()


@lru_cache
def get_upload_validator() -> UploadValidator:
    """Return the application's upload validator"""
    return UploadValidator(parser_registry=get_document_parser_registry())
