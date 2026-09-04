from __future__ import annotations

from functools import lru_cache

from app.core.ai.chunking.base import TextChunker
from app.core.ai.chunking.recursive import RecursiveTextChunker
from app.core.ai.context.assembler import ContextAssembler
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.ai.embeddings.gemini import GeminiEmbeddingGenerator
from app.core.ai.llm.base import LLMProvider
from app.core.ai.llm.gemini import GeminiLLMProvider
from app.core.ai.prompting.base import PromptBuilder
from app.core.ai.prompting.rag import RAGPromptBuilder
from app.core.auth.passwords import PasswordHasherService
from app.core.auth.session import SessionTokenService
from app.core.config import settings
from app.core.document.hasher import FileHasher
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.document.validator import UploadValidator
from app.core.pagination import CursorCodec
from app.core.storage import LocalFilesystemStorage, StorageService


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


@lru_cache
def get_storage_service() -> StorageService:
    """Return the application's storage service"""
    return LocalFilesystemStorage(root_directory=settings.upload_directory)


@lru_cache
def get_context_assembler() -> ContextAssembler:
    """Return the application's context assembler."""

    return ContextAssembler()


@lru_cache
def get_prompt_builder() -> PromptBuilder:
    """Return the application's RAG prompt builder."""

    return RAGPromptBuilder()


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Return the application's chat language model provider."""

    return GeminiLLMProvider()


@lru_cache
def get_cursor_codec() -> CursorCodec:
    """Return the application's signed cursor codec."""

    return CursorCodec(
        signing_key=settings.cursor_signing_key,
        previous_signing_key=settings.cursor_previous_signing_key,
    )


@lru_cache
def get_password_hasher() -> PasswordHasherService:
    return PasswordHasherService()


@lru_cache
def get_session_token_service() -> SessionTokenService:
    """Return the application's session token service."""
    return SessionTokenService()
