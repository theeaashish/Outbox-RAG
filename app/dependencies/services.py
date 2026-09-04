from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.core.ai.chunking.base import TextChunker
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.auth.passwords import PasswordHasherService
from app.core.auth.session import SessionTokenService
from app.core.document.hasher import FileHasher
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.document.validator import UploadValidator
from app.core.storage import StorageService
from app.dependencies.core import (
    get_document_parser_registry,
    get_embedding_generator,
    get_file_hasher,
    get_password_hasher,
    get_session_token_service,
    get_storage_service,
    get_text_chunker,
    get_upload_validator,
)
from app.dependencies.repositories import (
    DBSession,
    get_document_chunk_repository,
    get_document_repository,
    get_knowledge_base_repository,
    get_password_credential_repository,
    get_session_repository,
    get_user_repository,
)
from app.modules.auth.service import AuthService
from app.modules.document.service import DocumentService
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.password_credential import PasswordCredentialRepository
from app.repositories.session import SessionRepository
from app.repositories.user import UserRepository

KnowledgeBaseRepositoryDep = Annotated[
    KnowledgeBaseRepository,
    Depends(get_knowledge_base_repository),
]

DocumentRepositoryDep = Annotated[
    DocumentRepository,
    Depends(get_document_repository),
]

DocumentChunkRepositoryDep = Annotated[
    DocumentChunkRepository,
    Depends(get_document_chunk_repository),
]

TextChunkerDep = Annotated[
    TextChunker,
    Depends(get_text_chunker),
]

EmbeddingGeneratorDep = Annotated[
    EmbeddingGenerator,
    Depends(get_embedding_generator),
]

ParserRegistryDep = Annotated[
    DocumentParserRegistry,
    Depends(get_document_parser_registry),
]

UploadValidatorDep = Annotated[
    UploadValidator,
    Depends(get_upload_validator),
]

FileHasherDep = Annotated[
    FileHasher,
    Depends(get_file_hasher),
]

StorageServiceDep = Annotated[
    StorageService,
    Depends(get_storage_service),
]

UserRepositoryDep = Annotated[
    UserRepository,
    Depends(get_user_repository),
]

PasswordCredentialRepositoryDep = Annotated[
    PasswordCredentialRepository,
    Depends(get_password_credential_repository),
]

PasswordHasherDep = Annotated[
    PasswordHasherService,
    Depends(get_password_hasher),
]

SessionRepositoryDep = Annotated[
    SessionRepository,
    Depends(get_session_repository),
]

SessionTokenServiceDep = Annotated[
    SessionTokenService,
    Depends(get_session_token_service),
]


def get_document_service(
    db: DBSession,
    knowledge_base_repository: KnowledgeBaseRepositoryDep,
    document_repository: DocumentRepositoryDep,
    validator: UploadValidatorDep,
    hasher: FileHasherDep,
    storage: StorageServiceDep,
) -> DocumentService:
    """Return a configured DocumentService."""

    return DocumentService(
        db=db,
        knowledge_base_repository=knowledge_base_repository,
        document_repository=document_repository,
        validator=validator,
        hasher=hasher,
        storage=storage,
    )


DocumentServiceDep = Annotated[
    DocumentService,
    Depends(get_document_service),
]


def get_auth_service(
    db: DBSession,
    user_repository: UserRepositoryDep,
    password_credential_repository: PasswordCredentialRepositoryDep,
    session_repository: SessionRepositoryDep,
    password_hasher: PasswordHasherDep,
    session_token_service: SessionTokenServiceDep,
) -> AuthService:
    """Return a configured AuthService."""

    return AuthService(
        db=db,
        user_repository=user_repository,
        password_credential_repository=password_credential_repository,
        session_repository=session_repository,
        password_hasher=password_hasher,
        session_token_service=session_token_service,
    )


AuthServiceDep = Annotated[
    AuthService,
    Depends(get_auth_service),
]
