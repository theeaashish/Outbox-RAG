from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.core.ai.context.assembler import ContextAssembler
from app.core.ai.llm.base import LLMProvider
from app.core.ai.prompting.base import PromptBuilder
from app.core.config import settings
from app.dependencies.core import (
    get_context_assembler,
    get_llm_provider,
    get_prompt_builder,
)
from app.dependencies.repositories import (
    DBSession,
    get_conversation_repository,
    get_message_repository,
)
from app.dependencies.retrieval import RetrievalServiceDep
from app.modules.chat.controller import ChatController
from app.modules.chat.service import ChatService
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository

ConversationRepositoryDep = Annotated[
    ConversationRepository,
    Depends(get_conversation_repository),
]

MessageRepositoryDep = Annotated[
    MessageRepository,
    Depends(get_message_repository),
]

ContextAssemblerDep = Annotated[
    ContextAssembler,
    Depends(get_context_assembler),
]

PromptBuilderDep = Annotated[
    PromptBuilder,
    Depends(get_prompt_builder),
]

LLMProviderDep = Annotated[
    LLMProvider,
    Depends(get_llm_provider),
]


def get_chat_service(
    db: DBSession,
    conversation_repository: ConversationRepositoryDep,
    message_repository: MessageRepositoryDep,
    retrieval_service: RetrievalServiceDep,
    context_assembler: ContextAssemblerDep,
    prompt_builder: PromptBuilderDep,
    llm_provider: LLMProviderDep,
) -> ChatService:
    """Return a configured ChatService."""

    return ChatService(
        db=db,
        conversation_repository=conversation_repository,
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
        history_message_limit=settings.chat_history_message_limit,
        retrieval_limit=settings.default_top_k,
        similarity_threshold=settings.similarity_threshold,
        stream_max_buffered_characters=settings.chat_stream_max_buffered_characters,
    )


ChatServiceDep = Annotated[
    ChatService,
    Depends(get_chat_service),
]


def get_chat_controller(
    chat_service: ChatServiceDep,
) -> ChatController:
    """Return a configured ChatController."""

    return ChatController(chat_service=chat_service)


ChatControllerDep = Annotated[
    ChatController,
    Depends(get_chat_controller),
]
