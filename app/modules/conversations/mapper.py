from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from app.core.pagination import CursorCodec, CursorPage, CursorResource
from app.db.models import Conversation, Message, MessageFeedback, MessageSource
from app.db.models.enums import FeedbackRating
from app.modules.conversations.schemas import (
    ConversationCursorPageResponse,
    ConversationListResponse,
    ConversationResponse,
    CursorPageInfo,
    MessageCursorPageResponse,
    MessageFeedbackResponse,
    MessageListResponse,
    MessageResponse,
    MessageSourceResponse,
)


def to_message_source_response(source: MessageSource) -> MessageSourceResponse:
    """Map a MessageSource ORM model to its API response."""

    return MessageSourceResponse(
        id=source.id,
        citation=source.citation_number,
        is_cited=source.is_cited,
        source_order=source.source_order,
        document_id=source.document_id,
        document_chunk_id=source.document_chunk_id,
        document_name=source.document_name_snapshot,
        chunk_index=source.chunk_index_snapshot,
        score=source.retrieval_score,
        page_start=source.page_start_snapshot,
        page_end=source.page_end_snapshot,
    )


def to_message_feedback_response(
    feedback: MessageFeedback,
) -> MessageFeedbackResponse:
    """Map a MessageFeedback ORM model to its API response."""

    return MessageFeedbackResponse(
        id=feedback.id,
        message_id=feedback.message_id,
        user_id=feedback.user_id,
        rating=feedback.rating,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )


def to_conversation_response(conversation: Conversation) -> ConversationResponse:
    """Map a conversation orm model to its API response."""

    return ConversationResponse(
        id=conversation.id,
        knowledge_base_id=conversation.knowledge_base_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def to_conversation_list_response(
    conversations: Sequence[Conversation],
) -> ConversationListResponse:
    """Map a list of conversation orm models to their API response."""

    return ConversationListResponse(
        results=[
            to_conversation_response(conversation) for conversation in conversations
        ]
    )


def to_message_response(
    message: Message,
    *,
    feedback: FeedbackRating | None = None,
) -> MessageResponse:
    """Map a message orm model to its API response."""

    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        updated_at=message.updated_at,
        sources=[to_message_source_response(s) for s in message.sources],
        feedback=feedback,
    )


def to_message_list_response(
    messages: Sequence[Message],
    *,
    feedback_map: dict[UUID, FeedbackRating] | None = None,
) -> MessageListResponse:
    """Map a list of message orm models to their API response."""

    feedback_lookup = feedback_map or {}
    return MessageListResponse(
        results=[
            to_message_response(
                message,
                feedback=feedback_lookup.get(message.id),
            )
            for message in messages
        ]
    )


def to_conversation_cursor_page_response(
    *,
    page: CursorPage[Conversation],
    page_size: int,
    knowledge_base_id: UUID,
    cursor_codec: CursorCodec,
) -> ConversationCursorPageResponse:
    """Map a conversation keyset page to the v2 API contract."""

    return ConversationCursorPageResponse(
        items=[to_conversation_response(item) for item in page.items],
        page_info=_to_page_info(
            page=page,
            page_size=page_size,
            resource=CursorResource.CONVERSATIONS,
            scope_id=knowledge_base_id,
            cursor_codec=cursor_codec,
        ),
    )


def to_message_cursor_page_response(
    *,
    page: CursorPage[Message],
    page_size: int,
    conversation_id: UUID,
    cursor_codec: CursorCodec,
    feedback_map: dict[UUID, FeedbackRating] | None = None,
) -> MessageCursorPageResponse:
    """Map a message keyset page to the v2 API contract."""

    feedback_lookup = feedback_map or {}
    return MessageCursorPageResponse(
        items=[
            to_message_response(
                item,
                feedback=feedback_lookup.get(item.id),
            )
            for item in page.items
        ],
        page_info=_to_page_info(
            page=page,
            page_size=page_size,
            resource=CursorResource.MESSAGES,
            scope_id=conversation_id,
            cursor_codec=cursor_codec,
        ),
    )


def _to_page_info[ModelT: Conversation | Message](
    *,
    page: CursorPage[ModelT],
    page_size: int,
    resource: CursorResource,
    scope_id: UUID,
    cursor_codec: CursorCodec,
) -> CursorPageInfo:
    if not page.items:
        return CursorPageInfo(
            next_cursor=None,
            previous_cursor=None,
            has_next_page=False,
            has_previous_page=False,
            page_size=page_size,
        )

    def encode(item: ModelT) -> str:
        return cursor_codec.encode(
            resource=resource,
            scope_id=scope_id,
            created_at=item.created_at,
            entity_id=item.id,
            snapshot_timestamp=page.snapshot_timestamp,
        )

    return CursorPageInfo(
        next_cursor=encode(page.items[-1]) if page.has_next_page else None,
        previous_cursor=encode(page.items[0]) if page.has_previous_page else None,
        has_next_page=page.has_next_page,
        has_previous_page=page.has_previous_page,
        page_size=page_size,
    )
