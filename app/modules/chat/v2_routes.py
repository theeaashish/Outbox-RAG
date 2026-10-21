from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.core.exceptions import ValidationException
from app.dependencies.auth import CurrentUser
from app.dependencies.chat import ChatControllerDep, LLMProviderRegistryDep
from app.modules.chat.protocol.schemas import RunAgentInput
from app.modules.chat.schemas import ChatModelInfo, ChatModelsResponse

router = APIRouter(
    prefix="/conversations",
    tags=["Chat V2"],
)


@router.get(
    "/models",
    response_model=ChatModelsResponse,
)
def list_chat_models(
    current_user: CurrentUser,
    registry: LLMProviderRegistryDep,
) -> ChatModelsResponse:
    """Return available chat models and default selection for AG-UI / v2 clients."""

    return ChatModelsResponse(
        default_model=registry.default_model_id,
        models=[
            ChatModelInfo.model_validate(m) for m in registry.get_available_models()
        ],
    )


@router.post("/{conversation_id}/messages/stream")
async def stream_ag_ui_message(
    *,
    conversation_id: UUID,
    request: RunAgentInput,
    http_request: Request,
    current_user: CurrentUser,
    controller: ChatControllerDep,
) -> EventSourceResponse:
    """Submit an AG-UI run turn and return an AG-UI SSE event stream."""

    # 1. Validate threadId if provided
    if request.thread_id:
        try:
            thread_uuid = UUID(request.thread_id)
        except ValueError as exc:
            raise ValidationException("threadId must be a valid UUID") from exc
        if thread_uuid != conversation_id:
            raise ValidationException("threadId does not match the URL conversation_id")

    # 2. Extract latest user message
    if not request.messages:
        raise ValidationException("RunAgentInput must contain at least one message")

    last_message = request.messages[-1]
    if last_message.role.lower() != "user":
        raise ValidationException(
            "The latest message in RunAgentInput must have role 'user'"
        )

    content = last_message.content.strip()
    if not content:
        raise ValidationException("User message content cannot be blank")

    if len(content) > settings.chat_max_message_characters:
        raise ValidationException(
            f"User message exceeds maximum length of {settings.chat_max_message_characters} characters"
        )

    model: str | None = None
    if isinstance(request.forwarded_props, dict):
        prop_model = request.forwarded_props.get("model")
        if isinstance(prop_model, str):
            model = prop_model

    return await controller.stream_ag_ui_message(
        user_id=current_user.id,
        conversation_id=conversation_id,
        content=content,
        request=http_request,
        client_run_id=request.run_id,
        model=model,
    )
