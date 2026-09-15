from __future__ import annotations

from app.core.ai.llm.models import ChatMessage
from app.core.ai.prompting.conversational import ConversationalPromptBuilder
from app.core.ai.prompting.templates import CONVERSATIONAL_SYSTEM_PROMPT
from app.db.models.enums import MessageRole


def test_conversational_prompt_builder_structure():
    builder = ConversationalPromptBuilder()
    history = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there! How can I help?"),
    ]
    query = "Who are you?"

    messages = builder.build(conversation=history, user_query=query)

    assert len(messages) == 4
    # 1. System message
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[0].content == CONVERSATIONAL_SYSTEM_PROMPT

    # 2. History turns
    assert messages[1].role == MessageRole.USER
    assert messages[1].content == "Hello"
    assert messages[2].role == MessageRole.ASSISTANT
    assert messages[2].content == "Hi there! How can I help?"

    # 3. Final User turn
    assert messages[3].role == MessageRole.USER
    assert messages[3].content == "Who are you?"

    # Guarantee NO context tags in casual prompt
    assert "<retrieved_context>" not in messages[3].content
    assert "</retrieved_context>" not in messages[3].content


def test_conversational_prompt_builder_filters_non_dialogue_roles():
    builder = ConversationalPromptBuilder()
    history = [
        ChatMessage(role=MessageRole.SYSTEM, content="Internal system prompt leak"),
        ChatMessage(role=MessageRole.USER, content="Valid user message"),
        ChatMessage(role=MessageRole.ASSISTANT, content=""),  # blank content
        ChatMessage(role=MessageRole.ASSISTANT, content="Valid assistant message"),
    ]

    messages = builder.build(conversation=history, user_query="Hi")

    assert len(messages) == 4
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[0].content == CONVERSATIONAL_SYSTEM_PROMPT
    assert messages[1].role == MessageRole.USER
    assert messages[1].content == "Valid user message"
    assert messages[2].role == MessageRole.ASSISTANT
    assert messages[2].content == "Valid assistant message"
    assert messages[3].role == MessageRole.USER
    assert messages[3].content == "Hi"


def test_conversational_system_prompt_content():
    assert (
        "helpful, friendly, and professional conversational assistant"
        in CONVERSATIONAL_SYSTEM_PROMPT
    )
    assert (
        "Do not invent citations or use bracketed citation numbers"
        in CONVERSATIONAL_SYSTEM_PROMPT
    )
    # Verify no self-routing instruction
    assert "invite them to ask about those topics" not in CONVERSATIONAL_SYSTEM_PROMPT
