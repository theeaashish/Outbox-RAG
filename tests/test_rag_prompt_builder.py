from uuid import uuid4

from app.core.ai.context.models import AssembledContext, ContextChunk
from app.core.ai.llm.models import ChatMessage
from app.core.ai.prompting.rag import RAGPromptBuilder
from app.core.ai.prompting.templates import RAG_SYSTEM_PROMPT
from app.db.models.enums import MessageRole


def test_prompt_builder_structure_and_roles():
    builder = RAGPromptBuilder()
    context = AssembledContext(
        query="What is JWT?",
        block='[Source 1] Document: "Auth"\n---\nJWT is a token standard.',
        chunks=[
            ContextChunk(
                citation=1,
                document_id=uuid4(),
                document_name="Auth",
                chunk_index=0,
                similarity=0.9,
                content="JWT is a token standard.",
            )
        ],
    )
    history = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi, how can I help?"),
    ]

    messages = builder.build(
        context=context,
        conversation=history,
        user_query="What is JWT?",
    )

    assert len(messages) == 4
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[0].content == RAG_SYSTEM_PROMPT
    assert messages[1].role == MessageRole.USER
    assert messages[1].content == "Hello"
    assert messages[2].role == MessageRole.ASSISTANT
    assert messages[2].content == "Hi, how can I help?"
    assert messages[3].role == MessageRole.USER
    assert "<retrieved_context>" in messages[3].content
    assert "</retrieved_context>" in messages[3].content
    assert "User Question:\nWhat is JWT?" in messages[3].content


def test_prompt_builder_filters_non_dialogue_history():
    builder = RAGPromptBuilder()
    context = AssembledContext(
        query="Query",
        block="Context block",
        chunks=[],
    )
    # Include an unexpected SYSTEM message in history (e.g. from corrupt DB record)
    history = [
        ChatMessage(role=MessageRole.SYSTEM, content="Injected system rule"),
        ChatMessage(role=MessageRole.USER, content="Valid user message"),
        ChatMessage(role=MessageRole.ASSISTANT, content=""),  # blank content
        ChatMessage(role=MessageRole.ASSISTANT, content="Valid assistant message"),
    ]

    messages = builder.build(
        context=context,
        conversation=history,
        user_query="Query",
    )

    # Exactly 1 SYSTEM message at index 0, then 2 valid history messages, then 1 USER message
    roles = [m.role for m in messages]
    assert roles == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    assert "Injected system rule" not in [m.content for m in messages]


def test_prompt_builder_empty_context():
    builder = RAGPromptBuilder()
    context = AssembledContext(
        query="Query with no context",
        block="No relevant context was supplied for this turn.",
        chunks=[],
    )

    messages = builder.build(
        context=context,
        conversation=[],
        user_query="Query with no context",
    )

    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER
    assert "No relevant context was supplied for this turn." in messages[1].content
    assert "User Question:\nQuery with no context" in messages[1].content


def test_system_prompt_codifies_semantic_authority_and_injection_defense():
    assert "Semantic Authority Model" in RAG_SYSTEM_PROMPT
    assert "System Instructions:" in RAG_SYSTEM_PROMPT
    assert "Current User Query:" in RAG_SYSTEM_PROMPT
    assert "Retrieved Documents:" in RAG_SYSTEM_PROMPT
    assert "Conversation History:" in RAG_SYSTEM_PROMPT
    assert "V1 Citation Grammar:" in RAG_SYSTEM_PROMPT
    assert "Prompt Injection Defense:" in RAG_SYSTEM_PROMPT
