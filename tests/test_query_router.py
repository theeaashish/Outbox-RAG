from __future__ import annotations

import pytest

from app.core.ai.llm.models import ChatMessage
from app.core.ai.routing.models import ConversationMode, RoutingReason
from app.core.ai.routing.router import QueryRouter
from app.db.models.enums import MessageRole


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter()


def _user_msg(content: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content=content)


def _assistant_msg(content: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.ASSISTANT, content=content)


@pytest.mark.parametrize(
    "greeting",
    [
        "Hi",
        "hello",
        "Hey",
        "Good morning",
        "good afternoon",
        "good evening",
        "greetings!",
        "howdy",
        "sup?",
        "Hello there",
    ],
)
def test_pure_greetings_route_to_casual(router: QueryRouter, greeting: str):
    decision = router.route(query=greeting, conversation=[])
    assert decision.mode == ConversationMode.CASUAL
    assert decision.reason == RoutingReason.EXPLICIT_CASUAL


@pytest.mark.parametrize(
    "courtesy",
    [
        "thanks",
        "Thank you!",
        "thank you so much",
        "bye",
        "goodbye",
        "see you later",
        "have a nice day",
        "cheers",
        "ok thanks",
    ],
)
def test_courtesy_and_farewells_route_to_casual(router: QueryRouter, courtesy: str):
    decision = router.route(query=courtesy, conversation=[])
    assert decision.mode == ConversationMode.CASUAL
    assert decision.reason == RoutingReason.EXPLICIT_CASUAL


@pytest.mark.parametrize(
    "persona_query",
    [
        "Who are you?",
        "what is your name",
        "what are you?",
        "Are you an AI?",
        "are you a bot",
        "what can you do?",
        "How can you help me?",
        "how does this work?",
        "what are your features",
        "How are you doing?",
    ],
)
def test_persona_and_capabilities_route_to_casual(
    router: QueryRouter, persona_query: str
):
    decision = router.route(query=persona_query, conversation=[])
    assert decision.mode == ConversationMode.CASUAL
    assert decision.reason == RoutingReason.EXPLICIT_CASUAL


@pytest.mark.parametrize(
    "kb_query",
    [
        "What does the employee handbook say about remote work?",
        "Can you summarize the uploaded document?",
        "Search for the refund policy in the pdf",
        "What are the contract guidelines in section 4?",
        "According to the agreement, what is the notice period?",
        "Check our vacation policies",
    ],
)
def test_explicit_kb_signals_route_to_knowledge(router: QueryRouter, kb_query: str):
    decision = router.route(query=kb_query, conversation=[])
    assert decision.mode == ConversationMode.KNOWLEDGE
    assert decision.reason == RoutingReason.EXPLICIT_KNOWLEDGE


def test_mixed_greeting_and_kb_query_routes_to_knowledge(router: QueryRouter):
    mixed_explicit = (
        "Good morning! What does our travel policy say about flight bookings?"
    )
    decision = router.route(query=mixed_explicit, conversation=[])
    assert decision.mode == ConversationMode.KNOWLEDGE
    assert decision.reason == RoutingReason.EXPLICIT_KNOWLEDGE

    mixed_implicit = "Hello there, what is our session timeout limit?"
    decision_implicit = router.route(query=mixed_implicit, conversation=[])
    assert decision_implicit.mode == ConversationMode.KNOWLEDGE
    assert decision_implicit.reason == RoutingReason.DEFAULT_KNOWLEDGE


def test_word_boundary_safety(router: QueryRouter):
    # "this is useful" contains "hi", but word boundaries must prevent matching greeting
    decision = router.route(query="This is useful information", conversation=[])
    assert decision.mode == ConversationMode.KNOWLEDGE
    assert decision.reason == RoutingReason.DEFAULT_KNOWLEDGE


def test_followup_after_knowledge_user_turn_routes_to_knowledge(router: QueryRouter):
    history = [
        _user_msg("What is our session timeout policy?"),
        _assistant_msg("The session timeout is 30 minutes [1]."),
    ]
    followup = "Can you explain that in more detail?"
    decision = router.route(query=followup, conversation=history)
    assert decision.mode == ConversationMode.KNOWLEDGE
    assert decision.reason == RoutingReason.KNOWLEDGE_FOLLOW_UP

    followup_pronoun = "Is that configurable?"
    decision_pronoun = router.route(query=followup_pronoun, conversation=history)
    assert decision_pronoun.mode == ConversationMode.KNOWLEDGE
    assert decision_pronoun.reason == RoutingReason.KNOWLEDGE_FOLLOW_UP


def test_followup_after_casual_user_turn_routes_to_casual(router: QueryRouter):
    history = [
        _user_msg("Who are you?"),
        _assistant_msg("I am an AI assistant here to help."),
    ]
    followup = "Can you explain that?"
    decision = router.route(query=followup, conversation=history)
    assert decision.mode == ConversationMode.CASUAL
    assert decision.reason == RoutingReason.CASUAL_FOLLOW_UP


def test_multi_turn_followup_drift_defaults_conservatively_to_knowledge(
    router: QueryRouter,
):
    """
    Documents and verifies multi-turn follow-up drift behavior:
    1. Turn 1 (User): "Who are you?" (CASUAL)
       Turn 1 (Assistant): "I am an AI assistant."
    2. Turn 2 (User): "Tell me more."
       Evaluates previous user message "Who are you?" (explicit casual) -> CASUAL_FOLLOW_UP.
       Turn 2 (Assistant): "I help answer questions."
    3. Turn 3 (User): "Why?"
       Evaluates previous user message "Tell me more."
       Because "Tell me more." is a follow-up without explicit casual signals (like greetings)
       and without explicit knowledge signals, the non-recursive router conservatively
       treats it as not-explicitly-casual -> defaults to KNOWLEDGE_FOLLOW_UP.
    """
    # Turn 2 evaluates Turn 1 user message
    turn2_history = [
        _user_msg("Who are you?"),
        _assistant_msg("I am an AI assistant."),
    ]
    turn2_decision = router.route(query="Tell me more.", conversation=turn2_history)
    assert turn2_decision.mode == ConversationMode.CASUAL
    assert turn2_decision.reason == RoutingReason.CASUAL_FOLLOW_UP

    # Turn 3 evaluates Turn 2 user message ("Tell me more.")
    turn3_history = [
        _user_msg("Who are you?"),
        _assistant_msg("I am an AI assistant."),
        _user_msg("Tell me more."),
        _assistant_msg("I can assist with documentation and general inquiries."),
    ]
    turn3_decision = router.route(query="Why?", conversation=turn3_history)
    # Conservative V1: "Tell me more." is not an explicit casual greeting/persona pattern,
    # so the single-step router safely fails closed toward knowledge retrieval.
    assert turn3_decision.mode == ConversationMode.KNOWLEDGE
    assert turn3_decision.reason == RoutingReason.KNOWLEDGE_FOLLOW_UP


def test_followup_with_empty_history_defaults_to_knowledge(router: QueryRouter):
    decision = router.route(query="Can you explain that?", conversation=[])
    assert decision.mode == ConversationMode.KNOWLEDGE
    assert decision.reason == RoutingReason.DEFAULT_KNOWLEDGE


@pytest.mark.parametrize(
    "ambiguous_query",
    [
        "Session timeout limit",
        "Quarterly revenue projections",
        "Kubernetes deployment steps",
        "database connection pool size",
    ],
)
def test_ambiguous_factual_queries_default_to_knowledge(
    router: QueryRouter, ambiguous_query: str
):
    decision = router.route(query=ambiguous_query, conversation=[])
    assert decision.mode == ConversationMode.KNOWLEDGE
    assert decision.reason == RoutingReason.DEFAULT_KNOWLEDGE


def test_empty_query_defaults_to_knowledge(router: QueryRouter):
    decision = router.route(query="   ", conversation=[])
    assert decision.mode == ConversationMode.KNOWLEDGE
    assert decision.reason == RoutingReason.DEFAULT_KNOWLEDGE
