from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ConversationMode(StrEnum):
    """Execution mode for a chat conversation turn."""

    CASUAL = "casual"
    KNOWLEDGE = "knowledge"


class RoutingReason(StrEnum):
    """Controlled justification codes for query routing decisions."""

    EXPLICIT_KNOWLEDGE = "explicit_knowledge_signal"
    EXPLICIT_CASUAL = "explicit_casual_signal"
    KNOWLEDGE_FOLLOW_UP = "knowledge_follow_up"
    CASUAL_FOLLOW_UP = "casual_follow_up"
    DEFAULT_KNOWLEDGE = "default_knowledge"


@dataclass(frozen=True, slots=True)
class QueryRoutingDecision:
    """Type-safe outcome of evaluating a user turn at the retrieval gate."""

    mode: ConversationMode
    reason: RoutingReason
