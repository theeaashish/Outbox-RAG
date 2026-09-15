from __future__ import annotations

import re
from typing import ClassVar

from app.core.ai.llm.models import ChatMessage
from app.core.ai.routing.models import (
    ConversationMode,
    QueryRoutingDecision,
    RoutingReason,
)
from app.db.models.enums import MessageRole


class QueryRouter:
    """
    Deterministic retrieval gate.

    Decides whether a user turn requires knowledge-base retrieval (KNOWLEDGE)
    or is a conversational/persona turn that should skip retrieval (CASUAL).

    Invariant:
        Only skip retrieval when we have high confidence that the turn does not
        require knowledge-base grounding. When in doubt, default to KNOWLEDGE.
    """

    # Rule 1: Explicit knowledge-base signals (word-bounded)
    _KNOWLEDGE_NOUNS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(documents?|docs?|files?|pdf|polic(y|ies)|handbooks?|manuals?|"
        r"guidelines?|contracts?|agreements?|knowledge\s*bases?|kb|"
        r"uploaded|articles?|sections?|clauses?|pages?)\b",
        re.IGNORECASE,
    )

    _KNOWLEDGE_VERBS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(summariz(e|ing)|search(ing)?\s+for|look(ing)?\s+up|find\s+in|"
        r"extract(ing)?\s+from|according\s+to\s+(the|our))\b",
        re.IGNORECASE,
    )

    # Rule 2: High-confidence casual signals (anchored whole-turn patterns)
    _GREETING_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings|howdy|sup|yo)"
        r"(\s+there)?\b[!?.]*\s*$",
        re.IGNORECASE,
    )

    _COURTESY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*(thanks?(\s+you)?|thank\s+you(\s+so\s+much|\s+very\s+much)?|"
        r"bye|goodbye|see\s+you(\s+later)?|have\s+a\s+(good|nice|great)\s+(day|weekend|evening)|"
        r"cheers|ok\s+thanks?|great\s+thanks?|you['']?re\s+welcome|no\s+problem)\b[!?.]*\s*$",
        re.IGNORECASE,
    )

    _IDENTITY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*(who\s+are\s+you|what\s+is\s+your\s+name|what\s+are\s+you|"
        r"who\s+created\s+you|who\s+made\s+you|are\s+you\s+(an?\s+)?ai|are\s+you\s+a\s+bot)\b[!?.]*\s*$",
        re.IGNORECASE,
    )

    _CAPABILITY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*(what\s+can\s+you\s+do|how\s+can\s+you\s+help(\s+me)?|how\s+does\s+this\s+work|"
        r"what\s+are\s+your\s+(features|capabilities)|help(\s+me)?)\b[!?.]*\s*$",
        re.IGNORECASE,
    )

    _PLEASANTRY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*(how\s+are\s+you|how\s+are\s+you\s+doing|how['']?s\s+it\s+going|nice\s+to\s+meet\s+you)\b[!?.]*\s*$",
        re.IGNORECASE,
    )

    # Rule 3: Anaphoric follow-up cues
    _FOLLOW_UP_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(\b(can|could|would)\s+you\s+(explain|clarify|elaborate(\s+on)?|tell\s+me\s+more\s+about)\s+(that|this|it)\b|"
        r"^\s*(why(\s+is\s+that)?|how\s+so|how\s+come|what\s+about\s+(that|this|it)|tell\s+me\s+more|give\s+(me\s+)?an?\s+example|elaborate|clarify)\b[!?.]*\s*$|"
        r"\b(is|are|does|do|can)\s+(that|this|it|they|these|those)\s+(configurable|apply|mean|work|allowed|possible|required)\b|"
        r"\b(explain\s+that|what\s+about\s+that|why\s+that)\b)",
        re.IGNORECASE,
    )

    def route(
        self,
        *,
        query: str,
        conversation: list[ChatMessage],
    ) -> QueryRoutingDecision:
        """
        Evaluate a user query and recent history to decide if retrieval is required.
        """
        normalized_query = query.strip()
        if not normalized_query:
            return QueryRoutingDecision(
                mode=ConversationMode.KNOWLEDGE,
                reason=RoutingReason.DEFAULT_KNOWLEDGE,
            )

        # Rule 1: Strong Knowledge Signals (High Priority, beats greetings in mixed queries)
        if self._is_knowledge_signal(normalized_query):
            return QueryRoutingDecision(
                mode=ConversationMode.KNOWLEDGE,
                reason=RoutingReason.EXPLICIT_KNOWLEDGE,
            )

        # Rule 2: High-Confidence Casual Signals (only when no knowledge signals exist)
        if self._is_casual_signal(normalized_query):
            return QueryRoutingDecision(
                mode=ConversationMode.CASUAL,
                reason=RoutingReason.EXPLICIT_CASUAL,
            )

        # Rule 3: Follow-Up Analysis via Previous USER Turn
        if self._is_follow_up_query(normalized_query):
            previous_user_turn = self._find_previous_user_turn(conversation)
            if previous_user_turn is not None:
                # If the previous user turn was knowledge-oriented, this follow-up is knowledge-oriented
                if self._is_knowledge_signal(
                    previous_user_turn.content
                ) or not self._is_casual_signal(previous_user_turn.content):
                    return QueryRoutingDecision(
                        mode=ConversationMode.KNOWLEDGE,
                        reason=RoutingReason.KNOWLEDGE_FOLLOW_UP,
                    )
                return QueryRoutingDecision(
                    mode=ConversationMode.CASUAL,
                    reason=RoutingReason.CASUAL_FOLLOW_UP,
                )

        # Rule 4: Default Fallback -> KNOWLEDGE (Grounding preservation)
        return QueryRoutingDecision(
            mode=ConversationMode.KNOWLEDGE,
            reason=RoutingReason.DEFAULT_KNOWLEDGE,
        )

    @classmethod
    def _is_knowledge_signal(cls, text: str) -> bool:
        """Check whether text contains explicit knowledge or document references."""
        return bool(
            cls._KNOWLEDGE_NOUNS_PATTERN.search(text)
            or cls._KNOWLEDGE_VERBS_PATTERN.search(text)
        )

    @classmethod
    def _is_casual_signal(cls, text: str) -> bool:
        """Check whether text is a high-confidence greeting, courtesy, identity, or capability turn."""
        return bool(
            cls._GREETING_PATTERN.match(text)
            or cls._COURTESY_PATTERN.match(text)
            or cls._IDENTITY_PATTERN.match(text)
            or cls._CAPABILITY_PATTERN.match(text)
            or cls._PLEASANTRY_PATTERN.match(text)
        )

    @classmethod
    def _is_follow_up_query(cls, text: str) -> bool:
        """Check whether text contains anaphoric pronouns or elliptical follow-up structure."""
        return bool(cls._FOLLOW_UP_PATTERN.search(text))

    @staticmethod
    def _find_previous_user_turn(conversation: list[ChatMessage]) -> ChatMessage | None:
        """Scan backwards to find the most recent user turn with non-empty content."""
        for msg in reversed(conversation):
            if msg.role == MessageRole.USER and msg.content.strip():
                return msg
        return None
