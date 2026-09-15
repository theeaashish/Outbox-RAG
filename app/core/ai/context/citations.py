from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """
    Result of validating citation references in generated assistant text.

    Proves only:
    - citation syntax follows the V1 grammar ([n])
    - cited IDs exist in the current turn's context source set

    Does NOT prove semantic grounding or claim support.
    """

    content: str
    cited_citations: list[int]
    invalid_citations: list[int]
    is_valid: bool
    sanitized_content: str | None = None


class CitationValidator:
    """
    Validates citation markers in generated text against the V1 grammar.

    V1 Grammar:
        Citation ::= "[" [1-9][0-9]* "]"
    """

    _CITATION_REGEX = re.compile(r"\[(\d+)\]")

    def validate(
        self,
        *,
        content: str,
        valid_citations: set[int] | frozenset[int],
    ) -> CitationValidationResult:
        """
        Extract citation markers and verify that all referenced IDs belong to valid_citations.

        Non-destructive: content is preserved untouched in the result.
        """
        matches = self._CITATION_REGEX.findall(content)

        cited_citations: list[int] = []
        invalid_citations: list[int] = []
        seen_cited: set[int] = set()
        seen_invalid: set[int] = set()

        for match in matches:
            citation_id = int(match)
            if citation_id in valid_citations:
                if citation_id not in seen_cited:
                    seen_cited.add(citation_id)
                    cited_citations.append(citation_id)
            else:
                if citation_id not in seen_invalid:
                    seen_invalid.add(citation_id)
                    invalid_citations.append(citation_id)

        sanitized: str | None = None
        if invalid_citations:
            sanitized = self._CITATION_REGEX.sub(
                lambda m: "" if int(m.group(1)) in seen_invalid else m.group(0),
                content,
            )

        return CitationValidationResult(
            content=content,
            cited_citations=cited_citations,
            invalid_citations=invalid_citations,
            is_valid=len(invalid_citations) == 0,
            sanitized_content=sanitized,
        )
