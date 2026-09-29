from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """
    Result of validating citation references in generated assistant text.

    Proves only:
    - citation syntax strictly follows the V1 grammar ([n] where n in [1-9][0-9]*)
    - cited IDs exist in the current turn's context source set
    - rejected forms ([0], [01], [1,2], [1-3], [Source 1]) are caught as invalid

    Does NOT prove semantic grounding or claim support.
    """

    content: str
    cited_citations: list[int]
    invalid_citations: list[int | str]
    is_valid: bool


class CitationValidator:
    """
    Validates citation markers in generated text against the strict V1 grammar.

    V1 Grammar:
        Citation ::= "[" [1-9][0-9]* "]"

    Only citation-like forms ([n], [0], [01], [-1], [1,2], [1-3], [Source 1]) are
    inspected; arbitrary brackets like [foo], [abc 1], [1 foo] and markdown
    links [1](url) are intentionally ignored.

    Strictly rejects:
        - [0], [-1], [01] (zero, negative, leading zeros)
        - [1, 2], [1,2], [1-3], [1; 2] (multi-citations and ranges)
        - [Source 1], [doc 1], [p. 1] (prefixed/named citations)
        - Any citation [n] where n not in valid_citations

    Ignores:
        - Inline code `...` and fenced code blocks ```...```
        - Markdown links [text](url) and images ![text](url)
        - Arbitrary bracketed text that is not citation-like (e.g., [foo], [abc 1], [1 foo])
    """

    _CODE_BLOCK_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
    _BRACKET_REGEX = re.compile(r"\[([^\]\n]+)\]")
    _VALID_INT_REGEX = re.compile(r"^([1-9]\d*)$")
    _LEADING_ZERO_REGEX = re.compile(r"^0\d*$")
    _NEGATIVE_INT_REGEX = re.compile(r"^-\d+$")
    _COMPOUND_RANGE_REGEX = re.compile(r"^\d+\s*(?:[,\-;–—]\s*\d+)+$")
    _NAMED_PREFIX_REGEX = re.compile(
        r"^(?i:(?:source|doc|document|ref|reference|chunk|page|p\.?))\s*:?\s*\d+.*$"
    )

    def validate(
        self,
        *,
        content: str,
        valid_citations: set[int] | frozenset[int],
    ) -> CitationValidationResult:
        """
        Extract citation markers and verify that all referenced IDs adhere to the V1 grammar
        and belong to valid_citations.

        Non-destructive: content is preserved untouched in the result.
        """
        cited_citations: list[int] = []
        invalid_citations: list[int | str] = []
        seen_cited: set[int] = set()
        seen_invalid: set[int | str] = set()

        scan_text = self._CODE_BLOCK_RE.sub(" ", content)

        for match in self._BRACKET_REGEX.finditer(scan_text):
            # Skip markdown links e.g. [text](url) or images ![text](url)
            end_pos = match.end()
            if end_pos < len(scan_text) and scan_text[end_pos] == "(":
                continue
            start_pos = match.start()
            if start_pos > 0 and scan_text[start_pos - 1] == "!":
                continue

            raw_inner = match.group(1).strip()
            if not raw_inner:
                continue

            # 1. Valid positive integer without leading zeros
            if self._VALID_INT_REGEX.match(raw_inner):
                citation_id = int(raw_inner)
                if citation_id in valid_citations:
                    if citation_id not in seen_cited:
                        seen_cited.add(citation_id)
                        cited_citations.append(citation_id)
                else:
                    if citation_id not in seen_invalid:
                        seen_invalid.add(citation_id)
                        invalid_citations.append(citation_id)

            # 2. Leading zero or zero integer e.g. [0], [01]
            elif self._LEADING_ZERO_REGEX.match(raw_inner):
                item: int | str = 0 if raw_inner == "0" else raw_inner
                if item not in seen_invalid:
                    seen_invalid.add(item)
                    invalid_citations.append(item)

            # 3. Negative integer e.g. [-1]
            elif self._NEGATIVE_INT_REGEX.match(raw_inner):
                neg_item = int(raw_inner)
                if neg_item not in seen_invalid:
                    seen_invalid.add(neg_item)
                    invalid_citations.append(neg_item)

            # 4. Multi-citation, ranges, or prefixed/named citations
            elif self._COMPOUND_RANGE_REGEX.match(
                raw_inner
            ) or self._NAMED_PREFIX_REGEX.match(raw_inner):
                if raw_inner not in seen_invalid:
                    seen_invalid.add(raw_inner)
                    invalid_citations.append(raw_inner)

        return CitationValidationResult(
            content=content,
            cited_citations=cited_citations,
            invalid_citations=invalid_citations,
            is_valid=len(invalid_citations) == 0,
        )
