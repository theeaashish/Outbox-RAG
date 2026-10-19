from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """
    Result of validating citation references in generated assistant text.

    Proves:
    - Cited IDs exist in the current turn's context source set.
    - Compound lists ([1, 2], [1; 2]) and valid bounded ranges ([1-3]) are
      expanded into individual cited IDs while incrementing format_deviations.
    - Rejected forms ([0], [01], [-1], [Source 1], descending/out-of-bounds ranges)
      are caught in invalid_citations.

    Does NOT prove semantic grounding or claim support.
    """

    content: str
    cited_citations: list[int]
    invalid_citations: list[int | str]
    is_valid: bool
    format_deviations: int = 0


class CitationValidator:
    """
    Validates citation markers in generated text with tolerant parsing.

    Accepted Grammar:
        Single   ::= "[" [1-9][0-9]* "]"
        List     ::= "[" Int ( [,;] Int )+ "]"      (each item validated individually; +1 format_deviation)
        Range    ::= "[" Start [-–—] End "]"        (valid iff 1 <= Start < End and all in valid_citations; +1 format_deviation)

    Strictly rejects:
        - [0], [-1], [01] (zero, negative, leading zeros)
        - Invalid ranges: descending ([3-1]), zero-based ([0-2]), or partially/fully out of bounds ([2-7] when valid={1..5})
        - [Source 1], [doc 1], [p. 1] (prefixed/named citations)
        - Any citation ID not in valid_citations

    Ignores:
        - Inline code `...` and fenced code blocks ```...```
        - Markdown links [text](url) and images ![text](url)
        - Subscript/indexing brackets directly following an identifier char ([A-Za-z0-9_\\])]), e.g. matrix[1, 2] or arr[0]
        - Arbitrary bracketed text that is not citation-like (e.g., [foo], [abc 1], [1 foo])
    """

    _CODE_BLOCK_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
    _BRACKET_REGEX = re.compile(r"\[([^\]\n]+)\]")
    _IDENTIFIER_PRECEDING_RE = re.compile(r"[A-Za-z0-9_\])]")

    _VALID_INT_REGEX = re.compile(r"^([1-9]\d*)$")
    _LEADING_ZERO_REGEX = re.compile(r"^0\d*$")
    _NEGATIVE_INT_REGEX = re.compile(r"^-\d+$")

    _LIST_CANDIDATE_REGEX = re.compile(r"^-?\d+(?:\s*[,;]\s*-?\d+)+$")
    _RANGE_REGEX = re.compile(r"^([1-9]\d*)\s*[-–—]\s*([1-9]\d*)$")
    _FALLBACK_COMPOUND_REGEX = re.compile(r"^-?\d+\s*(?:[,\-;–—]\s*-?\d+)+$")
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
        Extract citation markers and verify that all referenced IDs belong to valid_citations.

        Non-destructive: content is preserved untouched in the result.
        """
        cited_citations: list[int] = []
        invalid_citations: list[int | str] = []
        seen_cited: set[int] = set()
        seen_invalid: set[int | str] = set()
        format_deviations = 0

        def record_cited(cid: int) -> None:
            if cid not in seen_cited:
                seen_cited.add(cid)
                cited_citations.append(cid)

        def record_invalid(item: int | str) -> None:
            if item not in seen_invalid:
                seen_invalid.add(item)
                invalid_citations.append(item)

        def validate_single_token(token: str) -> None:
            if self._VALID_INT_REGEX.match(token):
                citation_id = int(token)
                if citation_id in valid_citations:
                    record_cited(citation_id)
                else:
                    record_invalid(citation_id)
            elif self._LEADING_ZERO_REGEX.match(token):
                record_invalid(0 if token == "0" else token)
            elif self._NEGATIVE_INT_REGEX.match(token):
                record_invalid(int(token))
            else:
                record_invalid(token)

        scan_text = self._CODE_BLOCK_RE.sub(" ", content)

        for match in self._BRACKET_REGEX.finditer(scan_text):
            end_pos = match.end()
            if end_pos < len(scan_text) and scan_text[end_pos] == "(":
                continue

            start_pos = match.start()
            if start_pos > 0:
                prev_char = scan_text[start_pos - 1]
                if prev_char == "!" or self._IDENTIFIER_PRECEDING_RE.match(prev_char):
                    continue

            raw_inner = match.group(1).strip()
            if not raw_inner:
                continue

            # 1. Single valid positive integer without leading zeros
            if (
                self._VALID_INT_REGEX.match(raw_inner)
                or self._LEADING_ZERO_REGEX.match(raw_inner)
                or self._NEGATIVE_INT_REGEX.match(raw_inner)
            ):
                validate_single_token(raw_inner)

            # 4. Comma/semicolon separated list e.g. [1, 2], [1; 2]
            elif self._LIST_CANDIDATE_REGEX.match(raw_inner):
                format_deviations += 1
                for part in re.split(r"\s*[,;]\s*", raw_inner):
                    validate_single_token(part.strip())

            # 5. Numeric range e.g. [1-3], [1–3], [1—3]
            elif range_match := self._RANGE_REGEX.match(raw_inner):
                start_id = int(range_match.group(1))
                end_id = int(range_match.group(2))
                span_len = end_id - start_id + 1
                if (
                    1 <= start_id < end_id
                    and valid_citations
                    and span_len <= len(valid_citations)
                    and all(
                        cid in valid_citations for cid in range(start_id, end_id + 1)
                    )
                ):
                    format_deviations += 1
                    for cid in range(start_id, end_id + 1):
                        record_cited(cid)
                else:
                    record_invalid(raw_inner)

            # 6. Other malformed compound ranges or named prefixes e.g. [0-2], [1-2, 4], [Source 1]
            elif self._FALLBACK_COMPOUND_REGEX.match(
                raw_inner
            ) or self._NAMED_PREFIX_REGEX.match(raw_inner):
                record_invalid(raw_inner)

        return CitationValidationResult(
            content=content,
            cited_citations=cited_citations,
            invalid_citations=invalid_citations,
            is_valid=len(invalid_citations) == 0,
            format_deviations=format_deviations,
        )
