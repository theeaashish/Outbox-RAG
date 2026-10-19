from __future__ import annotations

import pytest

from app.core.ai.context.citations import CitationValidator


@pytest.fixture
def validator() -> CitationValidator:
    return CitationValidator()


def test_v1_grammar_valid():
    validator = CitationValidator()
    text = "JWT tokens are used for authentication. [1] They expire in 15 minutes. [2] Or up to 30 days. [10]"
    result = validator.validate(content=text, valid_citations={1, 2, 10})

    assert result.is_valid is True
    assert result.cited_citations == [1, 2, 10]
    assert result.invalid_citations == []
    assert result.content == text


def test_invalid_citation_id_detected():
    validator = CitationValidator()
    text = "Postgres is used. [1] Redis is also used. [99] SQLite is not used. [0] Negative [-1]"
    result = validator.validate(content=text, valid_citations={1, 2})

    assert result.is_valid is False
    assert result.cited_citations == [1]
    assert result.invalid_citations == [99, 0, -1]
    # Content is preserved untouched (non-destructive)
    assert result.content == text


def test_accepted_grammar_tolerates_lists_and_rejects_invalid_elements():
    validator = CitationValidator()
    # [1, 2] and [1,2] are accepted as list syntax; since valid_citations={1}, 2 is invalid.
    # [1-3] cannot expand because 2 and 3 are missing, so it is recorded as invalid.
    # [Source 1] and [01] remain strictly rejected.
    text = "Features [1, 2] and comma [1,2] and range [1-3] and named [Source 1] and leading zero [01] and valid [1]."
    result = validator.validate(content=text, valid_citations={1})

    assert result.is_valid is False
    assert result.cited_citations == [1]
    assert result.invalid_citations == [2, "1-3", "Source 1", "01"]
    assert result.format_deviations == 2  # [1, 2] and [1,2] list deviations


def test_compound_lists_and_semicolons():
    validator = CitationValidator()
    result_comma = validator.validate(content="See [1, 2]", valid_citations={1, 2})
    assert result_comma.is_valid is True
    assert result_comma.cited_citations == [1, 2]
    assert result_comma.invalid_citations == []
    assert result_comma.format_deviations == 1

    result_semi = validator.validate(content="See [1;2]", valid_citations={1, 2})
    assert result_semi.is_valid is True
    assert result_semi.cited_citations == [1, 2]
    assert result_semi.invalid_citations == []
    assert result_semi.format_deviations == 1


@pytest.mark.parametrize("marker", ["[1-3]", "[1–3]", "[1—3]"])
def test_valid_ranges_expand_and_count_format_deviation(
    validator: CitationValidator,
    marker: str,
) -> None:
    result = validator.validate(
        content=f"Covered across sections {marker}.",
        valid_citations={1, 2, 3, 4, 5},
    )
    assert result.is_valid is True
    assert result.cited_citations == [1, 2, 3]
    assert result.invalid_citations == []
    assert result.format_deviations == 1


@pytest.mark.parametrize("bad_range", ["3-1", "2-7", "1-999", "0-2"])
def test_invalid_or_out_of_bounds_ranges_fail_without_partial_citations(
    validator: CitationValidator,
    bad_range: str,
) -> None:
    result = validator.validate(
        content=f"Bad range [{bad_range}].",
        valid_citations={1, 2, 3, 4, 5},
    )
    assert result.is_valid is False
    assert result.cited_citations == []
    assert result.invalid_citations == [bad_range]
    assert result.format_deviations == 0


def test_compound_list_with_invalid_element():
    validator = CitationValidator()
    # [1, 9] with valid {1..5} -> cited [1], invalid [9]
    result = validator.validate(content="Cites [1, 9]", valid_citations={1, 2, 3, 4, 5})
    assert result.is_valid is False
    assert result.cited_citations == [1]
    assert result.invalid_citations == [9]
    assert result.format_deviations == 1


def test_prose_identifier_bracket_ignored():
    validator = CitationValidator()
    text = (
        "In Python, access matrix[1, 2] and arr[0] and tensor[1][2] and func()[1] directly. "
        "According to [1], standard 1-based indexing applies."
    )
    result = validator.validate(content=text, valid_citations={1})
    assert result.is_valid is True
    assert result.cited_citations == [1]
    assert result.invalid_citations == []
    assert result.format_deviations == 0


def test_empty_context_treats_all_citations_as_invalid():
    validator = CitationValidator()
    text = "The answer is 42. [1]"
    result = validator.validate(content=text, valid_citations=set())

    assert result.is_valid is False
    assert result.cited_citations == []
    assert result.invalid_citations == [1]


def test_repeated_citations_deduplicated_in_order():
    validator = CitationValidator()
    text = "Point A [2]. Point B [1]. Point C [2]."
    result = validator.validate(content=text, valid_citations={1, 2})

    assert result.is_valid is True
    assert result.cited_citations == [2, 1]
    assert result.invalid_citations == []


def test_repeated_invalid_citations_deduplicated_in_order():
    validator = CitationValidator()
    text = "Invalid A [99]. Valid [1]. Invalid B [50]. Invalid A repeat [99]."
    result = validator.validate(content=text, valid_citations={1})

    assert result.is_valid is False
    assert result.cited_citations == [1]
    assert result.invalid_citations == [99, 50]


def test_text_without_citations():
    validator = CitationValidator()
    text = "I do not know the answer based on the available documents."
    result = validator.validate(content=text, valid_citations={1, 2})

    assert result.is_valid is True
    assert result.cited_citations == []
    assert result.invalid_citations == []
    assert result.content == text


def test_citation_validator_ignores_code_blocks_and_inline_code():
    validator = CitationValidator()
    text = (
        "In Python, access the first item via `arr[0]` or `matrix[1, 2]`.\n"
        "```python\ndef get_first(items):\n    return items[0]\n```\n"
        "According to the docs [1], indexing is 0-based."
    )
    result = validator.validate(content=text, valid_citations={1})

    assert result.is_valid is True
    assert result.cited_citations == [1]
    assert result.invalid_citations == []


def test_citation_validator_ignores_markdown_links_and_images():
    validator = CitationValidator()
    text = (
        "Check [Official Documentation](https://example.com/docs) and "
        "![Architecture Diagram](https://example.com/arch.png) as noted in [1]."
    )
    result = validator.validate(content=text, valid_citations={1})

    assert result.is_valid is True
    assert result.cited_citations == [1]
    assert result.invalid_citations == []


def test_citation_validator_ignores_non_citation_brackets():
    validator = CitationValidator()
    text = "Here is some [foo] and some [1 foo] and [abc 1] with valid citation [1]."
    result = validator.validate(content=text, valid_citations={1})

    assert result.is_valid is True
    assert result.cited_citations == [1]
    assert result.invalid_citations == []
