from app.core.ai.context.citations import CitationValidator


def test_v1_grammar_valid():
    validator = CitationValidator()
    text = "JWT tokens are used for authentication. [1] They expire in 15 minutes. [2]"
    result = validator.validate(content=text, valid_citations={1, 2})

    assert result.is_valid is True
    assert result.cited_citations == [1, 2]
    assert result.invalid_citations == []
    assert result.content == text


def test_invalid_citation_id_detected():
    validator = CitationValidator()
    text = "Postgres is used. [1] Redis is also used. [99] SQLite is not used. [0]"
    result = validator.validate(content=text, valid_citations={1, 2})

    assert result.is_valid is False
    assert result.cited_citations == [1]
    assert result.invalid_citations == [99, 0]
    # Content is preserved untouched (non-destructive)
    assert result.content == text


def test_v1_grammar_does_not_recognize_unsupported_syntaxes():
    validator = CitationValidator()
    # [1, 2], [1-3], [Source 1] do not match r"\[(\d+)\]"
    text = "Features [1, 2] and range [1-3] and named [Source 1] and valid [1]."
    result = validator.validate(content=text, valid_citations={1})

    assert result.is_valid is True
    assert result.cited_citations == [1]
    assert result.invalid_citations == []


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
