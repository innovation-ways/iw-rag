"""Unit tests for the BM25 query sanitizer in sparse.py.

The ``_bm25_query`` function strips punctuation and lowercases terms so the
ParadeDB pg_search query parser never chokes on invoice IDs, URLs, or other
punctuation-heavy query strings.  Mutating the regex or the OR-join format must
break these tests.
"""

from __future__ import annotations

import pytest

from iw_rag.retrieval.sparse import _bm25_query


def test_invoice_id_strips_hyphen_and_slash_and_question_mark() -> None:
    """INV-123/total? → three clean terms joined by ' OR '."""
    result = _bm25_query("INV-123/total?")
    assert result == "content:inv OR content:123 OR content:total"


def test_lowercasing_applied_to_all_terms() -> None:
    """All extracted terms must be lowercased."""
    result = _bm25_query("Invoice TOTAL")
    assert result == "content:invoice OR content:total"


def test_plain_alphanumeric_query_unchanged() -> None:
    """A clean query passes straight through (no punctuation to strip)."""
    result = _bm25_query("invoice total amount")
    assert result == "content:invoice OR content:total OR content:amount"


def test_single_term_produces_no_or() -> None:
    """A single-word query produces exactly one content: clause."""
    result = _bm25_query("invoice")
    assert result == "content:invoice"


def test_punctuation_only_produces_empty_string() -> None:
    """Query with no alphanumeric chars → empty string (early-return branch)."""
    result = _bm25_query("---/???!!!")
    assert result == ""


def test_empty_string_produces_empty_string() -> None:
    """Empty input → empty string (no terms → empty join)."""
    result = _bm25_query("")
    assert result == ""


def test_whitespace_only_produces_empty_string() -> None:
    """Whitespace-only input has no alphanumeric terms."""
    result = _bm25_query("   \t\n  ")
    assert result == ""


def test_mixed_case_with_numbers_and_punctuation() -> None:
    """URL-style query is fully sanitized."""
    result = _bm25_query("REF#2024_Q1:report.pdf")
    assert result == "content:ref OR content:2024 OR content:q1 OR content:report OR content:pdf"


def test_or_join_exact_format() -> None:
    """The separator is exactly ' OR ' (space–OR–space), not '|' or 'OR'."""
    result = _bm25_query("foo bar")
    parts = result.split(" OR ")
    assert parts == ["content:foo", "content:bar"]


def test_terms_prefixed_with_content_colon() -> None:
    """Every extracted term must be prefixed with 'content:'."""
    result = _bm25_query("alpha beta gamma")
    for part in result.split(" OR "):
        assert part.startswith("content:"), f"term {part!r} missing 'content:' prefix"


def test_numbers_are_kept_as_terms() -> None:
    """Numeric tokens (e.g. invoice numbers) are preserved."""
    result = _bm25_query("PO 99887766")
    assert result == "content:po OR content:99887766"


# --- Early-return branch (sparse.py:61-62) ---


def test_empty_terms_causes_empty_return_from_bm25_query() -> None:
    """Verifies the falsy-string early-return path used by SparseRetriever.

    SparseRetriever checks ``if not bm25: return []`` (sparse.py:61-62).
    ``_bm25_query`` must return an empty string (falsy) when no terms exist so
    that guard fires correctly.
    """
    result = _bm25_query("!!! ??? ---")
    assert not result, "no alphanumeric terms must yield a falsy (empty) string"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hello", "content:hello"),
        ("Hello World", "content:hello OR content:world"),
        ("FOO-BAR", "content:foo OR content:bar"),
        ("abc123", "content:abc123"),
        ("a1b2 c3d4", "content:a1b2 OR content:c3d4"),
    ],
)
def test_parametrized_sanitization(raw: str, expected: str) -> None:
    """Parametrized sanity check: punctuation strips, lowercase, correct join."""
    assert _bm25_query(raw) == expected
