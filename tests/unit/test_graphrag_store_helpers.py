"""Unit tests for pure helper functions in AgeGraphStore (no DB, no AGE required).

Covers:
- ``_cypher_rel_type`` — free-text-to-identifier folding (load-bearing bug fixes
  noted in store.py comments): lowercasing, whitespace→``_``, ``RELATED_TO``
  fallback, leading-digit ``R_`` prefix.
- ``_escape_cypher`` — single-quote escaping for safe Cypher literal embedding.
- ``AgeGraphStore.__init__`` ``graph_name`` validator — raises on invalid names,
  accepts valid identifiers.

All tests run offline — no Postgres, no AGE extension, no network.

References:
- src/iw_rag/modes/graphrag/store.py (``_cypher_rel_type``, ``_escape_cypher``,
  ``AgeGraphStore.__init__``)
- lld-graphrag §2 (store interface), task-4a-graphrag-route §1
"""

from __future__ import annotations

import pytest

from iw_rag.modes.graphrag.store import AgeGraphStore, _cypher_rel_type, _escape_cypher

# ---------------------------------------------------------------------------
# Tests: _escape_cypher
# ---------------------------------------------------------------------------


def test_escape_cypher_no_quotes_unchanged() -> None:
    """A string without single quotes is returned unchanged."""
    assert _escape_cypher("hello world") == "hello world"


def test_escape_cypher_single_quote_escaped() -> None:
    """A single quote in the value is replaced with an escaped single quote."""
    assert _escape_cypher("it's") == "it\\'s"


def test_escape_cypher_multiple_quotes_all_escaped() -> None:
    """Every single quote in the value is individually escaped."""
    assert _escape_cypher("it's Ada's") == "it\\'s Ada\\'s"


def test_escape_cypher_only_quotes() -> None:
    """Two single-quotes are each individually escaped, producing backslash-quote pairs."""
    # "'" → "\'" so "''" → "\'\'" which in Python raw literals is "\\'\\''"
    assert _escape_cypher("''") == "\\'" + "\\'"


def test_escape_cypher_empty_string_unchanged() -> None:
    """An empty string is returned as-is (no substitution target)."""
    assert _escape_cypher("") == ""


def test_escape_cypher_no_modification_to_other_special_chars() -> None:
    """Characters other than single-quote are not modified (double-quote, backslash)."""
    assert _escape_cypher('say "hi"') == 'say "hi"'


# ---------------------------------------------------------------------------
# Tests: _cypher_rel_type — load-bearing normalisation (store.py comments)
# ---------------------------------------------------------------------------


def test_cypher_rel_type_plain_word_uppercased() -> None:
    """A simple lowercase word is returned as its upper-case equivalent."""
    assert _cypher_rel_type("uses") == "USES"


def test_cypher_rel_type_multi_word_whitespace_to_underscore() -> None:
    """Spaces between words are replaced by underscores and the result is uppercased."""
    assert _cypher_rel_type("request through") == "REQUEST_THROUGH"


def test_cypher_rel_type_is_a_phrase() -> None:
    """'is a' — a typical LLM extraction output — normalises to IS_A."""
    assert _cypher_rel_type("is a") == "IS_A"


def test_cypher_rel_type_works_at_phrase() -> None:
    """'works at' normalises to WORKS_AT."""
    assert _cypher_rel_type("works at") == "WORKS_AT"


def test_cypher_rel_type_already_valid_identifier_unchanged() -> None:
    """An already-valid upper-snake-case identifier passes through unmodified."""
    assert _cypher_rel_type("CALLS") == "CALLS"


def test_cypher_rel_type_mixed_case_uppercased() -> None:
    """Mixed-case input is uppercased to the normalised form."""
    assert _cypher_rel_type("Calls") == "CALLS"


def test_cypher_rel_type_hyphens_become_underscores() -> None:
    """Hyphens (non-identifier chars) are replaced with underscores."""
    assert _cypher_rel_type("related-to") == "RELATED_TO"


def test_cypher_rel_type_non_identifier_only_returns_fallback() -> None:
    """Input whose only characters are non-identifier chars returns RELATED_TO."""
    assert _cypher_rel_type("---") == "RELATED_TO"


def test_cypher_rel_type_empty_string_returns_fallback() -> None:
    """Empty string returns the RELATED_TO fallback."""
    assert _cypher_rel_type("") == "RELATED_TO"


def test_cypher_rel_type_whitespace_only_returns_fallback() -> None:
    """Whitespace-only input returns the RELATED_TO fallback (no identifier chars)."""
    assert _cypher_rel_type("   ") == "RELATED_TO"


def test_cypher_rel_type_leading_digit_prefixed_with_r_underscore() -> None:
    """A result whose first character would be a digit gets the R_ prefix."""
    assert _cypher_rel_type("3way") == "R_3WAY"


def test_cypher_rel_type_leading_digit_multi_segment() -> None:
    """Input like '2 phase' produces R_2_PHASE (digit-start after normalisation)."""
    assert _cypher_rel_type("2 phase") == "R_2_PHASE"


def test_cypher_rel_type_leading_underscore_stripped() -> None:
    """Leading underscores produced by leading non-identifier chars are stripped.

    Input: ' leading' → after sub → '_leading' → strip('_') → 'leading' → 'LEADING'
    """
    assert _cypher_rel_type(" leading") == "LEADING"


def test_cypher_rel_type_internal_special_chars_collapsed() -> None:
    """Special chars that are not underscores/alphanumerics become single underscores."""
    assert _cypher_rel_type("a@b!c") == "A_B_C"


def test_cypher_rel_type_tabs_and_newlines_to_underscore() -> None:
    """Tabs and newline characters are treated as non-identifier chars → underscore."""
    assert _cypher_rel_type("a\tb\nc") == "A_B_C"


# ---------------------------------------------------------------------------
# Tests: AgeGraphStore.__init__ graph_name validator
# ---------------------------------------------------------------------------


def test_graph_store_init_valid_simple_name_accepted() -> None:
    """A simple lowercase identifier is a valid graph_name."""
    store = AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db", graph_name="mygraph")
    assert store._graph_name == "mygraph"


def test_graph_store_init_valid_name_with_underscore_and_digits_accepted() -> None:
    """An identifier with underscores and trailing digits is a valid graph_name."""
    store = AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db", graph_name="iw_rag_graph_2")
    assert store._graph_name == "iw_rag_graph_2"


def test_graph_store_init_default_graph_name_accepted() -> None:
    """The default graph_name 'iw_rag_graph' passes the validator."""
    store = AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db")
    assert store._graph_name == "iw_rag_graph"


def test_graph_store_init_name_with_hyphen_raises() -> None:
    """A graph_name containing a hyphen is rejected with ValueError."""
    with pytest.raises(ValueError, match="invalid AGE graph_name"):
        AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db", graph_name="my-graph")


def test_graph_store_init_name_with_space_raises() -> None:
    """A graph_name containing a space is rejected with ValueError."""
    with pytest.raises(ValueError, match="invalid AGE graph_name"):
        AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db", graph_name="my graph")


def test_graph_store_init_name_with_leading_digit_raises() -> None:
    """A graph_name starting with a digit is rejected with ValueError."""
    with pytest.raises(ValueError, match="invalid AGE graph_name"):
        AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db", graph_name="1graph")


def test_graph_store_init_empty_name_raises() -> None:
    """An empty graph_name is rejected with ValueError."""
    with pytest.raises(ValueError, match="invalid AGE graph_name"):
        AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db", graph_name="")


def test_graph_store_init_name_with_dot_raises() -> None:
    """A graph_name containing a dot is rejected with ValueError."""
    with pytest.raises(ValueError, match="invalid AGE graph_name"):
        AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db", graph_name="schema.graph")


def test_graph_store_init_name_with_semicolon_raises() -> None:
    """A graph_name containing a SQL-injection character is rejected with ValueError."""
    with pytest.raises(ValueError, match="invalid AGE graph_name"):
        AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db", graph_name="g;DROP TABLE")


def test_graph_store_init_dsn_stored_as_provided() -> None:
    """The DSN is stored exactly as passed without modification at init time."""
    dsn = "postgresql+psycopg://iw_rag:secret@db:5432/iw_rag"
    store = AgeGraphStore(dsn=dsn, graph_name="iw_rag_graph")
    assert store._dsn == dsn


def test_graph_store_init_engine_is_none_before_first_use() -> None:
    """The async engine is not created at init — it is lazily initialised on first call."""
    store = AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db")
    assert store._engine is None


def test_graph_store_init_age_not_checked_before_first_use() -> None:
    """AGE availability is not checked at init — deferred to _ensure_age()."""
    store = AgeGraphStore(dsn="postgresql+psycopg://u:p@localhost/db")
    assert store._age_checked is False
