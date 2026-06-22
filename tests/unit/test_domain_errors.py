"""Sanity gate: the typed exception hierarchy is shaped as the LLDs assume."""

from __future__ import annotations

import pytest

from iw_rag.domain.errors import (
    AuthenticationError,
    AuthorizationError,
    ConfigError,
    EmbedError,
    GenerationError,
    IngestionError,
    IwRagError,
    ParseError,
    RerankError,
    RetrievalError,
)

_ALL = [
    ConfigError,
    ParseError,
    EmbedError,
    IngestionError,
    RetrievalError,
    RerankError,
    GenerationError,
    AuthenticationError,
    AuthorizationError,
]


@pytest.mark.parametrize("exc", _ALL)
def test_every_error_subclasses_root(exc: type[IwRagError]) -> None:
    assert issubclass(exc, IwRagError)


def test_rerank_is_a_retrieval_error() -> None:
    assert issubclass(RerankError, RetrievalError)
    assert issubclass(RerankError, IwRagError)


def test_root_is_an_exception() -> None:
    assert issubclass(IwRagError, Exception)
    with pytest.raises(IwRagError, match="boom"):
        raise RetrievalError("boom")
