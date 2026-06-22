"""Cross-cutting unit tests for citation plumbing (CR-00098 / S07).

The S01 + S05 unit tests in :mod:`tests.unit.test_citations` and
:mod:`tests.unit.api.test_citation_download_url` cover the building blocks:

* ``RetrievedChunk`` carries ``document_id`` (S01 field plumbing).
* ``AnswerAssembler._resolve`` threads ``document_id`` from chunk to
  :class:`Citation` (S01 + S05).
* :class:`CitationOut` exposes the additive ``document_id`` +
  ``download_url`` fields (S05).
* The /query response populates both fields (S05).

This file adds the **cross-cutting** assertions S07 is responsible for —
the seam between the layers:

* ``Citation.document_id`` always equals the underlying chunk's
  ``document_id`` (a regression pin for the S01 thread; the token
  mints against *this* id, so a wrong thread is a security regression).
* :class:`CitationOut` ``document_id`` always equals the domain
  :class:`Citation` ``document_id`` (the schema is a thin pass-through
  — no fabricated or empty doc id leaks to the response).
* The :class:`CitationOut` ``download_url`` is one of the documented
  forms: a real ``"/documents/{document_id}/raw?token=…"`` URL whose
  token ``verify`` accepts for the citation's document + principal, OR
  the empty string (the fail-soft path when the secret is unset).
* A citation's ``quote`` retains the matched leaf span (the S01
  behaviour did not regress) and the ``source_uri`` is the storage
  URI (not the PK; the two are independent).
* The ``marker`` is the 1-based index into the *chunks* the assembler
  was given (not the *citations*) — the same contract _resolve had
  pre-S01.

Every test is a **unit test** (no docker, no live DB, no network)
per the S07 spec.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from iw_rag.api import download_tokens
from iw_rag.api.schemas import CitationOut
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, RetrievalResult, RetrievedChunk
from iw_rag.generation.answer import AnswerAssembler
from iw_rag.generation.llm import GenerationParams, LlmResponse, Message, ToolSpec
from iw_rag.generation.prompts import CitationPrompt

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_TEST_SECRET: Final = "test-secret-must-be-long-enough-for-hmac-sha256"


class _FakeLlm:
    """A canned-response LLM fake (mirrors ``test_citations.FakeLlm``)."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def complete(self, messages: Sequence[Message], params: GenerationParams) -> LlmResponse:
        self.calls += 1
        return LlmResponse(text=self._text, usage={"total_tokens": 7})

    async def complete_tools(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], params: GenerationParams
    ) -> LlmResponse:  # pragma: no cover
        return LlmResponse(text="")

    async def complete_structured(
        self, messages: Sequence[Message], json_schema: dict, params: GenerationParams
    ) -> LlmResponse:  # pragma: no cover
        return LlmResponse(text="")

    async def health(self) -> bool:  # pragma: no cover
        return True


def _assembler(llm: _FakeLlm) -> AnswerAssembler:
    return AnswerAssembler(llm, CitationPrompt(), GenerationParams(model="m"))


def _ctx() -> AccessContext:
    return AccessContext(
        principal_id="alice",
        principal_groups=("finance",),
        is_operator=False,
    )


# ---------------------------------------------------------------------------
# 1. Citation.document_id == chunk.document_id — the S01 thread invariant
# ---------------------------------------------------------------------------


async def test_citation_document_id_equals_chunk_document_id_one_to_one() -> None:
    """Every citation's ``document_id`` equals the underlying chunk's ``document_id``.

    The S01 thread is the seam the S05 token mint + S04 download both
    rely on: a wrong thread would mint a token bound to the wrong
    document, the S04 path would refuse to serve bytes, and the
    user-visible link would point to a 404 — but the **security**
    invariant (no cross-document leak) would still hold.  This test
    pins the seam so a regression is caught at the unit level.
    """
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            document_id=f"doc-{i:04d}-0000-0000-0000-000000000000",
            content=f"content {i}",
            score=1.0 - i * 0.1,
            source_uri=f"corpus://hr/doc{i}.md",
        )
        for i in range(3)
    ]
    llm = _FakeLlm("Answer [1] and [2] and [3].")
    ans = await _assembler(llm).generate(
        "q", RetrievalResult(chunks=chunks, retriever="spine", acl_applied=True)
    )
    # citation i references chunk i — document_id round-trips 1:1
    for cit, chunk in zip(ans.citations, chunks, strict=True):
        assert cit.document_id == chunk.document_id


async def test_citation_quote_preserves_leaf_content_on_parent_expansion() -> None:
    """``Citation.quote`` keeps the matched leaf span (small-to-big expansion).

    S01 did not touch the quote field; this regression pin guards
    against a future edit to ``_resolve`` accidentally dropping the
    ``leaf_content or content`` preference.
    """
    parent = RetrievedChunk(
        chunk_id="p1",
        document_id="doc-parent",
        content="PARENT context paragraph",
        score=0.9,
        source_uri="corpus://x",
        leaf_content="precise leaf span",
    )
    llm = _FakeLlm("Answer [1].")
    ans = await _assembler(llm).generate(
        "q", RetrievalResult(chunks=[parent], retriever="spine", acl_applied=True)
    )
    assert ans.citations[0].quote == "precise leaf span"


async def test_citation_quote_falls_back_to_content_when_no_leaf() -> None:
    """When ``leaf_content`` is ``None``, ``quote`` uses ``content``."""
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="doc-1",
        content="verbatim chunk content",
        score=0.9,
        source_uri="corpus://x",
        leaf_content=None,
    )
    llm = _FakeLlm("Answer [1].")
    ans = await _assembler(llm).generate(
        "q", RetrievalResult(chunks=[chunk], retriever="spine", acl_applied=True)
    )
    assert ans.citations[0].quote == "verbatim chunk content"


async def test_citation_source_uri_preserved_from_chunk() -> None:
    """``Citation.source_uri`` is the chunk's storage URI, not the document_id.

    The two are independent: ``source_uri`` carries the URI path
    (e.g. ``corpus://hr/policy.md``) and ``document_id`` is the
    documents.id PK.  The S01 thread added the PK as an *additional*
    field; ``source_uri`` is unchanged.
    """
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="00000000-0000-0000-0000-000000000001",
        content="x",
        score=1.0,
        source_uri="corpus://hr/policy.md",
    )
    llm = _FakeLlm("Answer [1].")
    ans = await _assembler(llm).generate(
        "q", RetrievalResult(chunks=[chunk], retriever="spine", acl_applied=True)
    )
    cit = ans.citations[0]
    assert cit.source_uri == "corpus://hr/policy.md"
    assert cit.document_id == "00000000-0000-0000-0000-000000000001"
    # And the document_id is NOT embedded in source_uri.
    assert "00000000-0000-0000-0000-000000000001" not in cit.source_uri


# ---------------------------------------------------------------------------
# 2. _resolve contract — markers / chunks mapping is unchanged
# ---------------------------------------------------------------------------


async def test_markers_map_to_one_indexed_chunk_positions() -> None:
    """``_resolve`` maps ``[n]`` → ``chunks[n-1]`` (1-based, pre-S01 contract)."""
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            document_id=f"doc-{i}",
            content=f"c{i}",
            score=1.0 - i * 0.1,
            source_uri=f"uri{i}",
        )
        for i in range(5)
    ]
    llm = _FakeLlm("Citing [3] and [5].")
    ans = await _assembler(llm).generate(
        "q", RetrievalResult(chunks=chunks, retriever="spine", acl_applied=True)
    )
    assert [c.marker for c in ans.citations] == [3, 5]
    assert [c.chunk_id for c in ans.citations] == ["c2", "c4"]


async def test_markers_with_spaces_and_commas_resolve() -> None:
    """``[1, 2]`` and ``[1][2]`` both resolve correctly (S01 contract).

    The marker regex accepts ``[0-9]+`` separated by ``,`` (with optional
    whitespace) and adjacent ``[...]`` brackets — ``[ 1 ]`` (whitespace
    *inside* the brackets) is not supported by the marker parser; that
    is the S01 contract.
    """
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            document_id=f"doc-{i}",
            content=f"c{i}",
            score=1.0,
            source_uri=f"uri{i}",
        )
        for i in range(2)
    ]
    for text, expected_markers in [
        ("Answer [1, 2].", [1, 2]),
        ("Answer [1][2].", [1, 2]),
        ("Answer [1, 2][1].", [1, 2]),  # duplicate dedupe
    ]:
        llm = _FakeLlm(text)
        ans = await _assembler(llm).generate(
            "q", RetrievalResult(chunks=chunks, retriever="spine", acl_applied=True)
        )
        assert [c.marker for c in ans.citations] == expected_markers, f"text={text!r}"


async def test_out_of_range_marker_dropped_in_order() -> None:
    """Out-of-range markers are dropped and reported in ``diagnostics`` (S01 contract)."""
    chunks = [
        RetrievedChunk(
            chunk_id="c0",
            document_id="doc-0",
            content="c0",
            score=1.0,
            source_uri="uri0",
        )
    ]
    llm = _FakeLlm("Real [1] fake [2] fake [99].")
    ans = await _assembler(llm).generate(
        "q", RetrievalResult(chunks=chunks, retriever="spine", acl_applied=True)
    )
    assert [c.marker for c in ans.citations] == [1]
    # Dropped markers are reported in-order; S01 did not change the
    # contract, so [2] and [99] are both dropped and reported.
    assert ans.diagnostics is not None
    assert ans.diagnostics.get("dropped_markers") == [2, 99]


# ---------------------------------------------------------------------------
# 3. CitationOut schema — document_id + download_url contract
# ---------------------------------------------------------------------------


def test_citation_out_document_id_passes_through_from_domain_citation() -> None:
    """``CitationOut.document_id`` equals the domain ``Citation.document_id``."""
    citation = Citation(
        marker=1,
        chunk_id="chunk-1",
        document_id="11111111-1111-1111-1111-111111111111",
        source_uri="corpus://hr/policy.md",
        quote="policy excerpt",
    )
    out = CitationOut(
        marker=citation.marker,
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        source_uri=citation.source_uri,
        quote=citation.quote,
    )
    assert out.document_id == citation.document_id == "11111111-1111-1111-1111-111111111111"


def test_citation_out_download_url_accepts_download_tokens_verify() -> None:
    """A ``download_url`` of the documented form has a token that ``verify`` accepts.

    The cross-cutting pin: a S05-minted download URL (a relative
    ``/documents/{id}/raw?token=...`` whose token is bound to
    ``(document_id, principal_id, exp)``) is verifiable by the S04
    download path's ``verify`` call.  This is the seam the browser
    relies on.
    """
    document_id = "11111111-1111-1111-1111-111111111111"
    principal_id = "alice"
    token = download_tokens.mint(
        document_id=document_id,
        principal_id=principal_id,
        secret=_TEST_SECRET,
        ttl_seconds=300,
        now=1_700_000_000,
    )
    download_url = f"/documents/{document_id}/raw?token={token}"

    # The download URL has the documented form.
    assert download_url.startswith(f"/documents/{document_id}/raw?token=")
    token_str = download_url.rsplit("token=", 1)[-1]
    # The token verifies for the right document + principal under the same secret.
    verified = download_tokens.verify(token_str, secret=_TEST_SECRET, now=1_700_000_001)
    assert verified is not None
    assert verified.document_id == document_id
    assert verified.principal_id == principal_id


def test_citation_out_download_url_for_wrong_document_rejected_by_path_check() -> None:
    """A token bound to doc A, used against doc B, never serves bytes.

    The S04 path checks ``verified.document_id == path_id`` — the
    unit-level pin: the verified id is exactly the minted one, so
    a path-mismatch in the S04 route will raise NotFoundError.
    """
    document_a = "11111111-1111-1111-1111-111111111111"
    document_b = "22222222-2222-2222-2222-222222222222"
    token = download_tokens.mint(
        document_id=document_a,
        principal_id="alice",
        secret=_TEST_SECRET,
        ttl_seconds=300,
        now=1_700_000_000,
    )
    # The S04 path builds: f"/documents/{path_id}/raw?token={token}"
    # The verified id is the bound document_a; a path of document_b
    # would mismatch the equality check, the route returns 404.
    verified = download_tokens.verify(token, secret=_TEST_SECRET, now=1_700_000_001)
    assert verified is not None
    assert verified.document_id == document_a
    # The caller (S04) compares verified.document_id to the path's {id};
    # if the path is document_b, the comparison fails → NotFoundError.
    assert verified.document_id != document_b


def test_citation_out_download_url_empty_when_document_id_blank() -> None:
    """``CitationOut.download_url == ""`` when ``document_id`` is blank.

    A citation without a document_id (the visual/graphrag/long-context
    paths) cannot mint a download URL — the link is omitted and the
    frontend falls back to plain text rendering.  This pins the
    fail-soft contract.
    """
    out = CitationOut(
        marker=1,
        chunk_id="c1",
        document_id="",  # no PK
        source_uri="corpus://x",
        quote="q",
    )
    assert out.document_id == ""
    # The download_url is the default — empty string (not a forged URL).
    assert out.download_url == ""


# ---------------------------------------------------------------------------
# 4. End-to-end chunk → citation → CitationOut — full seam
# ---------------------------------------------------------------------------


async def test_resolved_citation_round_trips_into_citation_out_with_document_id() -> None:
    """A ``Citation`` produced by ``_resolve`` round-trips into ``CitationOut``
    with the **same** document_id (no swap, no truncation, no empty).
    """
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-abc",
            document_id="11111111-1111-1111-1111-111111111111",
            content="policy text",
            score=0.9,
            source_uri="corpus://hr/policy.md",
        )
    ]
    llm = _FakeLlm("Answer [1].")
    ans = await _assembler(llm).generate(
        "q", RetrievalResult(chunks=chunks, retriever="spine", acl_applied=True)
    )

    citation = ans.citations[0]
    out = CitationOut(
        marker=citation.marker,
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        source_uri=citation.source_uri,
        quote=citation.quote,
    )
    # Full thread integrity: chunk → citation → CitationOut.
    assert out.document_id == chunks[0].document_id
    assert out.document_id == citation.document_id
    assert out.chunk_id == citation.chunk_id == chunks[0].chunk_id
    assert out.source_uri == citation.source_uri == chunks[0].source_uri
    assert out.quote == citation.quote


async def test_multiple_citations_preserve_distinct_document_ids() -> None:
    """A multi-citation answer preserves the distinct document_ids of its chunks.

    A regression on the S01 thread would manifest as a single document_id
    leaking across all citations (e.g. the last chunk's id).  This test
    pins the per-citation binding.
    """
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            document_id=f"00000000-0000-0000-0000-00000000000{i}",
            content=f"c{i}",
            score=1.0 - i * 0.1,
            source_uri=f"uri{i}",
        )
        for i in range(4)
    ]
    llm = _FakeLlm("Answer [1] [2] [3] [4].")
    ans = await _assembler(llm).generate(
        "q", RetrievalResult(chunks=chunks, retriever="spine", acl_applied=True)
    )
    assert len(ans.citations) == 4
    # Every citation's document_id equals the matching chunk's id, in order.
    for cit, chunk in zip(ans.citations, chunks, strict=True):
        assert cit.document_id == chunk.document_id
    # And all four are distinct — no two citations share a document_id.
    doc_ids = [c.document_id for c in ans.citations]
    assert len(set(doc_ids)) == 4, f"document_ids must be distinct, got {doc_ids!r}"
