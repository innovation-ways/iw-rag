"""Unit tests for citation parsing, source mapping, and fail-closed refusal."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.generation.answer import AnswerAssembler, parse_citation_markers, strip_datamarks
from iw_rag.generation.llm import GenerationParams, LlmResponse, Message, ToolSpec
from iw_rag.generation.prompts import REFUSAL_TEXT, CitationPrompt


class FakeLlm:
    """Returns a canned response and records whether it was called."""

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


def _chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            document_id=f"d{i:08x}-0000-0000-0000-000000000000",
            content=f"content {i}",
            score=1.0 - i * 0.1,
            source_uri=f"uri{i}",
        )
        for i in range(n)
    ]


def _result(chunks: list[RetrievedChunk]) -> RetrievalResult:
    return RetrievalResult(chunks=chunks, retriever="spine", acl_applied=True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a [1] b", [1]),
        ("a [1, 2] b", [1, 2]),
        ("a [1][2] b", [1, 2]),
        ("no markers", []),
        ("dup [1] and [1]", [1]),
    ],
)
def test_parse_citation_markers(text: str, expected: list[int]) -> None:
    assert parse_citation_markers(text) == expected


def _assembler(llm: FakeLlm) -> AnswerAssembler:
    return AnswerAssembler(llm, CitationPrompt(), GenerationParams(model="m"))


async def test_markers_map_to_chunks() -> None:
    llm = FakeLlm("The answer is grounded [1] and also [2].")
    ans = await _assembler(llm).generate("q", _result(_chunks(3)))
    assert ans.grounded is True
    assert [c.marker for c in ans.citations] == [1, 2]
    assert {c.chunk_id for c in ans.citations} == {"c0", "c1"}


async def test_hallucinated_marker_dropped_and_flagged() -> None:
    llm = FakeLlm("Real [1] but fake [9].")  # only 2 sources
    ans = await _assembler(llm).generate("q", _result(_chunks(2)))
    assert [c.marker for c in ans.citations] == [1]
    assert ans.diagnostics.get("dropped_markers") == [9]


async def test_refusal_phrase_marks_ungrounded() -> None:
    llm = FakeLlm(REFUSAL_TEXT)
    ans = await _assembler(llm).generate("q", _result(_chunks(2)))
    assert ans.grounded is False
    assert ans.citations == []


async def test_empty_result_refuses_without_llm_call() -> None:
    llm = FakeLlm("should not be called")
    ans = await _assembler(llm).generate("q", _result([]))
    assert ans.grounded is False
    assert llm.calls == 0  # no LLM call on empty retrieval
    assert ans.answer == REFUSAL_TEXT


async def test_cite_required_uncited_answer_is_ungrounded() -> None:
    llm = FakeLlm("An answer with no markers at all.")
    ans = await _assembler(llm).generate("q", _result(_chunks(2)))
    assert ans.grounded is False  # cite_required → no citations means ungrounded


def test_strip_datamarks_removes_session_markers() -> None:
    # U+2063 INVISIBLE SEPARATOR + hex token interspersed by guards-on sanitize.
    marked = "The⁣82c8ed8abf74 total⁣82c8ed8abf74 due⁣82c8ed8abf74 is"
    assert strip_datamarks(marked) == "The total due is"
    # No-op when no markers present (guards-off path).
    assert strip_datamarks("plain text") == "plain text"


async def test_citation_quote_strips_datamarks() -> None:
    # Guards-on: the assembler receives sanitized (datamarked) chunk content; the
    # user-facing citation quote must never surface the invisible markers.
    marker = "⁣bb4e225dab29"
    chunk = RetrievedChunk(
        chunk_id="c0",
        document_id="d0000000-0000-0000-0000-000000000000",
        content=f"The{marker} total{marker} due{marker} is{marker} 15000{marker} USD",
        score=1.0,
        source_uri="corpus://finance/invoice_q3.md",
    )
    llm = FakeLlm("Answer [1].")
    ans = await _assembler(llm).generate("q", _result([chunk]))
    assert ans.citations[0].quote == "The total due is 15000 USD"
    assert "⁣" not in ans.citations[0].quote


async def test_leaf_quote_retained_on_parent_expansion() -> None:
    # parent-expanded chunk: content=parent, leaf_content=leaf → citation quote = leaf
    chunk = RetrievedChunk(
        chunk_id="c0",
        document_id="d0000000-0000-0000-0000-000000000000",
        content="PARENT context paragraph",
        score=1.0,
        source_uri="uri0",
        leaf_content="precise leaf span",
    )
    llm = FakeLlm("Answer [1].")
    ans = await _assembler(llm).generate("q", _result([chunk]))
    assert ans.citations[0].quote == "precise leaf span"


def test_retrieved_chunk_exposes_document_id() -> None:
    """RetrievedChunk must carry the owning ``document_id`` (the documents.id PK).

    The UUID embedded in ``source_uri`` is a *storage* uuid, not the primary
    key the download endpoint keys on; a downstream citation must address the
    real document. The new ``document_id`` field is the channel for that.
    """
    chunk = RetrievedChunk(
        chunk_id="c0",
        content="text",
        score=1.0,
        source_uri="corpus://hr/policy.md",
        document_id="11111111-1111-1111-1111-111111111111",
    )
    assert chunk.document_id == "11111111-1111-1111-1111-111111111111"


async def test_resolve_citation_carries_chunk_document_id() -> None:
    """``_resolve`` must thread the chunk's ``document_id`` onto every Citation.

    The Citation (and ultimately CitationOut) needs the owning document_id so
    the API layer can mint a per-citation download URL addressing
    ``GET /documents/{document_id}/raw``. Without this thread the field is
    unfillable end-to-end and the download link is unaddressable.
    """
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            content=f"content {i}",
            score=1.0 - i * 0.1,
            source_uri=f"uri{i}",
            document_id=f"doc-{i:08x}-0000-0000-0000-000000000000",
        )
        for i in range(2)
    ]
    llm = FakeLlm("Answer [1] and [2].")
    ans = await _assembler(llm).generate("q", _result(chunks))
    assert [c.document_id for c in ans.citations] == [
        "doc-00000000-0000-0000-0000-000000000000",
        "doc-00000001-0000-0000-0000-000000000000",
    ]
