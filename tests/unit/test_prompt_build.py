"""Unit tests for the citation-required prompt builder."""

from __future__ import annotations

from iw_rag.domain.models import RetrievedChunk
from iw_rag.generation.context_budget import ContextBudget, estimate_tokens
from iw_rag.generation.prompts import REFUSAL_TEXT, CitationPrompt


def _chunk(i: int, *, section: str | None = None, ctype: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc-chunk",
        chunk_id=f"c{i}",
        content=f"body {i}",
        score=1.0 - i * 0.1,
        source_uri=f"uri{i}",
        metadata={"section_path": section} if section else {},
    )


def test_sources_numbered_one_to_n() -> None:
    bundle = CitationPrompt().build("q?", [_chunk(0), _chunk(1), _chunk(2)])
    assert "[1] (uri0" in bundle.user
    assert "[2] (uri1" in bundle.user
    assert "[3] (uri2" in bundle.user
    assert len(bundle.used_chunks) == 3


def test_section_path_rendered_when_present() -> None:
    """Section path 'HR > Leave' appears verbatim in the rendered user message."""
    bundle = CitationPrompt().build("q?", [_chunk(0, section="HR > Leave")])
    # Rendered exactly once, verbatim, in the user message.
    assert bundle.user.count("HR > Leave") == 1
    # The single source is numbered [1].
    assert bundle.user.count("[1]") == 1


def test_refusal_phrase_in_system_prompt() -> None:
    bundle = CitationPrompt().build("q?", [_chunk(0)])
    assert REFUSAL_TEXT in bundle.system


def test_context_overflow_drops_lowest_ranked() -> None:
    chunks = [_chunk(i) for i in range(5)]  # ranked desc already
    bundle = CitationPrompt(max_sources=3).build("q?", chunks)
    assert len(bundle.used_chunks) == 3
    assert [c.chunk_id for c in bundle.used_chunks] == ["c0", "c1", "c2"]  # kept highest-ranked
    assert "uri4" not in bundle.user  # lowest-ranked dropped


def test_as_messages_shape() -> None:
    bundle = CitationPrompt().build("q?", [_chunk(0)])
    msgs = bundle.as_messages()
    assert [m["role"] for m in msgs] == ["system", "user"]


# ── Token-budgeting (I-CTXOVERFLOW) ─────────────────────────────────────────


def _big_chunk(i: int, *, size: int) -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc-chunk",
        chunk_id=f"c{i}",
        content="A" * size,
        score=1.0 - i * 0.01,
        source_uri=f"uri{i}",
    )


def test_budget_fits_generation_prompt_to_window() -> None:
    """With a budget, the assembled generation prompt provably fits the window —
    even with eight whole-document-sized sources that previously 400'd vLLM.
    """
    budget = ContextBudget(max_model_len=16384, reserved_output_tokens=1024, headroom_tokens=2048)
    chunks = [_big_chunk(i, size=50_000) for i in range(8)]
    bundle = CitationPrompt(budget=budget).build("explain everything", chunks)

    # Kept sources are a ranked prefix (lowest-ranked dropped first).
    kept_ids = [c.chunk_id for c in bundle.used_chunks]
    assert kept_ids == [f"c{i}" for i in range(len(kept_ids))]
    assert 0 < len(kept_ids) <= 8
    # THE invariant: the whole user message fits under the window.
    assert estimate_tokens(bundle.user) <= 16384 - 1024


def test_budget_truncates_lone_oversized_source() -> None:
    """A single source larger than the whole window is truncated, not dropped."""
    budget = ContextBudget(max_model_len=8192, reserved_output_tokens=1024, headroom_tokens=1024)
    bundle = CitationPrompt(budget=budget).build("q?", [_big_chunk(0, size=200_000)])
    assert len(bundle.used_chunks) == 1  # survives (truncated)
    assert estimate_tokens(bundle.user) <= 8192 - 1024


def test_no_budget_keeps_legacy_count_only_behaviour() -> None:
    """Without a budget the builder is unchanged (count cap only)."""
    chunks = [_big_chunk(i, size=50_000) for i in range(5)]
    bundle = CitationPrompt(max_sources=3).build("q?", chunks)
    assert len(bundle.used_chunks) == 3
    assert all("A" * 50_000 in s.content for s in bundle.used_chunks)
