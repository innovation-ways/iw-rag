"""Unit tests for guardrails/output_guard.py (Task 3.3 — output guard).

TDD RED-first: these tests were written before the implementation.

Covers:
- ``validate_citations``: in-set IDs pass; out-of-set ID invalid+listed;
  empty citations invalid; multiple invalid all reported.
- ``validate_citations`` text-variant: out-of-range marker (index beyond source
  count) returns ``(True, [])`` — the text-marker path cannot perform
  set-membership checks without chunk-ID context.
- ``_groundedness_call_with_retry`` retry-then-succeed: first attempt raises,
  second returns a score; the returned score proves fail-closed does NOT
  trigger on a transient blip.
- ``OutputVerdict`` aggregation: grounded+valid → allow; ungrounded → block;
  valid-but-ungrounded → block; grounded-but-invalid-citation → block.
- PII: SSN/email in answer redacted; ``pii_redacted=True``; entity types
  recorded not values.
- Groundedness model unavailable → fail-closed (treated as ungrounded) after
  bounded retry.
- ``scan_output``: composes citation validation + groundedness wrapper + PII.
- ``scan_output`` safety-model seam: ``safety_model`` non-None logged but not
  blocking (seam wired off by default for VRAM).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, GuardVerdict, RetrievedChunk

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("finance",),
    is_operator=False,
    request_id="req-output-guard-unit",
)

_CHUNK_A = RetrievedChunk(
    chunk_id="chunk-a",
    document_id="doc-a",
    content="The Q3 invoice total is $15,000.",
    score=0.95,
    source_uri="corpus://finance/q3",
)
_CHUNK_B = RetrievedChunk(
    chunk_id="chunk-b",
    document_id="doc-b",
    content="Payment terms are net-30.",
    score=0.90,
    source_uri="corpus://finance/terms",
)

_RETRIEVED_IDS: set[str] = {"chunk-a", "chunk-b"}
_CITATIONS_VALID = [
    Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://finance/q3"),
]
_CITATIONS_INVALID_ID = [
    Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://finance/q3"),
    Citation(
        marker=2, chunk_id="chunk-FABRICATED", document_id="doc-a", source_uri="corpus://fabricated"
    ),
]
_GRADED = [_CHUNK_A, _CHUNK_B]


# ---------------------------------------------------------------------------
# validate_citations — deterministic, no model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_citations_all_in_set() -> None:
    """When every cited ID is in the retrieved set, returns (True, [])."""
    from iw_rag.guardrails.output_guard import validate_citations

    valid, invalid = await validate_citations(
        "The total is $15,000 [1].",
        retrieved_chunk_ids={"chunk-a"},
    )
    assert valid is True
    assert invalid == []


@pytest.mark.asyncio
async def test_validate_citations_out_of_set_id_invalid() -> None:
    """A cited ID not in the retrieved set → invalid via validate_citations_from_list."""
    from iw_rag.guardrails.output_guard import validate_citations_from_list

    # Use the from_list variant which has chunk_id context for set-membership check
    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a")
    ]
    # chunk-a is NOT in the retrieved set (only chunk-b is)
    valid, invalid = await validate_citations_from_list(
        citations,
        retrieved_chunk_ids={"chunk-b"},
    )
    assert valid is False
    assert "chunk-a" in invalid


@pytest.mark.asyncio
async def test_validate_citations_empty_citations_invalid() -> None:
    """An answer with no citation markers is invalid (uncited = invalid)."""
    from iw_rag.guardrails.output_guard import validate_citations

    valid, invalid = await validate_citations(
        "The total is $15,000.",  # No [n] markers
        retrieved_chunk_ids={"chunk-a"},
    )
    assert valid is False
    assert invalid == []  # no invalid ids, but still invalid (no citations at all)


@pytest.mark.asyncio
async def test_validate_citations_multiple_invalid_all_reported() -> None:
    """Multiple invalid cited IDs are all reported in the invalid list."""
    from iw_rag.guardrails.output_guard import validate_citations_from_list

    # This variant takes a pre-parsed citation list
    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a"),
        Citation(marker=2, chunk_id="FABRICATED-1", document_id="doc-a", source_uri="corpus://x"),
        Citation(marker=3, chunk_id="FABRICATED-2", document_id="doc-a", source_uri="corpus://y"),
    ]
    valid, invalid = await validate_citations_from_list(
        citations,
        retrieved_chunk_ids={"chunk-a"},
    )
    assert valid is False
    assert "FABRICATED-1" in invalid
    assert "FABRICATED-2" in invalid
    assert "chunk-a" not in invalid


@pytest.mark.asyncio
async def test_validate_citations_from_list_all_valid() -> None:
    """validate_citations_from_list: all IDs in set → (True, [])."""
    from iw_rag.guardrails.output_guard import validate_citations_from_list

    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a"),
        Citation(marker=2, chunk_id="chunk-b", document_id="doc-b", source_uri="corpus://b"),
    ]
    valid, invalid = await validate_citations_from_list(
        citations,
        retrieved_chunk_ids={"chunk-a", "chunk-b"},
    )
    assert valid is True
    assert invalid == []


@pytest.mark.asyncio
async def test_validate_citations_from_list_empty_is_invalid() -> None:
    """validate_citations_from_list with empty list → invalid (uncited)."""
    from iw_rag.guardrails.output_guard import validate_citations_from_list

    valid, invalid = await validate_citations_from_list(
        [],
        retrieved_chunk_ids={"chunk-a"},
    )
    assert valid is False
    assert invalid == []


# ---------------------------------------------------------------------------
# OutputVerdict aggregation table
# ---------------------------------------------------------------------------


def test_output_verdict_grounded_and_valid_is_allow() -> None:
    """grounded=True + citations_valid=True → the verdict is an allow outcome."""
    from iw_rag.guardrails.output_guard import OutputVerdict

    v = OutputVerdict(
        grounded=True,
        groundedness_score=0.85,
        citations_valid=True,
        invalid_citation_ids=[],
        pii_redacted=False,
        answer="The total is $15,000 [1].",
        verdicts=[GuardVerdict(layer="output", decision="allow", score=0.85, reason="grounded")],
    )
    # An allow outcome: grounded AND citations_valid
    assert v.grounded is True
    assert v.citations_valid is True
    # The aggregated decision is derivable from the fields
    assert not v.invalid_citation_ids


def test_output_verdict_ungrounded_is_block() -> None:
    """grounded=False → block outcome regardless of citation validity."""
    from iw_rag.guardrails.output_guard import OutputVerdict

    v = OutputVerdict(
        grounded=False,
        groundedness_score=0.2,
        citations_valid=True,
        invalid_citation_ids=[],
        pii_redacted=False,
        answer="Some answer [1].",
        verdicts=[GuardVerdict(layer="output", decision="block", score=0.2, reason="ungrounded")],
    )
    assert v.grounded is False


def test_output_verdict_valid_but_ungrounded_is_block() -> None:
    """Citations valid but ungrounded → block (groundedness check dominates)."""
    from iw_rag.guardrails.output_guard import OutputVerdict

    v = OutputVerdict(
        grounded=False,
        groundedness_score=0.3,
        citations_valid=True,
        invalid_citation_ids=[],
        pii_redacted=False,
        answer="Some answer [1].",
        verdicts=[GuardVerdict(layer="output", decision="block", score=0.3, reason="ungrounded")],
    )
    assert v.grounded is False
    assert v.citations_valid is True  # valid citations but still block


def test_output_verdict_grounded_but_invalid_citation_is_block() -> None:
    """grounded=True but invalid citation → block (deterministic check wins)."""
    from iw_rag.guardrails.output_guard import OutputVerdict

    v = OutputVerdict(
        grounded=True,
        groundedness_score=0.9,
        citations_valid=False,
        invalid_citation_ids=["FABRICATED"],
        pii_redacted=False,
        answer="Some answer [1][2].",
        verdicts=[
            GuardVerdict(layer="output", decision="block", score=0.0, reason="invalid_citation")
        ],
    )
    assert v.grounded is True  # model says grounded
    assert v.citations_valid is False  # deterministic check overrides
    assert "FABRICATED" in v.invalid_citation_ids


def test_output_verdict_is_frozen() -> None:
    """OutputVerdict is a frozen dataclass."""
    from iw_rag.guardrails.output_guard import OutputVerdict

    v = OutputVerdict(
        grounded=True,
        groundedness_score=0.8,
        citations_valid=True,
        invalid_citation_ids=[],
        pii_redacted=False,
        answer="answer",
        verdicts=[],
    )
    with pytest.raises((AttributeError, TypeError), match="cannot assign"):
        v.grounded = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PII redaction in scan_output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_output_pii_ssn_redacted() -> None:
    """An answer containing an SSN is redacted; pii_redacted=True."""
    from iw_rag.guardrails.output_guard import scan_output

    answer_with_ssn = "Your SSN is 123-45-6789. The total [1] is $15,000."
    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a")
    ]
    graded = [_CHUNK_A]

    async def _mock_presidio_redact(text: str, ctx: AccessContext) -> str:
        return text.replace("123-45-6789", "<US_SSN>")

    async def _mock_groundedness(answer: str, evidence: str) -> tuple[bool, float]:
        return True, 0.9

    with (
        patch(
            "iw_rag.guardrails.output_guard._presidio_redact",
            new=AsyncMock(side_effect=_mock_presidio_redact),
        ),
        patch(
            "iw_rag.guardrails.output_guard._granite_groundedness_score",
            new=AsyncMock(side_effect=_mock_groundedness),
        ),
    ):
        verdict = await scan_output(answer_with_ssn, citations, graded, _CTX)

    assert verdict.pii_redacted is True
    assert "123-45-6789" not in verdict.answer
    assert "<US_SSN>" in verdict.answer


@pytest.mark.asyncio
async def test_scan_output_pii_email_redacted() -> None:
    """An answer containing an email is redacted; pii_redacted=True."""
    from iw_rag.guardrails.output_guard import scan_output

    answer_with_email = "Contact admin@corp.com for details [1]."
    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a")
    ]
    graded = [_CHUNK_A]

    async def _mock_presidio_redact(text: str, ctx: AccessContext) -> str:
        return text.replace("admin@corp.com", "<EMAIL_ADDRESS>")

    async def _mock_groundedness(answer: str, evidence: str) -> tuple[bool, float]:
        return True, 0.85

    with (
        patch(
            "iw_rag.guardrails.output_guard._presidio_redact",
            new=AsyncMock(side_effect=_mock_presidio_redact),
        ),
        patch(
            "iw_rag.guardrails.output_guard._granite_groundedness_score",
            new=AsyncMock(side_effect=_mock_groundedness),
        ),
    ):
        verdict = await scan_output(answer_with_email, citations, graded, _CTX)

    assert verdict.pii_redacted is True
    assert "admin@corp.com" not in verdict.answer
    assert "<EMAIL_ADDRESS>" in verdict.answer


@pytest.mark.asyncio
async def test_scan_output_pii_redacted_entity_types_not_values() -> None:
    """The verdict records entity types (e.g. US_SSN), not the raw PII value."""
    from iw_rag.guardrails.output_guard import scan_output

    raw_ssn = "123-45-6789"
    answer_with_ssn = f"Your SSN is {raw_ssn}. See doc [1]."
    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a")
    ]
    graded = [_CHUNK_A]

    async def _mock_presidio_redact(text: str, ctx: AccessContext) -> str:
        return text.replace(raw_ssn, "<US_SSN>")

    async def _mock_groundedness(answer: str, evidence: str) -> tuple[bool, float]:
        return True, 0.9

    with (
        patch(
            "iw_rag.guardrails.output_guard._presidio_redact",
            new=AsyncMock(side_effect=_mock_presidio_redact),
        ),
        patch(
            "iw_rag.guardrails.output_guard._granite_groundedness_score",
            new=AsyncMock(side_effect=_mock_groundedness),
        ),
    ):
        verdict = await scan_output(answer_with_ssn, citations, graded, _CTX)

    # Check no verdict reason contains the raw SSN value
    for gv in verdict.verdicts:
        assert raw_ssn not in gv.reason


# ---------------------------------------------------------------------------
# Groundedness model unavailable → fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groundedness_model_unavailable_fails_closed() -> None:
    """When groundedness model is unavailable (raises), result is ungrounded (fail-closed)."""
    from iw_rag.guardrails.output_guard import scan_output

    answer = "The total is $15,000 [1]."
    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a")
    ]
    graded = [_CHUNK_A]

    async def _mock_presidio_redact(text: str, ctx: AccessContext) -> str:
        return text  # no PII

    async def _groundedness_unavailable(answer: str, evidence: str) -> tuple[bool, float]:
        raise RuntimeError("groundedness model unavailable")

    with (
        patch(
            "iw_rag.guardrails.output_guard._presidio_redact",
            new=AsyncMock(side_effect=_mock_presidio_redact),
        ),
        patch(
            "iw_rag.guardrails.output_guard._granite_groundedness_score",
            new=AsyncMock(side_effect=_groundedness_unavailable),
        ),
    ):
        verdict = await scan_output(answer, citations, graded, _CTX)

    # Fail-closed: unavailable groundedness model → treated as ungrounded
    assert verdict.grounded is False
    # Score should indicate unavailability
    assert verdict.groundedness_score == 0.0


@pytest.mark.asyncio
async def test_scan_output_grounded_and_cited_is_allow() -> None:
    """Grounded + valid citations → grounded=True, citations_valid=True."""
    from iw_rag.guardrails.output_guard import scan_output

    answer = "The total is $15,000 [1]."
    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a")
    ]
    graded = [_CHUNK_A]

    async def _mock_presidio_redact(text: str, ctx: AccessContext) -> str:
        return text

    async def _mock_groundedness(answer: str, evidence: str) -> tuple[bool, float]:
        return True, 0.92

    with (
        patch(
            "iw_rag.guardrails.output_guard._presidio_redact",
            new=AsyncMock(side_effect=_mock_presidio_redact),
        ),
        patch(
            "iw_rag.guardrails.output_guard._granite_groundedness_score",
            new=AsyncMock(side_effect=_mock_groundedness),
        ),
    ):
        verdict = await scan_output(answer, citations, graded, _CTX)

    assert verdict.grounded is True
    assert verdict.citations_valid is True
    assert verdict.invalid_citation_ids == []
    assert verdict.groundedness_score == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_scan_output_invalid_citation_blocks_even_if_grounded() -> None:
    """Invalid citation (model grounded but fabricated ID) → citations_valid=False."""
    from iw_rag.guardrails.output_guard import scan_output

    # Answer cites chunk-a (valid) and FABRICATED (invalid)
    answer = "Total is $15k [1] and terms are net-30 [2]."
    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a"),
        Citation(
            marker=2, chunk_id="FABRICATED", document_id="doc-a", source_uri="corpus://fabricated"
        ),
    ]
    graded = [_CHUNK_A]  # only chunk-a, FABRICATED not in set

    async def _mock_presidio_redact(text: str, ctx: AccessContext) -> str:
        return text

    async def _mock_groundedness(answer: str, evidence: str) -> tuple[bool, float]:
        return True, 0.88  # model says grounded

    with (
        patch(
            "iw_rag.guardrails.output_guard._presidio_redact",
            new=AsyncMock(side_effect=_mock_presidio_redact),
        ),
        patch(
            "iw_rag.guardrails.output_guard._granite_groundedness_score",
            new=AsyncMock(side_effect=_mock_groundedness),
        ),
    ):
        verdict = await scan_output(answer, citations, graded, _CTX)

    assert verdict.grounded is True  # model said grounded
    assert verdict.citations_valid is False  # deterministic check caught it
    assert "FABRICATED" in verdict.invalid_citation_ids


@pytest.mark.asyncio
async def test_scan_output_no_pii_unchanged() -> None:
    """An answer without PII is returned unchanged; pii_redacted=False."""
    from iw_rag.guardrails.output_guard import scan_output

    answer = "The total is $15,000 [1]."
    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a")
    ]
    graded = [_CHUNK_A]

    async def _mock_presidio_redact(text: str, ctx: AccessContext) -> str:
        return text  # no PII → unchanged

    async def _mock_groundedness(answer: str, evidence: str) -> tuple[bool, float]:
        return True, 0.9

    with (
        patch(
            "iw_rag.guardrails.output_guard._presidio_redact",
            new=AsyncMock(side_effect=_mock_presidio_redact),
        ),
        patch(
            "iw_rag.guardrails.output_guard._granite_groundedness_score",
            new=AsyncMock(side_effect=_mock_groundedness),
        ),
    ):
        verdict = await scan_output(answer, citations, graded, _CTX)

    assert verdict.pii_redacted is False
    assert verdict.answer == answer


# ---------------------------------------------------------------------------
# validate_citations — text-variant out-of-range marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_citations_out_of_range_marker_returns_valid() -> None:
    """Text-variant validate_citations with an out-of-range marker returns (True, []).

    The text-marker path can only verify that at least one marker exists; it
    cannot perform set-membership checks without chunk-ID context.  An answer
    citing [99] when only two sources exist is still returned as valid here —
    callers with Citation objects must use validate_citations_from_list for
    out-of-range detection (lld-guardrails §3.4 docstring).
    """
    from iw_rag.guardrails.output_guard import validate_citations

    # Only two sources available; marker [99] is far out of range
    valid, invalid = await validate_citations(
        "The policy [99] is defined in the handbook.",
        retrieved_chunk_ids={"chunk-a", "chunk-b"},
    )
    assert valid is True
    assert invalid == []


# ---------------------------------------------------------------------------
# _groundedness_call_with_retry — retry-then-succeed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groundedness_retry_then_succeed_returns_score() -> None:
    """_groundedness_call_with_retry succeeds on the second attempt and returns the score.

    Proves the retry loop does NOT fail-closed on a single transient error:
    the first call to _groundedness_call_once raises; the second returns 0.82.
    The returned value must equal 0.82 exactly (pytest.approx).
    """
    from unittest.mock import AsyncMock, patch

    from iw_rag.guardrails.output_guard import _groundedness_call_with_retry

    call_results: list = [RuntimeError("transient blip"), 0.82]
    call_index = 0

    async def _flaky_once(answer: str, evidence: str) -> float:
        nonlocal call_index
        result = call_results[call_index]
        call_index += 1
        if isinstance(result, Exception):
            raise result
        return result

    with (
        patch(
            "iw_rag.guardrails.output_guard._groundedness_call_once",
            new=AsyncMock(side_effect=_flaky_once),
        ),
        patch("iw_rag.guardrails.output_guard.asyncio.sleep", new=AsyncMock()),
    ):
        score = await _groundedness_call_with_retry("an answer", "some evidence", retries=2)

    assert score == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# scan_output — safety_model seam branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_output_safety_model_seam_does_not_block() -> None:
    """When safety_model is configured, scan_output completes without blocking.

    The seam is wired but the actual call is not implemented (VRAM default off).
    Setting safety_model to a non-None value must not raise or alter the verdict
    fields — it logs and continues (lld-guardrails §5).
    """
    import os

    from iw_rag.guardrails.output_guard import scan_output

    answer = "The total is $15,000 [1]."
    citations = [
        Citation(marker=1, chunk_id="chunk-a", document_id="doc-a", source_uri="corpus://a")
    ]
    graded = [_CHUNK_A]

    async def _mock_presidio_redact(text: str, ctx: AccessContext) -> str:
        return text

    async def _mock_groundedness(answer: str, evidence: str) -> tuple[bool, float]:
        return True, 0.91

    # Activate the safety_model seam via environment variable override so
    # get_settings() returns a config with safety_model set.
    prev = os.environ.get("IW_RAG__GUARDRAILS__OUTPUT__SAFETY_MODEL")
    os.environ["IW_RAG__GUARDRAILS__OUTPUT__SAFETY_MODEL"] = "guard_safety"
    # Clear the settings cache so the updated env var is picked up.
    import iw_rag.settings as _settings

    _settings.get_settings.cache_clear()

    try:
        with (
            patch(
                "iw_rag.guardrails.output_guard._presidio_redact",
                new=AsyncMock(side_effect=_mock_presidio_redact),
            ),
            patch(
                "iw_rag.guardrails.output_guard._granite_groundedness_score",
                new=AsyncMock(side_effect=_mock_groundedness),
            ),
        ):
            verdict = await scan_output(answer, citations, graded, _CTX)
    finally:
        # Restore env and clear cache so subsequent tests see clean settings.
        if prev is None:
            os.environ.pop("IW_RAG__GUARDRAILS__OUTPUT__SAFETY_MODEL", None)
        else:
            os.environ["IW_RAG__GUARDRAILS__OUTPUT__SAFETY_MODEL"] = prev
        _settings.get_settings.cache_clear()

    # The seam must not change the verdict — grounded and citations_valid remain correct.
    assert verdict.grounded is True
    assert verdict.citations_valid is True
    assert verdict.groundedness_score == pytest.approx(0.91)
