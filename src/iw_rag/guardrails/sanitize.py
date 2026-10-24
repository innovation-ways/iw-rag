"""Context sanitization — datamarking, per-chunk injection scan, Presidio PII redaction.

Implements the sanitize layer (lld-guardrails §3.3) that sits between ``retrieve``
and ``grade_documents`` in the LangGraph loop.  Every retrieved chunk is treated as
**untrusted data** and passed through a fixed pipeline:

1. **Per-chunk injection scan** (Prompt Guard 2, reused from Task 3.1) — chunks
   flagged as injection are dropped (``on_poison=drop``, default) or marked
   (``on_poison=flag``) and excluded from further processing.
2. **PII redaction** (Presidio Analyzer + Anonymizer, CPU-only) — survivors are
   clearance-scoped: only the PII entity types in the access context's policy are
   redacted.  This is a *second*, field-level layer strictly additive to the
   document-level ACL pre-filter (which is upstream — this module must NOT re-filter
   by entitlement).
3. **Datamarking / spotlighting** (pure string ops, no model) — a session-randomized
   marker is interspersed every N words so embedded instruction sequences are broken
   and the generator treats chunk text as *data*, not instructions.

Order is fixed: injection-scan → PII-redact → datamark.  Scanning first avoids
spending Presidio on poisoned chunks; redacting before datamarking prevents markers
from fragmenting PII spans; datamarking last produces the final generator-visible form.

**Presidio is lazily imported** inside :func:`_presidio_redact` so this module can be
collected by pytest even when the ``guardrails`` dependency group is not installed.
Tests mock ``_presidio_redact`` directly.

ACL is enforced UPSTREAM (Task 3.0).  This module must NOT filter by entitlement.
``principal_groups`` in ``AccessContext`` are used only for clearance-scoped PII
policy and audit attribution; they are NEVER derived from chunk text or model output.

References:
- lld-guardrails §3.3 (sanitize pipeline + order invariant)
- lld-guardrails §5 (all-dropped → fail-closed)
- lld-guardrails §6 (config: every_n_words, on_poison, pii entity types)
- task-3.2-context-sanitization §1–§6
- CONVENTIONS.md §1 (RetrievedChunk contract), §2 (AccessContext invariant)
- R-00108 F2 (indirect injection is the #1 RAG threat; datamarking drops ASR
  ~50%→~3% — figures are GPT-era, to be re-validated locally per Task 3.5)
"""

from __future__ import annotations

import logging
import secrets
from functools import lru_cache
from typing import Any, Literal

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import GuardVerdict, RetrievedChunk, SanitizedContext
from iw_rag.guardrails.input_guard import prompt_guard_score
from iw_rag.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default PII entity types masked when no per-clearance policy is configured.
# A deployment policy maps clearance levels to entity-type sets; this is the
# safe production default (mask common PII for all requesters).
# ---------------------------------------------------------------------------
_DEFAULT_PII_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IBAN_CODE",
        "US_SSN",
        "UK_NHS",
        "PERSON",
        "LOCATION",
    }
)

# Injection threshold: P(jailbreak) at or above this → flagged per chunk.
# Matches the input-guard threshold from settings (IW_RAG__GUARDRAILS__INPUT__THRESHOLD).
_INJECTION_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Datamarking / spotlighting (pure string ops — no model)
# ---------------------------------------------------------------------------


def datamark(text: str, *, marker: str, every_n_words: int = 1) -> str:
    """Intersperse ``marker`` after every Nth word (spotlighting / datamarking).

    Breaks embedded instruction sequences so the generator treats the text as
    *data*, not instructions.  Datamarking drops injection attack success rates
    ~50%→~3% with no measurable task accuracy hit (R-00108 F2).

    The marker is appended directly to the word (no space between word and marker)
    so the total tokenization is predictable and ``strip_marker`` can remove it
    as a simple substring.

    Args:
        text: The chunk content to mark.
        marker: The session-randomized token to intersperse.  Must be the same
            token passed to :func:`strip_marker` for round-trip correctness.
        every_n_words: Insert the marker after every Nth word.  Default 1
            (marks every word, maximum density — tuned via config).

    Returns:
        The marked text.  Empty when ``text`` contains no words.
    """
    words = text.split()
    if not words:
        return ""
    return " ".join(
        w + (marker if (i + 1) % every_n_words == 0 else "") for i, w in enumerate(words)
    )


def strip_marker(text: str, marker: str) -> str:
    """Remove all occurrences of ``marker`` from ``text``.

    Used before any chunk text is echoed back to the user (citations, previews)
    to ensure the session marker does not leak into responses.

    Args:
        text: The datamarked text.
        marker: The marker string to strip (must match what was passed to
            :func:`datamark`).

    Returns:
        The text with all marker occurrences removed.
    """
    return text.replace(marker, "")


def generate_session_marker() -> str:
    """Generate a session-randomized unguessable marker token.

    Called once per request; the same marker is threaded through
    :func:`datamark` for all chunks in the session and stripped before any
    text is echoed to the user.  Randomisation prevents an attacker from
    pre-empting the marker with a fixed token.

    Returns:
        A cryptographically random hex token (12 hex characters, 6 bytes of
        entropy — sufficient to prevent pre-emption for a single session).
    """
    # 6 bytes = 12 hex chars = 72 bits of randomness per session
    return f"⁣{secrets.token_hex(6)}"


# ---------------------------------------------------------------------------
# clearance_filter — PII entity-type scoping
# ---------------------------------------------------------------------------


def clearance_filter(
    analyzer_results: list[Any],
    ctx: AccessContext,
    *,
    pii_entity_types: set[str] | None = None,
) -> list[Any]:
    """Filter Presidio analyzer results to only the clearance-scoped entity types.

    This is the *second*, field-level PII layer on top of the document-level ACL
    pre-filter.  It is strictly additive — not a substitute for ACL enforcement.
    ``principal_groups`` from ``ctx`` are used for audit attribution; the PII
    policy itself is passed in as ``pii_entity_types`` (a deployment concern, not
    hardwired here).

    Args:
        analyzer_results: List of ``RecognizerResult`` objects from
            ``presidio_analyzer.AnalyzerEngine.analyze``.
        ctx: The authenticated ``AccessContext`` (for audit attribution; not used
            to derive ``principal_groups`` from text or model output).
        pii_entity_types: The set of entity type strings that must be masked for
            this requester's clearance (e.g. ``{"EMAIL_ADDRESS", "PHONE_NUMBER"}``).
            ``None`` → use the module-level default set.

    Returns:
        A filtered list containing only results whose ``entity_type`` is in
        ``pii_entity_types``.
    """
    effective_types: frozenset[str] = (
        frozenset(pii_entity_types) if pii_entity_types is not None else _DEFAULT_PII_ENTITY_TYPES
    )
    return [r for r in analyzer_results if r.entity_type in effective_types]


# ---------------------------------------------------------------------------
# provenance-aware carve-out — keep PII the user themselves supplied
# ---------------------------------------------------------------------------


def _drop_allowlisted(text: str, results: list[Any], allow_text: str | None) -> list[Any]:
    """Drop analyzer results whose detected span appears in ``allow_text``.

    Provenance-aware PII carve-out (chat memory): PII the user themselves
    supplied earlier in the *same* conversation — recalled into the answer via
    conversation-as-source memory — must NOT be redacted, or "Sergio" would be
    scrubbed out of "your name is Sergio" and the memory would be pointless.
    Only spans **literally present** in the user's own conversation text are
    kept; genuinely-new PII (e.g. an address leaked from a document) still gets
    redacted because it never appears in ``allow_text``.

    Matching is case-insensitive substring containment against the recalled
    conversation text — deliberately narrow: the allow source is the user's own
    turns, scoped to this principal's conversation, never the corpus.

    Args:
        text: The text the analyzer ran over (to slice each span out of).
        results: Presidio ``RecognizerResult`` objects (need ``.start``/``.end``).
        allow_text: The recalled conversation text whose PII is exempt; ``None``
            or empty → no carve-out (every result is kept, legacy behaviour).

    Returns:
        ``results`` minus the spans whose text is found in ``allow_text``.
    """
    if not allow_text:
        return results
    allow_lc = allow_text.lower()
    kept: list[Any] = []
    for r in results:
        span = text[r.start : r.end].strip().lower()
        if span and span in allow_lc:
            continue  # the user's own recalled PII — leave it in the answer
        kept.append(r)
    return kept


# ---------------------------------------------------------------------------
# _presidio_redact — lazy Presidio import (mirrors ragas_runner pattern)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _presidio_engines() -> tuple[Any, Any]:
    """Build the Presidio analyzer/anonymizer once per process.

    ``AnalyzerEngine()`` loads the spaCy model (~1 s), so constructing it on every
    call would add that to each redaction. An ``ImportError`` (guardrails group
    missing) is raised, not cached, so installing the group takes effect without
    a restart.

    Returns:
        ``(AnalyzerEngine, AnonymizerEngine)``.

    Raises:
        ImportError: Presidio is not installed.
    """
    # Lazy import — presidio lives in the optional guardrails dep group.
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    return AnalyzerEngine(), AnonymizerEngine()


async def _presidio_redact(text: str, ctx: AccessContext, *, allow_text: str | None = None) -> str:
    """Redact PII from ``text`` using Presidio (CPU-only, lazily imported).

    Presidio is lazily imported inside this function so the module can be
    collected by pytest even when the ``guardrails`` dependency group is absent.
    Tests mock this function directly.

    The clearance-scoped policy is the module default (``_DEFAULT_PII_ENTITY_TYPES``);
    a production deployment should pass a per-principal policy here.

    Args:
        text: The chunk content to scan for PII.
        ctx: The authenticated ``AccessContext`` (for clearance scoping + audit).
        allow_text: Optional provenance carve-out — PII spans whose text appears
            in this string are NOT redacted (chat memory: the user's own recalled
            words). ``None`` (default) → redact everything in policy (legacy
            behaviour, so the sanitize-pipeline + existing callers are unchanged).

    Returns:
        The text with PII entities replaced by ``<ENTITY_TYPE>`` placeholders.
        Returns ``text`` unchanged if Presidio is unavailable (logged as a warning;
        accepted residual risk — R-00108 F4).
    """
    try:
        analyzer, anonymizer = _presidio_engines()
    except ImportError:
        logger.warning(
            "presidio_analyzer/presidio_anonymizer not installed; PII redaction skipped. "
            "Add the 'guardrails' dependency group: uv sync --group guardrails"
        )
        return text

    results = analyzer.analyze(text=text, language="en")
    filtered = clearance_filter(results, ctx)
    # Provenance carve-out: keep PII the user supplied earlier in this same
    # conversation (recalled via chat memory) — never the corpus.
    filtered = _drop_allowlisted(text, filtered, allow_text)
    if not filtered:
        # No PII entities in scope — return unchanged.
        return text

    anonymized = anonymizer.anonymize(text=text, analyzer_results=filtered)
    return str(anonymized.text)


# ---------------------------------------------------------------------------
# sanitize_context — the main pipeline
# ---------------------------------------------------------------------------


async def sanitize_context(
    chunks: list[RetrievedChunk],
    access_context: AccessContext,
    *,
    on_poison: Literal["drop", "flag"] = "drop",
    session_marker: str | None = None,
) -> SanitizedContext:
    """Per-chunk: injection scan → PII redact → datamark.  Returns a SanitizedContext.

    ACL is already applied upstream — this function does NOT filter by
    entitlement (lld-guardrails §3.3).  The ``access_context`` is threaded
    through for clearance-scoped PII policy selection and audit attribution.
    ``principal_groups`` are never derived from chunk text or model output.

    Pipeline order is invariant (lld-guardrails §3.3):
    1. **Injection scan** (Prompt Guard 2, reused from input_guard) — flagged
       chunks are dropped (``on_poison=drop``) or marked (``on_poison=flag``).
       Scanning first avoids spending Presidio on poisoned chunks.
    2. **PII redact** (Presidio, CPU-only, lazy import) — clearance-scoped.
       Redacting before datamarking prevents markers from fragmenting PII spans.
    3. **Datamark** (string ops, session-randomized marker) — applied last so
       the generator-visible text is the final, marked, redacted form.

    When all chunks are poisoned, ``chunks=[]`` is returned → ``grade_documents``
    sees an empty set → loop treats as incorrect → rewrite/fail-closed.  No
    ungrounded generation from a fully-poisoned context (lld-guardrails §5).

    When the master guardrails switch is disabled
    (``IW_RAG__GUARDRAILS__ENABLED=false``), the function returns chunks unchanged
    with empty verdicts and dropped_chunk_ids — used by the loop-mechanics unit
    tests (agent/conftest.py) to bypass the guard stack without serving models.

    Args:
        chunks: Retrieved chunks to sanitize.  Must be the ACL-pre-filtered
            spine output.  ACL is NOT re-applied here.
        access_context: Authenticated principal context (from auth, never the
            LLM or user free-text).  Used for PII policy + audit attribution.
        on_poison: ``"drop"`` (default, production) removes flagged chunks from
            the surviving set.  ``"flag"`` keeps them with
            ``metadata["poisoned"]=True`` for eval/demo comparisons.
        session_marker: Optional pre-generated marker for deterministic testing.
            When ``None`` (default), a fresh random marker is generated via
            :func:`generate_session_marker`.

    Returns:
        A :class:`~iw_rag.domain.models.SanitizedContext` with:
        - ``chunks``: injection-scanned, PII-redacted, datamarked survivors.
        - ``verdicts``: per-chunk ``GuardVerdict`` objects (injection + PII).
        - ``dropped_chunk_ids``: ids of chunks excluded under ``on_poison=drop``.
    """
    # ------------------------------------------------------------------
    # Master switch: guardrails disabled → pass through unchanged.
    # This path is used by the loop-mechanics unit tests (agent/conftest.py)
    # which set IW_RAG__GUARDRAILS__ENABLED=false via monkeypatch to bypass
    # the guard stack without serving Presidio or Prompt Guard 2.
    # ------------------------------------------------------------------
    cfg = get_settings()
    if not cfg.guardrails.enabled:
        return SanitizedContext(
            chunks=list(chunks),
            verdicts=[],
            dropped_chunk_ids=[],
        )

    # Generate a session marker for this sanitization pass.
    marker: str = session_marker if session_marker is not None else generate_session_marker()

    # Read every_n_words from settings (IW_RAG__GUARDRAILS__SANITIZE__EVERY_N_WORDS).
    every_n_words: int = cfg.guardrails.sanitize.every_n_words

    # Injection threshold from the input guard settings (shared threshold).
    threshold: float = cfg.guardrails.input.threshold

    surviving_chunks: list[RetrievedChunk] = []
    verdicts: list[GuardVerdict] = []
    dropped_chunk_ids: list[str] = []

    # Read injection_scan flag (IW_RAG__GUARDRAILS__SANITIZE__INJECTION_SCAN).
    # When False the per-chunk Prompt Guard 2 scan is skipped entirely —
    # the "gate has teeth" test uses this to prove the gate detects regressions
    # (Task 3.5 / lld-guardrails §6).
    injection_scan_enabled: bool = cfg.guardrails.sanitize.injection_scan

    for chunk in chunks:
        # ------------------------------------------------------------------
        # Step 1: Per-chunk injection scan (Prompt Guard 2)
        # Skipped when injection_scan=False (config-controlled; task 3.5).
        # ------------------------------------------------------------------
        injection_flagged = False
        if injection_scan_enabled:
            try:
                pg_score = await prompt_guard_score(chunk.content)
            except Exception:
                # Fail-closed: if the guard client is unavailable, drop the chunk.
                logger.warning(
                    "sanitize: prompt_guard_score unavailable for chunk_id=%r"
                    " — dropping (fail-closed)",
                    chunk.chunk_id,
                )
                pg_score = 1.0  # treat as flagged

            injection_flagged = pg_score >= threshold
        else:
            pg_score = 0.0  # scan skipped — no injection score

        if injection_flagged:
            if on_poison == "drop":
                # Drop the chunk; record a block verdict; skip PII + datamark.
                dropped_chunk_ids.append(chunk.chunk_id)
                verdicts.append(
                    GuardVerdict(
                        layer="sanitize",
                        decision="block",
                        score=pg_score,
                        reason=f"injection:{pg_score:.3f}",
                        detail={"chunk_id": chunk.chunk_id},
                    )
                )
                logger.info(
                    "sanitize: chunk_id=%r DROPPED (injection score=%.3f)",
                    chunk.chunk_id,
                    pg_score,
                )
                # Do NOT PII-scan or datamark this chunk.
                continue
            else:
                # flag: keep but mark poisoned in metadata
                verdicts.append(
                    GuardVerdict(
                        layer="sanitize",
                        decision="flag",
                        score=pg_score,
                        reason=f"injection:{pg_score:.3f}",
                        detail={"chunk_id": chunk.chunk_id},
                    )
                )
                # Update metadata to signal poisoning; replace chunk in-loop.
                new_metadata = dict(chunk.metadata)
                new_metadata["poisoned"] = True
                chunk = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=chunk.score,
                    source_uri=chunk.source_uri,
                    metadata=new_metadata,
                    leaf_content=chunk.leaf_content,
                )
                logger.info(
                    "sanitize: chunk_id=%r FLAGGED (injection score=%.3f, on_poison=flag)",
                    chunk.chunk_id,
                    pg_score,
                )

        # ------------------------------------------------------------------
        # Step 2: PII redaction (Presidio, CPU-only, lazy import)
        # Must run BEFORE datamarking so markers don't fragment PII spans.
        # ------------------------------------------------------------------
        original_content = chunk.content
        redacted_content = await _presidio_redact(original_content, access_context)

        if redacted_content != original_content:
            # Record what was redacted (entity types, not values).
            verdicts.append(
                GuardVerdict(
                    layer="sanitize",
                    decision="redact",
                    score=1.0,
                    reason="pii_redacted",
                    detail={"chunk_id": chunk.chunk_id},
                )
            )
            logger.info(
                "sanitize: chunk_id=%r PII redacted (content changed)",
                chunk.chunk_id,
            )
            # Rebuild chunk with redacted content.
            chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=redacted_content,
                score=chunk.score,
                source_uri=chunk.source_uri,
                metadata=dict(chunk.metadata),
                leaf_content=chunk.leaf_content,
            )

        # ------------------------------------------------------------------
        # Step 3: Datamark (pure string ops, session-randomized marker)
        # Applied LAST so the final generator-visible text is marked + redacted.
        # ------------------------------------------------------------------
        marked_content = datamark(chunk.content, marker=marker, every_n_words=every_n_words)
        chunk = RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=marked_content,
            score=chunk.score,
            source_uri=chunk.source_uri,
            metadata=dict(chunk.metadata),
            leaf_content=chunk.leaf_content,
        )

        surviving_chunks.append(chunk)

    logger.debug(
        "sanitize: %d/%d chunks survived (dropped=%d, on_poison=%r)",
        len(surviving_chunks),
        len(chunks),
        len(dropped_chunk_ids),
        on_poison,
    )

    return SanitizedContext(
        chunks=surviving_chunks,
        verdicts=verdicts,
        dropped_chunk_ids=dropped_chunk_ids,
    )
