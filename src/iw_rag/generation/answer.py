"""Answer assembly — citation parsing, source mapping, fail-closed refusal.

Structural validation only: every emitted ``[n]`` must resolve to a real source;
hallucinated markers are dropped and flagged. *Deep* groundedness (does the cited
source actually support the claim?) is the Phase-3 output guard's job. An empty
``RetrievalResult`` is an honest refusal with NO LLM call. Citation ``quote``
retains the matched leaf span when content was parent-expanded (small-to-big).
"""

from __future__ import annotations

import re

from iw_rag.domain.models import Citation, GeneratedAnswer, RetrievalResult, RetrievedChunk
from iw_rag.generation.llm import GenerationParams, LlmClient
from iw_rag.generation.prompts import (
    GENERAL_KNOWLEDGE_DISCLAIMER,
    REFUSAL_TEXT,
    CitationPrompt,
    build_general_knowledge_messages,
)

_MARKER = re.compile(r"\[([0-9]+(?:\s*,\s*[0-9]+)*)\]")

#: Reasoning-model scratchpad block (e.g. Qwen3 ``<think>…</think>``). Stripped
#: from the user-facing answer; model-agnostic (a no-op when absent).
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

#: An UNTERMINATED reasoning block — a ``<think>`` whose ``</think>`` never arrived
#: (the model overthought and hit the token limit before answering). Without this,
#: the raw chain-of-thought leaks to the user. Stripped from the opening tag to EOS.
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)

#: Datamark spotlighting token left in sanitized chunk content: U+2063 INVISIBLE
#: SEPARATOR followed by the session hex marker (see
#: ``guardrails.sanitize.generate_session_marker``). It is interspersed into
#: retrieved content to break embedded injection sequences and MUST be stripped
#: before any of that content is echoed back to the user (citation quotes). The
#: separator is a Unicode control char that never occurs in real document text,
#: so a pattern strip is safe and avoids a generation→guardrails import.
_DATAMARK = re.compile("⁣[0-9a-f]*")


def strip_datamarks(text: str) -> str:
    """Remove guardrails datamark spotlighting tokens from ``text``.

    Sanitized chunk content carries an interspersed ``U+2063``+hex marker
    (guards-on mode). The generated *answer* rarely echoes it, but citation
    ``quote`` values are sliced directly from the marked content and would
    otherwise surface the invisible markers to the user. No-op when absent.

    Args:
        text: Possibly-datamarked text.

    Returns:
        The text with all datamark tokens removed.
    """
    return _DATAMARK.sub("", text)


def strip_reasoning(text: str) -> str:
    """Remove a leading ``<think>…</think>`` scratchpad block from model output.

    Reasoning models (Qwen3 et al.) prepend their chain-of-thought in a
    ``<think>`` block that must never reach the user or the citation parser.
    No-op for models that emit none. Also drops an *unterminated* ``<think>``
    block (overthought past the token limit) so raw chain-of-thought never leaks.

    Args:
        text: Raw model output.

    Returns:
        The answer text with any reasoning block removed and outer whitespace trimmed.
    """
    text = _THINK.sub("", text)
    text = _THINK_OPEN.sub("", text)
    return text.strip()


def parse_citation_markers(text: str) -> list[int]:
    """Extract citation markers from answer text (tolerant of ``[1]``/``[1,2]``/``[1][2]``).

    Args:
        text: The generated answer.

    Returns:
        The unique 1-based markers in first-seen order.
    """
    seen: dict[int, None] = {}
    for group in _MARKER.findall(text):
        for part in group.split(","):
            seen.setdefault(int(part.strip()), None)
    return list(seen)


def _is_refusal(text: str) -> bool:
    """Whether the model emitted the explicit refusal phrase."""
    return REFUSAL_TEXT.lower() in text.lower()


def _resolve(markers: list[int], chunks: list[RetrievedChunk]) -> tuple[list[Citation], list[int]]:
    """Map markers to chunks; return (citations, dropped_out_of_range_markers).

    Threads the chunk's ``document_id`` (owning ``documents.id`` PK) onto each
    :class:`Citation` so the API layer can mint a per-citation
    ``download_url`` addressing ``GET /documents/{document_id}/raw``
    (CR-00098).  The UUID embedded in ``source_uri`` is a *storage* uuid, not
    the PK — ``document_id`` is the only addressable identifier for the
    download flow.  Marker resolution semantics are unchanged: in-range markers
    become citations, out-of-range markers are dropped and reported separately.
    """
    citations: list[Citation] = []
    dropped: list[int] = []
    for marker in markers:
        idx = marker - 1
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_uri=chunk.source_uri,
                    # small-to-big: prefer the precise matched leaf span for the
                    # quote; strip datamark spotlighting tokens (guards-on mode)
                    # so the user-facing quote never shows the invisible markers.
                    quote=strip_datamarks(chunk.leaf_content or chunk.content),
                )
            )
        else:
            dropped.append(marker)
    return citations, dropped


class AnswerAssembler:
    """Builds a grounded, cited :class:`GeneratedAnswer` from a RetrievalResult."""

    def __init__(
        self,
        llm: LlmClient,
        prompt: CitationPrompt,
        params: GenerationParams,
        *,
        cite_required: bool = True,
    ) -> None:
        """Wire the assembler.

        Args:
            llm: The generator client.
            prompt: The citation-required prompt builder.
            params: Generation params (temperature/model/...).
            cite_required: When True, an answer with no resolved citation is
                treated as ungrounded.
        """
        self._llm = llm
        self._prompt = prompt
        self._params = params
        self._cite_required = cite_required

    async def generate(
        self, query: str, result: RetrievalResult, *, allow_ungrounded: bool = False
    ) -> GeneratedAnswer:
        """Generate a grounded, cited answer (or an honest refusal).

        Args:
            query: The user question.
            result: The retrieval result (its chunks are the only allowed sources).
            allow_ungrounded: Assist mode. When True and the documents do not
                support a grounded answer, fall back to a general-knowledge
                answer (``grounded=False``, ``diagnostics["general_knowledge"]``)
                clearly marked as not document-derived, instead of refusing.
                Defaults False (strict fail-closed).

        Returns:
            A :class:`GeneratedAnswer`; ``grounded=False`` is a fail-closed refusal
            in strict mode, or a marked general-knowledge answer in assist mode.
        """
        if not result.chunks:
            # No evidence at all. Strict → honest refusal (no LLM call). Assist →
            # answer from general knowledge, clearly marked as not document-derived.
            if allow_ungrounded:
                return await self._general_knowledge(query, reason="no_chunks")
            return GeneratedAnswer(
                answer=REFUSAL_TEXT, grounded=False, diagnostics={"reason": "no_chunks"}
            )

        bundle = self._prompt.build(query, result.chunks)
        resp = await self._llm.complete(bundle.as_messages(), self._params)
        # Strip any reasoning scratchpad (Qwen3 <think>…</think>) before it can
        # reach the user or the citation parser.
        answer_text = strip_datamarks(strip_reasoning(resp.text))
        if not answer_text:
            # Nothing left after stripping (e.g. the model only emitted an
            # unterminated reasoning block). Strict → refuse; assist → general
            # knowledge so the user still gets a useful, marked answer.
            if allow_ungrounded:
                return await self._general_knowledge(query, reason="empty_generation")
            return GeneratedAnswer(
                answer=REFUSAL_TEXT, grounded=False, diagnostics={"reason": "empty_generation"}
            )
        if _is_refusal(answer_text):
            # The model refused to answer from the documents. Strict → surface the
            # refusal. Assist → fall back to general knowledge (marked).
            if allow_ungrounded:
                return await self._general_knowledge(query, reason="grounded_refusal")
            return GeneratedAnswer(answer=answer_text, grounded=False, usage=resp.usage)

        markers = parse_citation_markers(answer_text)
        citations, dropped = _resolve(markers, bundle.used_chunks)
        grounded = bool(citations) if self._cite_required else True
        diagnostics: dict[str, object] = {}
        if dropped:
            diagnostics["dropped_markers"] = dropped
        return GeneratedAnswer(
            answer=answer_text,
            grounded=grounded,
            citations=citations,
            used_chunk_ids=[c.chunk_id for c in citations],
            usage=resp.usage,
            diagnostics=diagnostics,
        )

    async def _general_knowledge(self, query: str, *, reason: str) -> GeneratedAnswer:
        """Assist-mode fallback: answer from the model's own knowledge (marked).

        The retrieved documents could not ground an answer; the caller opted into
        assist mode, so we answer from general knowledge with an explicit
        disclaimer prepended. The result carries ``grounded=False`` and
        ``diagnostics["general_knowledge"]=True`` so downstream guards and the UI
        treat it as document-unsupported. It still flows through the output guard
        (PII redaction) and every upstream safeguard.

        Args:
            query: The user's original question.
            reason: Why we fell back (``"no_chunks"`` or ``"grounded_refusal"``).

        Returns:
            A general-knowledge :class:`GeneratedAnswer` (no citations).
        """
        resp = await self._llm.complete(build_general_knowledge_messages(query), self._params)
        answer_text = strip_datamarks(strip_reasoning(resp.text))
        if not answer_text or _is_refusal(answer_text):
            # Even general knowledge had nothing — keep an honest, helpful note.
            answer_text = (
                "I don't have enough information to answer that, either from your "
                "documents or from general knowledge. Try rephrasing or asking "
                "something more specific."
            )
        body = f"{GENERAL_KNOWLEDGE_DISCLAIMER}\n\n{answer_text}"
        return GeneratedAnswer(
            answer=body,
            grounded=False,
            citations=[],
            used_chunk_ids=[],
            usage=resp.usage,
            diagnostics={"general_knowledge": True, "reason": reason},
        )
