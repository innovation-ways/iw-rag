"""Reranker lab: replay a recorded real spine run, changing what the spine's knobs decide.

``traces/rerank.json`` holds, for every lab question, the fused candidate list the
production dense and BM25 arms produced, the cross-encoder's score for every candidate,
and what the production ``RetrievalSpine`` really returned in three configurations (see
``scripts/learn_record_rerank_trace.py``).  The reranker scores each chunk on its own, so
the reader can change how many to keep, switch the reranker off (the spine then falls back
to the fused order) or change the expansion setting, and this module re-applies each rule:
the order from the recorded scores, and the small-to-big rule with the production
:func:`~iw_rag.retrieval.spine.expand_to_parent`.  Nothing here needs a GPU or a database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from iw_rag.domain.models import RetrievedChunk
from iw_rag.learn.rerank_corpus import QUERIES, RerankQuery
from iw_rag.retrieval.spine import expand_to_parent, keep_candidate, merge_repeated_sections

TRACE_PATH = Path(__file__).parent / "traces" / "rerank.json"
SHOWN = 12
MIN_TOP_K, MAX_TOP_K = 1, 10
MIN_BUDGET, MAX_BUDGET = 500, 8000
CLAMP_CHARS = 600


@dataclass(frozen=True, slots=True)
class Candidate:
    """One chunk in the fused shortlist, before and after the reranker."""

    chunk_id: str
    kind: str
    document: str
    section: str
    text: str
    fused_rank: int
    rerank_rank: int
    rerank_score: float
    dense_rank: int | None
    sparse_rank: int | None
    is_answer: bool


@dataclass(frozen=True, slots=True)
class Selected:
    """One chunk handed to the generator, and the text it is handed."""

    chunk_id: str
    kind: str
    document: str
    section: str
    rank: int
    matched_text: str
    sent_text: str
    sent_chars: int
    expanded: bool
    note: str
    is_answer: bool
    merged_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RerankLabResult:
    """Everything the reranker lab shows for one question and one set of knobs."""

    query: dict[str, str]
    settings: dict[str, Any]
    candidates_total: int
    fused: tuple[Candidate, ...]
    reranked: tuple[Candidate, ...]
    kept_ids: tuple[str, ...]
    selected: tuple[Selected, ...]
    facts: dict[str, Any]
    notes: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def load_trace() -> dict[str, Any]:
    """Load the recorded trace (cached); raises ``FileNotFoundError`` if not recorded."""
    data: dict[str, Any] = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    return data


def get_query(query_id: str) -> RerankQuery | None:
    """Return the lab question for ``query_id``, or ``None``."""
    return next((q for q in QUERIES if q.query_id == query_id), None)


def _index(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for leaf in trace["leaves"]:
        out[leaf["id"]] = {**leaf, "kind": "leaf"}
    for parent in trace["parents"]:
        out[parent["id"]] = {**parent, "kind": "section", "parent": None}
    return out


def _order(candidates: list[dict[str, Any]], reranker_on: bool) -> list[dict[str, Any]]:
    if not reranker_on:
        return list(candidates)
    return sorted(candidates, key=lambda c: -c["rerank"])


def run_rerank_lab(  # noqa: PLR0914 - one pass over the pipeline reads best as one function
    query: RerankQuery,
    *,
    top_k: int = 8,
    reranker_on: bool = True,
    expansion: str = "parent",
    budget_chars: int = 4000,
    sections: bool = True,
) -> RerankLabResult:
    """Replay the recorded run for ``query`` with the given spine settings.

    Args:
        query: The lab question.
        top_k: How many chunks to keep after reranking, clamped to 1..10.
        reranker_on: ``False`` replays the spine's fallback when the reranker is down:
            the fused order, cut at ``top_k``.
        expansion: ``"parent"`` (small-to-big) or ``"leaf"`` (no expansion).
        budget_chars: The per-candidate expansion budget, clamped to 500..8000.
        sections: The ``retrieval.section_hits`` switch: whether a section row may match on its
            own and be a candidate.

    Returns:
        The fused and reranked lists, the chunks handed to the generator and the facts
        that show what the choices cost or bought.
    """
    trace = load_trace()
    top_k = max(MIN_TOP_K, min(MAX_TOP_K, top_k))
    budget_chars = max(MIN_BUDGET, min(MAX_BUDGET, budget_chars))
    expansion = "leaf" if expansion == "leaf" else "parent"
    info = _index(trace)
    recorded = next(q for q in trace["queries"] if q["id"] == query.query_id)
    cands: list[dict[str, Any]] = recorded["candidates"]

    fused_rank = {c["id"]: i for i, c in enumerate(cands, start=1)}
    # The production rule for which fused candidates go on to the reranker (CR-00103).
    section_ids = frozenset(p["id"] for p in trace["parents"])
    eligible: list[dict[str, Any]] = []
    set_aside: dict[str, int] = {}
    for c in cands:
        why = keep_candidate(
            c["id"],
            info[c["id"]]["text"],
            section_ids,
            expansion_mode=expansion,
            section_hits=sections,
            budget_chars=budget_chars,
        )
        if why is None:
            eligible.append(c)
        else:
            set_aside[why] = set_aside.get(why, 0) + 1
    reranked_all = _order(eligible, True)
    rerank_rank = {c["id"]: i for i, c in enumerate(reranked_all, start=1)}
    # Every chunk whose text holds the answer carries it to the generator: the leaf, and the
    # section around it.  The *answer leaf* is the leaf among them.
    carrier_ids = {cid for cid, c in info.items() if query.answer in c["text"]}
    leaf_ids = {cid for cid in carrier_ids if info[cid]["kind"] == "leaf"}

    def candidate(c: dict[str, Any]) -> Candidate:
        meta = info[c["id"]]
        return Candidate(
            chunk_id=c["id"],
            kind=meta["kind"],
            document=meta["doc"],
            section=meta["section"],
            text=meta["text"],
            fused_rank=fused_rank[c["id"]],
            rerank_rank=rerank_rank[c["id"]],
            rerank_score=c["rerank"],
            dense_rank=c["dense_rank"],
            sparse_rank=c["sparse_rank"],
            is_answer=c["id"] in carrier_ids,
        )

    # Walk every eligible candidate in rank order, expand each leaf, then merge repeats and cut to
    # top_k: the same steps, in the same order, as RetrievalSpine.retrieve.
    expanded_chunks: list[RetrievedChunk] = []
    for c in _order(eligible, reranker_on):
        meta = info[c["id"]]
        parent_text = info[meta["parent"]]["text"] if meta.get("parent") else None
        base = RetrievedChunk(
            chunk_id=c["id"],
            document_id=meta["doc"],
            content=meta["text"],
            score=c["rerank"],
            source_uri=f"learn://{meta['doc']}",
            metadata={"parent_id": meta["parent"]} if meta.get("parent") else {},
        )
        expanded_chunks.append(
            expand_to_parent(base, parent_text, budget_chars) if expansion == "parent" else base
        )
    picked, merged = merge_repeated_sections(expanded_chunks, section_ids, top_k)
    kept_ids = tuple(c.chunk_id for c in picked)

    selected: list[Selected] = []
    for rank, sent in enumerate(picked, start=1):
        meta = info[sent.chunk_id]
        merged_ids = tuple(sent.metadata.get("merged_chunk_ids", ()))
        expanded = sent.leaf_content is not None and sent.content != sent.leaf_content
        parent_text = info[meta["parent"]]["text"] if meta.get("parent") else None
        if meta["kind"] == "section":
            note = "A whole section matched on its own and fits the budget, so it is sent as it is."
        elif expansion == "leaf":
            note = "Expansion is off: only the matched leaf is sent."
        elif expanded:
            note = (
                f"The leaf's section ({len(sent.content):,} characters) fits the budget and "
                "replaces it."
            )
        else:
            note = (
                f"The leaf's section ({len(parent_text or ''):,} characters) is over the "
                f"{budget_chars:,}-character budget, so only the leaf is sent."
            )
        if merged_ids:
            note += f" It also stands for {', '.join(merged_ids)}, which carried the same text."
        selected.append(
            Selected(
                chunk_id=sent.chunk_id,
                kind=meta["kind"],
                document=meta["doc"],
                section=meta["section"],
                rank=rank,
                matched_text=meta["text"] if sent.leaf_content is None else sent.leaf_content,
                sent_text=sent.content,
                sent_chars=len(sent.content),
                expanded=expanded,
                note=note,
                is_answer=sent.chunk_id in carrier_ids or any(m in carrier_ids for m in merged_ids),
                merged_ids=merged_ids,
            )
        )

    sent_texts = [s.sent_text for s in selected]
    total_chars = sum(len(t) for t in sent_texts)
    unique_chars = _unique_chars(sent_texts)
    context = "\n".join(sent_texts)
    gold_ranks = {
        "fused": min((fused_rank[g] for g in leaf_ids if g in fused_rank), default=None),
        "reranked": min((rerank_rank[g] for g in leaf_ids if g in rerank_rank), default=None),
    }
    eligible_ids = {c["id"] for c in eligible}
    carrier_fused = min((fused_rank[g] for g in carrier_ids if g in eligible_ids), default=None)
    answer_in_context = query.answer in context
    context_in_context = (query.context in context) if query.context else None
    facts = {
        "gold_ranks": gold_ranks,
        "carrier_fused_rank": carrier_fused,
        "answer_in_context": answer_in_context,
        "context_phrase": query.context or None,
        "context_phrase_in_context": context_in_context,
        "chars_sent": total_chars,
        "chars_unique": unique_chars,
        "chars_duplicated": total_chars - unique_chars,
        "chunks_sent": len(selected),
        "repeats_merged": merged,
        "sections_set_aside": set_aside,
        "eligible": len(eligible),
    }
    notes = _notes(facts, reranker_on, top_k)
    ordered_fused = [candidate(c) for c in eligible]
    ordered_reranked = [candidate(c) for c in reranked_all]
    return RerankLabResult(
        query={"id": query.query_id, "text": query.text, "lesson": query.lesson},
        settings={
            "top_k": top_k,
            "reranker_on": reranker_on,
            "expansion": expansion,
            "budget_chars": budget_chars,
            "sections": sections,
        },
        candidates_total=len(cands),
        fused=tuple(ordered_fused[:SHOWN]),
        reranked=tuple(ordered_reranked[:SHOWN] if reranker_on else ordered_fused[:SHOWN]),
        kept_ids=kept_ids,
        selected=tuple(selected),
        facts=facts,
        notes=tuple(notes),
        meta={
            "recorded": trace["recorded"],
            "embedder": trace["embedder"],
            "reranker": trace["reranker"],
            "embedding_note": trace["embedding_note"],
            "spine_top_k": trace["top_k"],
            "spine_budget": trace["parent_budget_chars"],
        },
    )


def _unique_chars(texts: list[str]) -> int:
    """Characters sent once: a text contained in another selected text counts only once."""
    total = 0
    for i, text in enumerate(texts):
        repeated = any(
            text in other and (len(other) > len(text) or j < i)
            for j, other in enumerate(texts)
            if j != i
        )
        if not repeated:
            total += len(text)
    return total


def _notes(facts: dict[str, Any], reranker_on: bool, top_k: int) -> list[str]:
    """Plain-language observations about what this combination of settings did."""
    notes: list[str] = []
    fused, reranked = facts["gold_ranks"]["fused"], facts["gold_ranks"]["reranked"]
    carrier = facts["carrier_fused_rank"]
    if reranker_on and fused is not None and reranked is not None:
        if reranked < fused:
            notes.append(
                f"The reranker moved the answer leaf from #{fused} to #{reranked}, by reading the "
                "question and each chunk together."
            )
        elif reranked > fused:
            notes.append(f"The reranker moved the answer leaf down from #{fused} to #{reranked}.")
    if carrier is not None and fused is not None and carrier < fused:
        notes.append(
            f"The answer leaf is buried at #{fused}, but a section that contains it ranks "
            f"#{carrier} in the fused order, so the answer can also reach the generator through "
            "that section."
        )
    position = fused if not reranker_on else reranked
    if position is not None and position > top_k:
        tail = (
            "It still reaches the generator inside a section that was kept."
            if facts["answer_in_context"]
            else "It never reaches the generator."
        )
        notes.append(f"With top-k = {top_k} the answer leaf (#{position}) is cut. {tail}")
    if not reranker_on:
        notes.append(
            "The reranker is off, so the spine uses the fused order. This is what happens when the "
            "reranker is down: the query still works, with weaker ordering."
        )
    if (
        facts["answer_in_context"]
        and facts["context_phrase"]
        and facts["context_phrase_in_context"] is False
    ):
        notes.append(
            f"The answer reaches the generator, but without “{facts['context_phrase']}…”, the "
            "text that says what the answer is about."
        )
    if facts["repeats_merged"] > 0:
        notes.append(
            f"{facts['repeats_merged']} repeated "
            f"cop{'y' if facts['repeats_merged'] == 1 else 'ies'} merged: a section and its own "
            "leaves carried the same text, so the generator reads it once and the freed slots "
            "went to the next passages."
        )
    aside = facts["sections_set_aside"]
    if aside.get("oversize"):
        notes.append(
            f"{aside['oversize']} section(s) were set aside before reranking because they are over "
            "the budget and could never be sent whole; their leaves were still candidates."
        )
    if aside.get("leaf_only") or aside.get("section_hits_off"):
        count = aside.get("leaf_only", 0) + aside.get("section_hits_off", 0)
        notes.append(f"{count} section row(s) were not candidates: only leaves can match.")
    if not facts["answer_in_context"]:
        notes.append("The answer is not in anything the generator receives.")
    return notes


__all__ = [
    "MAX_BUDGET",
    "MAX_TOP_K",
    "MIN_BUDGET",
    "MIN_TOP_K",
    "SHOWN",
    "Candidate",
    "RerankLabResult",
    "Selected",
    "get_query",
    "load_trace",
    "run_rerank_lab",
]
