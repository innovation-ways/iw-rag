"""Guards lab: replay recorded runs of the production graph, one checkpoint at a time.

``traces/guards.json`` holds every node the real LangGraph graph visited for each
scenario in :mod:`iw_rag.learn.guards_scenarios` (see
``scripts/learn_record_guards_trace.py``).  This module turns those raw node updates into
steps a reader can follow: what each checkpoint decided, why, and how trustworthy the
signal behind it is (a recorded classifier score, a scripted model step, or plain
production logic).  It also folds the updates into the running state, so the lab can show
the retry budget being spent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from iw_rag.learn.guards_scenarios import SCENARIOS, Scenario, get_scenario

TRACE_PATH = Path(__file__).parent / "traces" / "guards.json"

Source = Literal["recorded", "scripted", "logic"]
Outcome = Literal["pass", "stop", "retry", "info"]

SOURCE_LABELS: dict[Source, str] = {
    "recorded": "Recorded from the live classifier",
    "scripted": "Scripted language-model step; production code decides",
    "logic": "Production code, no model",
}

_TITLES = {
    "input_guard": "Input guard",
    "reject": "Reject",
    "router": "Router",
    "retrieve": "Search",
    "sanitize": "Sanitise the documents",
    "grade_documents": "Grade the evidence",
    "consume_retry": "Spend a retry",
    "rewrite_query": "Rewrite the question",
    "generate": "Generate the draft",
    "grade_generation": "Self-check the draft",
    "output_guard": "Output guard",
    "done": "Answer",
    "fail_closed": "Fail closed",
    "audit": "Audit",
}

_SOURCES: dict[str, Source] = {
    "input_guard": "recorded",
    "sanitize": "recorded",
    "router": "scripted",
    "retrieve": "scripted",
    "grade_documents": "scripted",
    "rewrite_query": "scripted",
    "generate": "scripted",
    "grade_generation": "scripted",
}


@dataclass(frozen=True, slots=True)
class LabStep:
    """One checkpoint the graph visited, explained."""

    index: int
    node: str
    title: str
    source: Source
    outcome: Outcome
    headline: str
    detail: str
    items: tuple[str, ...] = ()
    marked_sample: str | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    """The card shown for a scenario before it is played."""

    scenario_id: str
    title: str
    question: str
    story: str
    disposition: str
    decided_by: str


@dataclass(frozen=True, slots=True)
class GuardsLabResult:
    """Everything the guards lab shows for one scenario."""

    scenario: ScenarioSummary
    steps: tuple[LabStep, ...]
    final: dict[str, Any]
    max_retries: int
    threshold: float
    meta: dict[str, Any]


@lru_cache(maxsize=1)
def load_trace() -> dict[str, Any]:
    """Load the recorded trace (cached); raises ``FileNotFoundError`` if not recorded."""
    data: dict[str, Any] = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    return data


def _recorded(scenario_id: str) -> dict[str, Any]:
    return next(s for s in load_trace()["scenarios"] if s["id"] == scenario_id)


def _decided_by(steps: list[dict[str, Any]]) -> str:
    nodes = [s["node"] for s in steps]
    if "reject" in nodes:
        return "Input guard"
    if "fail_closed" in nodes:
        return "Fail closed"
    return "Output guard"


def summaries() -> tuple[ScenarioSummary, ...]:
    """Return the card for every scenario, with the outcome taken from the recording."""
    out = []
    for sc in SCENARIOS:
        rec = _recorded(sc.scenario_id)
        out.append(
            ScenarioSummary(
                scenario_id=sc.scenario_id,
                title=sc.title,
                question=sc.question,
                story=sc.story,
                disposition=str(rec["final"].get("disposition", "")),
                decided_by=_decided_by(rec["steps"]),
            )
        )
    return tuple(out)


def _chunk_texts(scenario: Scenario) -> dict[str, str]:
    return {c.chunk_id: c.text for retrieval in scenario.retrievals for c in retrieval}


def _score(x: float) -> str:
    return f"{x:.3f}" if x >= 0.001 else "under 0.001"


def _describe(  # noqa: PLR0911, PLR0912 - one branch per graph node reads best flat
    node: str,
    update: dict[str, Any],
    run: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Outcome, str, str, tuple[str, ...]]:
    """Return ``(outcome, headline, detail, items)`` for one visited node."""
    threshold = ctx["threshold"]
    retries_now = run["retries"]
    max_retries = ctx["max_retries"]
    verdicts = update.get("verdicts") or [{}]
    v = verdicts[0]
    if node == "input_guard":
        score = float(v.get("score") or 0.0)
        blocked = update.get("input_decision") != "allow"
        return (
            "stop" if blocked else "pass",
            f"Prompt Guard 2 scores the question {_score(score)}. The limit is {threshold}: "
            + ("blocked." if blocked else "allowed."),
            "The score is the classifier's probability that the question is a jailbreak or an "
            "injection. Granite Guardian also screens the question in the live system; its "
            "container was off when this was recorded, so only Prompt Guard 2 spoke here.",
            (),
        )
    if node == "reject":
        return (
            "stop",
            "Request rejected. Nothing was searched and no model wrote a word.",
            f"The user sees: “{update.get('refusal')}”. Any verdict other than an explicit "
            "'allow' routes here, including a guard that is down.",
            (),
        )
    if node == "router":
        return (
            "info",
            f"Route chosen: {update.get('route')} ({update.get('query_type')} question).",
            "A scripted router decides the question needs the documents.",
            (),
        )
    if node == "retrieve":
        ids = ", ".join(update.get("chunks", [])) or "nothing"
        return (
            "info",
            f"Search returned {len(update.get('chunks', []))} chunk(s): {ids}.",
            "A scripted search stands in for the spine here: the previous labs show the real "
            "one. The documents now go through the guards before any model reads them.",
            (),
        )
    if node == "sanitize":
        scanned = run["last_retrieved"]
        kept = set(update.get("chunks", []))
        texts, scores = ctx["texts"], ctx["scores"]
        items = tuple(
            f"{cid} · injection score {_score(scores[texts[cid]])} → "
            + ("kept" if cid in kept else "DROPPED")
            for cid in scanned
        )
        dropped = update.get("dropped_ids", [])
        detail = (
            "Every retrieved chunk is treated as untrusted data: Prompt Guard 2 scores it, "
            "poisoned chunks are dropped, and the survivors are datamarked (a marker after "
            "every word, shown here as ◆) so the model can tell data from instructions."
        )
        if dropped:
            detail += (
                " A chunk over the limit is dropped even when it is innocent: the guard "
                "fails closed, so a false positive costs a document, never a leak."
            )
        return (
            "stop" if dropped else "pass",
            f"Kept {len(kept)} of {len(scanned)} chunk(s); dropped {len(dropped)}.",
            detail,
            items,
        )
    if node == "grade_documents":
        verdict = update.get("doc_verdict")
        kept = update.get("graded", [])
        if verdict == "incorrect" and retries_now < max_retries:
            outcome: Outcome = "retry"
            tail = "Nothing relevant, and a retry is left: the loop will rewrite the question."
        elif verdict == "incorrect":
            outcome = "stop"
            tail = (
                "Nothing relevant, and the retries are spent: the draft step will have "
                "nothing to write from."
            )
        else:
            outcome = "pass"
            tail = "Enough relevant evidence to go on; only the relevant chunks continue."
        return (
            outcome,
            f"Evidence graded {verdict}: {v.get('reason', '')}.",
            tail,
            tuple(f"kept {cid}" for cid in kept),
        )
    if node == "consume_retry":
        return (
            "retry",
            f"Retry {update.get('retries')} of {max_retries} spent.",
            "One counter, shared by weak evidence and weak drafts, so a question can never get "
            f"more than {max_retries} correction passes by alternating failures.",
            (),
        )
    if node == "rewrite_query":
        return (
            "retry",
            f"The question becomes: “{update.get('question')}”.",
            "The rewrite starts from the user's original wording, so repeated rewrites cannot "
            "drift away from what was asked.",
            (),
        )
    if node == "generate":
        n = len(update.get("citations", []))
        return (
            "info" if n else "stop",
            f"Draft: “{update.get('draft')}”",
            f"{n} citation(s) resolved to real chunks."
            if n
            else "No citation resolved. A citation is required, so this can never count as "
            "grounded, whatever the text says.",
            tuple(f"[{c['marker']}] → {c['chunk_id']}" for c in update.get("citations", [])),
        )
    if node == "grade_generation":
        verdict = update.get("gen_verdict")
        if verdict == "grounded":
            outcome = "pass"
            tail = "The draft is supported by the evidence and answers the question."
        elif retries_now < max_retries:
            outcome = "retry"
            tail = "Not grounded, and a retry is left: back to a new search."
        else:
            outcome = "stop"
            tail = "Not grounded, and the retries are spent: the system must refuse."
        return (outcome, f"Self-check: {verdict} ({v.get('reason', '')}).", tail, ())
    if node == "output_guard":
        ok = update.get("output_guard_decision") == "allow"
        return (
            "pass" if ok else "stop",
            "Citations valid and answer grounded: allowed." if ok else "Blocked.",
            "Deterministic citation validation: an empty citation list, or a citation outside "
            "the graded set, is a hard block no model can override. Groundedness is the "
            "self-check's verdict, read once.",
            (),
        )
    if node == "done":
        return (
            "pass",
            "Answered.",
            "Only this node can turn a draft into an answer, and only when the verdict is "
            "'grounded'.",
            (),
        )
    if node == "fail_closed":
        return (
            "stop",
            "Refused: " + str(update.get("refusal")),
            "The honest refusal, never a guess. " + str(v.get("reason", "")),
            (),
        )
    # audit
    return (
        "info",
        "Outcome handed to the audit log.",
        "In the live system every outcome, answered, refused or rejected, is written to an "
        "append-only audit log, and an answer that cannot be audited is withheld. This "
        "recording runs without a database, so nothing was written.",
        (),
    )


def run_guards_lab(scenario: Scenario) -> GuardsLabResult:
    """Replay the recorded run for ``scenario`` as explained steps with running state."""
    trace = load_trace()
    rec = _recorded(scenario.scenario_id)
    ctx = {
        "threshold": trace["pg2_threshold"],
        "max_retries": trace["max_retries"],
        "scores": trace["pg2_scores"],
        "texts": _chunk_texts(scenario),
    }
    run: dict[str, Any] = {
        "retries": 0,
        "last_retrieved": [],
        "kept": [],
        "dropped": [],
        "doc_verdict": None,
        "gen_verdict": None,
        "question": scenario.question,
        "disposition": None,
    }
    steps: list[LabStep] = []
    for i, step in enumerate(rec["steps"]):
        node, update = step["node"], step["update"]
        if node == "retrieve":
            run["last_retrieved"] = list(update.get("chunks", []))
            run["dropped"] = []
        # Describe against the state *before* this node's own retries change, except
        # for consume_retry, which reports the counter it just set.
        if "retries" in update:
            run["retries"] = update["retries"]
        outcome, headline, detail, items = _describe(node, update, run, ctx)
        if node == "sanitize":
            run["kept"] = list(update.get("chunks", []))
            run["dropped"] = list(update.get("dropped_ids", []))
        if node == "grade_documents":
            run["kept"] = list(update.get("graded", []))
            run["doc_verdict"] = update.get("doc_verdict")
        for key in ("gen_verdict", "question", "disposition"):
            if key in update:
                run[key] = update[key]
        steps.append(
            LabStep(
                index=i,
                node=node,
                title=_TITLES.get(node, node),
                source=_SOURCES.get(node, "logic"),
                outcome=outcome,
                headline=headline,
                detail=detail,
                items=items,
                marked_sample=update.get("first_chunk_marked") if node == "sanitize" else None,
                state={
                    "retries": run["retries"],
                    "question": run["question"],
                    "kept": list(run["kept"]),
                    "dropped": list(run["dropped"]),
                    "doc_verdict": run["doc_verdict"],
                    "gen_verdict": run["gen_verdict"],
                    "disposition": run["disposition"],
                },
            )
        )
    card = next(s for s in summaries() if s.scenario_id == scenario.scenario_id)
    return GuardsLabResult(
        scenario=card,
        steps=tuple(steps),
        final=dict(rec["final"]),
        max_retries=trace["max_retries"],
        threshold=trace["pg2_threshold"],
        meta={
            "recorded": trace["recorded"],
            "provenance": trace["provenance"],
            "source_labels": SOURCE_LABELS,
        },
    )


__all__ = [
    "SOURCE_LABELS",
    "GuardsLabResult",
    "LabStep",
    "ScenarioSummary",
    "get_scenario",
    "load_trace",
    "run_guards_lab",
    "summaries",
]
