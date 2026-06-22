"""Agent prompts and JSON schemas for the Adaptive-RAG agentic loop.

Contains:
- Router classifier prompt + schema (Task 2.1).
- ISREL document-relevance grader prompt + schema (Task 2.3 / CRAG).
- Rewrite-query prompt (Task 2.3).
- ISSUP (groundedness) + ISUSE (usefulness) grader prompts + schemas (Task 2.4).

Kept as plain strings + plain dicts so every prompt is auditable, diffable,
and version-controlled without any rendering overhead.  This module is distinct
from ``generation/prompts.py`` which owns the citation-required answer prompt.

All graders run on the generator endpoint (Qwen3-14B via vLLM ``guided_json``
constrained decoding) — a separate model is a config swap, never a code change
(CONVENTIONS.md §4: ``grader_model="generator"`` default).

References:
- task-2.1-router-node §1 (router schema + system prompt)
- task-2.3-grade-documents-and-rewrite §1 (ISREL grader + rewrite prompt)
- task-2.4-generate-and-grade-generation §2 (ISSUP + ISUSE grader prompts)
- lld-agentic-loop §3.1 (router), §3.2 (CRAG/ISREL), §3.3 (rewrite), §3.5 (Self-RAG)
- CONVENTIONS.md §4 (grader/router on generator via constrained decode)
- R-00105 F2/F3/F5/F8 (externalized graders, classification, batching)
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# ROUTER_SCHEMA — the JSON schema for the constrained decode
# ---------------------------------------------------------------------------

ROUTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "enum": ["skip", "retrieve", "graph", "long_context", "visual"],
        },
        "query_type": {
            "enum": ["factual", "multihop", "wholedoc", "relational"],
        },
    },
    "required": ["route", "query_type"],
}
"""JSON schema for the router structured-output call.

Fields:
    route: Branch selection — ``skip`` | ``retrieve`` | ``graph`` |
        ``long_context`` | ``visual``.
    query_type: Complexity label — ``factual`` | ``multihop`` | ``wholedoc`` |
        ``relational``.

``route`` is listed first **on purpose**: ``guided_json`` decodes properties in
schema order, so the model commits to the route — the field the prompt defines in
detail and the only one the loop branches on — before the secondary complexity
label.  The previous order (query_type first) let the model lock in an
under-specified ``relational`` type and then drag every route to ``graph``
(live finding 2026-06-14; the graph collapse to the spine masked it in answers).

No free-text ``reason`` field: under ``guided_json`` an unbounded string can
consume the whole ``ROUTER_MAX_TOKENS`` budget and truncate the object before
its closing brace, which forced the safe-but-wrong vector fallback.  The router
only needs the two routing labels; rationale lives in tracing spans instead.
"""

# ---------------------------------------------------------------------------
# ROUTER_SYSTEM — the classifier system prompt
# ---------------------------------------------------------------------------

ROUTER_SYSTEM: str = (
    "Classify the user question for a retrieval router. Choose the route first, "
    "then the query_type.\n"
    "ROUTE — how to fetch evidence:\n"
    "- retrieve: a normal question answered from the document corpus. THIS IS THE "
    "DEFAULT — use it for almost every question, including single-fact lookups, "
    "definitions grounded in the corpus, amounts/dates, policies, and support/how-to "
    "questions.\n"
    "- skip: trivial/parametric questions answerable with NO corpus (greetings, "
    "general arithmetic, common-knowledge definitions).\n"
    "- graph: ONLY questions that explicitly require connecting MULTIPLE named "
    "entities or multi-hop reasoning across relationships (e.g. 'How is X related to "
    "Y through Z?'). A single-fact lookup is NEVER graph.\n"
    "- long_context: questions about a whole single document or a small static "
    "corpus (e.g. 'summarise the entire contract').\n"
    "- visual: questions explicitly about images, charts, figures, diagrams, or "
    "visual content.\n"
    "When in doubt, choose retrieve.\n"
    "QUERY_TYPE — complexity label (secondary): factual = one fact; multihop = "
    "several facts combined; wholedoc = whole-document/global; relational = explicit "
    "entity-relationship reasoning. Most questions are factual + retrieve."
)
"""System prompt for the Adaptive-RAG router classifier.

Bias toward ``retrieve`` is deliberate: a wrong classification that picks
``retrieve`` still goes through the corrective loop and grades its own work,
while a wrong ``skip`` bypasses retrieval entirely.  The prompt defines BOTH the
route and the query_type labels — the original prompt described only the routes,
so the model defaulted query_type to ``relational`` and dragged every route to
``graph`` (live finding 2026-06-14).  ``graph`` is explicitly reserved for
multi-entity / multi-hop questions; a single-fact lookup must stay ``retrieve``.
"""

# ---------------------------------------------------------------------------
# ANSWER_DIRECT_SYSTEM — the direct-answer system prompt (skip branch)
# ---------------------------------------------------------------------------

ANSWER_DIRECT_SYSTEM: str = (
    "Answer the question briefly using only your parametric knowledge. "
    "If you are not confident, say so explicitly rather than speculating. "
    "Do not fabricate facts — it is better to say 'I don't know' than to guess."
)
"""System prompt for the ``answer_direct`` node (skip branch).

Used when the router classifies the question as trivial/parametric — no
retrieval corpus is consulted.  The answer still flows to ``grade_generation``
(Task 2.4) to check quality; ISSUP groundedness is bypassed when ``graded``
is empty.
"""

# ---------------------------------------------------------------------------
# GRADER_PARAMS — tight generation params for the router (not a full generation)
# ---------------------------------------------------------------------------

#: Served model name placeholder — resolved at runtime from the ``grader_model``
#: settings key.  The router caller must supply the real model name.
GRADER_MODEL_PLACEHOLDER = "generator"

#: Low temperature (near-deterministic) for the router; the classifier should
#: always pick the same label for the same question.
ROUTER_TEMPERATURE: float = 0.0

#: Small max_tokens — the router emits a short JSON label, not prose.
ROUTER_MAX_TOKENS: int = 64


# ---------------------------------------------------------------------------
# ISREL grader — per-candidate relevance (CRAG/grade_documents, Task 2.3)
# ---------------------------------------------------------------------------

DOC_GRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relevant": {"type": "boolean"},
                },
                "required": ["relevant"],
            },
        },
    },
    "required": ["grades"],
}
"""JSON schema for the batched ISREL document-relevance grader.

The model receives all candidates in a single user message and returns a
``grades`` array where element ``i`` corresponds to candidate ``i``.  Batching
is mandatory so vLLM continuous-batching amortises all per-chunk judgements
into one request rather than one round-trip per chunk (R-00105 F8).

No free-text ``reason`` field (I-CTXOVERFLOW perf): under ``guided_json`` a
per-candidate rationale string is the bulk of the decoded output — for 8
candidates it filled the ``DOC_GRADE_MAX_TOKENS`` budget with prose that
``_parse_grade_response`` then **discards** (it reads only ``relevant``),
costing ~25-35s of slow guided decode per grade. The grader only needs the
boolean labels; rationale lives in tracing spans.  (Mirrors the same fix on the
router schema, where an unbounded ``reason`` truncated the constrained decode.)

Fields:
    grades: Array of per-candidate grade objects.
    grades[i].relevant: Whether candidate ``i`` is relevant to the question.
"""

DOC_GRADE_SYSTEM: str = (
    "You are a relevance judge. For each numbered document in the user's list, "
    "decide whether it contains information that helps answer the question. "
    "Relevant = the document directly supports answering the question. "
    "Return a JSON object with a 'grades' array, one element per document, "
    "each with a single boolean 'relevant' field. "
    "Do NOT include any other text or fields — output only the JSON object."
)
"""System prompt for the batched ISREL document-relevance grader.

Instructs the model to grade each candidate in the user's list and return a
parallel JSON array.  The grade is binary (relevant/irrelevant) so constrained
decoding can validate the output cheaply.

Grader strategy (CONVENTIONS §4, lld-agentic-loop §3.2):
- All candidates are submitted in one request (batch) so vLLM folds them.
- Constrained decode (``guided_json``) enforces the schema.
- Low temperature, small max_tokens — this is a label, not prose generation.
"""

#: Max tokens for the ISREL grader.  Small — with no per-candidate reason field
#: the output is just N booleans, so a tight ceiling both bounds latency and
#: prevents runaway guided decode.  ~10 tokens/candidate is ample for the array.
DOC_GRADE_MAX_TOKENS: int = 160

#: Temperature for the ISREL grader — near-deterministic classification.
DOC_GRADE_TEMPERATURE: float = 0.0


# ---------------------------------------------------------------------------
# Rewrite-query prompt (Task 2.3 / R-00105 F4)
# ---------------------------------------------------------------------------

REWRITE_SYSTEM: str = (
    "You are a search query rewriter. "
    "The user's original question could not be answered because the retrieved "
    "documents were not relevant. "
    "Reformulate the original question into a better search query that may find "
    "more relevant documents. "
    "Rules: (1) stay true to the original question's intent — do NOT change the topic; "
    "(2) use different keywords or phrasing; "
    "(3) output ONLY the rewritten query, no explanation, no quotation marks, no prefix."
)
"""System prompt for the query-rewrite node (single-hop reformulation).

Single-hop reformulation is the default inner-loop correction (R-00105 F4).
The rewrite is anchored to ``original_question`` so successive passes do not
drift off the user's intent.  Multi-hop decomposition is a Phase-4 router mode.

The node writes only ``state["question"]`` — the original_question remains
immutable throughout the loop.
"""

#: Max tokens for the rewrite.  One query line, not an essay.
REWRITE_MAX_TOKENS: int = 128

#: Temperature for the rewrite node — slightly warmer than classifiers to allow
#: lexical variation, but low enough to stay on-topic.
REWRITE_TEMPERATURE: float = 0.2


# ---------------------------------------------------------------------------
# ISSUP — groundedness/support grader (Task 2.4 / Self-RAG externalized)
# ---------------------------------------------------------------------------

ISSUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
    },
    "required": ["supported"],
}
"""JSON schema for the ISSUP groundedness/support grader.

ISSUP asks: "Is every claim in the answer supported by the provided evidence?"
The grader is separate from the generator (R-00105 F2) — it is NOT the generator's
own self-reflection tokens.  Phase 2 uses Qwen3-14B constrained decoding; Phase 3
swaps this for Granite Guardian groundedness as the output guard (lld-agentic-loop
§3.5 [ASSUMPTION]).

No free-text ``reason`` field (I-CTXOVERFLOW perf): under ``guided_json`` an
unbounded rationale string fills the tight ``ISSUP_MAX_TOKENS`` budget and
**truncates the JSON object** before/around the boolean.  ``_parse_bool`` then
hits a ``JSONDecodeError`` and fails closed to ``supported=False``, mislabelling a
correct, cited answer as ``"unsupported"`` (badged "not grounded" in assist mode).
The grader only needs the boolean; rationale lives in tracing spans.  (Mirrors the
same fix on ``ROUTER_SCHEMA`` / ``DOC_GRADE_SCHEMA``.)

Fields:
    supported: True only if the evidence grounds every claim in the answer.
"""

ISSUP_SYSTEM: str = (
    "Is every claim in the answer supported by the provided evidence? "
    "Answer true only if the evidence actually grounds the answer. "
    "If any claim is hallucinated or not present in the evidence, answer false. "
    "Output only a JSON object with a single boolean 'supported' field. "
    "Do NOT include any other text or fields — output only the JSON object."
)
"""System prompt for the ISSUP groundedness grader.

Receives: ``EVIDENCE: <concatenated graded chunks>\\n\\nANSWER: <draft>`` as the user message.
Returns: ``{"supported": bool}`` (constrained decode).

This grader is deliberately separate from the generator (off-the-shelf Qwen3-14B
does not self-grade — R-00105 F2).  It is batched alongside ISUSE in a single
vLLM request set so continuous-batching amortises both (R-00105 F5/F8).
"""

#: Generation params max_tokens for ISSUP — short boolean label, not prose.
ISSUP_MAX_TOKENS: int = 64

#: Near-deterministic temperature for ISSUP classification.
ISSUP_TEMPERATURE: float = 0.0


# ---------------------------------------------------------------------------
# ISUSE — answer-quality/usefulness grader (Task 2.4 / Self-RAG externalized)
# ---------------------------------------------------------------------------

ISUSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers": {"type": "boolean"},
    },
    "required": ["answers"],
}
"""JSON schema for the ISUSE answer-quality/usefulness grader.

ISUSE asks: "Does the answer actually address the user's original question?"
Unlike ISSUP, ISUSE ignores whether claims are grounded — it judges usefulness
and relevance only.  This distinction matters when an answer is technically
grounded but off-topic (which ``_aggregate_gen_verdict`` maps to ``not_answering``).

No free-text ``reason`` field (I-CTXOVERFLOW perf): under ``guided_json`` an
unbounded rationale truncates the constrained decode within ``ISUSE_MAX_TOKENS``,
so ``_parse_bool`` fails closed to ``answers=False``.  The grader only needs the
boolean; rationale lives in tracing spans.  (Mirrors ``ISSUP_SCHEMA``.)

Fields:
    answers: True if the answer is useful and directly addresses the question.
"""

ISUSE_SYSTEM: str = (
    "Does the answer actually address the user's original question? "
    "Ignore whether the answer is grounded — judge usefulness and relevance only. "
    "Answer true if the answer is helpful and directly addresses what was asked. "
    "Output only a JSON object with a single boolean 'answers' field. "
    "Do NOT include any other text or fields — output only the JSON object."
)
"""System prompt for the ISUSE answer-quality/usefulness grader.

Receives: ``QUESTION: <original_question>\\n\\nANSWER: <draft>`` as the user message.
Returns: ``{"answers": bool}`` (constrained decode).

Batched alongside ISSUP so both graders run in a single vLLM request set
(R-00105 F5/F8; lld-agentic-loop §3.5).
"""

#: Generation params max_tokens for ISUSE — short boolean label, not prose.
ISUSE_MAX_TOKENS: int = 64

#: Near-deterministic temperature for ISUSE classification.
ISUSE_TEMPERATURE: float = 0.0


# ---------------------------------------------------------------------------
# GEN_GRADE_SCHEMA — combined schema for reference / doc purposes
# ---------------------------------------------------------------------------

GEN_GRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "answers": {"type": "boolean"},
    },
    "required": ["supported", "answers"],
}
"""Combined JSON schema documenting the generation-grade semantic contract.

In practice ISSUP and ISUSE are run as SEPARATE structured calls (batched
together) — see ``ISSUP_SCHEMA`` / ``ISUSE_SCHEMA``.  This combined schema
exists for documentation, audit, and potential future single-call optimisation.

Aggregation table (_aggregate_gen_verdict):
    (supported=True,  answers=True)  → "grounded"
    (supported=False, answers=*)     → "unsupported"
    (supported=True,  answers=False) → "not_answering"
"""
