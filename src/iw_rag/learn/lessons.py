# ruff: noqa: E501
"""Lesson content for the Learn section: how IW-RAG itself works.

Pure, typed data (stdlib only) so the text can be reviewed and translated
independently of the HTML that renders it.  Each lesson follows one real step of
the system, in order, and answers four questions: what it does, how it does it,
*why* it was built that way, and what that choice costs.  Numbers quoted under
``evidence`` come from the project's own measured experiments (``guide/08``); keep
them in sync with ``eval/reports`` when those are re-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Phase = Literal["prepare", "answer", "prove"]


@dataclass(frozen=True, slots=True)
class Lesson:
    """One step of IW-RAG explained for someone who wants to understand this system.

    Attributes:
        slug: URL-safe id, used in ``/console/learn/{slug}``.
        title: Step name.
        short: Compact label for the pipeline map.
        phase: ``prepare`` (ingestion, once per document), ``answer`` (per question)
            or ``prove`` (modes, measurement, hardware).
        question: The plain-language problem this step solves.
        what: What the step does, in two or three sentences.
        how: The concrete mechanics, one short point per item.
        why: Why IW-RAG does it this way (the design decision and its reasoning).
        gains: What the choice buys.
        losses: What it costs, or where it is limited.
        evidence: Measured result backing the choice, when there is one.
        try_it: ``(label, href)`` console pages where the step can be seen working.
        code: Source paths to read.
    """

    slug: str
    title: str
    short: str
    phase: Phase
    question: str
    what: str
    how: tuple[str, ...]
    why: str
    gains: tuple[str, ...]
    losses: tuple[str, ...]
    evidence: str | None = None
    try_it: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    code: tuple[str, ...] = field(default_factory=tuple)


LESSONS: tuple[Lesson, ...] = (
    # ------------------------------------------------------------------ prepare
    Lesson(
        slug="parsing",
        title="Parsing",
        short="Parse",
        phase="prepare",
        question="How does a PDF or a markdown file become something the system can use?",
        what=(
            "Ingestion reads each file into an ordered list of blocks (headings, text, "
            "tables, figure captions) and remembers which section each block sits in. "
            "Nothing is searched yet; this step only makes the document legible."
        ),
        how=(
            "A router picks the parser: an explicit override first, then the heavy parser for "
            "documents flagged as hard, then a plain-text parser for markdown and text, and "
            "otherwise Docling.",
            "Headings update a running section path, so every later chunk knows where it came from.",
            "Tables stay tables, and figure captions are kept as their own blocks.",
        ),
        why=(
            "Docling is MIT-licensed, layout-aware and runs air-gapped, which fits a system "
            "that must never call a hosted service. A dependency-free text parser handles "
            "born-digital markdown, so the demo corpus ingests without the heavy toolchain."
        ),
        gains=(
            "Section paths and tables survive, so chunks and citations keep their context.",
            "Everything runs on the box; no document is sent to an external parser.",
        ),
        losses=(
            "The escalation parser for the hardest documents (dense tables, formulas, scans) "
            "is a stub today and raises an error if selected.",
            "Parsing is the slow, offline part of the system: it never runs on the answer path.",
        ),
        try_it=(("Upload a document", "/console/upload"),),
        code=("src/iw_rag/ingestion/parse.py", "guide/03-ingestion.md"),
    ),
    Lesson(
        slug="chunking",
        title="Chunking",
        short="Chunk",
        phase="prepare",
        question="How is a document cut into pieces, and why in two sizes?",
        what=(
            "Each document becomes a tree: small leaf chunks that are searched, and larger "
            "parent chunks (one per section) that are handed to the model. Search finds the "
            "leaf; the answer is written from its parent."
        ),
        how=(
            "Leaves target 384 tokens. Documents uploaded in the console use leaves of about "
            "1,600 characters instead.",
            "One parent is built per section; its text is its leaves joined together.",
            "Heading text counts as context for the token stream but does not become a leaf.",
            "The tokenizer is injected and must match the embedding model, so chunk boundaries "
            "line up exactly with what gets embedded.",
        ),
        why=(
            "The design review found that parent-document retrieval is the highest-leverage "
            "chunking choice, while semantic chunking repeatedly failed to justify its cost. "
            "So the effort goes into the hierarchy rather than into clever boundaries."
        ),
        gains=(
            "Small leaves rank precisely; large parents give the generator real context.",
            "The citation can still point at the exact leaf that matched.",
        ),
        losses=(
            "A section with no headings becomes one huge parent. Parents larger than 4,000 "
            "characters are not expanded, and the leaf is used as is.",
            "Fixed-size leaves can still cut a thought in half.",
        ),
        try_it=(("Upload a document", "/console/upload"),),
        code=("src/iw_rag/ingestion/chunk.py", "guide/03-ingestion.md"),
    ),
    Lesson(
        slug="embedding",
        title="Embedding",
        short="Embed",
        phase="prepare",
        question="How do chunks become searchable by meaning?",
        what=(
            "Every leaf is turned into a 1024-number vector by BGE-M3 so that passages with "
            "similar meaning sit close together. IW-RAG does this with late chunking: the "
            "whole document is read once, and each leaf's vector is taken from that "
            "full-context pass."
        ),
        how=(
            "BGE-M3 reads up to 8,192 tokens at once, so every token sees the whole document.",
            "Each leaf's vector is the mean of its tokens' vectors, so it carries document context.",
            "Longer documents are split into overlapping windows (512 tokens of overlap, never less "
            "than one leaf), and each leaf is pooled from the one window that fully contains it.",
            "A parent's vector is the normalised mean of its leaves' vectors, so parents are searchable too.",
        ),
        why=(
            "The alternative, Contextual Retrieval, asks an LLM to write context for every chunk. "
            "On one GPU that competes with the answer model. Late chunking gets most of the "
            "benefit from one embedding pass and zero LLM calls. BGE-M3 is MIT-licensed, "
            "multilingual, and one model serves both the document and the query side."
        ),
        gains=(
            "Leaves keep document-level context without any extra model calls.",
            "Ingestion never competes with the generator for the GPU.",
        ),
        losses=(
            "The dimension is fixed at 1024, so changing the embedder means re-embedding everything.",
            "BGE-M3 is below some newer models on pure dense quality.",
            "Late chunking needs token-level hidden states, so ingestion runs the model "
            "directly instead of through the served endpoint.",
        ),
        code=("src/iw_rag/ingestion/embed.py", "guide/03-ingestion.md"),
    ),
    Lesson(
        slug="storage",
        title="Access tags and storage",
        short="Tag + store",
        phase="prepare",
        question="Who may read each chunk, and where does it all live?",
        what=(
            "Every chunk is stamped with the groups allowed to read it, then written, with its "
            "vector and text, into a single Postgres database in one transaction."
        ),
        how=(
            "Access tags come only from the source system's metadata: the corpus manifest or "
            "the uploader. Neither the document text nor any model can influence them.",
            "A document with no tags is invisible to everyone (it never becomes public by accident).",
            "Vectors (pgvector), the BM25 index (ParadeDB), metadata, tags and the graph all "
            "live in the same Postgres.",
            "Ingestion is idempotent: an unchanged (source, checksum) pair is skipped, and one "
            "failing document never aborts the run or leaves partial rows.",
            "Console uploads are 'specific' (owner only) or 'common' (public).",
        ),
        why=(
            "One transactional store means a retrieval and its access check are a single "
            "decision with a single audit trail. That compliance story is the point of the design."
        ),
        gains=(
            "A leak through the model or the text is impossible because tags are set before any model runs.",
            "Access decisions are auditable in one place.",
        ),
        losses=(
            "Single Postgres instance: no scale-out, no multi-tenancy. This is a showcase, not a cluster.",
            "Getting a document's tags wrong at ingestion is a data-governance problem the system cannot detect.",
        ),
        try_it=(("Manage documents", "/console/upload"),),
        code=("src/iw_rag/ingestion/enrich.py", "src/iw_rag/ingestion/pipeline.py"),
    ),
    # ------------------------------------------------------------------- answer
    Lesson(
        slug="input-guard",
        title="Input guard",
        short="Input guard",
        phase="answer",
        question="What happens to a question before the system acts on it?",
        what=(
            "The first node of the loop screens the user's message for prompt injection, "
            "jailbreaks and harmful requests. Anything not explicitly allowed is rejected."
        ),
        how=(
            "Two local models read the user's turn: Prompt Guard 2 (86M) and Granite Guardian 3.3 (8B).",
            "The decision must be exactly 'allow' to continue. A block, a missing value or an "
            "unexpected value all route to reject.",
            "A rejected request still produces an audit record.",
        ),
        why=(
            "Guards are layered rather than a single gate, and every one of them fails closed. "
            "They run locally because 'no hosted API' includes the guards."
        ),
        gains=(
            "Obvious attacks never reach retrieval or the generator.",
            "Refusing on doubt is the safe default for a corpus of private documents.",
        ),
        losses=(
            "Guard models add latency and VRAM, so they are on in serving mode and off in benchmark mode.",
            "A guard that is down means every query is rejected, by design.",
        ),
        try_it=(
            ("Walk a jailbreak through the guards", "/console/learn/fail-closed"),
            ("Ask something in Chat", "/console/chat"),
        ),
        code=("src/iw_rag/guardrails/input_guard.py", "guide/06-guardrails.md"),
    ),
    Lesson(
        slug="router",
        title="Router",
        short="Router",
        phase="answer",
        question="Does this question even need the documents?",
        what=(
            "A small structured-output call classifies the question, then picks a route: skip "
            "retrieval, retrieve, or one of the optional modes (graph, long context, visual)."
        ),
        how=(
            "The classifier returns a query type (factual, multihop, wholedoc, relational) and a route.",
            "An unknown label defaults to 'retrieve'. 'Skip' is downgraded to 'retrieve' when disabled.",
            "Routes for the optional modes are recorded as intended but collapse to 'retrieve' "
            "unless that mode is switched on.",
            "Skipped answers still pass through the generation self-check.",
        ),
        why=(
            "This is Adaptive-RAG: deciding whether to retrieve is the cheapest latency win "
            "available. About 8% of queries (greetings, definitions, arithmetic) never touch the corpus."
        ),
        gains=("Trivial questions are answered fast, without searching.",),
        losses=(
            "A wrong route costs quality: sending a corpus question to 'skip' answers from memory. "
            "Route accuracy has its own evaluation for exactly that reason.",
            "It is one more model call on every question.",
        ),
        code=("src/iw_rag/agent/nodes.py", "guide/05-agentic-loop.md"),
    ),
    Lesson(
        slug="search",
        title="Hybrid search",
        short="Search",
        phase="answer",
        question="Which chunks might answer the question?",
        what=(
            "Two searches run under the same access filter, one by meaning and one by exact "
            "words, and their rankings are merged. The access filter is applied first, inside "
            "the search itself."
        ),
        how=(
            "Access pre-filter: the SQL predicate acl_tags && principal_groups is bound into both "
            "searches. The groups come from sign-in, never from the question or the model.",
            "Dense arm: the query is embedded and pgvector (HNSW, cosine) returns the 40 nearest chunks.",
            "Sparse arm: ParadeDB BM25 returns the 40 best keyword matches. It catches exact "
            "identifiers such as INV-2026-0931 that vectors blur.",
            "Reciprocal Rank Fusion (k = 60) merges the two lists by rank and keeps the best 80.",
        ),
        why=(
            "Pure vector search quietly fails on IDs, SKUs, error codes and rare terms, which "
            "a business corpus is full of. Fusing by rank means the two score scales never need "
            "to be normalised against each other. And the filter is a pre-filter because a "
            "post-filter can leak the existence of a document or starve the top-k."
        ),
        gains=(
            "Each search covers the other's blind spot.",
            "Raising the candidate budget can never leak a document the user may not see.",
        ),
        losses=(
            "Two searches cost more than one; they run one after the other in a single session.",
            "Only BM25 is wired as the keyword arm; learned-sparse is an untested experiment.",
        ),
        evidence=(
            "Ablation on a 108-document, 37-query corpus (recall@10 / nDCG@10 / MRR): "
            "dense only 0.691 / 0.646 / 0.687; hybrid 0.713 / 0.715 / 0.802. "
            "The jump is biggest in MRR: BM25 ranks exact identifiers far higher."
        ),
        try_it=(
            ("Run a benchmark", "/console/benchmark"),
            ("Ask something in Chat", "/console/chat"),
        ),
        code=(
            "src/iw_rag/retrieval/spine.py",
            "src/iw_rag/retrieval/fusion.py",
            "guide/04-retrieval-spine.md",
        ),
    ),
    Lesson(
        slug="rerank",
        title="Reranking and expansion",
        short="Rerank",
        phase="answer",
        question="Of 80 candidates, which 8 really answer the question?",
        what=(
            "A cross-encoder reads the question and each candidate together and re-scores them. "
            "The best 8 are then swapped for their larger parent sections before going on."
        ),
        how=(
            "bge-reranker-v2-m3 (568M parameters) scores each (question, chunk) pair jointly; the top 8 are kept.",
            "Each surviving leaf is replaced by its parent section's text (small-to-big), while "
            "the matched leaf is kept for citation.",
            "If the reranker is down the spine degrades to the fused order and flags it; it does "
            "not fail the query (configurable).",
            "Sections are searchable too: a section can rank on its own, and is then sent to the "
            "generator as it is, provided it fits the budget. A section over the budget is set "
            "aside before the reranker (its leaves stay candidates), and retrieval.section_hits "
            "can switch section matching off.",
            "Leaves of one section that would all carry the same text, and the section itself, "
            "are merged into one chunk before the cut to top-k, so the freed slots go to the next "
            "passages. The merged chunk keeps the matched leaf for the citation.",
        ),
        why=(
            "Search compares the question and a chunk independently; a cross-encoder reads them "
            "together and resolves fine distinctions search can only approximate. The design review "
            "called it the single highest-return precision stage."
        ),
        gains=(
            "Much better ordering where many near-duplicate chunks compete.",
            "The generator sees enough surrounding text, and the citation stays exact.",
        ),
        losses=(
            "Reading 80 candidates in full is the most expensive step of retrieval.",
            "It can only reorder what search found; a missed chunk stays missed.",
            "One extra lookup per search finds which candidates are sections.",
            "Expansion only helps while a section fits the budget: with the offline defaults a "
            "document with no headings is one giant section, so its leaves are sent unexpanded.",
        ),
        evidence=(
            "Adding the reranker lifts recall@10 from 0.713 to 0.779, nDCG@10 from 0.715 to "
            "0.788 and MRR from 0.802 to 0.853, mostly by separating 52 near-duplicate invoices. "
            "0.779 is just under the project's own 0.80 floor, and the floor was not lowered. "
            "Merging repeats (CR-00103) was measured on the same benchmark over 34 gold questions: "
            "of the 8 chunks returned, 4.2 distinct passages on average became 7.9, repeated texts "
            "fell from 130 to 3, and retrieval quality was unchanged (recall@10 0.779 before and "
            "after, nDCG@5 0.778 and 0.777, MRR 0.853 and 0.846)."
        ),
        try_it=(("Compare runs", "/console/compare"), ("Scoreboard", "/console/scoreboard")),
        code=("src/iw_rag/retrieval/rerank.py",),
    ),
    Lesson(
        slug="sanitize",
        title="Documents are data",
        short="Sanitize",
        phase="answer",
        question="What if a document tries to give the model instructions?",
        what=(
            "Everything retrieved is treated as untrusted data. Before the model sees it, each "
            "chunk is scanned, scrubbed and marked."
        ),
        how=(
            "Injection scan: Prompt Guard 2 scores each chunk; poisoned chunks are dropped.",
            "PII scrub: Presidio redacts personal data, scoped to the user's clearance.",
            "Datamarking: an invisible marker (U+2063) is woven through the text so the model "
            "can tell data from instructions. The marks are stripped again before the user sees a quote.",
            "The order is fixed (scan, then scrub, then mark). It runs on every retrieval pass, "
            "including retries.",
        ),
        why=(
            "The top RAG-specific threat (OWASP LLM01:2025) is indirect injection: the user types "
            "nothing malicious, but the corpus carries hidden instructions. The documents are the attack surface."
        ),
        gains=("A poisoned document is dropped before it can steer the answer.",),
        losses=(
            "The classifier is not perfect, and the guard fails closed: an innocent chunk that "
            "scores over the limit is dropped too. In the guards lab a harmless note about meal "
            "allowances is dropped this way.",
            "Datamarking inflates the token count, so the context has to be budgeted by tokens.",
            "A guard model scoring every chunk costs GPU time on every question.",
        ),
        try_it=(("Watch a poisoned document get dropped", "/console/learn/fail-closed"),),
        code=("src/iw_rag/guardrails/sanitize.py", "guide/06-guardrails.md"),
    ),
    Lesson(
        slug="grade-evidence",
        title="Grading the evidence",
        short="Grade evidence",
        phase="answer",
        question="Is what we found actually relevant?",
        what=(
            "Before writing anything, a grader model judges each retrieved chunk: relevant to "
            "the question or not. The overall verdict decides what happens next."
        ),
        how=(
            "Each chunk gets a yes/no relevance grade; the grades go to the model as one batch.",
            "'correct': all relevant, go and generate. 'ambiguous': some relevant, generate with the relevant ones.",
            "'incorrect': none relevant, so use one retry (rewrite the question and search again) if any are left.",
            "Only the relevant chunks are passed on to the generator.",
        ),
        why=(
            "This is the corrective-RAG idea: notice weak retrieval before spending a generation "
            "on it. The web-search fallback of the original method is out of scope, because "
            "nothing may leave the box. Rewriting and re-searching is the local substitute."
        ),
        gains=("Weak evidence is caught early, and the generator sees cleaner context.",),
        losses=(
            "Another model call per question, with grading judgements that can themselves be wrong.",
            "'Ambiguous' goes ahead anyway, trading some risk for latency.",
        ),
        code=("src/iw_rag/agent/nodes.py",),
    ),
    Lesson(
        slug="generate",
        title="Generating with citations",
        short="Generate",
        phase="answer",
        question="How is the answer written, and how are citations made real?",
        what=(
            "Qwen3-14B (AWQ-INT4) writes the answer from the graded chunks only, and every claim "
            "must carry a numbered marker. The assembler checks that every marker points at a "
            "real chunk."
        ),
        how=(
            "Markers like [1], [1,2] and [1][2] are parsed and mapped to real chunks.",
            "A marker with no matching chunk (a hallucinated [7] over five chunks) is dropped, not cited.",
            "Each citation's quote is the exact matched leaf, even though the model saw the parent.",
            "Hidden reasoning (the <think> block) and datamark characters are stripped before parsing.",
            "An answer with no resolved citation is marked ungrounded.",
        ),
        why=(
            "There is exactly one answer path, so the rules cannot drift between code paths. "
            "The generator is off the shelf and does not grade itself: all reflection is done by "
            "separate grader calls, which keeps the checking independent from the writing."
        ),
        gains=(
            "Every shown citation resolves to a real passage you can open.",
            "A fabricated source cannot survive the assembler.",
        ),
        losses=(
            "This is structural validation: it proves the source exists, not that it supports the claim. That is the next step's job.",
            "A 14B model on one GPU bounds the answer quality.",
        ),
        try_it=(("Ask something in Chat", "/console/chat"),),
        code=("src/iw_rag/generation/answer.py", "guide/05-agentic-loop.md"),
    ),
    Lesson(
        slug="self-check",
        title="Self-check and retries",
        short="Self-check",
        phase="answer",
        question="Is the draft really supported, and how many second chances does it get?",
        what=(
            "Two graders read the draft: is it supported by the evidence, and does it answer the "
            "question? If not, the loop rewrites the question and tries again, within a hard budget."
        ),
        how=(
            "No citation at all: unsupported, decided deterministically before any model is called.",
            "Supported and answering: 'grounded', so go to the output guard.",
            "Not supported: retry to find better evidence. Supported but off-topic: retry with a better question.",
            "Retries are counted in exactly one place and shared by evidence and answer failures: at "
            "most 2 correction passes per question.",
            "LangGraph's recursion limit (25) is a backstop; hitting it becomes a refusal, never an error page.",
        ),
        why=(
            "The loop's correctness is its edge logic, so the limits are plain, unit-tested "
            "predicates, not model judgement. Two brakes beat one: if a cycle ever escapes the "
            "counter, the recursion limit still stops it and the system fails closed."
        ),
        gains=(
            "A bad first pass gets a bounded chance to improve.",
            "The loop can never run away or alternate failure types to double its budget.",
        ),
        losses=(
            "Each retry costs a full retrieval and generation, so a hard question is slow.",
            "Measured honestly: on the project's golden sets the loop and a single pass tied (see evidence).",
        ),
        evidence=(
            "Experiment: loop on versus off over the same golden set gave the same grounded rate "
            "(8/9 on the demo corpus, 15/21 on the benchmark corpus). The loop earns its keep on "
            "noisy, adversarial corpora that the synthetic sets do not contain."
        ),
        code=("src/iw_rag/agent/graph.py", "src/iw_rag/agent/budgets.py"),
    ),
    Lesson(
        slug="fail-closed",
        title="Output guard, fail closed, audit",
        short="Output + audit",
        phase="answer",
        question="What stops a wrong answer from reaching you, and what is recorded?",
        what=(
            "The last checkpoint validates the citations and the groundedness verdict. If either "
            "fails, the system refuses honestly instead of answering. Every outcome, answered, "
            "refused or rejected, is written to an audit log."
        ),
        how=(
            "Deterministic citation validation: an empty citation list, or any citation not in the "
            "graded set, is a hard block that nothing can override.",
            "Groundedness is read from the self-check verdict: one authority, no second model call.",
            "PII is redacted from the final answer.",
            "Only the 'done' node can turn a draft into an answer, and only when the verdict is 'grounded'.",
            "Every path ends in an append-only audit record. If the audit write fails, the answer is withheld.",
        ),
        why=(
            "An ungrounded answer that looks right is worse than a refusal, so trust is engineered "
            "first and the refusal is a feature. An answer that cannot be audited is never emitted."
        ),
        gains=(
            "You get a cited answer or an honest 'not enough grounded evidence', never a confident guess.",
            "Every decision can be reconstructed afterwards.",
        ),
        losses=(
            "A correct answer is sometimes refused because the evidence looked too thin.",
            "Refusing is silent about why unless you read the verdict trail.",
        ),
        evidence=(
            "Judged faithfulness over the golden sets: 1.000 mean (8/8 demo, 18/18 benchmark). Read "
            "that carefully: honest 'I do not have enough information' refusals are trivially faithful, "
            "so the number reflects the grounded-or-refuse floor, and its calibration is checked by "
            "judge-versus-human agreement (Cohen's kappa)."
        ),
        try_it=(("Ask something in Chat", "/console/chat"),),
        code=("src/iw_rag/guardrails/output_guard.py", "src/iw_rag/guardrails/audit.py"),
    ),
    # -------------------------------------------------------------------- prove
    Lesson(
        slug="modes",
        title="Optional modes",
        short="Modes",
        phase="prove",
        question="When does the plain spine fall short, and what else is there?",
        what=(
            "Three extra routes sit behind the router: GraphRAG (a knowledge graph), visual "
            "(page images) and long-context (read the whole document). Each returns the same "
            "result type as the spine, and each was measured before being trusted."
        ),
        how=(
            "GraphRAG builds an entity graph offline in Apache AGE and seeds searches by fuzzy entity names.",
            "Visual retrieval matches page images (ColVision late interaction) or captions made offline by a vision model.",
            "Long-context feeds the whole document to the generator at a native 32K window.",
            "Each is gated: only used when switched on and chosen by the router, otherwise it falls back to the spine.",
        ),
        why=(
            "Every capability is justified by a repeatable experiment, and the negatives are published. "
            "Heavy extraction work stays offline so it never competes with the resident generator."
        ),
        gains=("Each mode covers a case the spine cannot, and says so with numbers.",),
        losses=(
            "GraphRAG: still scores 0 on procedural and policy questions.",
            "Visual: the two methods tied, so the simpler caption route is the recommendation.",
            "Long-context: accurate, but costs about 21K tokens of prefill per query.",
        ),
        evidence=(
            "GraphRAG first measured 0.000 (two bugs), then won on entity-centric queries: nDCG@5 0.395 "
            "versus the spine's 0.292. Visual: both methods scored 1.000 on image-only facts that text "
            "retrieval cannot find at all (about 0). Long-context: recovered an obliquely worded needle "
            "in a 21K-token document that RAG missed, at every depth and length tried."
        ),
        code=("src/iw_rag/modes/", "guide/07-optional-modes.md"),
    ),
    Lesson(
        slug="benchmarks",
        title="Reading the benchmarks",
        short="Benchmarks",
        phase="prove",
        question="What do all those numbers on the Benchmark pages mean?",
        what=(
            "Benchmark Studio runs a golden set of questions through the live pipeline and saves "
            "one immutable report per run. Reports can then be compared. Metrics come in three tiers, "
            "plus speed and memory."
        ),
        how=(
            "Tier 1, retrieval (no judge): recall@k is the share of questions where a correct passage is in the top k. "
            "MRR rewards a correct passage appearing early. nDCG rewards a good ordering of the whole list.",
            "Tier 2, judged without a reference answer: faithfulness (does the answer stick to its "
            "sources) and context precision (how much of the context was useful).",
            "Tier 3, judged against a reference: context recall and factual correctness.",
            "Speed and memory: per-stage latency, tokens per second, VRAM and model load time.",
            "Compare runs only on the same dataset. Differences come with bootstrap 95% confidence "
            "intervals, and an A/B result is flagged as underpowered when the sample is too small to trust.",
            "The judges are local models from a different family than the generator, to avoid a "
            "model favouring its own style. Cohen's kappa measures judge-versus-human agreement.",
        ),
        why=(
            "'Looks right' is treated as a liability. Every claim in this system is meant to be a "
            "measurement you can reproduce, including the ones that came out negative."
        ),
        gains=(
            "A change to a model or a setting can be judged by evidence, not by feel.",
            "Reports are append-only, so history cannot be rewritten.",
        ),
        losses=(
            "Numbers are only as good as the golden set: a small or easy set saturates and hides differences "
            "(the 3-document demo corpus scores 1.000 on every arm).",
            "Judged metrics need the GPU, so benchmark mode and chat cannot run at the same time.",
        ),
        evidence=(
            "The experiments quoted across these lessons are listed in guide/08-trust-and-eval.md; the full "
            "reports are in eval/reports."
        ),
        try_it=(
            ("Run a benchmark", "/console/benchmark"),
            ("Browse results", "/console/results"),
            ("Compare runs", "/console/compare"),
            ("Scoreboard", "/console/scoreboard"),
        ),
        code=("src/iw_rag/eval/benchmark/", "guide/08-trust-and-eval.md"),
    ),
    Lesson(
        slug="one-gpu",
        title="One GPU",
        short="One GPU",
        phase="prove",
        question="How does all of this fit on a single graphics card?",
        what=(
            "Everything runs on one RTX 5090 with 32 GB: the generator, embedder and reranker stay "
            "resident, and the heavy work is moved to times when it will not get in the way."
        ),
        how=(
            "Two clocks: heavy work (parsing, embedding, graph extraction, captioning, judging, benchmarking) is "
            "offline; the answer path stays inside the resident-VRAM and latency budget.",
            "Serving mode keeps the generator and guards loaded. Benchmark mode frees them so a model-swap "
            "controller (llama-swap) can load different models one at a time.",
            "The two modes are mutually exclusive, and the console shows which one is active.",
            "The System page shows live VRAM, powers components on and off, and unloads idle models.",
        ),
        why=(
            "'Local-only, open-weight, single GPU' is the product. No hosted API is called anywhere, so "
            "the memory budget is a first-class design constraint, not an afterthought."
        ),
        gains=(
            "Sensitive documents never leave the machine, and the system can be air-gapped.",
            "A benchmark can compare model A and model B on the same card with no redeploy.",
        ),
        losses=(
            "Roughly 2 to 5 concurrent users, bound by VRAM and the KV cache.",
            "Chat is disabled while benchmark mode is active.",
            "No high availability, no scale-out: a second GPU is a documented graduation, not a deliverable.",
        ),
        evidence=(
            "Measured peak resident memory with generator, embedder and reranker loaded: 23.9 GB of 32 GB, "
            "leaving about 8 GB of headroom."
        ),
        try_it=(("Open System", "/console/system"),),
        code=("src/iw_rag/serving/", "guide/09-operations.md"),
    ),
)

_BY_SLUG = {lesson.slug: lesson for lesson in LESSONS}


def get_lesson(slug: str) -> Lesson | None:
    """Return the lesson for ``slug``, or ``None`` if there is none."""
    return _BY_SLUG.get(slug)


__all__ = ["LESSONS", "Lesson", "Phase", "get_lesson"]
