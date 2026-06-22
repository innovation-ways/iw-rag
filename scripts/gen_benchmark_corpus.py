"""Deterministic synthetic benchmark corpus generator (experiment #1/#3/#4 fuel).

The 3-doc/18-chunk demo corpus exercises the harness but is too small/easy to show
the spine's dense→hybrid→rerank lift. This script emits a larger, *harder*,
fully-synthetic corpus (no LLM, no external data → reproducible byte-for-byte) plus
a **matching golden set**, engineered so the experiments have something to measure:

  - **hybrid > dense**: many queries key on *rare exact-match tokens* (invoice IDs,
    PO numbers, SKU codes, incident IDs, policy codes) that dense embeddings blur
    together but BM25/learned-sparse nails.
  - **rerank > hybrid**: near-duplicate distractor docs (12 structurally-identical
    invoices, parallel policies) so first-stage fusion returns lexical near-misses
    that the cross-encoder reranker reorders correctly.
  - **loop earns its keep (#4)**: a band of unanswerable / under-specified queries
    that a single pass may fabricate on but the verification loop fails closed.

Every fact is planted with a **distinctive keyword** so the (source_uri, keyword)
golden model resolves to exactly the right chunk(s) — no recall inflation.

ACL groups: finance · hr · eng · public (legal/marketing). ACLs come from the
manifest (source-system metadata), never inferred from content.

Usage::

    uv run python scripts/gen_benchmark_corpus.py            # -> data/benchmark_corpus/
    uv run python scripts/gen_benchmark_corpus.py --out data/benchmark_corpus

Then ingest (augments the live DB — additive, idempotent on (uri, checksum))::

    set -a; . ./.env; set +a
    uv run --group ingest python scripts/ingest_corpus.py data/benchmark_corpus/manifest.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Pinned-v1 contract (F-00096 S06).                                             #
#                                                                               #
# A frozen reference the Benchmark Studio compares against: a versioned         #
# ``v1`` golden file + a deterministic corpus so a run today and a run next   #
# month compare like-for-like (no silent drift — the very risk the existing    #
# generator's own comments call out, "eval golden set drifted from a silent     #
# corpus re-gen").                                                              #
#                                                                               #
# The pinned v1 golden lives at ``eval/golden/benchmark_golden_v1.yaml`` and   #
# is stamped with ``dataset.version: v1``.  The script can also emit the       #
# non-pinned legacy ``eval/golden/benchmark_golden.yaml`` for the existing     #
# default workflow.                                                             #
# --------------------------------------------------------------------------- #

DATASET_VERSION_V1: str = "v1"
"""The frozen dataset version the Benchmark Studio pins to (``"v1"``)."""

PINNED_GOLDEN_PATH: str = "eval/golden/benchmark_golden_v1.yaml"
"""Canonical path (relative to CWD) of the pinned v1 golden file."""

GOLDEN_HEADER_V1: str = (
    "# Pinned v1 benchmark golden set (F-00096 S06).\n"
    "#\n"
    "# DO NOT EDIT BY HAND — this file is a frozen reference the Benchmark\n"
    "# Studio compares against.  Regenerate exclusively via:\n"
    "#\n"
    "#     uv run python scripts/gen_benchmark_corpus.py --version v1\n"
    "#\n"
    "# Relevance = (source_uri, keyword): a chunk is relevant when it belongs to\n"
    "# `source` AND its content contains `keyword` (case-insensitive).  Keywords\n"
    "# are distinctive per doc so recall is not inflated.\n"
)
"""Header comment block for the pinned v1 golden file (byte-stable prefix)."""

GOLDEN_HEADER_LEGACY: str = (
    "# Synthetic benchmark golden set (generated alongside data/benchmark_corpus).\n"
    "# Relevance = (source_uri, keyword): a chunk is relevant when it belongs to\n"
    "# `source` AND its content contains `keyword` (case-insensitive). Keywords are\n"
    "# distinctive per doc so recall is not inflated. Regenerate with\n"
    "# scripts/gen_benchmark_corpus.py to stay in sync with the corpus.\n"
)
"""Header comment block for the legacy (non-pinned) golden file."""

# --------------------------------------------------------------------------- #
# Deterministic source data (fixed tables — no randomness).                     #
# --------------------------------------------------------------------------- #

# 12 invoices. Distinct customer / PO / totals / support-cost so every exact-match
# query resolves to a single doc. Customers repeat across invoices on purpose so a
# customer-name query is a *distractor* set the reranker must disambiguate by ID.
_CUSTOMERS = [
    ("Globex Corp", "Net 30", "VEND-01"),
    ("Initech LLC", "Net 45", "VEND-02"),
    ("Umbrella Inc", "Net 60", "VEND-03"),
    ("Soylent Co", "Net 30", "VEND-04"),
    ("Hooli Group", "Net 45", "VEND-05"),
    ("Stark Industries", "Net 15", "VEND-06"),
]
# (invoice_no, customer_idx, po_no, license_cost, support_cost, total)
_INVOICES = [
    ("INV-2026-0901", 0, "PO-44120", 12000, 3000, 15000),
    ("INV-2026-0902", 1, "PO-44121", 8000, 2000, 10000),
    ("INV-2026-0903", 2, "PO-44122", 24000, 6000, 30000),
    ("INV-2026-0904", 3, "PO-44123", 5000, 1250, 6250),
    ("INV-2026-0905", 4, "PO-44124", 16000, 4000, 20000),
    ("INV-2026-0906", 5, "PO-44125", 40000, 9000, 49000),
    ("INV-2026-0907", 1, "PO-44126", 9500, 2500, 12000),
    ("INV-2026-0908", 0, "PO-44127", 21000, 5250, 26250),
    ("INV-2026-0909", 2, "PO-44128", 13500, 3375, 16875),
    ("INV-2026-0910", 4, "PO-44129", 7000, 1750, 8750),
    ("INV-2026-0911", 3, "PO-44130", 30000, 7500, 37500),
    ("INV-2026-0912", 5, "PO-44131", 18000, 4500, 22500),
]

# HR bands → distinctive annual-leave + perks. Distinct numbers per band.
_BANDS = [
    ("Junior", 20, 1.67, "HR-POL-01"),
    ("Mid", 22, 1.83, "HR-POL-02"),
    ("Senior", 25, 2.08, "HR-POL-03"),
    ("Principal", 30, 2.50, "HR-POL-04"),
]

# Eng incidents → distinctive root cause + INC id.
_INCIDENTS = [
    ("INC-2026-001", "search latency spike", "HNSW ef_search misconfiguration", "P2"),
    ("INC-2026-002", "ingestion backlog", "embedder OOM under batch size 256", "P1"),
    ("INC-2026-003", "citation mismatch", "datamark stripping regression", "P3"),
    ("INC-2026-004", "ACL leak false alarm", "stale principal_groups cache", "P2"),
]

# Eng API endpoints → distinctive path + rate limit.
_ENDPOINTS = [
    ("/v2/search", "POST", "60 req/min", "hybrid retrieval over the index"),
    ("/v2/ingest", "POST", "10 req/min", "submit a document for offline ingestion"),
    ("/v2/documents", "GET", "120 req/min", "list documents visible to the principal"),
    ("/v2/health", "GET", "600 req/min", "liveness and resident-VRAM probe"),
]


# --------------------------------------------------------------------------- #
# Document renderers.                                                            #
# --------------------------------------------------------------------------- #


def _invoice_doc(inv: tuple) -> str:
    no, cidx, po, lic, sup, total = inv
    cust, terms, vend = _CUSTOMERS[cidx]
    return f"""# Invoice {no}

Billing period: Q3 2026. Customer: {cust} ({vend}). Purchase order {po}.
Payment terms: {terms}.

## Line Items

| Code | Description | Qty | Unit | Total |
| --- | --- | --- | --- | --- |
| SKU-001 | Platform license | 1 | {lic} | {lic} |
| SKU-014 | Premium support | 1 | {sup} | {sup} |

## Summary

The total due on invoice {no} is {total} USD, payable per {terms} from the
invoice date. Remit against purchase order {po}.
"""


def _vendor_master() -> str:
    rows = "\n".join(
        f"| {c} | {v} | {t} | accounts-{v.lower()}@example.invalid |" for c, t, v in _CUSTOMERS
    )
    return f"""# Vendor Master File

Authoritative customer → payment-terms mapping (finance source of truth).

| Customer | Vendor ID | Payment Terms | Billing Contact |
| --- | --- | --- | --- |
{rows}

Payment terms on every invoice MUST match this master. Disputes route to the
billing contact above.
"""


def _expense_policy() -> str:
    return """# Expense Reimbursement Policy

Policy code FIN-POL-07. Expenses are reimbursed within 14 business days of an
approved claim.

- Per-diem for domestic travel: 75 USD per day.
- Per-diem for international travel: 120 USD per day.
- Single expenses above 2500 USD require director approval (DIR-APPROVAL).
- Mileage is reimbursed at 0.67 USD per mile.

Submit claims through the finance portal; attach receipts for any item above
25 USD.
"""


def _po_log() -> str:
    rows = "\n".join(
        f"| {po} | {no} | {_CUSTOMERS[cidx][0]} | {total} |"
        for (no, cidx, po, _l, _s, total) in _INVOICES
    )
    return f"""# Purchase Order Log

Cross-reference of purchase orders to invoices (finance reconciliation).

| Purchase Order | Invoice | Customer | Amount (USD) |
| --- | --- | --- | --- |
{rows}
"""


def _leave_policy() -> str:
    rows = "\n".join(f"| {b} | {d} | {acc} | {code} |" for b, d, acc, code in _BANDS)
    return f"""# Annual Leave Policy

Policy code HR-POL-00. Paid annual leave is granted by band and accrues monthly.

| Band | Annual Days | Monthly Accrual | Policy Code |
| --- | --- | --- | --- |
{rows}

Sick leave is separate from annual leave, does not carry over year to year, and
is capped at 10 days. Annual leave accrues each month and unused days above 5 are
forfeited at year end.
"""


def _benefits() -> str:
    return """# Employee Benefits

Policy code HR-POL-10.

- Health plan: three tiers — Bronze, Silver, Gold. Gold covers 90% co-insurance.
- Retirement: the company matches 401k contributions up to 6% of salary.
- Wellness stipend: 800 USD per year, claimable quarterly.
- Learning budget: 1500 USD per year for courses and conferences.
"""


def _directory() -> str:
    people = [
        ("Ada Lovelace", "Principal", "Engineering"),
        ("Alan Turing", "Senior", "Engineering"),
        ("Grace Hopper", "Principal", "Finance"),
        ("Katherine Johnson", "Senior", "Finance"),
        ("Dorothy Vaughan", "Mid", "HR"),
        ("Mary Jackson", "Junior", "HR"),
    ]
    rows = "\n".join(f"| {n} | {band} | {dept} |" for n, band, dept in people)
    return f"""# Employee Directory

Internal directory (band drives leave + benefits — see HR-POL-00).

| Name | Band | Department |
| --- | --- | --- |
{rows}
"""


def _conduct() -> str:
    return """# Code of Conduct

Policy code HR-POL-20. All employees must complete annual conduct training.

- Conflicts of interest must be declared within 5 business days.
- Confidential data must never leave approved systems.
- Violations are reviewed by the ethics committee (ETHICS-REV) within 30 days.
"""


def _api_reference() -> str:
    rows = "\n".join(f"| `{p}` | {m} | {rl} | {desc} |" for p, m, rl, desc in _ENDPOINTS)
    return f"""# API Reference (v2)

Internal service API. All endpoints require a bearer token; the principal's ACL
groups are server-injected and never read from the request body.

| Endpoint | Method | Rate Limit | Description |
| --- | --- | --- | --- |
{rows}

Authentication failures return 401; ACL-filtered empty results return 200 with an
empty list (never 403, to avoid existence oracles).
"""


def _incidents() -> str:
    blocks = "\n\n".join(
        f"## {iid} — {title} ({sev})\n\nRoot cause: {cause}. "
        f"Resolved by rolling back the offending change and adding a regression test."
        for iid, title, cause, sev in _INCIDENTS
    )
    return f"""# Incident Postmortems

Engineering incident log. Each entry has a distinct incident id and root cause.

{blocks}
"""


def _config_reference() -> str:
    return """# Configuration Reference

Settings are environment-driven (`IW_RAG__SECTION__KEY`).

- `IW_RAG__RETRIEVAL__RRF_K`: reciprocal-rank-fusion constant, default 60.
- `IW_RAG__RETRIEVAL__TOP_K`: candidates before rerank, default 40.
- `IW_RAG__GENERATION__MAX_TOKENS`: answer cap, default 1024.
- `IW_RAG__AGENT__MAX_RETRIES`: loop budget, default 2.
- `IW_RAG__GUARDRAILS__ENABLED`: input/output guards, default true.
"""


def _runbook() -> str:
    return """# Operations Runbook

On-call procedures for the retrieval service.

- To reindex after an embedder change, run the ingestion pipeline with
  `--reembed` during the maintenance window.
- If search latency exceeds 800 ms p95, check HNSW `ef_search` (see INC-2026-001).
- Resident VRAM must stay under 32 GB; the health probe fails closed otherwise.
"""


def _product_intro() -> str:
    return """# Acme Search — Product Introduction

Acme Search is a fully-local, open-weight retrieval-augmented assistant. It
answers questions grounded in your own documents with citations and document-level
access control, without a byte leaving your hardware.

For help, open a support ticket through the in-app help menu.
"""


def _tos() -> str:
    return """# Terms of Service

These public terms govern evaluation use of Acme Search.

- The evaluation licence is granted for 90 days and is non-transferable.
- The software is provided "as is" without warranty.
- Governing law for this evaluation agreement is the State of Delaware.
"""


def _privacy() -> str:
    return """# Privacy Notice

Acme Search processes all data locally. No telemetry, prompts, or documents are
transmitted to any third party. Audit logs are retained for 365 days and then
purged.
"""


# --------------------------------------------------------------------------- #
# Corpus assembly: (relative path, acl group, renderer).                        #
# --------------------------------------------------------------------------- #


def _build_docs() -> list[tuple[str, str, str]]:
    docs: list[tuple[str, str, str]] = []
    for inv in _INVOICES:
        docs.append((f"finance/{inv[0].lower()}.md", "finance", _invoice_doc(inv)))
    docs += [
        ("finance/vendor_master.md", "finance", _vendor_master()),
        ("finance/expense_policy.md", "finance", _expense_policy()),
        ("finance/po_log.md", "finance", _po_log()),
        ("hr/leave_policy.md", "hr", _leave_policy()),
        ("hr/benefits.md", "hr", _benefits()),
        ("hr/directory.md", "hr", _directory()),
        ("hr/conduct.md", "hr", _conduct()),
        ("eng/api_reference.md", "eng", _api_reference()),
        ("eng/incidents.md", "eng", _incidents()),
        ("eng/config_reference.md", "eng", _config_reference()),
        ("eng/runbook.md", "eng", _runbook()),
        ("public/product_intro.md", "public", _product_intro()),
        ("public/tos.md", "public", _tos()),
        ("public/privacy.md", "public", _privacy()),
    ]
    return docs


def _uri(rel: str) -> str:
    return f"corpus://bench/{rel}"


# --------------------------------------------------------------------------- #
# Golden set: (question, source rel-path, keyword, query_type).                 #
# keyword MUST occur in the target doc and be distinctive to it.                #
# --------------------------------------------------------------------------- #


def _build_golden() -> list[dict]:
    g: list[dict] = []

    # --- exact_match: rare-token ID lookups (favors hybrid/sparse) ---
    g += [
        {
            "q": "What is the total due on invoice INV-2026-0908?",
            "src": "finance/inv-2026-0908.md",
            "kw": "26250",
            "t": "exact_match",
        },
        {
            "q": "Which purchase order is on invoice INV-2026-0911?",
            "src": "finance/inv-2026-0911.md",
            "kw": "PO-44130",
            "t": "exact_match",
        },
        {
            "q": "What is the premium support cost on invoice INV-2026-0903?",
            "src": "finance/inv-2026-0903.md",
            "kw": "6000",
            "t": "exact_match",
        },
        {
            "q": "Which invoice references purchase order PO-44127?",
            "src": "finance/inv-2026-0908.md",
            "kw": "PO-44127",
            "t": "exact_match",
        },
        {
            "q": "What is the root cause of incident INC-2026-002?",
            "src": "eng/incidents.md",
            "kw": "embedder OOM",
            "t": "exact_match",
        },
        {
            "q": "What rate limit applies to the /v2/ingest endpoint?",
            "src": "eng/api_reference.md",
            "kw": "10 req/min",
            "t": "exact_match",
        },
        {
            "q": "What is the policy code for the expense reimbursement policy?",
            "src": "finance/expense_policy.md",
            "kw": "FIN-POL-07",
            "t": "exact_match",
        },
    ]

    # --- distractor: many near-duplicate invoices; must pick the right one (favors rerank) ---
    g += [
        {
            "q": "What is the platform license cost on invoice INV-2026-0906?",
            "src": "finance/inv-2026-0906.md",
            "kw": "40000",
            "t": "distractor",
        },
        {
            "q": "Who is the customer on invoice INV-2026-0905?",
            "src": "finance/inv-2026-0905.md",
            "kw": "Hooli Group",
            "t": "distractor",
        },
        {
            "q": "What is the total on invoice INV-2026-0910?",
            "src": "finance/inv-2026-0910.md",
            "kw": "8750",
            "t": "distractor",
        },
    ]

    # --- semantic: paraphrase, no shared rare token (favors dense) ---
    g += [
        {
            "q": "How many paid days off does a principal engineer get each year?",
            "src": "hr/leave_policy.md",
            "kw": "30",
            "t": "semantic",
        },
        {
            "q": "How much does the company contribute to retirement savings?",
            "src": "hr/benefits.md",
            "kw": "6%",
            "t": "semantic",
        },
        {
            "q": "How much can I claim per day when travelling abroad?",
            "src": "finance/expense_policy.md",
            "kw": "120 USD per day",
            "t": "semantic",
        },
        {
            "q": "What should I do if search is responding slowly?",
            "src": "eng/runbook.md",
            "kw": "ef_search",
            "t": "semantic",
        },
        {
            "q": "How do I reach support?",
            "src": "public/product_intro.md",
            "kw": "ticket",
            "t": "semantic",
        },
        {
            "q": "How long are audit records kept before deletion?",
            "src": "public/privacy.md",
            "kw": "365 days",
            "t": "semantic",
        },
    ]

    # --- multi_hop: bridge two docs; gold = the doc with the final answer keyword ---
    g += [
        {
            "q": "What are the payment terms for the customer on invoice INV-2026-0902?",
            "src": "finance/inv-2026-0902.md",
            "kw": "Net 45",
            "t": "multi_hop",
        },
        {
            "q": "What annual leave does Grace Hopper receive given her band?",
            "src": "hr/leave_policy.md",
            "kw": "30",
            "t": "multi_hop",
        },
    ]

    # --- unanswerable: zero-gold, must fail closed (#4 / groundedness) ---
    g += [
        {
            "q": "What is the revenue forecast for 2030?",
            "src": None,
            "kw": None,
            "t": "unanswerable",
        },
        {
            "q": "What is the CEO's personal mobile number?",
            "src": None,
            "kw": None,
            "t": "unanswerable",
        },
        {
            "q": "Which invoice will we issue next quarter?",
            "src": None,
            "kw": None,
            "t": "unanswerable",
        },
    ]

    return [
        {
            "question": e["q"],
            "source": _uri(e["src"]) if e["src"] else None,
            "keyword": e["kw"],
            "query_type": e["t"],
        }
        for e in g
    ]


# --------------------------------------------------------------------------- #
# Scale-up: ~85 more deterministic docs (additive — the 26 above + their golden #
# stay byte-identical so existing baselines hold). More near-duplicate distractors #
# = harder retrieval = a more benchmark-grade lift. All keywords stay distinctive. #
# --------------------------------------------------------------------------- #


# 40 more invoices (INV-2026-0913..0952), cycling the existing customers so
# vendor_master stays authoritative; totals are computed-unique (no collision with
# the 12 round-number originals).
def _scale_invoices() -> list[tuple]:
    out: list[tuple] = []
    for i in range(40):
        no = f"INV-2026-{913 + i:04d}"
        cidx = i % len(_CUSTOMERS)
        po = f"PO-44{132 + i}"
        lic = 5113 + i * 371  # odd, monotonic → unique and non-round
        sup = lic // 4
        out.append((no, cidx, po, lic, sup, lic + sup))
    return out


_SCALE_INVOICES = _scale_invoices()

# 24 individual incident docs (INC-2026-005..028), each with a distinctive root cause.
_SCALE_INCIDENTS = [
    ("connection pool exhaustion under burst load", "P2"),
    ("reranker latency spike from cold TEI cache", "P3"),
    ("duplicate chunk ids after a botched re-ingest", "P2"),
    ("ACL pre-filter bypassed by a null principal", "P1"),
    ("Phoenix trace export silently dropped spans", "P3"),
    ("vLLM OOM at 32K context with two concurrent users", "P1"),
    ("BM25 index corruption after an unclean shutdown", "P2"),
    ("citation markers misaligned by a datamark off-by-one", "P3"),
    ("guard model timeout cascaded into fail-closed storms", "P2"),
    ("HNSW recall collapse after an ef_construction change", "P2"),
    ("audit log write contention under high QPS", "P3"),
    ("stale embedding cache served pre-reindex vectors", "P2"),
    ("RRF fusion divided by zero on an empty sparse arm", "P3"),
    ("prompt-guard false positives blocked benign finance queries", "P2"),
    ("checkpoint thread_id collision resumed the wrong session", "P2"),
    ("Caddy edge dropped SSE keep-alives after 60s idle", "P3"),
    ("Postgres autovacuum stalled the HNSW build", "P2"),
    ("infinity reranker returned scores out of [0,1]", "P3"),
    ("long-context route enabled YaRN against the invariant", "P1"),
    ("graph extraction emitted relation types with spaces", "P3"),
    ("visual caption job leaked GPU memory across batches", "P2"),
    ("eval golden set drifted from a silent corpus re-gen", "P3"),
    ("migration down_revision hand-chained and broke head", "P2"),
    ("judge endpoint co-resided with the generator and OOM'd", "P1"),
]


def _scale_incident_doc(iid: str, cause: str, sev: str) -> str:
    return (
        f"# Incident {iid} ({sev})\n\n"
        f"Root cause: {cause}. The on-call engineer mitigated within the SLA window, "
        f"opened a follow-up to add a regression test, and recorded the postmortem here. "
        f"This incident is distinct from all others by its {iid} identifier and root cause."
    )


# 18 KB articles (KB-2026-001..018), each with one distinctive how-to fact.
_SCALE_KB = [
    (
        "rotate a service API key without downtime",
        "rotate the key in the portal, deploy, then revoke the old key after 24 hours",
    ),
    (
        "tune HNSW for higher recall",
        "raise ef_search at query time; ef_construction is fixed at index build",
    ),
    (
        "re-embed the corpus after a model swap",
        "run the ingestion pipeline with --reembed during a maintenance window",
    ),
    (
        "read a Phoenix trace for one query",
        "open the trace_ref.trace_id from the response diagnostics in the Phoenix UI",
    ),
    (
        "add a new ACL group",
        "register the group in the static-token map; never infer ACLs from content",
    ),
    (
        "interpret a fail-closed refusal",
        "the loop exhausted its retry budget without grounding; it refuses by design",
    ),
    (
        "enable the guards locally",
        "set IW_RAG__GUARDRAILS__ENABLED=true and serve the guard models",
    ),
    (
        "back up the single Postgres store",
        "snapshot the volume; one store holds vectors, BM25, graph, and audit",
    ),
    (
        "debug a 422 from the query API",
        "the query field enforces 1..4000 characters; empty or over-long inputs are rejected",
    ),
    (
        "measure judge-human agreement",
        "run the kappa harness over the calibration slice with the judge served",
    ),
    (
        "choose between dense and hybrid retrieval",
        "hybrid wins on rare exact-match identifiers; the reranker disambiguates near-duplicates",
    ),
    (
        "set the agentic loop retry budget",
        "IW_RAG__AGENT__MAX_RETRIES defaults to 2 with a recursion_limit of 25",
    ),
    (
        "redact PII from answers",
        "Presidio runs in the output guard; synthetic SSNs are masked before display",
    ),
    (
        "serve the offline faithfulness judge",
        "bring up Lynx-8B on the judge endpoint; it never co-resides with the hot path",
    ),
    (
        "understand the two-clocks rule",
        "ingestion, eval, and indexing are offline/batch; the hot path stays within VRAM budget",
    ),
    (
        "verify citations in an answer",
        "every bracketed marker must resolve to a retrieved chunk; the output guard enforces it",
    ),
    (
        "scope a query to one tenant",
        "the principal's ACL groups pre-filter retrieval; cross-tenant chunks are unreachable",
    ),
    (
        "recover from an unclean shutdown",
        "restart the generator before the guards to avoid a KV-cache memory race",
    ),
]


def _scale_kb_doc(code: str, title: str, fact: str) -> str:
    return (
        f"# {code}: How to {title}\n\n"
        f"This knowledge-base article explains how to {title}. The key step: {fact}. "
        f"For related procedures see the operations runbook and the configuration reference."
    )


def _build_scale_docs() -> list[tuple[str, str, str]]:
    docs: list[tuple[str, str, str]] = []
    for inv in _SCALE_INVOICES:
        docs.append((f"finance/{inv[0].lower()}.md", "finance", _invoice_doc(inv)))
    for i, (cause, sev) in enumerate(_SCALE_INCIDENTS):
        iid = f"INC-2026-{5 + i:03d}"
        docs.append((f"eng/{iid.lower()}.md", "eng", _scale_incident_doc(iid, cause, sev)))
    for i, (title, fact) in enumerate(_SCALE_KB):
        code = f"KB-2026-{1 + i:03d}"
        docs.append((f"eng/kb/{code.lower()}.md", "eng", _scale_kb_doc(code, title, fact)))
    return docs


def _build_scale_golden() -> list[dict]:
    g: list[dict] = []
    # exact-match on new invoices (total + PO — rare tokens, favor hybrid/sparse)
    for idx in (3, 11, 19, 27, 35):  # spread across the 40 new invoices
        no, _cidx, po, _lic, _sup, total = _SCALE_INVOICES[idx]
        rel = f"finance/{no.lower()}.md"
        g.append(
            {
                "q": f"What is the total due on invoice {no}?",
                "src": rel,
                "kw": str(total),
                "t": "exact_match",
            }
        )
        g.append(
            {
                "q": f"Which purchase order is on invoice {no}?",
                "src": rel,
                "kw": po,
                "t": "exact_match",
            }
        )
    # exact-match on new incidents (root cause by INC id)
    for i in (2, 9, 18):
        iid = f"INC-2026-{5 + i:03d}"
        cause = _SCALE_INCIDENTS[i][0]
        kw = cause.split()[0] + " " + cause.split()[1]  # distinctive leading bigram
        g.append(
            {
                "q": f"What was the root cause of incident {iid}?",
                "src": f"eng/{iid.lower()}.md",
                "kw": kw,
                "t": "exact_match",
            }
        )
    # semantic on KB articles (paraphrased how-to → favors dense)
    for i, q in (
        (1, "How do I improve retrieval recall from the vector index?"),
        (8, "Why would the query endpoint reject my request with a 422?"),
        (16, "How do I make sure a query only sees one tenant's documents?"),
    ):
        code = f"KB-2026-{1 + i:03d}"
        kw = _SCALE_KB[i][1].split(";")[0].split(",")[0][:24]
        g.append({"q": q, "src": f"eng/kb/{code.lower()}.md", "kw": kw, "t": "semantic"})
    return [
        {
            "question": e["q"],
            "source": _uri(e["src"]) if e["src"] else None,
            "keyword": e["kw"],
            "query_type": e["t"],
        }
        for e in g
    ]


def build_golden_doc(
    docs: list[tuple[str, str, str]],
    include_scale: bool = False,
    version: str = DATASET_VERSION_V1,
) -> dict[str, Any]:
    """Build the golden-set document in a byte-stable, deterministic form.

    Args:
        docs: The corpus doc list (``(relative_path, acl_group, body)``
            triples) — used to derive the sorted ``principal_groups`` set.
        include_scale: When True, append the scale-up golden queries so the
            set covers the full ~85-doc corpus.
        version: The dataset version stamp (default: the pinned v1).

    Returns:
        A dict with ``dataset`` (``name`` + ``version``), the sorted
        ``principal_groups`` list, and the ``queries`` list — deterministic
        iteration order, no timestamps, no randomness.
    """
    golden = list(_build_golden())
    if include_scale:
        golden = golden + list(_build_scale_golden())
    groups = sorted({d[1] for d in docs})
    return {
        "dataset": {"name": "benchmark-golden", "version": version},
        "principal_groups": groups,
        "queries": golden,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchmark_corpus")
    ap.add_argument("--scale", action="store_true", help="also emit the ~85-doc scale-up set")
    ap.add_argument(
        "--golden",
        default=None,
        help=(
            "where to write the matching golden set "
            "(default: depends on --version; legacy non-pinned path when unset)"
        ),
    )
    ap.add_argument(
        "--version",
        dest="dataset_version",
        default=None,
        choices=[DATASET_VERSION_V1],
        help=(
            "stamp a dataset version on the emitted golden and write to the "
            f"matching pinned path (only {DATASET_VERSION_V1!r} is supported)"
        ),
    )
    args = ap.parse_args()

    pinned = args.dataset_version is not None
    golden_path = args.golden or (
        PINNED_GOLDEN_PATH if pinned else "eval/golden/benchmark_golden.yaml"
    )
    header = GOLDEN_HEADER_V1 if pinned else GOLDEN_HEADER_LEGACY
    dataset_version = args.dataset_version or DATASET_VERSION_V1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = _build_docs()
    if args.scale:
        docs += _build_scale_docs()

    manifest_docs = []
    for rel, group, content in docs:
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        manifest_docs.append(
            {"uri": _uri(rel), "path": rel, "mime": "text/markdown", "acl_groups": [group]}
        )

    manifest = {"documents": manifest_docs}
    (out / "manifest.yaml").write_text(
        "# Synthetic benchmark corpus manifest (generated by scripts/gen_benchmark_corpus.py).\n"
        "# ACL groups are source-system metadata, never content-inferred. Public/synthetic only.\n"
        + yaml.safe_dump(manifest, sort_keys=False)
    )

    golden_doc = build_golden_doc(
        docs=docs,
        include_scale=args.scale,
        version=dataset_version,
    )
    golden = golden_doc["queries"]
    gp = Path(golden_path)
    gp.parent.mkdir(parents=True, exist_ok=True)
    gp.write_text(header + yaml.safe_dump(golden_doc, sort_keys=False, width=100))

    n_gold = sum(1 for q in golden if q["source"])
    n_zero = len(golden) - n_gold
    print(f"Wrote {len(docs)} docs ({len(manifest_docs)} manifest entries) → {out}")
    print(f"  groups: {golden_doc['principal_groups']}")
    print(
        f"Wrote golden set: {len(golden)} queries ({n_gold} gold, {n_zero} zero-gold) "
        f"[version={dataset_version}] → {gp}"
    )
    print("\nNext: ingest (augments the live DB):")
    print("  set -a; . ./.env; set +a")
    print(f"  uv run --group ingest python scripts/ingest_corpus.py {out}/manifest.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
