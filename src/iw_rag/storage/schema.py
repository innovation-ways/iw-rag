"""SQLAlchemy Core table definitions for the ``iw_rag`` schema.

Alembic owns the DDL (lld-data-model §9); this module is the typed reflection
the repositories build queries against, and the ``target_metadata`` Alembic
autogenerate compares against. Column shapes mirror the migration files exactly
(lld-data-model §3). Indexes are managed in the migration, not declared here,
to keep autogenerate quiet about the special HNSW/BM25 index types.

Task 3.4 adds ``audit_log`` (``alembic/versions/0003_audit_log.py``): the
append-only decision trail. Column shapes mirror the migration; the table object
is the insert surface used by ``AuditRepository.append``.

WI-031 adds ``visual_pages`` (``alembic/versions/0004_visual_pages.py``): the
ColVision multi-vector store for the visual late-interaction path.  Column shapes
mirror the migration and the ``PostgresCandidateStore`` SQL exactly.

F-00093 adds ``benchmark_reports`` (``alembic/versions/0008_benchmark_reports.py``):
the Benchmark Studio measurement-core summary table — one row per eval run
holding the rolled-up RAG quality (T1 retrieval, T2 reference-free judged, T3
reference-required judged), latency, and resource measurements.  Column shapes
mirror the migration; the table object is the insert surface used by
``BenchmarkReportRepository`` (S06).

F-00096 adds ``config_change_log`` (``alembic/versions/0009_config_change_log.py``):
the Operator Console live-config audit trail.  Append-only (Invariant 2),
mirroring the ``audit_log`` immutability pattern (trigger + REVOKE).  One
row per applied config edit; ``principal_id`` is sourced exclusively from
the authenticated ``AccessContext`` (Invariant 3 — never from a request
body); ``applied_mode`` is the ``hot`` vs ``pending_restart`` classification
that the console surfaces to the operator as a badge.
"""

from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

SCHEMA = "iw_rag"
EMBEDDING_DIMS = 1024

metadata = MetaData(schema=SCHEMA)


def _uuid_pk() -> Column[Any]:
    """A uuid primary-key column defaulting to ``gen_random_uuid()``."""
    return Column(
        "id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


def _created_at() -> Column[Any]:
    """A ``created_at timestamptz`` column defaulting to ``now()``."""
    return Column(
        "created_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )


ingestion_runs = Table(
    "ingestion_runs",
    metadata,
    _uuid_pk(),
    Column("corpus_label", Text),
    Column("parser", Text),
    Column("chunker_config", Text),
    Column("embed_model", Text),
    Column("status", Text, nullable=False, server_default=text("'running'")),
    Column("started_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")),
    Column("finished_at", TIMESTAMP(timezone=True)),
)

documents = Table(
    "documents",
    metadata,
    _uuid_pk(),
    Column("source_uri", Text, nullable=False),
    Column("title", Text),
    Column("content_type", Text, nullable=False, server_default=text("'document'")),
    Column("checksum", Text, nullable=False),
    Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("ingestion_run_id", UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.ingestion_runs.id")),
    # WI-DOCMGMT — user-facing document management (owner + visibility).
    # ``owner_id`` is the uploading principal id (NULL for system/corpus docs
    # ingested via ``make ingest``).  ``visibility`` is ``common`` (visible to
    # all principals → acl tag ``public``) or ``specific`` (owner-only → acl tag
    # ``user:<owner_id>``).  NULL on both means a corpus doc managed offline, not
    # surfaced in the management UI.  These columns are metadata for listing +
    # delete-authorization; retrieval visibility is still driven solely by
    # ``chunks.acl_tags`` (the pre-filter), never by these columns.
    Column("owner_id", Text),
    Column("visibility", Text),
    _created_at(),
    UniqueConstraint("source_uri", "checksum", name="documents_source_uri_checksum_key"),
)

chunks = Table(
    "chunks",
    metadata,
    _uuid_pk(),
    Column(
        "document_id",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.documents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("parent_id", UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.chunks.id", ondelete="CASCADE")),
    Column("ordinal", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("context_prefix", Text),
    Column("dense_embedding", Vector(EMBEDDING_DIMS), nullable=False),
    Column("sparse_weights", JSONB),
    Column("content_type", Text, nullable=False, server_default=text("'text'")),
    Column("section_path", Text),
    Column("acl_tags", ARRAY(Text), nullable=False, server_default=text("'{}'")),
    Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    _created_at(),
)

document_acl = Table(
    "document_acl",
    metadata,
    _uuid_pk(),
    Column(
        "document_id",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.documents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("principal", Text, nullable=False),
    Column("permission", Text, nullable=False, server_default=text("'read'")),
)

# ---------------------------------------------------------------------------
# ephemeral_sessions (WI-DOCMGMT — chat-scoped attachment ownership)
# ---------------------------------------------------------------------------
# Binds a chat session id to the principal that created it so the API can
# validate, server-side, that a request asking to "see" a session's ephemeral
# attachments actually owns that session before granting the synthetic
# ``session:<id>`` acl group.  This keeps the ACL keystone honest: the extra
# group is granted by the server from a verified ownership record, never blindly
# from request input.  Rows (and the attachment docs they gate) are purged on
# chat end and by a TTL sweep (``expires_at``).
ephemeral_sessions = Table(
    "ephemeral_sessions",
    metadata,
    Column("session_id", Text, primary_key=True),
    Column("principal_id", Text, nullable=False),
    Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")),
    Column("expires_at", TIMESTAMP(timezone=True), nullable=False),
)

# ---------------------------------------------------------------------------
# conversations + conversation_messages (chat memory — multi-turn history)
# ---------------------------------------------------------------------------
# Server-side persisted chat history so the agent can answer follow-ups that
# depend on what was said earlier in the SAME conversation ("what is my name?"
# after "my name is Sergio").  Prior turns are replayed into the agent loop as
# citable, principal-scoped synthetic chunks (conversation-as-source), so the
# fail-closed grounding contract is preserved: a memory answer cites the chat
# turn it came from rather than bypassing the grounding gate.
#
# ACL: a conversation is owned by exactly one ``principal_id`` (sourced from the
# authenticated AccessContext, never request input).  History is only ever
# loaded for the owning principal; the per-message FK + ON DELETE CASCADE means
# clearing a conversation purges its turns in one statement (the "Clear" button).
conversations = Table(
    "conversations",
    metadata,
    _uuid_pk(),
    Column("principal_id", Text, nullable=False),
    Column("title", Text),
    _created_at(),
    Column(
        "last_active_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
)

conversation_messages = Table(
    "conversation_messages",
    metadata,
    _uuid_pk(),
    Column(
        "conversation_id",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    # role: 'user' | 'assistant' (CHECK in migration)
    Column("role", Text, nullable=False),
    Column("content", Text, nullable=False),
    # citations: the assistant turn's resolved citations, jsonb (so the GET
    # history endpoint can re-render sources without re-running the query).
    Column("citations", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    # disposition: answered | refused | rejected (assistant turns only; NULL for user).
    Column("disposition", Text),
    # clock_timestamp() (NOT now()): a turn's user + assistant rows are inserted
    # in the SAME transaction, where now() returns the identical transaction-start
    # time for both — making (created_at)-ordering ambiguous within a turn.
    # clock_timestamp() advances per statement, giving a stable replay order.
    Column(
        "created_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("clock_timestamp()"),
    ),
)

eval_datasets = Table(
    "eval_datasets",
    metadata,
    _uuid_pk(),
    Column("name", Text, nullable=False),
    Column("version", Text, nullable=False),
    _created_at(),
)

eval_queries = Table(
    "eval_queries",
    metadata,
    _uuid_pk(),
    Column(
        "dataset_id",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.eval_datasets.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("question", Text, nullable=False),
    Column("reference_answer", Text),
    Column(
        "relevant_chunk_ids",
        ARRAY(UUID(as_uuid=True)),
        nullable=False,
        server_default=text("'{}'"),
    ),
    Column("query_type", Text),
)

eval_runs = Table(
    "eval_runs",
    metadata,
    _uuid_pk(),
    Column("dataset_id", UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.eval_datasets.id")),
    Column("experiment", Text, nullable=False),
    Column("git_sha", Text),
    Column("config", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    _created_at(),
)

eval_results = Table(
    "eval_results",
    metadata,
    _uuid_pk(),
    Column(
        "run_id",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.eval_runs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("query_id", UUID(as_uuid=True), ForeignKey(f"{SCHEMA}.eval_queries.id")),
    Column("retrieval_metrics", JSONB),
    Column("generation_metrics", JSONB),
    Column("trace_ref", JSONB),
)

# ---------------------------------------------------------------------------
# audit_log (Task 3.4 — append-only decision trail)
# ---------------------------------------------------------------------------

audit_log = Table(
    "audit_log",
    metadata,
    _uuid_pk(),
    Column("request_id", Text, nullable=False),
    Column("principal", Text, nullable=False),
    Column("query", Text, nullable=False),
    # acl_filter: principal_groups applied at retrieval (provenance, Task 3.0)
    Column("acl_filter", ARRAY(Text), nullable=False),
    # retrieved_chunk_ids: text[] (chunk ids flow as str through state)
    Column(
        "retrieved_chunk_ids",
        ARRAY(Text),
        nullable=False,
        server_default=text("'{}'"),
    ),
    # guard_verdicts: full decision trail serialised as jsonb
    Column(
        "guard_verdicts",
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    ),
    # redactions: entity types/counts only — never raw PII values
    Column(
        "redactions",
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    ),
    # disposition: answered | refused | rejected (CHECK in migration)
    Column("disposition", Text, nullable=False),
    _created_at(),
)

# ---------------------------------------------------------------------------
# visual_pages (WI-031 — ColVision multi-vector store, task-4b §5)
# ---------------------------------------------------------------------------
#: Proxy vector dimension: ColQwen2.5 and ColNomic-3B both produce 128-d patch vectors.
VISUAL_PROXY_DIMS = 128

visual_pages = Table(
    "visual_pages",
    metadata,
    _uuid_pk(),
    # Back-reference to the source chunk (chunk ids are str per the RetrievedChunk contract)
    Column("chunk_id", Text, nullable=False),
    Column("source_uri", Text, nullable=False),
    Column("page", Integer, nullable=False),
    Column("content", Text, nullable=False),
    # ACL entitlement tags — future stage-1 pre-filter (currently post-filter in VisualRetriever)
    Column("acl_tags", ARRAY(Text), nullable=False, server_default=text("'{}'")),
    # Pooled+binary multi-vectors stored as JSONB (no native pgvector multi-vector; ADR-010)
    Column("pooled_vectors", JSONB, nullable=False),
    Column("quantized_vectors", JSONB, nullable=False),
    # Mean-pooled single-vector proxy for stage-1 HNSW ANN (fixed 128-d for HNSW index)
    Column("proxy_vector", Vector(VISUAL_PROXY_DIMS), nullable=False),
    Column("bbox", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("page_image_ref", Text, nullable=False),
    Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    # UNIQUE constraint matches the ON CONFLICT target in PostgresCandidateStore.upsert
    UniqueConstraint("chunk_id", "page", name="visual_pages_chunk_id_page_key"),
)

# ---------------------------------------------------------------------------
# benchmark_reports (F-00093 — Benchmark Studio measurement core)
# ---------------------------------------------------------------------------
# Append-only history of rolled-up RAG evaluation reports.  One row per
# ``eval_runs`` row, keyed by ``run_id`` (FK → ``eval_runs.id``); every
# column is written by the S06 report-assembly path that combines the
# ``MetricsEngine`` output (T1 / T2 / T3 means + per-tier blob) with the
# S05 perf collectors (latency / resource JSONB) and a 95% bootstrap CI
# (S07).  No UPDATE / DELETE surface exists in
# ``BenchmarkReportRepository`` (Invariant 2 — reports are append-only
# history) and the FKs do not cascade-delete.  ``created_by`` is the
# authenticated principal id passed in by the auth layer (Invariant 3 —
# never derived from the LLM or from a request body field).
benchmark_reports = Table(
    "benchmark_reports",
    metadata,
    _uuid_pk(),
    # ``run_id`` FK to ``eval_runs.id`` — NO ON DELETE CASCADE (reports
    # outlive their run if the run is purged).  S01 migration mirrors
    # this with REFERENCES … (no ON DELETE clause).
    Column(
        "run_id",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.eval_runs.id"),
        nullable=False,
    ),
    # ``dataset_id`` FK to ``eval_datasets.id`` — NO ON DELETE CASCADE.
    Column(
        "dataset_id",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.eval_datasets.id"),
        nullable=False,
    ),
    _created_at(),
    Column("report_label", Text, nullable=False),
    Column("git_sha", Text),
    # Provenance from auth — never from the LLM or user-supplied request
    # body.  NULLable because the offline batch path may not have an
    # authenticated principal (the orchestrator sets it).
    Column("created_by", Text),
    Column("model_config", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    # T1 — deterministic retrieval metrics, promoted floats.
    Column("recall_at_5", Float),
    Column("recall_at_10", Float),
    Column("ndcg_at_5", Float),
    Column("ndcg_at_10", Float),
    Column("mrr", Float),
    # T2 — reference-free judged metrics.
    Column("faithfulness_mean", Float),
    Column("faithfulness_kappa", Float),
    Column("context_precision", Float),
    # T3 — reference-required judged metrics; NULL when the dataset has
    # no reference_answer (Invariant 5 — engine MUST NOT call a judge
    # for them in that case).
    Column("context_recall", Float),
    Column("factual_correctness", Float),
    # Latency / resource / CI JSONB columns (R-00122 extension of the
    # R-00121 §5.2 base schema).
    Column("latency", JSONB),
    Column("resource", JSONB),
    Column("ci_95", JSONB),
    Column("n_queries", Integer, nullable=False),
    Column("n_judged", Integer),
    # Full per-tier blob — a superset of the promoted columns, including
    # per-query vectors paired by index for the S07 stats module's
    # paired A/B tests.
    Column("metrics", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
)

# ---------------------------------------------------------------------------
# config_change_log (F-00096 — Operator Console live-config audit trail)
# ---------------------------------------------------------------------------
# Append-only history of applied config edits surfaced by the Operator
# Console.  Every successful ``PUT /api/config/{key}`` writes exactly one
# row capturing the authenticated principal, the touched key, the
# stringified prior value, the new value, and whether the change was
# applied hot (next request) or flagged pending-restart (startup-baked
# services).  Invariants enforced at the storage layer:
#
#   2.  Append-only — no UPDATE/DELETE path exists in code or DB
#       (trigger + REVOKE, mirroring ``audit_log``).
#   3.  ``principal_id`` comes ONLY from the authenticated
#       ``AccessContext``; the repository surface never accepts a
#       principal id from a request body, and the column is NOT NULL
#       so a stray anonymous insert cannot land.
#
# The ``applied_mode`` column carries a ``hot`` vs ``pending_restart``
# enum (enforced by a CHECK constraint in the migration DDL) so the
# console can render the correct badge without a join.  ``old_value`` is
# nullable: a brand-new overlay key has no prior value.
config_change_log = Table(
    "config_change_log",
    metadata,
    _uuid_pk(),
    # ``ts`` — append timestamp, server default now() (no separate
    # _created_at() helper because that helper emits a column named
    # ``created_at``; the design pins the column name to ``ts``).
    Column("ts", TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")),
    # ``principal_id`` — the authenticated principal (Invariant 3).  NOT
    # NULL so the DB itself rejects any insert that lacks auth provenance.
    Column("principal_id", Text, nullable=False),
    Column("config_key", Text, nullable=False),
    # ``old_value`` — stringified prior value; NULL on the first ever
    # overlay write for a given key.
    Column("old_value", Text),
    Column("new_value", Text, nullable=False),
    # ``applied_mode`` — hot | pending_restart (CHECK enforced in the
    # migration DDL; mirror of ``audit_log.disposition``).
    Column("applied_mode", Text, nullable=False),
    # ``trace_id`` — optional correlation id (request/htmx swap id).
    Column("trace_id", Text),
)
