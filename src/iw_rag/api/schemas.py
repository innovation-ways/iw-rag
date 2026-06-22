"""Request/response models for the API.

The security keystone is structural: ``QueryRequest`` has **no** groups/acl field,
so a client cannot assert its own access — ``principal_groups`` flow only from
verified auth into the ``AccessContext`` (lld-api §5).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """A query request. Deliberately carries NO acl/groups field."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["auto", "retrieve", "graph", "long_context", "visual"] = "auto"
    # Assist mode. When True, an answer the evidence does not support is RETURNED
    # (clearly marked ungrounded) instead of fail-closed refused. Groundedness
    # becomes *informational*, never a gate — but every other safeguard (input
    # guard, injection, safety, citation-validity, PII) stays fully enforced.
    # Defaults False so the strict fail-closed contract is preserved for every
    # existing caller (evals, red-team gate, tests); the frontend opts in.
    allow_ungrounded: bool = False
    # Chat session id for ephemeral attachments. When set AND the session is owned
    # by the authenticated caller (re-validated server-side), the caller's ACL is
    # widened with the synthetic ``session:<id>`` group so files attached to THIS
    # chat become retrievable — and only within this chat. Never trusted blindly:
    # an unknown/foreign session id is ignored (no group granted).
    session_id: str | None = Field(default=None, max_length=128)
    # Multi-turn chat memory. When set AND owned by the authenticated caller
    # (re-validated server-side), the prior turns of THIS conversation are
    # replayed into the agent loop as citable, principal-scoped evidence so a
    # follow-up ("what is my name?") can be answered + cited from earlier turns.
    # When None the server starts a new conversation and returns its id on the
    # response. Like ``session_id`` it is never trusted blindly: a foreign /
    # unknown id is ignored and a fresh conversation is created instead.
    conversation_id: str | None = Field(default=None, max_length=64)


class CitationOut(BaseModel):
    """A resolved citation in the response.

    Additive CR-00098 fields (S05):

    - ``document_id`` — the cited chunk's owning ``documents.id`` primary
      key (UUID-as-string).  Carried so the frontend can address the
      download endpoint (``/documents/{document_id}/raw``) without
      parsing the URI.  Always populated when the underlying
      :class:`~iw_rag.domain.models.Citation` carries one; defaults to
      ``""`` for the legacy / non-corpus paths (visual, graphrag,
      long-context) where the store does not yet surface the PK.
    - ``download_url`` — a **relative** URL the browser can use as an
      ``<a href>`` to download the original source file without
      dragging the caller's bearer token through a plain navigation.
      The form is ``"/documents/{document_id}/raw?token=<signed>"``.
      The token is a short-lived HMAC capability bound to
      ``(document_id, principal_id, exp[, session_id])`` minted by the
      S05 :func:`~iw_rag.api.routes.build_citation_out` helper.
      ``""`` (empty) when no token could be minted — e.g. the env-only
      ``IW_RAG__API__DOWNLOAD_TOKEN_SECRET`` is unset, the citation
      carries no document id, or minting raised.  The frontend treats
      ``""`` as "fall back to plain text rendering".
    """

    marker: int
    chunk_id: str
    source_uri: str
    quote: str = ""
    document_id: str = ""
    download_url: str = ""


class QueryResponse(BaseModel):
    """The answer payload. ``grounded=False`` is a fail-closed refusal (HTTP 200)."""

    answer: str
    grounded: bool
    citations: list[CitationOut] = Field(default_factory=list)
    used_chunk_ids: list[str] = Field(default_factory=list)
    retriever: str = "spine"
    trace_id: str = ""
    diagnostics: dict[str, Any] | None = None
    # The conversation this turn belongs to. Echoed back so the client can
    # persist it (localStorage) and send it on the next turn for memory. Empty
    # only when chat memory is not wired (no conversation service).
    conversation_id: str = ""


class ConversationCreateResponse(BaseModel):
    """The body of ``POST /conversations`` — a freshly created chat thread."""

    conversation_id: str


class ConversationMessageOut(BaseModel):
    """One persisted turn as surfaced by ``GET /conversations/{id}``."""

    role: str
    content: str
    citations: list[CitationOut] = Field(default_factory=list)
    disposition: str | None = None
    created_at: str = ""


class ConversationHistoryResponse(BaseModel):
    """The body of ``GET /conversations/{id}`` — the thread's turns, oldest-first."""

    conversation_id: str
    messages: list[ConversationMessageOut] = Field(default_factory=list)


class LoginRequest(BaseModel):
    """Username/password login (demo auth) — exchanged for a bearer token."""

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class LoginResponse(BaseModel):
    """The issued bearer token + resolved identity (groups for UI display only)."""

    token: str
    principal_id: str
    groups: list[str] = Field(default_factory=list)
    is_operator: bool = False


class DocumentOut(BaseModel):
    """A document as surfaced to the management UI."""

    id: str
    filename: str
    title: str | None = None
    mime: str
    visibility: str | None = None
    owner_id: str | None = None
    size_bytes: int = 0
    chunk_count: int = 0
    created_at: str = ""
    is_owner: bool = False
    kind: str = "upload"


class AttachmentOut(BaseModel):
    """The result of attaching an ephemeral file to a chat session."""

    document_id: str
    session_id: str
    filename: str
    chunk_count: int = 0


class HealthResponse(BaseModel):
    """Deep health payload naming each dependency."""

    status: Literal["ok", "degraded"]
    checks: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """An error body — carries no content/PII, only a code + trace id."""

    error: str
    trace_id: str = ""


# ---------------------------------------------------------------------------
# F-00094 — model-swap catalog (GET /models, POST /models/activate)
#
# The ActivateRequest carries ONLY ``model_id`` — NEVER groups/acl — so a
# caller cannot assert its own access (mirrors the QueryRequest keystone).
# The activation gating group is read from the resolved AccessContext, not
# from the body.
# ---------------------------------------------------------------------------


class VramBudgetOut(BaseModel):
    """The server-computed VRAM budget assessment for one catalog candidate.

    Invariant 7 (F-00096): the math lives server-side; the console
    only renders.  The triple is derived from the candidate's catalog
    ``expected_vram_gb`` + the always-resident Infinity footprint +
    the 32 GB GPU ceiling.

    Boundary cases (AC1 / Boundary row *"VRAM over budget"*):

    * A candidate whose ``expected_vram_gb + resident Infinity``
      equals the 32 GB ceiling **exactly** reports
      ``fits=True`` and ``headroom_gb=0.0``.
    * A candidate whose footprint exceeds the ceiling by any
      positive delta reports ``fits=False`` and a negative
      ``headroom_gb``.
    * A candidate in the unreachable placeholder row carries
      ``fits=True``, ``budget_used_gb=0.0``, ``headroom_gb=0.0`` —
      the UI treats these as "no measurement" (the swap-controller
      is down; the operator must wait or re-activate).  Per
      F-00094 Boundary row 3 the route MUST NOT 5xx.

    Attributes:
        fits: ``True`` when ``budget_used_gb <= ceiling_gb``
            (equivalently ``headroom_gb >= 0``).
        budget_used_gb: ``resident_infinity_gb + expected_vram_gb``
            (rounded to the same float precision the route computes;
            not rounded on the wire).
        headroom_gb: ``ceiling_gb - budget_used_gb``.  Negative when
            the candidate would not fit.
    """

    fits: bool
    budget_used_gb: float
    headroom_gb: float


class CatalogModelOut(BaseModel):
    """A single catalog row annotated with its current state.

    Attributes:
        id: The catalog id.
        served_name: The HF repo id passed to vLLM.
        quantization: The quantisation scheme (v1: ``"awq"``).
        expected_vram_gb: The expected VRAM budget for the model.
        max_context: vLLM ``--max-model-len``.
        state: One of ``"available"``, ``"loading"``, ``"active"``,
            ``"unreachable"``.  ``unreachable`` is the fallback when the
            swap-controller sidecar (or, behind it, llama-swap) is down.
        vram_budget: The server-computed VRAM budget assessment for the
            candidate (Invariant 7: the math lives server-side, the
            console only renders).  ``VramBudgetOut`` carries the
            ``fits`` / ``budget_used_gb`` / ``headroom_gb`` triple
            derived from this row's ``expected_vram_gb`` + the
            always-resident Infinity footprint + the 32 GB GPU ceiling.
    """

    id: str
    served_name: str
    quantization: str
    expected_vram_gb: float
    max_context: int
    state: str
    vram_budget: VramBudgetOut


class ModelsResponse(BaseModel):
    """Response body for ``GET /models``.

    Attributes:
        models: The catalog annotated with per-model state.
        active_model_id: The id of the currently resident generator, or
            ``None`` when the swap-controller is idle or unreachable.
    """

    models: list[CatalogModelOut]
    active_model_id: str | None = None


class ActivateRequest(BaseModel):
    """The body of ``POST /models/activate``.

    Deliberately carries NO acl/groups field — the keystone is structural
    (mirrors :class:`QueryRequest`).  The eval-operator gating is read from
    the resolved :class:`~iw_rag.domain.acl.AccessContext`, not from the
    request body.

    Attributes:
        model_id: The catalog id to activate.
    """

    model_id: str = Field(min_length=1, max_length=128)


class ActivateResponse(BaseModel):
    """Response body for ``POST /models/activate``.

    Mirrors the swap-controller sidecar's status payload; on success the
    ``state`` is ``"ready"`` and ``active_model_id`` names the resident
    generator.  On failure ``state`` is ``"failed"`` and ``reason`` carries
    the cause (``"timeout"`` or ``"error"``).

    Attributes:
        state: The activation state (``"idle"`` / ``"swapping"`` /
            ``"ready"`` / ``"failed"``).
        active_model_id: The id of the resident generator after the
            activation attempt (mirrors the sidecar's
            ``active_model_id`` field).  ``None`` when the sidecar
            surfaces a stateless error.
        reason: Optional failure reason (``"timeout"`` or ``"error"``);
            ``None`` for success paths.
    """

    state: str
    active_model_id: str | None = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# F-00095 — Benchmark Studio (POST /benchmark/run, GET reports / compare)
#
# Mirror the ``QueryRequest`` / ``ActivateRequest`` keystone: the
# ``BenchmarkRunRequest`` body carries NO groups / acl field — the
# eval-operator gating is read from the resolved ``AccessContext``
# (auth-only), never from the request body (Invariant 3, AC2).  The
# ``compare`` response carries the per-metric delta table the S03
# Studio page renders, plus the ``winner`` flag and the n≈37
# underpowered caveat (AC3).
# ---------------------------------------------------------------------------


class BenchmarkRunRequest(BaseModel):
    """The body of ``POST /benchmark/run``.

    Carries the per-step ``model_config`` snapshot, the run parameters,
    and the golden dataset id.  Deliberately carries **no** groups /
    acl / ``principal_groups`` field — the eval-operator gate is read
    from the resolved :class:`~iw_rag.domain.acl.AccessContext`, never
    from the request body (mirrors :class:`QueryRequest` /
    :class:`ActivateRequest` — the structural keystone).

    Attributes:
        generator: Logical name of the generator in
            ``config/model_catalog.yaml`` (the F-00094 catalog id).
        embedder: Logical name of the embedder in ``models.yaml``.
        reranker: Logical name of the reranker in ``models.yaml``.
        grader: Logical name of the grader judge in ``models.yaml``.
        dataset_id: The golden dataset id to evaluate against (default
            ``"benchmark_golden"`` matches the S01 runner default).
        params: A free-form per-run parameter bag (top_k, temperature,
            max_tokens, etc.).  Forwarded verbatim to the runner and
            persisted on the row.
    """

    generator: str = Field(min_length=1, max_length=128)
    embedder: str = Field(min_length=1, max_length=128)
    reranker: str = Field(min_length=1, max_length=128)
    grader: str = Field(min_length=1, max_length=128)
    dataset_id: str = Field(default="benchmark_golden", min_length=1, max_length=128)
    params: dict[str, Any] = Field(default_factory=dict)


class BenchmarkReportOut(BaseModel):
    """A single ``benchmark_reports`` row as surfaced to the Studio.

    Mirrors the F-00093 storage schema's JSONB columns and the
    promoted float columns.  A row's ``id`` is the persisted report's
    uuid-stringified form (the canonical id the S03 page links to).

    Note: ``model_config`` is the F-00093 column name on the storage
    row.  On the API surface we expose it as ``config_snapshot`` to
    avoid colliding with Pydantic's reserved ``model_config`` class
    attribute (which would break field detection); the S03 page maps
    it back to ``model_config`` for display.

    The ``metrics`` JSONB column is the per-query vector form
    (``{"ndcg_at_5": [0.5, 0.6, 0.7], ...}``) the F-00093 stats
    module uses for paired A/B tests.  The S03 page reads **scalar**
    values, so the API also surfaces the promoted scalar columns
    (``recall_at_5``, ``ndcg_at_5``, ``mrr``, ``faithfulness_mean``,
    ``judge_kappa``, etc.) directly so the page can render metric
    tiers without needing to mean-reduce vectors client-side.

    Attributes:
        id: The report's id.
        run_id: The :class:`eval_runs` id this report rolls up.
        dataset_id: The :class:`eval_datasets` id.
        report_label: Human-readable label.
        created_by: The authenticated principal id (auth-only).
        config_snapshot: The static model configuration snapshot
            (the F-00093 ``model_config`` JSONB column).
        n_queries: The number of golden queries in the run.
        n_judged: The number of unique queries sent to a judge.
        metrics: The full per-tier metrics blob (per-query vectors
            for the F-00093 stats module's paired A/B tests).
        recall_at_5: T1 promoted mean (retrieval recall@5).
        recall_at_10: T1 promoted mean (retrieval recall@10).
        ndcg_at_5: T1 promoted mean (retrieval nDCG@5).
        ndcg_at_10: T1 promoted mean (retrieval nDCG@10).
        mrr: T1 promoted mean (retrieval MRR).
        faithfulness_mean: T2 promoted mean (judged faithfulness).
        faithfulness_kappa: T2 judge κ (inter-judge agreement).
        context_precision: T2 promoted mean (judged context precision).
        context_recall: T3 promoted mean (judged context recall;
            ``None`` when the dataset has no reference answers).
        factual_correctness: T3 promoted mean (judged factual
            correctness; ``None`` when the dataset has no reference
            answers).
        latency: The latency JSONB (per-stage ms + ``model_load_ms``).
        resource: The resource JSONB (GPU + vLLM server).
        ci_95: The per-metric bootstrap 95% CI dict, or ``None``.
        created_at: ISO-8601 timestamp the row was inserted.
    """

    id: str
    seq: int | None = None
    display_id: str = ""
    run_id: str | None = None
    dataset_id: str | None = None
    # Human-readable dataset identity resolved from ``eval_datasets`` (the
    # row stores only the opaque uuid ``dataset_id``). ``None`` when the FK
    # is dangling — display sites then fall back to the uuid.
    dataset_name: str | None = None
    dataset_version: str | None = None
    report_label: str = ""
    created_by: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    n_queries: int = 0
    n_judged: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    # T1 + T2 + T3 promoted scalar columns (F-00093 S06 storage schema).
    # The S03 page reads these directly so it can render the metric
    # tiers without re-implementing vector→mean reduction client-side.
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    ndcg_at_5: float | None = None
    ndcg_at_10: float | None = None
    mrr: float | None = None
    faithfulness_mean: float | None = None
    faithfulness_kappa: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    factual_correctness: float | None = None
    latency: dict[str, Any] | None = None
    resource: dict[str, Any] | None = None
    ci_95: dict[str, Any] | None = None
    created_at: str = ""


class BenchmarkCompareDeltaOut(BaseModel):
    """One row of the A/B compare delta table.

    Mirrors the F-00093 stats module's :class:`MetricDelta` row, plus a
    flattened ``p_corrected` / ``ci_95`` pair for the page.

    Attributes:
        metric: The metric name (e.g. ``"ndcg_at_5"``, ``"faithfulness"``).
        val_a: The mean of A across the paired observations.
        val_b: The mean of B across the paired observations.
        delta: ``val_b - val_a``; positive means B is better.
        p_value: The raw (uncorrected) p-value of the paired test.
        p_corrected: The Bonferroni-Holm-corrected p-value.
        ci: The 95% bootstrap CI of the mean difference as ``(lo, hi)``.
        significant: ``True`` when ``p_corrected < 0.05`` AND the CI
            excludes 0.
    """

    metric: str
    val_a: float | None = None
    val_b: float | None = None
    delta: float | None = None
    p_value: float | None = None
    p_corrected: float | None = None
    ci: tuple[float, float] | None = None
    significant: bool = False


class BenchmarkCompareResponse(BaseModel):
    """The body of ``GET /benchmark/compare?a=..&b=..``.

    Carries the per-metric delta table, the ``winner`` flag (set only
    when one config is better on BOTH nDCG@5 AND faithfulness), the
    underpowered caveat (the n≈37 / n≪150 detectable-effect-size
    warning, AC3), and the A/B dataset ids for client-side correlation.

    Attributes:
        a: The A report's id.
        b: The B report's id.
        dataset_id: The shared dataset id both reports cover.
        rows: The per-metric :class:`BenchmarkCompareDeltaOut` rows.
        winner: ``"A"`` or ``"B"`` when the dual-metric rule is
            satisfied, else ``None``.
        underpowered: ``True`` when n is below the significance floor.
        caveat: Human-readable caveat describing the underpowered
            state + detectable effect size.
        reasoning: A human-readable summary of the comparison.
    """

    a: str
    b: str
    dataset_id: str | None = None
    rows: list[BenchmarkCompareDeltaOut] = Field(default_factory=list)
    winner: str | None = None
    underpowered: bool = False
    caveat: str | None = None
    reasoning: str = ""


# ---------------------------------------------------------------------------
# N-way (2..5 run) multi-compare — the Compare view + its PDF
# ---------------------------------------------------------------------------


class BenchmarkCompareSampleOut(BaseModel):
    """One run's persisted sample for a single compared item.

    The generated answer + retrieved contexts the Compare view reveals
    in the expandable per-item qualitative diff.  ``None`` (at the item
    level) when a run pre-dates sample persistence.

    Attributes:
        answer: The run's generated answer for this item (truncated at
            persistence time), or ``None``.
        contexts: The retrieved context strings handed to the generator
            (truncated + capped at persistence time).
    """

    answer: str | None = None
    contexts: list[str] = Field(default_factory=list)


class BenchmarkCompareRunOut(BaseModel):
    """One run (column) in the N-way compare, with its identity + stats.

    Attributes:
        id: The report id.
        display_id: The stable ``BENCH-000007`` execution number.
        label: The human report label.
        is_baseline: ``True`` for the single baseline run.
        color: The stable palette colour assigned to this run, reused in
            every chart of the view + the PDF.
        config_snapshot: The module composition snapshot (generator /
            embedder / reranker / grader / dataset).
        dataset_name: The resolved human dataset name (or ``None``).
        created_at: ISO-8601 run timestamp.
        n_queries: The run's query count.
        means: Per-metric mean for this run (``None`` when absent).
        deltas: The baseline-relative per-metric delta rows (empty for
            the baseline run itself).
    """

    id: str
    display_id: str = ""
    label: str = ""
    is_baseline: bool = False
    color: str = ""
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    dataset_name: str | None = None
    created_at: str = ""
    n_queries: int = 0
    means: dict[str, float | None] = Field(default_factory=dict)
    deltas: list[BenchmarkCompareDeltaOut] = Field(default_factory=list)


class BenchmarkCompareItemOut(BaseModel):
    """One compared item (row) across all runs.

    Attributes:
        index: The 0-based query index (the cross-run pairing key).
        question: The item's question text (``None`` when no run
            persisted a sample).
        values: Per-metric per-run value list, aligned to the run order
            (``None`` where a run has no value for that item/metric).
        samples: Per-run sample (answer + contexts), aligned to the run
            order (``None`` where a run persisted none).
    """

    index: int
    question: str | None = None
    values: dict[str, list[float | None]] = Field(default_factory=dict)
    samples: list[BenchmarkCompareSampleOut | None] = Field(default_factory=list)


class BenchmarkMultiCompareResponse(BaseModel):
    """The body of ``GET /benchmark/compare/multi?ids=..&baseline=..``.

    Carries the per-run identity + colours, the per-run baseline-relative
    delta tables, the per-item value matrix + qualitative samples, and the
    shared underpowered caveat.

    Attributes:
        dataset_id: The single dataset all runs share.
        baseline_id: The id of the baseline run.
        n_queries: The paired query count.
        underpowered: ``True`` when n is below the significance floor.
        caveat: The shared underpowered caveat, or ``None``.
        metrics: The metric names surfaced (means + per-item matrix).
        runs: The compared runs, in selection order (includes baseline).
        items: The per-item rows.
    """

    dataset_id: str | None = None
    baseline_id: str = ""
    n_queries: int = 0
    underpowered: bool = False
    caveat: str | None = None
    metrics: list[str] = Field(default_factory=list)
    runs: list[BenchmarkCompareRunOut] = Field(default_factory=list)
    items: list[BenchmarkCompareItemOut] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# F-00096 — Operator Console Config API (GET /api/config,
# PUT /api/config/{key})
#
# The ``ConfigUpdateRequest`` body carries ONLY ``value`` — NEVER
# ``principal_id`` / ``groups`` / ``acl`` — mirroring the
# ``QueryRequest`` / ``ActivateRequest`` / ``BenchmarkRunRequest``
# keystone (Invariant 3, AC2, AC7).  The eval-operator gate is read
# from the resolved :class:`~iw_rag.domain.acl.AccessContext`, never
# from the request body.
# ---------------------------------------------------------------------------


class ConfigItemOut(BaseModel):
    """One config key as surfaced to the Operator Console.

    Mirrors the S03 :class:`~iw_rag.api.config_service.ActiveConfigEntry`
    fields.  Locked keys' ``value`` field carries the redacted
    placeholder (e.g. ``"***"`` for ``database.dsn``); the live secret
    is never echoed (Invariant 5; design Notes "no raw secrets").
    The ``min`` / ``max`` / ``choices`` bounds metadata is surfaced
    only when the registry entry declares it (so a non-range / non-enum
    key simply omits the field rather than carrying a meaningless
    ``null``).

    Attributes:
        key: The dotted config key (e.g. ``"retrieval.top_k"``).
        value: The active resolved value (redacted for secret-shaped
            locked keys; the raw secret is never surfaced).
        explanation: The per-key one-line description from the S03
            registry (a generic ``"(locked) — ..."`` for locked keys).
        editable: ``True`` when the key is in the editable allowlist
            and NOT in the locked set (i.e. the operator can change it).
        locked: ``True`` when the key is in the locked set (display-only).
        pending_restart: ``True`` when the key is editable but a
            startup-baked client owns it — the edit will apply after
            restart (AC4).
        min: The lower bound (range-style keys only); ``None`` when the
            entry is not a numeric range.
        max: The upper bound (range-style keys only); ``None`` when the
            entry is not a numeric range.
        choices: The permitted string values (enum-style keys only);
            ``None`` when the entry is not an enum.
    """

    key: str
    value: Any
    explanation: str
    editable: bool
    locked: bool
    pending_restart: bool
    min: float | int | None = None
    max: float | int | None = None
    choices: list[str] | None = None


class ConfigResponse(BaseModel):
    """The body of ``GET /api/config``.

    Attributes:
        items: One :class:`ConfigItemOut` per registered key (editable
            allowlist + locked set, in registry-stable order).
    """

    items: list[ConfigItemOut] = Field(default_factory=list)


class ConfigUpdateRequest(BaseModel):
    """The body of ``PUT /api/config/{key}``.

    Carries the new value ONLY.  Deliberately carries NO
    ``principal_id`` / ``groups`` / ``acl`` / ``principal_groups``
    field — the eval-operator gating is read from the resolved
    :class:`~iw_rag.domain.acl.AccessContext`, never from the request
    body (mirrors :class:`QueryRequest` / :class:`ActivateRequest` /
    :class:`BenchmarkRunRequest` — the structural keystone, Invariant 3).

    Attributes:
        value: The new value.  Type-checked + bounds-checked +
            Settings-validated by the S03 service before apply.
    """

    value: Any


class ConfigUpdateResponse(BaseModel):
    """The body of ``PUT /api/config/{key}`` on success.

    Attributes:
        key: The dotted config key that was edited.
        applied_mode: ``"hot"`` (the next per-request ``get_settings()``
            returns the new value) or ``"pending_restart"`` (a
            startup-baked client owns the key — restart required).
        old_value: The prior value (as observed before the apply).
        new_value: The new value that was applied.
    """

    key: str
    applied_mode: str
    old_value: Any = None
    new_value: Any = None


# ---------------------------------------------------------------------------
# F-00096 — Operator Console Scoreboard (GET /benchmark/scoreboard)
#
# The scoreboard is the offline ranking of the ``benchmark_reports``
# history across six per-metric leaderboards plus a deterministic
# composite.  The S05 ``ScoreboardService`` (in
# :mod:`iw_rag.eval.benchmark.scoreboard`) is the **sole** authority on
# the per-row ordering; the S08 route is a thin HTTP facade over it.
# The field set mirrors the S05 :class:`ScoreboardEntry` row shape
# (``report_id``, ``model_config``, ``created_at``, ``metric_value``)
# plus a 1-based ``rank`` the API assigns as it serialises.
# ---------------------------------------------------------------------------


class ScoreboardEntryOut(BaseModel):
    """One ranked report row on a scoreboard leaderboard.

    Mirrors the S05 :class:`ScoreboardEntry` row shape
    (``report_id``, ``config_snapshot``, ``created_at``, ``metric_value``)
    plus a 1-based ``rank`` the route assigns in the order the
    service emitted (the service is the sole authority on the
    ordering — AC6: descending faithfulness).  The console renders
    one row per entry; ``metric_value`` is ``None`` when the source
    column / JSONB key was NULL (the UI renders the
    "no measurement" state per row).

    Note: the F-00093 storage row's JSONB column is ``model_config``;
    on the API surface we expose it as ``config_snapshot`` to avoid
    colliding with Pydantic's reserved ``model_config`` class
    attribute (which would break field detection).  The S14 page
    maps it back to ``model_config`` for display (mirrors
    :class:`BenchmarkReportOut`).

    Attributes:
        report_id: The report's id.
        rank: The 1-based rank assigned by the route in the order the
            service emitted (best row = 1).
        config_snapshot: The static model configuration snapshot
            (JSONB from the row).
        created_at: The run timestamp (string or datetime — the S05
            service round-trips the row's value verbatim).
        metric_value: The numeric value of the board's metric, or
            ``None`` when the source was NULL.
    """

    report_id: str
    seq: int | None = None
    display_id: str = ""
    rank: int
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    # Human dataset name resolved from ``eval_datasets`` (e.g. ``benchmark_golden``);
    # ``""`` when unresolved — the page falls back to the snapshot's uuid.
    dataset_name: str = ""
    created_at: str = ""
    metric_value: float | None = None


class ScoreboardResponse(BaseModel):
    """The body of ``GET /benchmark/scoreboard?board=<name>``.

    The route forwards the ``board`` query string to the S05 service
    and selects the matching per-board list (``faithfulness``,
    ``ndcg_at_5``, ``tok_per_s``, ``latency``, ``vram_efficiency``,
    ``composite``).  With **0 persisted reports** the service returns
    an empty list per board; the route surfaces it verbatim (Boundary
    row *"Empty report history → explicit empty state, not an error"*).

    Attributes:
        board: The board name echoed back to the caller (one of the
            six S05 leaderboards; never ``None``).
        entries: The per-row :class:`ScoreboardEntryOut` list, ordered
            best-first per the S05 service's direction (descending
            for ``faithfulness`` / ``ndcg_at_5`` / ``tok_per_s`` /
            ``vram_efficiency`` / ``composite``; ascending for
            ``latency``).
    """

    board: str
    entries: list[ScoreboardEntryOut] = Field(default_factory=list)
