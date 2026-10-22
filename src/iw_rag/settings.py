"""Application settings — single typed Settings object with model registry.

Provides one authoritative ``Settings`` instance loaded at startup.  All
configuration flows through this module; nothing else hardcodes endpoints,
ports, or model names.  Layered precedence: ``config/default.yaml`` (lowest) <
``.env`` < process env (highest), consistent with 12-factor / lld-config §4.

The model registry (``config/models.yaml``) maps logical model roles to served
endpoints.  Endpoint values may contain ``${ENV_VAR}`` placeholders resolved at
load time from process env.  A missing required value raises ``ConfigError``
immediately so misconfig surfaces before any traffic arrives.

Usage::

    from iw_rag.settings import get_settings

    cfg = get_settings()
    entry = cfg.models.get("generator")   # ModelEntry

Import budget: pydantic / pydantic-settings / stdlib + domain.errors only.
No other iw_rag layer may be imported here (this module is the base of the
dependency cone).
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from iw_rag.domain.errors import ConfigError

# Canonical env prefix + nested delimiter exposed as module constants so tests
# and tooling can reference them without re-importing Settings.
ENV_PREFIX = "IW_RAG__"
ENV_NESTED_DELIMITER = "__"


# ---------------------------------------------------------------------------
# Overlay slot — registered by iw_rag.api.config_service (live-edit overlay)
# ---------------------------------------------------------------------------
# :class:`Settings` consults a single registered callable for the
# live-edit overlay (F-00096).  The callable returns a nested init-
# kwargs dict (e.g. ``{"guardrails": {"input": {"threshold": 0.7}}}``)
# that Pydantic type-coerces and validates on the way in.  Layering
# note: this module is the base of the dependency cone and cannot
# import :mod:`iw_rag.api.config_service`; the API module imports THIS
# module and registers the provider at import time via
# :func:`set_overlay_source`.
OverlaySourceFn = Callable[[], dict[str, Any]]
_overlay_source: OverlaySourceFn | None = None


def set_overlay_source(fn: OverlaySourceFn | None) -> None:
    """Register (or clear) the overlay provider called by the Settings source.

    Args:
        fn: A zero-arg callable returning a nested init-kwargs dict, or
            ``None`` to clear the registration.  Called once per
            :class:`Settings` construction (which itself happens once
            per ``cache_clear()`` cycle).
    """
    global _overlay_source
    _overlay_source = fn


def _get_overlay_kwargs() -> dict[str, Any]:
    """Return the current overlay's nested init-kwargs (empty if unset)."""
    if _overlay_source is None:
        return {}
    return _overlay_source()


class _OverlaySettingsSource(InitSettingsSource):
    """Settings source for the F-00096 live-edit overlay.

    Reads the registered overlay callable (see :func:`set_overlay_source`)
    and returns its nested init-kwargs (Pydantic handles the
    nested-model init-kwargs shape).  Subclasses
    :class:`InitSettingsSource` rather than
    :class:`PydanticBaseSettingsSource` directly because the init
    source already implements the abstract ``get_field_value`` hook
    the way pydantic-settings expects, and the overlay's values are
    init-style kwargs (no env-var coercion needed).

    Sits between ``init_settings`` and ``env_settings`` in the source
    priority (i.e. overlay > env > .env > config/default.yaml) — the
    live edit should always win over the env-baked value when the
    overlay has an entry.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        """Bind to the Settings class (init kwargs are read lazily in ``__call__``)."""
        super().__init__(settings_cls, init_kwargs={})

    def __call__(self) -> dict[str, Any]:
        """Return the overlay's nested init-kwargs (rooted at Settings fields)."""
        return _get_overlay_kwargs()


# ---------------------------------------------------------------------------
# Sub-settings models (lld-config §2)
# ---------------------------------------------------------------------------


class DatabaseSettings(BaseModel):
    """Postgres connection and pool settings.

    Attributes:
        dsn: Full psycopg DSN — secret, must come from env only.
        pool_min: Minimum pool connections kept alive.
        pool_max: Maximum pool connections.
        statement_timeout_ms: Per-statement timeout in milliseconds.
    """

    dsn: str
    pool_min: int = 1
    pool_max: int = 10
    statement_timeout_ms: int = 30_000


class IngestionSettings(BaseModel):
    """Offline ingestion pipeline configuration (lld-ingestion §7).

    Attributes:
        parser_default: Default document parser name.
        parser_hard: Fallback heavy parser for complex PDFs.
        force_parser: Override both parsers when set.
        leaf_tokens: Target token count for leaf chunks.
        parent_strategy: How parent chunks are defined.
        late_chunking: Enable late-chunking embeddings.
        contextual_retrieval: Enable contextual-retrieval enrichment.
        embed_window_tokens: Context window for late-chunking embedding calls.
        sparse_arm: Which sparse retrieval arm to use by default.
        embed_retries: Number of embedding retries on transient failure.
        parse_timeout_s: Per-document parse timeout in seconds.
        concurrency: Number of concurrent ingest workers.
        supersede: When True, re-ingest replaces existing document rows.
        online_chunk_leaf_tokens: Target leaf size for the online upload path
            (I-00131).  Larger than the offline default because the online
            path uses ``CharTokenizer`` (1 token == 1 char) and the Late-
            Chunking embedder pools at the doc level anyway.  Picked to give
            reasonable leaf granularity on user-uploaded PDFs without
            exploding the leaf count.
    """

    parser_default: str = "docling"
    parser_hard: str = "mineru"
    force_parser: str | None = None
    leaf_tokens: int = 384
    parent_strategy: Literal["section", "window"] = "section"
    late_chunking: bool = True
    contextual_retrieval: bool = False
    embed_window_tokens: int = 8192
    sparse_arm: Literal["bm25", "learned_sparse"] = "bm25"
    embed_retries: int = 3
    parse_timeout_s: int = 120
    concurrency: int = 2
    supersede: bool = False
    online_chunk_leaf_tokens: int = 1600


class RetrievalSettings(BaseModel):
    """Retrieval spine configuration (lld-retrieval-spine §9).

    Attributes:
        candidate_k: Number of candidates retrieved from each arm.
        fused_k: Fused candidate budget after RRF (before rerank).
        top_k: Final top-k returned to the generator.
        fusion_type: Fusion algorithm — rrf (Reciprocal Rank Fusion) or rsf.
        rrf_k: The RRF constant k; canonical default is 60 (lld-config).
        rrf_weights: Weights for (dense, sparse) arms in RRF.
        sparse_arm: Which sparse arm to use at retrieval time.
        expansion_mode: Parent-doc expansion strategy.
        parent_expansion_max_chars: Per-candidate char budget enforced during
            small-to-big parent expansion.  A leaf is expanded to its parent's
            full content only when the parent fits within this budget;
            otherwise the leaf's own content is retained (matched text is
            always present, no candidate ever exceeds the budget).  Sized so
            that ``top_k * (parent_expansion_max_chars / 4 chars-per-token) +
            ~2k system/question/JSON-schema overhead`` stays well under the
            generator's ``GenerationSettings.max_model_len`` (16,384 by
            default) — see I-00131 design §"Budget sizing constraint".
        rerank_on_failure: Behaviour when the reranker is unavailable.
        rerank_max_doc_chars: Truncate each candidate's text to this many chars
            before cross-encoder scoring.  The reranker judges relevance from the
            head; an oversized whole-doc candidate otherwise pushed rerank
            latency to a flat ~30s/query (bge-reranker chewing past its ~8k
            window).  ``<= 0`` disables truncation.
        rerank_model: Logical name of the reranker in models.yaml.
        embed_model: Logical name of the embedder in models.yaml.
    """

    candidate_k: int = 40
    fused_k: int = 80
    top_k: int = 8
    fusion_type: Literal["rrf", "rsf"] = "rrf"
    rrf_k: int = 60
    rrf_weights: tuple[float, float] = (1.0, 1.0)
    sparse_arm: Literal["bm25", "learned_sparse"] = "bm25"
    expansion_mode: Literal["leaf", "parent", "window"] = "parent"
    parent_expansion_max_chars: int = 4000
    rerank_on_failure: Literal["degrade", "fail"] = "degrade"
    rerank_max_doc_chars: int = 2048
    rerank_model: str = "reranker"
    embed_model: str = "embedder"


class GenerationSettings(BaseModel):
    """Answer-generation configuration (lld-generation §7).

    Attributes:
        model: Logical model name → models.yaml.
        temperature: Sampling temperature.
        max_tokens: Max tokens in generated answer.
        max_model_len: Max context window length for the served model.
        context_headroom_tokens: Tokens reserved, on top of ``max_tokens``, for
            the parts of every prompt that are NOT budgeted chunk content — the
            system prompt, the question, the chat-template scaffolding, plus a
            safety margin.  Token-budgeting (``generation.context_budget``) fits
            retrieved chunks into ``max_model_len - max_tokens -
            context_headroom_tokens`` so the assembled prompt can never overflow
            the window (I-CTXOVERFLOW).  2048 comfortably covers the grader /
            citation system prompts (~250 tokens) and the question.
        enable_thinking: Enable chain-of-thought reasoning tokens.
        cite_required: Require citation markers in generated answers.
        prompt_version: Template version string.
        gen_retries: Number of generation retries on failure.
        tool_call_parser: vLLM tool-call parser name.
    """

    model: str = "generator"
    temperature: float = 0.1
    max_tokens: int = 1024
    max_model_len: int = 16384
    context_headroom_tokens: int = 2048
    enable_thinking: bool = False
    cite_required: bool = True
    prompt_version: str = "v1"
    gen_retries: int = 2
    tool_call_parser: str = "hermes"


class DemoUserCfg(BaseModel):
    """A demo user record for username/password login + token auth (WI-DOCMGMT).

    Used by :class:`~iw_rag.api.auth.DemoUserAuthenticator` to support a real
    document-management login: a username/password resolves to a stable bearer
    ``token`` and a stable ``principal_id`` (the username), so uploaded documents
    have a durable owner.  The authenticator always augments ``groups`` with
    ``public`` and ``user:<username>`` so the per-user ACL (``specific`` →
    ``user:<owner>``, ``common`` → ``public``) works without configuring those
    synthetic groups by hand.

    Attributes:
        password: The demo password (plaintext — demo auth only; OIDC in prod).
        token: The bearer token issued on successful login.
        groups: The user's real entitlement groups (e.g. ``["finance"]``).
        operator: Whether this user may invoke operator-only routes.
    """

    password: str
    token: str
    groups: list[str] = []
    operator: bool = False


class ApiSettings(BaseModel):
    """FastAPI delivery layer settings (lld-api §6).

    Attributes:
        host: Bind address for the API server.
        port: TCP port for the API server.
        auth_mode: Authentication mode.
        request_timeout_s: Per-request timeout in seconds.
        static_tokens: Legacy token→groups demo map (StaticTokenAuthenticator).
        operator_tokens: Which static tokens are operators.
        demo_users: Username→DemoUserCfg map for the richer login-capable demo
            auth (DemoUserAuthenticator).  When non-empty it supersedes
            ``static_tokens`` so principals have a stable ``principal_id`` (the
            username) for document ownership.  Provided as JSON via
            ``IW_RAG__API__DEMO_USERS``.
        upload_dir: Filesystem directory where uploaded document bytes are
            staged (so the ingestion pipeline can read them and the view/download
            endpoint can serve them back).
        max_upload_mb: Maximum accepted upload size in megabytes.
        ephemeral_ttl_minutes: Lifetime of a chat-attachment session before the
            TTL sweep purges its ephemeral chunks.
        download_token_secret: HMAC-SHA256 secret used to mint and verify
            signed download tokens for the CR-00098 ``/documents/{id}/raw?token=``
            path.  **Secret, env-only** — provided via
            ``IW_RAG__API__DOWNLOAD_TOKEN_SECRET`` (mirrors the
            ``DatabaseSettings.dsn`` "secret, env only" convention; do NOT
            commit a real value to ``config/default.yaml``).  An empty/blank
            value is the **fail-safe**: ``verify`` rejects every token
            (the feature is off) and ``mint`` raises (the caller decides
            whether to surface a plain URL or omit the link).
        download_token_ttl_seconds: Lifetime in seconds of a freshly minted
            download token (default 300 s = 5 minutes).  The TTL is
            surfaced in ``config/default.yaml`` (this is a non-secret
            knob); only the secret is env-only.  Shorter values reduce
            the exposure window of a leaked token; longer values trade
            that for a longer browser-handle window.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    auth_mode: Literal["oidc", "static_token", "mtls", "trusted_header"] = "static_token"
    request_timeout_s: int = 60
    # static_token demo auth: token → group tags, plus which tokens are operators.
    # Configured (never inferred); provided as JSON via IW_RAG__API__STATIC_TOKENS.
    static_tokens: dict[str, list[str]] = {}
    operator_tokens: list[str] = []
    # Richer login-capable demo auth (document management).
    demo_users: dict[str, DemoUserCfg] = {}
    upload_dir: str = "data/uploads"
    max_upload_mb: int = 50
    ephemeral_ttl_minutes: int = 240
    # CR-00098 / S02 — signed download-token primitive.
    # ``download_token_secret`` is env-only (mirror the DatabaseSettings.dsn
    # convention); ``download_token_ttl_seconds`` is surfaced in default.yaml.
    download_token_secret: str = ""
    download_token_ttl_seconds: int = 300


class ObservabilitySettings(BaseModel):
    """Phoenix/OTEL observability settings (lld-eval-observability).

    Attributes:
        phoenix_endpoint: OTLP endpoint for Arize Phoenix (None = disabled).
        trace_sample_rate: Fraction of traces to export (1.0 = all).
        log_level: Root log level string.
    """

    phoenix_endpoint: str | None = None
    trace_sample_rate: float = 1.0
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Model registry (lld-config §3)
# ---------------------------------------------------------------------------


class ModelEntry(BaseModel):
    """A single entry in the model registry mapping a logical role to a server.

    Attributes:
        served_name: The model identifier on the serving engine (e.g. HF repo id).
        role: Functional role of the model.
        engine: Serving engine identifier.  ``"infinity"`` is the sanctioned
            Blackwell (sm_120) fallback for TEI (Task 0.0 finding 2026-06-06).
        endpoint: Resolved HTTP endpoint for the model server.
        dims: Embedding dimensionality — required for embedder role.
        quantization: Quantisation scheme (awq, gptq, …) when applicable.
        kv_cache_dtype: KV cache dtype override for vLLM (fp8, auto, …).
        extra: Additional model-specific keys (e.g. tool_call_parser, emits).
    """

    served_name: str
    role: Literal[
        "generation", "embedding", "rerank", "judge", "guard", "visual_caption", "graph_extraction"
    ]
    # "infinity" is the Blackwell Q-TEI fallback alongside the canonical tei
    engine: Literal["vllm", "tei", "ollama", "transformers", "infinity"]
    endpoint: str
    dims: int | None = None
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    extra: dict[str, Any] = {}


class ModelRegistry(BaseModel):
    """Registry mapping logical model names to their ``ModelEntry`` records.

    Attributes:
        models: Dict of logical name → entry; loaded from config/models.yaml.
    """

    models: dict[str, ModelEntry]

    def get(self, logical_name: str) -> ModelEntry:
        """Resolve a logical model name to its entry.

        Args:
            logical_name: Logical role key (e.g. ``"generator"``).

        Returns:
            The resolved ``ModelEntry``.

        Raises:
            ConfigError: If the logical name is not present in the registry.
        """
        if logical_name not in self.models:
            raise ConfigError(
                f"Unknown model logical name '{logical_name}'. Available: {sorted(self.models)}"
            )
        return self.models[logical_name]


# ---------------------------------------------------------------------------
# Env-variable interpolation helper for models.yaml
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _interpolate_env(value: str) -> str:
    """Expand ``${VAR}``, ``${VAR:-default}``, or ``$VAR`` placeholders from env.

    ``${VAR:-default}`` resolves to ``default`` when ``VAR`` is unset — used for
    eval-only endpoints (judges) so the hot-path startup (API, ingest) does not
    require them (two-clocks). A bare ``${VAR}`` with no default stays strict
    (unset → ConfigError), which is how the hot-path generator/embedder/reranker
    endpoints behave.

    Args:
        value: A string potentially containing placeholders.

    Returns:
        The string with all env placeholders expanded.

    Raises:
        ConfigError: If a bare (default-less) referenced env variable is unset.
    """

    def replacer(match: re.Match[str]) -> str:
        token = match.group(1) or match.group(2)
        if match.group(1) is not None and ":-" in token:
            var_name, default = token.split(":-", 1)
            return os.environ.get(var_name, default)
        val = os.environ.get(token)
        if val is None:
            raise ConfigError(
                f"Endpoint env variable '{token}' referenced in "
                f"config/models.yaml is not set. "
                f"Add IW_RAG__MODELS__*__ENDPOINT to .env or process env."
            )
        return val

    return _ENV_PATTERN.sub(replacer, value)


def _interpolate_dict(data: Any) -> Any:
    """Recursively interpolate env placeholders in a dict/list/str structure.

    Args:
        data: YAML-decoded Python object (dict, list, str, …).

    Returns:
        The same structure with all string values interpolated.
    """
    if isinstance(data, dict):
        return {k: _interpolate_dict(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_interpolate_dict(item) for item in data]
    if isinstance(data, str):
        return _interpolate_env(data)
    return data


def _load_model_registry(registry_path: str) -> ModelRegistry:
    """Load and validate config/models.yaml into a ModelRegistry.

    Performs env-variable interpolation on endpoint fields before validation
    so unresolved endpoints surface as a ConfigError at startup.

    Args:
        registry_path: Path to the models.yaml file (relative to cwd or abs).

    Returns:
        A validated ``ModelRegistry`` instance.

    Raises:
        ConfigError: If the file is missing, unparseable, or an endpoint env
            var is unset.
    """
    path = Path(registry_path)
    if not path.exists():
        raise ConfigError(
            f"models.yaml not found at '{registry_path}'. "
            "Check models_registry_path in default.yaml or cwd."
        )
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "models" not in raw:
        raise ConfigError("config/models.yaml must have a top-level 'models:' key.")
    # Interpolate env vars throughout the YAML tree before pydantic validation
    interpolated: dict[str, Any] = _interpolate_dict(raw)
    return ModelRegistry(**interpolated)


# ---------------------------------------------------------------------------
# Settings source that hides the `models` registry field from env parsing
# ---------------------------------------------------------------------------


class _DropTopLevelKeys(PydanticBaseSettingsSource):
    """Delegating settings source that removes chosen top-level keys.

    The model registry uses ``${IW_RAG__MODELS__*__ENDPOINT}`` interpolation,
    but pydantic-settings would otherwise read those same env vars into the
    ``models`` field and fail (``ModelRegistry`` expects a ``models`` mapping,
    not ``models.generator``). Dropping the ``models`` key from the env/dotenv
    sources lets the registry be loaded from ``models.yaml`` instead, while the
    endpoint env vars remain readable by :func:`_interpolate_env` via os.environ.
    """

    def __init__(self, inner: PydanticBaseSettingsSource, drop: set[str]) -> None:
        super().__init__(inner.settings_cls)
        self._inner = inner
        self._drop = drop

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Unused for merged-dict sources; required by the ABC."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in self._inner().items() if k not in self._drop}


# ---------------------------------------------------------------------------
# Main Settings class
# ---------------------------------------------------------------------------


class InputGuardSettings(BaseModel):
    """Per-model config for the input guard layer (lld-guardrails §6).

    Attributes:
        prompt_guard_model: Logical name of the injection classifier model in
            ``models.yaml`` (default: ``"guard_injection"``).
        harm_model: Logical name of the harm-risk model in ``models.yaml``
            (default: ``"guard_safety"``).
        threshold: Block threshold — turns with ``max(pg, gg) >= threshold``
            are rejected.  Default ``0.5`` is a starting point; tuned via
            eval (Task 3.5 / R-00108 F2/F3).
    """

    prompt_guard_model: str = "guard_injection"
    harm_model: str = "guard_safety"
    threshold: float = 0.5


class SanitizeSettings(BaseModel):
    """Per-layer sanitize configuration (lld-guardrails §6, Task 3.2).

    Loaded from ``IW_RAG__GUARDRAILS__SANITIZE__*`` env vars or
    ``config/default.yaml``.

    Attributes:
        injection_scan: Whether to run per-chunk Prompt Guard 2 injection scan.
        on_poison: What to do when a chunk is flagged as injection.
            ``"drop"`` (default, production) excludes it; ``"flag"`` retains
            it with ``metadata["poisoned"]=True`` for eval/demo comparisons.
        every_n_words: Datamarking density — insert the session marker after
            every Nth word.  Default ``1`` (mark every word, maximum density).
            Tuned against the local Qwen3-14B (R-00108 F2 / Task 3.5).
        pii: Whether to run Presidio PII redaction on surviving chunks.
        pii_entity_types: PII entity types to redact for all requesters when
            no per-clearance policy override is configured.  An empty list
            means "use the module default set" (all common PII types).
    """

    injection_scan: bool = True
    on_poison: Literal["drop", "flag"] = "drop"
    every_n_words: int = 1
    pii: bool = True
    pii_entity_types: list[str] = []


class OutputGuardSettings(BaseModel):
    """Per-layer output guard configuration (lld-guardrails §6, Task 3.3).

    Loaded from ``IW_RAG__GUARDRAILS__OUTPUT__*`` env vars or
    ``config/default.yaml``.

    Attributes:
        groundedness_model: Logical name of the groundedness model in
            ``models.yaml`` (default: ``"guard_safety"`` = Granite Guardian 3.3-8B).
            Shared with the input guard workhorse — one co-resident model for
            harm + groundedness, minimising VRAM (R-00108 F6).
        groundedness_threshold: Below this score → ungrounded → fail-closed.
            Default 0.5; tuned via eval (Task 3.5).
        citation_validation: Whether to run deterministic citation validation.
            Non-disablable in production (hard block on invalid citations).
        pii: Whether to run Presidio PII redaction on the final answer.
        safety_model: Optional output safety model (Llama Guard 4 seam).
            ``None`` (default) = off for VRAM.  Set to a models.yaml logical name
            to enable (Phase-3 guards-on profile only).
    """

    groundedness_model: str = "guard_safety"
    groundedness_threshold: float = 0.5
    citation_validation: bool = True
    pii: bool = True
    safety_model: str | None = None


class AuditSettings(BaseModel):
    """Audit log configuration (lld-guardrails §3.5, Task 3.4).

    Loaded from ``IW_RAG__GUARDRAILS__AUDIT__*`` env vars or
    ``config/default.yaml``.

    Attributes:
        store: The audit sink — ``"postgres"`` (default) persists to the same
            Postgres instance as the data model (single-auditable-store,
            architecture §2 principle 7).
    """

    store: str = "postgres"


class GuardrailsSettings(BaseModel):
    """Master guardrails configuration (lld-guardrails §6).

    Loaded from ``IW_RAG__GUARDRAILS__*`` env vars or ``config/default.yaml``.

    Attributes:
        enabled: Master switch for all guardrails layers.  When ``False``
            the input/sanitize/output guard nodes are bypassed (Phase-3 off).
        input: Per-layer input guard configuration (threshold, model roles).
        sanitize: Per-layer sanitize configuration (datamarking, PII, injection
            scan; Task 3.2).
        output: Per-layer output guard configuration (groundedness model,
            threshold, citation validation, PII; Task 3.3).
        audit: Audit log sink configuration (Task 3.4).
    """

    enabled: bool = True
    input: InputGuardSettings = InputGuardSettings()
    sanitize: SanitizeSettings = SanitizeSettings()
    output: OutputGuardSettings = OutputGuardSettings()
    audit: AuditSettings = AuditSettings()


class AgentSettings(BaseModel):
    """Agentic-loop configuration (Phase 2; lld-agentic-loop §8).

    Loaded from ``IW_RAG__AGENT__*`` env vars or ``config/default.yaml``.

    Attributes:
        max_retries: Correction passes before fail-closed (demo: 1–2; R-00105 F5).
        recursion_limit: LangGraph super-step safety net.
        checkpointer: Which checkpointer to build at app startup.
        grader_model: Which model role fills the grader seat (CONVENTIONS §4).
            ``"generator"`` = share the Qwen3-14B endpoint (Phase 2 default, one
            resident model); ``"granite_guardian"`` = dedicated guard model
            (Phase 3 optional, requires spike headroom confirmation).
        router_enabled: Whether to run the Adaptive-RAG router.
        skip_enabled: Whether the router may choose the ``skip`` route.
    """

    max_retries: int = 2
    recursion_limit: int = 25
    checkpointer: Literal["in_memory", "postgres"] = "in_memory"
    grader_model: str = "generator"
    router_enabled: bool = True
    skip_enabled: bool = True

    class Router(BaseModel):
        """Router sub-settings.

        Attributes:
            enabled: Whether the Adaptive-RAG router runs.
            skip_enabled: Whether the router may select the ``skip`` route.
        """

        enabled: bool = True
        skip_enabled: bool = True

    class Grade(BaseModel):
        """Grade sub-settings.

        Attributes:
            min_relevant: Minimum number of relevant chunks to proceed to
                generation (CRAG ``doc_verdict`` threshold).
        """

        min_relevant: int = 1

    class Hitl(BaseModel):
        """Human-in-the-loop sub-settings.

        Attributes:
            enabled: Whether ``interrupt()`` is wired before the final emit.
                Off by default (would block the latency benchmark).
        """

        enabled: bool = False

    router: Router = Router()
    grade: Grade = Grade()
    hitl: Hitl = Hitl()


class VisualCaptionSettings(BaseModel):
    """Visual caption mode sub-settings (lld-multimodal §7).

    Attributes:
        model_role: Logical model role name in ``models.yaml``
            (default ``"visual_caption"`` → Qwen2.5-VL-7B-Instruct).
        max_assets_per_doc: Maximum visual assets captioned per document;
            ``0`` means unlimited.
        max_tokens: Maximum tokens in the generated caption.
    """

    model_role: str = "visual_caption"
    max_assets_per_doc: int = 0
    max_tokens: int = 512


class VisualLateInteractionSettings(BaseModel):
    """Late-interaction (ColVision) sub-settings for the visual mode (WI-031).

    Loaded from ``IW_RAG__MODES__VISUAL__LATE_INTERACTION__*`` env vars or
    ``config/default.yaml`` under ``modes.visual.late_interaction``.

    Attributes:
        backend: The multi-vector store backend.  ``"postgres_candidates"``
            (default) uses a mean-pooled HNSW proxy for stage-1 and in-process
            MaxSim for stage-2.  ``"qdrant"`` is the headline T1 trigger (ADR-007).
        pooled_vectors_per_page: Target number of pooled vectors per page
            (~38, per R-00107/F3).
        prefetch_n: Stage-1 candidate budget (pooled/binary ANN prefetch).
        top_k: Final top-k pages returned by the visual retriever.
        model_role: Logical ColVision encoder role in models.yaml.
    """

    backend: Literal["postgres_candidates", "qdrant"] = "postgres_candidates"
    pooled_vectors_per_page: int = 38
    prefetch_n: int = 200
    top_k: int = 8
    model_role: str = "visual_retrieval"


class VisualModeSettings(BaseModel):
    """Visual mode configuration (modes.visual, lld-multimodal §7).

    Loaded from ``IW_RAG__MODES__VISUAL__*`` env vars or
    ``config/default.yaml`` under ``modes.visual``.

    Attributes:
        enabled: Master switch.  Off by default until experiment #7 proves
            the visual win (lld-multimodal §8, task-4b acceptance criteria).
        caption: Visual caption sub-settings (model role, token budget, etc.).
        late_interaction: Late-interaction ColVision sub-settings (WI-031).
    """

    enabled: bool = False
    caption: VisualCaptionSettings = VisualCaptionSettings()
    late_interaction: VisualLateInteractionSettings = VisualLateInteractionSettings()


class GraphRagIndexSettings(BaseModel):
    """GraphRAG offline indexer sub-settings (lld-graphrag §7, task-4a §2).

    Loaded from ``IW_RAG__MODES__GRAPHRAG__INDEX__*`` env vars or
    ``config/default.yaml`` under ``modes.graphrag.index``.

    Attributes:
        extraction_model: Logical model role in ``models.yaml``
            (default ``"graphrag-extract"`` → Qwen3-4B, batch endpoint).
        skip_community_summaries: Skip community-summary generation (default
            ``True`` — dominant cost; only needed for global-summary demos,
            R-00106/F5).
        incremental: Enable incremental dedup on ``(source_uri, checksum)``
            (default ``True`` — only re-extract changed chunks, R-00106/F3).
        max_gleanings: Bounded re-prompt count per chunk (default 1, R-00106/F5).
    """

    extraction_model: str = "graphrag-extract"
    skip_community_summaries: bool = True
    incremental: bool = True
    max_gleanings: int = 1


class GraphRagModeSettings(BaseModel):
    """GraphRAG mode configuration (modes.graphrag, lld-graphrag §7, task-4a).

    Loaded from ``IW_RAG__MODES__GRAPHRAG__*`` env vars or
    ``config/default.yaml`` under ``modes.graphrag``.

    Attributes:
        enabled: Master switch.  Off by default until experiment #6 proves the
            multi-hop win (task-4a acceptance criteria, ADR-009).
        default_mode: Default LightRAG query mode; ``"mix"`` bounds misroute
            downside on simple queries (R-00106/F6).
        top_k: Final top-k chunks returned to the generator.
        enable_global: Whether the ``global`` community-summary mode is allowed
            (off by default; requires ``skip_community_summaries=False`` at index
            time and significant extra cost).
        acl_postfilter: Whether to post-filter resolved chunks by ACL tags
            (always ``True`` — lld-graphrag §6, ACL caveat).
        timeout_s: Per-query timeout in seconds; exceeded → fallback to spine.
        index: Offline indexer sub-settings.
    """

    enabled: bool = False
    default_mode: str = "mix"
    top_k: int = 10
    enable_global: bool = False
    acl_postfilter: bool = True
    timeout_s: float = 20.0
    index: GraphRagIndexSettings = GraphRagIndexSettings()


class ServingSwapSettings(BaseModel):
    """Model-swap serving configuration (F-00094, Benchmark Studio).

    Settings that govern the swap-controller sidecar (the FastAPI process
    that loads ``config/model_catalog.yaml`` and drives ``llama-swap``) and
    how the main API proxies to it.  The catalog is file-backed; the swap
    authority (which model is currently resident) lives in llama-swap, so
    ``health_timeout_s`` MUST match llama-swap's ``healthCheckTimeout`` to
    keep the activation gate coherent (R-00123 §2).

    Env mapping: ``IW_RAG__SERVING__*``.

    Attributes:
        catalog_path: Path to the user-facing generator catalog YAML
            (default ``"config/model_catalog.yaml"``).  The S02
            swap-controller loader reads this file and validates each
            entry against the schema; missing/malformed → ``ConfigError``
            at startup, mirroring the ``models.yaml`` loader contract.
        llama_swap_base_url: Base URL of the llama-swap proxy
            (default ``"http://llama-swap:8080"`` — the internal Compose
            network name; NEVER set this to a host-accessible URL: the
            llama-swap container holds the host docker socket and must
            only be reachable on the internal network
            (F-00094 design §"Hard constraints", R-00123 Limitations 2).
        swap_controller_base_url: Base URL of the swap-controller sidecar
            (default ``"http://swap-controller:8090"``).  The main API's
            ``GET /models`` and ``POST /models/activate`` are thin facades
            that proxy over HTTP to this sidecar via a typed async client
            (``src/iw_rag/serving/swap_controller_client.py``).  The
            sidecar is internal-only and not exposed to the host.
        health_timeout_s: Readiness poll timeout in seconds (default
            ``300`` — matches ``llama-swap.yaml`` ``healthCheckTimeout: 300``).
            The S02 swap-controller polls the model's ``/v1/models``
            endpoint until HTTP 200 OR this timeout; on timeout,
            activation transitions to ``failed`` (AC2 / Boundary row 1).
        activation_group: The ACL group required to activate a model
            (default ``"eval-operator"``).  The S03 API reads this from
            ``Settings`` and enforces the group check against
            ``AccessContext.principal_groups`` — NEVER from the request
            body (auth-only ACL keystone, F-00094 design §"ACL keystone").
    """

    catalog_path: str = "config/model_catalog.yaml"
    llama_swap_base_url: str = "http://llama-swap:8080"
    swap_controller_base_url: str = "http://swap-controller:8090"
    health_timeout_s: int = 300
    activation_group: str = "eval-operator"


class PowerComponentCfg(BaseModel):
    """One managed VRAM-holding component the power controller can stop/start.

    A component is a named group of Docker containers that, together, hold a
    chunk of VRAM and are powered on/off as a unit (e.g. ``"generator"`` =
    the vLLM container; ``"embedder"`` = the Infinity container serving the
    BGE-M3 embedder + reranker; ``"guards"`` = the input/injection guard
    sidecars).  The container names are deployment-specific (Compose project
    name + service), so they live in config rather than code (mirrors the
    model-agnostic catalog ethos — adding/renaming a managed container is a
    config change, never a code change).

    Attributes:
        containers: Docker container names this component owns.  All are
            stopped on power-off and started on power-on.  A name that does
            not resolve to a real container is skipped (not an error) so the
            same config works whether or not the optional guard overlay is up.
        label: Human-facing label for the console (defaults to the component
            key when blank).
        description: One-line explanation shown under the toggle.
        idle_unload: Whether the idle monitor auto-unloads this component
            after the idle timeout.  Default ``False``: only the generator
            (the dominant, cheapest-to-reload footprint) opts in, so the
            embedder/guards stay warm and every cold start after idle reloads
            only the LLM.
    """

    containers: list[str] = []
    label: str = ""
    description: str = ""
    idle_unload: bool = False


def _default_power_components() -> dict[str, PowerComponentCfg]:
    """Default serving-mode component → container mapping (single 32 GB box).

    Mirrors the canonical serving-mode container names (Compose project
    ``iw-rag`` + service): the pinned vLLM generator (``iw-rag-vllm-1``), the
    always-on Infinity embedder+reranker (``iw-rag-infinity-1``), and the
    optional guard sidecars (``iw-rag-vllm-guard`` / ``injguard``, present only
    when the guards overlay is up).  Overridable wholesale via
    ``IW_RAG__POWER__COMPONENTS`` (JSON) for a differently-named deployment.
    """
    return {
        "generator": PowerComponentCfg(
            # Serving mode pins ``iw-rag-vllm-1``; benchmark mode's generator is
            # llama-swap's dynamic ``vllm-<served-name>`` container.  The
            # vendor-family patterns match every generator llama-swap can spawn
            # (config/llama-swap.yaml) WITHOUT matching the judges
            # (``vllm-judge-*``) or unrelated projects — so one map works in both
            # modes.  Override via IW_RAG__POWER__COMPONENTS if a non-iw-rag
            # container shares one of these vendor prefixes on the box.
            containers=[
                "iw-rag-vllm-1",
                "vllm-qwen*",
                "vllm-llama*",
                "vllm-hermes*",
                "vllm-mistral*",
            ],
            label="Generator (LLM)",
            description="The vLLM generator — the dominant VRAM consumer (~10-18 GB).",
            idle_unload=True,
        ),
        "embedder": PowerComponentCfg(
            containers=["iw-rag-infinity-1"],
            label="Embedder + reranker",
            description="Infinity BGE-M3 embedder + reranker (~4 GB). Needed for retrieval.",
            idle_unload=False,
        ),
        "judges": PowerComponentCfg(
            # Benchmark two-phase judges (Lynx faithfulness + Selene grading),
            # spawned by llama-swap.  Loaded during scoring; a resident judge can
            # hold ~16-18 GB, so it is worth being able to free explicitly.
            containers=["vllm-judge-lynx", "vllm-judge-selene"],
            label="Benchmark judges",
            description="Lynx + Selene judges (benchmark scoring only; ~16-18 GB when resident).",
            idle_unload=False,
        ),
        "guards": PowerComponentCfg(
            containers=["iw-rag-vllm-guard", "injguard"],
            label="Guard models",
            description="Input/injection guard sidecars (~1-2 GB). Present only when guards-on.",
            idle_unload=False,
        ),
    }


class PowerModePlan(BaseModel):
    """The container + endpoint plan for one box mode (serving | benchmark).

    A mode switch is: stop ``stop`` containers, start ``start`` containers
    (wait until ready), and re-point the API's generator at ``generator_endpoint``
    with ``guards`` enabled/disabled — the in-UI equivalent of the
    ``make serving-mode`` / ``make benchmark-mode`` targets.  Container names
    that do not exist are skipped, so the optional guard containers being
    absent is not an error.

    Attributes:
        start: Containers to start (and wait to ready) when entering this mode.
        stop: Containers to stop when entering this mode.
        generator_endpoint: The generator endpoint the API rebuilds onto.
        guards: Whether guardrails are enabled in this mode.
    """

    start: list[str] = []
    stop: list[str] = []
    generator_endpoint: str = ""
    guards: bool = True


def _default_power_modes() -> dict[str, PowerModePlan]:
    """Default serving/benchmark switch plans (mirror the make targets)."""
    swap_stack = ["iw-rag-llama-swap-1", "iw-rag-swap-controller-1"]
    serving_stack = ["iw-rag-vllm-1", "iw-rag-vllm-guard", "injguard"]
    return {
        "serving": PowerModePlan(
            start=serving_stack,
            stop=swap_stack,
            generator_endpoint="http://vllm:8001/v1",
            guards=True,
        ),
        "benchmark": PowerModePlan(
            start=swap_stack,
            stop=serving_stack,
            generator_endpoint="http://llama-swap:8080/v1",
            guards=False,
        ),
    }


class PowerControlSettings(BaseModel):
    """System power-control configuration (F — VRAM power switch).

    Governs the ``system-control`` sidecar (the privileged FastAPI process
    that owns the Docker socket and starts/stops the VRAM-holding containers)
    and how the main API proxies to it, runs the idle-unload monitor, and
    blocks-and-warms the first query after an auto-unload.

    **Privilege boundary.**  The main API stays unprivileged — it NEVER
    touches the Docker socket.  Only the sidecar (``base_url``) holds the
    socket and runs ``docker stop/start``.  This mirrors the swap-controller
    split: policy (idle timer, ACL gate) lives in the API; execution lives in
    the sidecar.

    Env mapping: ``IW_RAG__POWER__*``.

    Attributes:
        enabled: Master feature switch.  When ``False`` the API wires no
            power client, runs no idle monitor, and the System view degrades
            to "power control not enabled".
        base_url: Base URL of the ``system-control`` sidecar (internal
            Compose network only — NEVER published to the host, since it
            holds the Docker socket).
        docker_socket: Path to the Docker Unix socket the sidecar mounts
            (read by the sidecar only).
        power_group: The ACL group required to flip the power switch
            (default ``"eval-operator"``).  Read from
            ``AccessContext.principal_groups`` (auth only — never the request
            body), mirroring the swap-controller activation keystone.
        idle_unload_enabled: Whether the idle monitor auto-unloads idle
            components after ``idle_unload_minutes``.
        idle_unload_minutes: Idle threshold in minutes before auto-unload
            (default 15).  The monitor unloads only components whose
            :attr:`PowerComponentCfg.idle_unload` is ``True``.
        ready_timeout_s: Per-component readiness budget after a start
            (poll the container's running/health state until ready or this
            timeout).  Default 300 to cover a real cold vLLM load.
        poll_interval_s: Sleep between readiness polls in seconds.
        block_and_warm: When ``True`` (default), a query that arrives while
            the generator is powered off transparently powers it back on and
            waits, rather than failing.  The first query after idle pays the
            cold-start; subsequent queries are fast.
        vram_ceiling_gb: The single-GPU VRAM ceiling in GB (32 for the
            RTX 5090) — used only for the console gauge's full-scale.
        components: The managed component → container mapping.
    """

    enabled: bool = True
    base_url: str = "http://system-control:8091"
    docker_socket: str = "/var/run/docker.sock"
    power_group: str = "eval-operator"
    idle_unload_enabled: bool = True
    idle_unload_minutes: int = 15
    ready_timeout_s: int = 300
    poll_interval_s: float = 1.0
    block_and_warm: bool = True
    vram_ceiling_gb: float = 32.0
    mode_switch_enabled: bool = True
    components: dict[str, PowerComponentCfg] = Field(default_factory=_default_power_components)
    modes: dict[str, PowerModePlan] = Field(default_factory=_default_power_modes)


class LongContextModeSettings(BaseModel):
    """Long-context route configuration (modes.long_context, lld-long-context §6).

    Loaded from ``IW_RAG__MODES__LONG_CONTEXT__*`` env vars or
    ``config/default.yaml`` under ``modes.long_context``.

    **YaRN invariant:** ``enable_yarn`` MUST remain ``False``.  The
    ``@field_validator`` on :class:`~iw_rag.modes.long_context.types.LongContextConfig`
    already enforces this at the domain level; this settings model mirrors the
    constraint so that misconfiguration surfaces at startup (not mid-request).

    Attributes:
        enabled: Master switch.  Off by default until experiment #8 proves the
            whole-doc win within the VRAM budget (task-4c acceptance criteria,
            ADR-012).
        lc_max_input: Target content cap in tokens (well under native 32K;
            default 24 000).  Tuned via experiment #8 NoLiMa/RULER probe.
        hard_ceiling: Absolute maximum tokens; native 32K window only (default
            30 000).  Post-assembly overflow → ``LongContextOverBudget``
            → fallback to ``retrieve``.
        enable_yarn: INVARIANT — must remain ``False``.  Any ``True`` value
            raises ``ConfigError`` at startup (R-00109/F3).
        require_document_level_attribution_only: When ``True`` (default),
            only queries needing ≤ document-level attribution are eligible.
            Per-claim attribution queries are forced to RAG (T4).
    """

    enabled: bool = False
    lc_max_input: int = 24_000
    hard_ceiling: int = 30_000
    enable_yarn: bool = False
    require_document_level_attribution_only: bool = True

    @model_validator(mode="after")
    def _reject_yarn(self) -> LongContextModeSettings:
        """Reject enable_yarn=True at startup (YaRN invariant, R-00109/F3).

        Returns:
            Self when ``enable_yarn=False``.

        Raises:
            ConfigError: If ``enable_yarn=True`` — the long-context route
                must NEVER enable the 131K YaRN window.
        """
        if self.enable_yarn:
            raise ConfigError(
                "IW_RAG__MODES__LONG_CONTEXT__ENABLE_YARN=true is forbidden. "
                "The long-context route operates strictly in the native 32K window. "
                "YaRN degrades all short RAG traffic ~15-20% (R-00109/F3). "
                "Set IW_RAG__MODES__LONG_CONTEXT__ENABLE_YARN=false."
            )
        return self


class ModesSettings(BaseModel):
    """Optional-mode configuration block (Phase 4; lld-multimodal/lld-graphrag).

    Each mode is independently switchable — all default to ``enabled=false``
    until their headline experiment (Phase 4 exit gate) proves they earn it.

    Attributes:
        visual: Visual fallback (parse-then-caption) and late-interaction settings.
        graphrag: GraphRAG optional retrieval mode (WI-028 indexer + WI-029 query).
        long_context: Long-context gated 4th route (native 32K, YaRN off, WI-032).
    """

    visual: VisualModeSettings = VisualModeSettings()
    graphrag: GraphRagModeSettings = GraphRagModeSettings()
    long_context: LongContextModeSettings = LongContextModeSettings()


class EvalRetrievalSettings(BaseModel):
    """Judge-free retrieval-metric thresholds (lld-eval-observability §10).

    Attributes:
        ks: The k cutoffs scored (recall@k / nDCG@k).
        recall_at_10_min: The CI hard gate on Recall@10 (R-00100, ADR-013).
    """

    ks: tuple[int, ...] = (5, 10)
    recall_at_10_min: float = 0.8


class EvalJudgeSettings(BaseModel):
    """Offline judge configuration (lld-eval-observability §5, §10).

    Both judges are a DIFFERENT model family from the Qwen3-14B generator —
    self-preference de-bias (ADR-013, CONVENTIONS §4).  The logical model names
    reference entries in ``config/models.yaml`` (role: judge).

    Attributes:
        faithfulness_model: Logical model name for the faithfulness judge
            (Patronus Lynx-8B; role=judge_faithfulness in models.yaml).
        general_model: Logical model name for the general grading judge
            (Atla Selene-Mini-8B; role=judge_general in models.yaml).
        position_swap: Default for GradingJudge.grade — call twice with
            swapped option order and merge (bias control).
        rubric_anchored: Whether judges use rubric-anchored pointwise prompts
            rather than free-form scoring (bias control).
    """

    faithfulness_model: str = "judge_faithfulness"
    general_model: str = "judge_general"
    position_swap: bool = True
    rubric_anchored: bool = True


class EvalGuardrailsSettings(BaseModel):
    """Guard-quality and red-team threshold configuration (ADR-013, Task 3.5).

    Thresholds are frozen as the CI regression gate for experiment #5
    (guardrail efficacy / red-team).  All values are defensible starting points;
    numeric bars are owned by ADR-013 / eval LLD and tuned once live red-team
    numbers (5a Qwen3-14B ASR, promptfoo/garak full runs) are available.

    Attributes:
        acl_leak_rate_max: Maximum tolerated ACL cross-tenant leak rate.
            Target is structural zero (0.0) — the pre-filter is server-injected,
            never LLM-derived (lld-guardrails §3.1).
        attack_catch_rate_min: Minimum fraction of the adversarial corpus that
            the guard chain must intercept (block or neutralize).
        groundedness_accuracy_min: Minimum Granite Guardian groundedness
            classification accuracy on a labeled grounded/ungrounded set.
        pii_recall_min: Minimum Presidio PII recall on the synthetic PII corpus
            (emails/SSNs/phones, incl. adversarial paraphrase probes).
        asr_max: Maximum tolerated Attack-Success-Rate per attack class.
            Attacks that defeat all guard layers must remain ≤ this fraction.
        fpr_max: Maximum tolerated False-Positive-Rate (FAR) — guard blocking
            benign queries; over-block destroys utility (combined OGP balance).
        fail_closed_rate_min: Minimum fraction of attacks that defeat earlier
            layers that the groundedness/citation check converts into a safe
            refusal.  The headline trust number (playbook §6).
        latency_overhead_ms_max: Maximum added latency from the hot guards
            (Prompt Guard 2 + Presidio + datamark; one Granite call shared with
            the loop and counted once).  Single-digit-ms hot guards; ~30 ms
            Granite Guardian; 50 ms is a conservative CI budget.
    """

    acl_leak_rate_max: float = 0.0
    attack_catch_rate_min: float = 0.85
    groundedness_accuracy_min: float = 0.80
    pii_recall_min: float = 0.80
    asr_max: float = 0.15
    fpr_max: float = 0.10
    fail_closed_rate_min: float = 0.95
    latency_overhead_ms_max: float = 50.0


class EvalCiSettings(BaseModel):
    """CI regression gate configuration (lld-eval-observability §10, ADR-013).

    Controls the two gate behaviours: the judge-free Recall@10 hard floor
    (always runs) and the judged faithfulness no-regress check (skip-guarded
    when the judge endpoint is absent).

    Env mapping: ``IW_RAG__EVAL__CI__DEEPEVAL_FAIL_ON_REGRESSION`` etc.

    Attributes:
        deepeval_fail_on_regression: When ``True`` (default), a metric
            regression fails the build.  When ``False``, a regression is
            recorded as a warning but the gate still passes — useful for
            temporarily unblocking CI while investigating a dip.
        faithfulness_regress_tolerance: Maximum absolute drop in faithfulness
            that is tolerated before the gate fires.  Default 0.03 (3 pp)
            matches the Phase-0 baseline file (lld-eval-observability §10).
    """

    deepeval_fail_on_regression: bool = True
    faithfulness_regress_tolerance: float = 0.03


class EvalBenchmarkSettings(BaseModel):
    """Benchmark Studio measurement-core knobs (F-00093, S06).

    Settings that govern the Benchmark Studio perf collectors + stats
    module: the S07 bootstrap resample count, the S05 GPU sampler
    interval, the vLLM ``/metrics`` endpoint, the latency percentiles
    to extract, and the underpowered-caveat threshold (the n below
    which paired A/B comparisons flag the result as underpowered
    rather than reporting a p-value crash — per R-00121 §4 / Boundary
    Behavior row 5).

    Env mapping: ``IW_RAG__EVAL__BENCHMARK__*``.

    Attributes:
        bootstrap_resamples: Number of BCa bootstrap resamples for the
            S07 stats module's per-metric 95% CI.  Default 10 000
            matches the R-00121 §4 recipe; lower values trade CI
            smoothness for compute.
        gpu_sampler_interval_s: S05 :class:`GpuSampler` background
            thread interval in seconds.  Default 0.5 s (500 ms) per
            R-00122 §5.
        vllm_metrics_endpoint: Local vLLM Prometheus exposition
            endpoint.  Default ``"http://localhost:8001/metrics"`` —
            the S05 :class:`VllmMetricsScraper` GETs this URL; offline
            tests inject a fetcher instead.
        latency_percentiles: Percentiles the report assembly extracts
            from the vLLM ``/metrics`` text.  Default ``(50, 95, 99)``
            mirrors lld-eval-observability §10.
        min_queries_for_significance: The n below which the S07 stats
            module's A/B compare flags the result as underpowered
            (no p-value, no CI).  Default 150 matches R-00121 §4's
            "n<≈150 underpowered" caveat — smaller golden sets cannot
            trust pairwise significance.
    """

    bootstrap_resamples: int = 10_000
    gpu_sampler_interval_s: float = 0.5
    vllm_metrics_endpoint: str = "http://localhost:8001/metrics"
    latency_percentiles: tuple[int, ...] = (50, 95, 99)
    min_queries_for_significance: int = 150
    # ------------------------------------------------------------------
    # F-00095 / S01 — run-orchestration knobs.
    #
    # ``default_dataset_id`` is the dataset the S01 runner resolves
    # when the submitted ``model_config`` does not carry one; it
    # matches the default in :mod:`iw_rag.eval.benchmark.runner`.
    # ``health_timeout_s`` defaults to the same value as
    # ``Settings.serving.health_timeout_s`` so the runner's wait-
    # for-ready budget is coherent with the swap-controller's
    # ``llama-swap.yaml`` ``healthCheckTimeout`` (R-00123 §2).
    # ``per_query_timeout_s`` is the per-query wall-clock budget
    # the S01 runner threads through to its instrumentation; the
    # S05 perf collectors and the per-stage latency JSONB read
    # from this knob.
    # ------------------------------------------------------------------
    default_dataset_id: str = "benchmark_golden"
    health_timeout_s: int = 300
    per_query_timeout_s: int = 120


class EvalSettings(BaseModel):
    """Phase-1/Phase-3 eval/trust configuration (sibling section, additive)."""

    retrieval: EvalRetrievalSettings = EvalRetrievalSettings()
    judges: EvalJudgeSettings = EvalJudgeSettings()
    guardrails: EvalGuardrailsSettings = EvalGuardrailsSettings()
    ci: EvalCiSettings = EvalCiSettings()
    benchmark: EvalBenchmarkSettings = EvalBenchmarkSettings()


class Settings(BaseSettings):
    """The single authoritative configuration object for iw-rag.

    Loaded once at startup via ``get_settings()``; injected into modules —
    never imported ad-hoc at call sites (lld-config §1).

    Attributes:
        environment: Runtime environment label.
        models_registry_path: Path to config/models.yaml (relative to cwd).
        database: Postgres connection + pool config (DSN from env).
        ingestion: Offline pipeline tuning knobs.
        retrieval: Spine retrieval parameters (candidate_k, rrf_k=60, …).
        generation: LLM generation parameters.
        api: FastAPI binding + auth mode.
        observability: Phoenix/OTEL settings.
        agent: Agentic-loop (Phase 2) configuration.
        eval: Eval/trust configuration (retrieval metrics, judges, and Phase-3
            guardrail quality / red-team thresholds).
        guardrails: Phase-3 guardrails layer configuration (input guard,
            sanitize, output guard).
        modes: Phase-4 optional mode configuration (visual, graphrag, …).
        serving: Model-swap serving configuration (F-00094, Benchmark Studio).
            Holds the catalog path, llama-swap base URL, swap-controller
            sidecar URL, readiness timeout, and the ``eval-operator`` ACL
            group required to activate a model.
        models: Validated model registry loaded from models_registry_path.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        # env_file is deliberately NOT set here: the .env file contains
        # docker-compose vars (POSTGRES_PASSWORD, VLLM_URL, etc.) that do not
        # carry the IW_RAG__ prefix.  pydantic-settings 2.x passes ALL dotenv
        # keys to the model even without a matching prefix, which conflicts with
        # extra="forbid".  Process env (sourced from .env by the shell or by
        # python-dotenv in get_settings) is the correct delivery mechanism.
        extra="forbid",
    )

    environment: Literal["dev", "demo", "test"] = "dev"
    models_registry_path: str = "config/models.yaml"

    database: DatabaseSettings
    ingestion: IngestionSettings = IngestionSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    generation: GenerationSettings = GenerationSettings()
    api: ApiSettings = ApiSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    agent: AgentSettings = AgentSettings()
    eval: EvalSettings = EvalSettings()
    guardrails: GuardrailsSettings = GuardrailsSettings()
    modes: ModesSettings = ModesSettings()
    serving: ServingSwapSettings = ServingSwapSettings()
    power: PowerControlSettings = PowerControlSettings()
    # Populated by the post-init validator; default is an empty registry so
    # pydantic does not try to parse it from env/yaml.
    models: ModelRegistry = ModelRegistry(models={})

    @model_validator(mode="after")
    def _load_and_validate_models(self) -> Settings:
        """Load the model registry after base field validation.

        Reads models_registry_path, interpolates endpoint env vars, validates.

        Returns:
            Self with models populated from models.yaml.

        Raises:
            ConfigError: If models.yaml is missing or any endpoint env var is unset.
        """
        registry = _load_model_registry(self.models_registry_path)
        object.__setattr__(self, "models", registry)
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Override source priority: init > env > .env > config/default.yaml.

        The ``YamlConfigSettingsSource`` provides lowest-priority defaults from
        ``config/default.yaml`` (lld-config §4 precedence). Process env wins.

        Args:
            settings_cls: The Settings class being built.
            init_settings: Constructor kwargs source (highest priority).
            env_settings: Process env source.
            dotenv_settings: .env file source.
            file_secret_settings: /run/secrets style source (unused here).

        Returns:
            Tuple of sources ordered highest-priority-first.
        """
        # Hide the `models` registry field from env/dotenv parsing — it is
        # loaded from models.yaml by the post-init validator (see class docstring).
        drop = {"models"}
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            # F-00096 overlay source — sits between init kwargs and env so a
            # successful live-edit ``apply_edit`` call is observed by the
            # next per-request ``get_settings()`` (after ``cache_clear()``).
            # The source is a no-op when no overlay provider is registered
            # (the production / test paths that don't use the Operator
            # Console).
            _OverlaySettingsSource(settings_cls),
            _DropTopLevelKeys(env_settings, drop),
            _DropTopLevelKeys(dotenv_settings, drop),
        ]
        yaml_path = Path("config/default.yaml")
        if yaml_path.exists():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        return tuple(sources)


# ---------------------------------------------------------------------------
# Cached accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and return the singleton Settings (lru_cache — called once).

    Loads ``.env`` via dotenv so IW_RAG__* vars reach process env before
    pydantic-settings reads them.  This avoids the pydantic-settings 2.x
    limitation where ALL dotenv keys (including non-prefixed compose vars)
    are passed to the model with ``extra="forbid"``.

    Validates eagerly: missing dsn or unresolved endpoint raises ``ConfigError``
    at startup, not mid-request.

    Returns:
        The validated, cached Settings instance.

    Raises:
        ConfigError: If required config is missing or invalid.
    """
    # Load .env into process env (only sets vars not already set, so process
    # env wins; IW_RAG__* vars are then visible to pydantic-settings EnvSource)
    _load_dotenv_safe()

    try:
        return Settings()  # type: ignore[call-arg]
    except ConfigError:
        raise
    except Exception as exc:
        # Wrap pydantic validation errors that mention dsn as ConfigError
        msg = str(exc)
        if "dsn" in msg.lower() or ("database" in msg.lower() and "missing" in msg.lower()):
            raise ConfigError(
                f"Missing required setting: IW_RAG__DATABASE__DSN. "
                f"Set it in .env or process env. Original error: {exc}"
            ) from exc
        raise


def _load_dotenv_safe() -> None:
    """Load only IW_RAG__* vars from .env into process env (non-overriding).

    Process env always wins (12-factor).  Non-prefixed vars (POSTGRES_PASSWORD,
    VLLM_URL, etc.) are skipped so they never reach the Settings model.
    """
    if os.environ.get("IW_RAG_SKIP_DOTENV"):
        return
    env_file = Path(".env")
    if not env_file.exists():
        return
    with env_file.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            # Skip comments and blank lines
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Only load IW_RAG__* vars; others are compose-level
            if key.startswith(ENV_PREFIX) and key not in os.environ:
                os.environ[key] = value
