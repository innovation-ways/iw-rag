"""Config registry — editable-key allowlist + the hard locked set.

The registry is the **security keystone** of the Operator Console's
live-edit Config view (F-00096).  It encodes, server-side, the
"an edit can NEVER set a locked (security/infra) key" invariant
(F-00096 §Invariants 1 & 5) and exposes the curated allowlist of
RAG-behaviour knobs the operator can edit through the Console.

A pure, importable module — no DB, no I/O, no FastAPI.  Depends only
on stdlib (pydantic-free, kept leaf-light so the service layer can
import it cheaply).

The registry has two co-equal parts:

* :data:`_REGISTRY` — the editable allowlist.  Each entry maps a
  dotted key (e.g. ``"retrieval.top_k"``) to a :class:`RegistryEntry`
  with the per-key description, the pre-Settings bounds, and the
  ``hot_reloadable`` flag.
* :data:`_LOCKED_KEYS` — a hard ``frozenset`` of dotted keys that can
  **NEVER** be edited.  Per the design Notes, this set is the only
  place the "fail-closed / ACL / local-only / DSN / model endpoints"
  invariants are encoded at the application layer.  Locked wins:
  a key that appears in BOTH sets is classified as ``"locked"``.

The service layer (``iw_rag.api.config_service``) consults the
registry on every edit.  A locked key is rejected with a typed
:exc:`ValidationError` **before** any overlay row or audit row is
written (AC3 / boundary "Config edit to a locked key").  An unknown
key is rejected the same way.  Only an editable key with a value
inside its declared :class:`Bounds` and a value that ``Settings``
itself can validate proceeds to the apply step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Typed structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bounds:
    """The pre-Settings bounds for an editable key.

    Used to reject out-of-bounds values *before* ``Settings`` validation
    runs, so a noisy "top_k = -1" never reaches pydantic.  The service
    still validates the proposed value through ``Settings`` afterwards
    (defence in depth — pydantic's own constraints apply on top of
    these bounds).

    Attributes:
        kind: ``"int_range"`` for inclusive ``[min, max]`` integer
            bounds, ``"float_range"`` for inclusive ``[min, max]`` float
            bounds, or ``"enum"`` for a fixed ``choices`` tuple.
        min: Inclusive lower bound (int_range / float_range only).
        max: Inclusive upper bound (int_range / float_range only).
        choices: Permitted string values (enum only).
    """

    kind: Literal["int_range", "float_range", "enum"]
    min: int | float | None = None
    max: int | float | None = None
    choices: tuple[str, ...] = ()

    def check(self, value: Any) -> bool:
        """Return True iff ``value`` is in-bounds for this entry.

        Args:
            value: The candidate value (must already be of the right
                Python type — the service validates through ``Settings``
                afterwards).

        Returns:
            ``True`` when the value satisfies the declared bounds,
            ``False`` otherwise.
        """
        if self.kind == "int_range":
            return (
                isinstance(value, int)
                and not isinstance(value, bool)
                and self.min is not None
                and self.max is not None
                and self.min <= value <= self.max
            )
        if self.kind == "float_range":
            # Accept ints as floats (1 → 1.0); reject bools.
            v = float(value) if not isinstance(value, bool) else None
            if v is None or self.min is None or self.max is None:
                return False
            return self.min <= v <= self.max
        if self.kind == "enum":
            return isinstance(value, str) and value in self.choices
        # Unknown kind — fail closed.
        return False


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """One editable allowlist entry.

    Attributes:
        description: One-line operator-facing description (shown in the
            Config view next to the editable form).
        bounds: The pre-Settings bounds for the value.
        hot_reloadable: ``True`` if the key is read per-request via
            ``get_settings()`` and the change can be applied hot by
            clearing the cache; ``False`` if a startup-baked client owns
            the value (e.g. the spine's :class:`RetrievalConfig`) → the
            edit is flagged ``pending_restart``.
    """

    description: str
    bounds: Bounds
    hot_reloadable: bool


# ---------------------------------------------------------------------------
# Module-level allowlist
# ---------------------------------------------------------------------------
# Seeded with safe RAG-behaviour keys named in the design.  Each key
# genuinely exists on :class:`iw_rag.settings.Settings` and is reachable
# via dotted attribute access (e.g. ``settings.retrieval.top_k``).
#
# ``hot_reloadable=True`` keys are read per-request via ``get_settings()``
# — the guard layer's threshold / datamark knobs are the canonical
# example.  ``hot_reloadable=False`` keys are baked into startup-time
# clients (the spine's ``RetrievalConfig``, the generator client, the
# model catalog); the running request path is unchanged until restart
# (AC4) — a pending-restart badge is rendered, never a silent
# stale-value.
_REGISTRY: dict[str, RegistryEntry] = {
    # --- Retrieval spine knobs (baked into RetrievalConfig at startup) ---
    "retrieval.candidate_k": RegistryEntry(
        description=(
            "Per-arm candidate budget fed into the RRF fusion stage. "
            "Baked into the spine's RetrievalConfig at startup — "
            "changes apply after restart."
        ),
        bounds=Bounds(kind="int_range", min=1, max=500),
        hot_reloadable=False,
    ),
    "retrieval.fused_k": RegistryEntry(
        description=(
            "Fused candidate budget after RRF (before rerank). Baked "
            "into the spine at startup — changes apply after restart."
        ),
        bounds=Bounds(kind="int_range", min=1, max=1000),
        hot_reloadable=False,
    ),
    "retrieval.top_k": RegistryEntry(
        description=(
            "Final top-k returned to the generator. Baked into the "
            "spine's RetrievalConfig at startup — changes apply after "
            "restart."
        ),
        bounds=Bounds(kind="int_range", min=1, max=100),
        hot_reloadable=False,
    ),
    "retrieval.rrf_k": RegistryEntry(
        description=(
            "RRF constant k (canonical 60). Baked into the spine at "
            "startup — changes apply after restart."
        ),
        bounds=Bounds(kind="int_range", min=1, max=1000),
        hot_reloadable=False,
    ),
    "retrieval.rerank_on_failure": RegistryEntry(
        description=(
            "Behaviour when the reranker is unavailable: 'degrade' "
            "(return the fused order) or 'fail' (refuse the query). "
            "Baked into the spine at startup — changes apply after "
            "restart."
        ),
        bounds=Bounds(kind="enum", choices=("degrade", "fail")),
        hot_reloadable=False,
    ),
    "retrieval.parent_expansion_max_chars": RegistryEntry(
        description=(
            "Per-candidate char budget during small-to-big parent "
            "expansion. Baked into the spine at startup — changes apply "
            "after restart."
        ),
        bounds=Bounds(kind="int_range", min=256, max=65536),
        hot_reloadable=False,
    ),
    # --- Guard knobs (read per-request via get_settings() in the
    # guard layer) — these are the headline hot-reloadable keys. ---
    "guardrails.input.threshold": RegistryEntry(
        description=(
            "Block threshold for the input guard. Turns with "
            "max(prompt_guard_score, granite_harm_score) >= threshold "
            "are rejected. Read per-request — applies hot on the next "
            "query."
        ),
        bounds=Bounds(kind="float_range", min=0.0, max=1.0),
        hot_reloadable=True,
    ),
    "guardrails.output.groundedness_threshold": RegistryEntry(
        description=(
            "Granite Guardian groundedness threshold. Answers with "
            "groundedness_score < threshold are refused (fail-closed). "
            "Read per-request — applies hot on the next query."
        ),
        bounds=Bounds(kind="float_range", min=0.0, max=1.0),
        hot_reloadable=True,
    ),
    "guardrails.sanitize.every_n_words": RegistryEntry(
        description=(
            "Datamarking density — insert the session marker after "
            "every Nth word (1 = every word, max density). Read "
            "per-request — applies hot on the next query."
        ),
        bounds=Bounds(kind="int_range", min=1, max=20),
        hot_reloadable=True,
    ),
}


# ---------------------------------------------------------------------------
# Locked set — explicit frozenset of dotted keys that can NEVER be edited
# ---------------------------------------------------------------------------
# Covers the design-named invariants (AC3, F-00096 §Invariants 1 & 5):
#
#   * ``guardrails.enabled`` — the master guardrails switch.  Editing
#     this would let a hostile or inattentive operator turn the
#     fail-closed layer off (Invariant 5).
#   * ``database.dsn`` — secret, env-only.  Editing this would either
#     redirect the store to a hostile DB or surface a cleartext secret
#     in an audit row.
#   * ``models`` — the model registry namespace.  Every client
#     (generator, embedder, reranker, judge, guard, visual_caption) is
#     constructed at startup in ``app.py:_build_query_service``; the
#     endpoints and the served model names are baked in.  Locking the
#     parent prefix rejects every ``models.<logical_name>.*`` path.
#   * ``serving.llama_swap_base_url`` /
#     ``serving.swap_controller_base_url`` — internal-network URLs;
#     must never be redirectable via the overlay.
#   * ``api.static_tokens`` / ``api.operator_tokens`` /
#     ``api.demo_users`` / ``api.download_token_secret`` — auth
#     credentials / ACL inputs.
#   * ``serving.activation_group`` — the ACL group that gates
#     model-swap activation.  Editing this would widen operator
#     entitlements.
#
# A locked key MUST be rejected even if (by mistake) it also appears
# in the editable allowlist — ``classify_key`` implements locked-wins.
_LOCKED_KEYS: frozenset[str] = frozenset(
    {
        # AC3 — master guardrails switch.
        "guardrails.enabled",
        # DSN — env-only secret.
        "database.dsn",
        # Model registry namespace (parent prefix; matches every
        # models.<logical_name>.* path via prefix check).
        "models",
        # Internal-network URLs (F-00094 swap controller).
        "serving.llama_swap_base_url",
        "serving.swap_controller_base_url",
        # ACL inputs / auth (Invariant 5 — "ACL inputs").
        "serving.activation_group",
        "api.static_tokens",
        "api.operator_tokens",
        "api.demo_users",
        # Secret / token-issuance keys.
        "api.download_token_secret",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_locked(key: str) -> bool:
    """Return True if ``key`` is in the locked set (exact or parent-prefix).

    The locked set stores both exact keys (``"guardrails.enabled"``) and
    parent-prefix keys (``"models"`` — covers every
    ``models.<logical_name>.*`` sub-path).  A key whose parent prefix is
    locked is also locked.

    Args:
        key: Dotted config key.

    Returns:
        ``True`` if the key (or any of its parent prefixes) is in the
        locked set.
    """
    if key in _LOCKED_KEYS:
        return True
    # Parent-prefix check: walk the dotted components from the root
    # down, asking "is this prefix locked?".  A locked parent prefix
    # (e.g. ``"models"``) rejects every descendant.
    parts = key.split(".")
    for i in range(1, len(parts)):
        prefix = ".".join(parts[:i])
        if prefix in _LOCKED_KEYS:
            return True
    return False


def classify_key(key: str) -> Literal["editable", "locked", "unknown"]:
    """Classify a dotted config key as ``editable``, ``locked`` or ``unknown``.

    Locked-wins: a key present in BOTH the editable allowlist and the
    locked set must classify as ``"locked"`` (Invariant 1).  The locked
    set is the only place the security invariant is encoded.

    Args:
        key: Dotted config key (e.g. ``"retrieval.top_k"``).

    Returns:
        ``"locked"`` when the key (or any of its parent prefixes) is in
        the locked set, ``"editable"`` when it has a registry entry,
        ``"unknown"`` otherwise.
    """
    if _is_locked(key):
        return "locked"
    if key in _REGISTRY:
        return "editable"
    return "unknown"


def get_registry_entry(key: str) -> RegistryEntry | None:
    """Return the registry entry for ``key`` (or None for non-editable keys).

    The locked set is a security gate, not an editable allowlist — this
    helper deliberately does not return an entry for a locked key, so
    the service's "key not editable" path stays simple.

    Args:
        key: Dotted config key.

    Returns:
        The :class:`RegistryEntry` for editable keys, ``None`` for
        locked or unknown keys.
    """
    return _REGISTRY.get(key)


def iter_editable_entries() -> list[tuple[str, RegistryEntry]]:
    """Return every editable allowlist entry as ``(key, entry)`` pairs.

    Public surface for callers that need to enumerate the editable
    allowlist (e.g. the service's ``read_active``) without reaching
    into the module-level :data:`_REGISTRY` dict.  The result is a
    fresh list, sorted by key for stable display ordering; callers
    may mutate it without affecting the registry.

    Returns:
        A list of ``(dotted_key, RegistryEntry)`` pairs, sorted by key.
    """
    return sorted(_REGISTRY.items())


def iter_locked_keys() -> list[str]:
    """Return every locked-set key as a sorted list.

    Public surface for callers that need to enumerate the locked set
    (e.g. the service's ``read_active``) without reaching into the
    module-level :data:`_LOCKED_KEYS` frozenset.  The result is a
    fresh list, sorted for stable display ordering; callers may
    mutate it without affecting the locked set.

    Returns:
        A sorted list of dotted keys in the locked set.
    """
    return sorted(_LOCKED_KEYS)
