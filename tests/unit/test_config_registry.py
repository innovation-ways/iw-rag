"""Unit tests for ``iw_rag.config_registry`` (F-00096, S03).

The config registry is the **security keystone** of the Operator Console's
live-edit Config view: the locked set is the only place the "an edit can
NEVER set a locked (security/infra) key" invariant (F-00096 §Invariants 1
& 5) is encoded server-side.  These tests pin:

* Classification of a known editable key → ``"editable"`` with bounds +
  ``hot_reloadable`` flag.
* Classification of a known locked key → ``"locked"``.
* Locked-wins: a key that (by mistake) appears in BOTH the editable
  allowlist and the locked set must classify as ``"locked"`` (Invariant 1).
* Unknown keys classify as ``"unknown"`` (no entry, no fallback).
* ``classify_key`` / ``get_registry_entry`` accept the dotted-path form
  the API serialises (e.g. ``"retrieval.top_k"``).
* A bounds helper is exposed and rejects out-of-range values the design
  calls out (``top_k = -1``).
* The locked set is a ``frozenset`` (immutable) — no caller can mutate it
  and widen the locked set at runtime.
"""

from __future__ import annotations

import dataclasses

import pytest

from iw_rag.config_registry import (
    _LOCKED_KEYS,
    _REGISTRY,
    Bounds,
    RegistryEntry,
    classify_key,
    get_registry_entry,
)

# ---------------------------------------------------------------------------
# RegistryEntry / Bounds — typed structures
# ---------------------------------------------------------------------------


def test_registry_entry_is_typed_frozen_dataclass() -> None:
    """RegistryEntry is a typed (frozen) structure — not a loose dict."""
    assert dataclasses.is_dataclass(RegistryEntry)
    assert RegistryEntry.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
    entry = RegistryEntry(
        description="top_k value",
        bounds=Bounds(kind="int_range", min=1, max=1000),
        hot_reloadable=False,
    )
    # Immutable — security keystone: a caller cannot widen the allowlist
    # by mutating a returned entry.
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
        entry.hot_reloadable = True  # type: ignore[misc]


def test_bounds_int_range_rejects_out_of_range() -> None:
    """Bounds(kind='int_range', ...) rejects values < min or > max."""
    b = Bounds(kind="int_range", min=1, max=1000)
    assert b.check(1) is True
    assert b.check(8) is True
    assert b.check(1000) is True
    assert b.check(0) is False
    assert b.check(-1) is False
    assert b.check(1001) is False


def test_bounds_enum_rejects_unlisted_value() -> None:
    """Bounds(kind='enum', choices=...) accepts only listed values."""
    b = Bounds(kind="enum", choices=("degrade", "fail"))
    assert b.check("degrade") is True
    assert b.check("fail") is True
    assert b.check("other") is False


# ---------------------------------------------------------------------------
# classify_key — editable | locked | unknown
# ---------------------------------------------------------------------------


def test_classify_editable_key_returns_editable() -> None:
    """A known editable key classifies as 'editable' with its entry."""
    # ``retrieval.top_k`` is named in the design as an editable RAG-behaviour
    # key (baked into the spine at startup → pending-restart, not hot).
    assert classify_key("retrieval.top_k") == "editable"
    # And a known hot key too.
    assert classify_key("guardrails.input.threshold") == "editable"


def test_classify_locked_key_returns_locked() -> None:
    """A known locked key classifies as 'locked'."""
    # ``guardrails.enabled`` is the design's named locked key (AC3).
    assert classify_key("guardrails.enabled") == "locked"
    # The DSN is locked too (Invariant 5).
    assert classify_key("database.dsn") == "locked"


def test_classify_unknown_key_returns_unknown() -> None:
    """A key not in either the registry or the locked set classifies as 'unknown'."""
    assert classify_key("this.key.does.not.exist") == "unknown"
    assert classify_key("totally.bogus.field") == "unknown"


def test_classify_locked_wins_when_key_is_both() -> None:
    """Locked-wins: a key present in BOTH registry and locked set is 'locked'.

    Per the design (Invariant 1) a locked key MUST be rejected even if it
    also appears in the editable allowlist (e.g. an operator fat-fingers
    the configuration file).  The classification must be deterministic and
    locked-wins.
    """
    # Simulate the misconfiguration by mutating the in-memory allowlist to
    # include a locked key, then classify — the locked set must still win.
    saved = _REGISTRY.get("guardrails.enabled")
    _REGISTRY["guardrails.enabled"] = RegistryEntry(  # type: ignore[index]
        description="intentionally collides",
        bounds=Bounds(kind="enum", choices=("true", "false")),
        hot_reloadable=True,
    )
    try:
        assert classify_key("guardrails.enabled") == "locked"
    finally:
        if saved is None:
            _REGISTRY.pop("guardrails.enabled", None)  # type: ignore[attr-defined]
        else:
            _REGISTRY["guardrails.enabled"] = saved  # type: ignore[index]


# ---------------------------------------------------------------------------
# get_registry_entry — typed fetch
# ---------------------------------------------------------------------------


def test_get_registry_entry_returns_entry_for_known_editable() -> None:
    """get_registry_entry returns the RegistryEntry for a known editable key."""
    entry = get_registry_entry("retrieval.top_k")
    assert entry is not None
    assert isinstance(entry, RegistryEntry)
    # Baked into RetrievalConfig at startup → not hot-reloadable.
    assert entry.hot_reloadable is False
    assert entry.bounds.kind == "int_range"
    assert entry.bounds.check(8) is True
    assert entry.bounds.check(-1) is False
    assert entry.bounds.check(0) is False


def test_get_registry_entry_returns_none_for_locked_or_unknown() -> None:
    """get_registry_entry returns None for non-editable keys.

    The locked set is a security gate, not an editable allowlist — the
    helper deliberately does not return an entry for a locked key, so the
    service's "key not editable" path stays simple.
    """
    assert get_registry_entry("guardrails.enabled") is None
    assert get_registry_entry("database.dsn") is None
    assert get_registry_entry("not.a.real.key") is None


# ---------------------------------------------------------------------------
# _LOCKED_KEYS — frozenset, non-empty, covers the named invariants
# ---------------------------------------------------------------------------


def test_locked_set_is_frozenset() -> None:
    """The locked set is a frozenset — immutable, not extendable at runtime."""
    # Use type() comparison (stronger than isinstance which can be fooled
    # by exact subclasses, but a frozenset is a terminal type with no
    # realistic subclass).
    assert type(_LOCKED_KEYS) is frozenset


def test_locked_set_includes_named_security_keys() -> None:
    """The locked set covers the keys the design explicitly names (AC3 / Invariant 5)."""
    # AC3 names guardrails.enabled; Invariant 5 extends to guard/judge/model
    # endpoints + DSN + ACL inputs.  Use set membership with a strong
    # equality (operator `in`) — the locked set MUST contain them.
    assert ("guardrails.enabled" in _LOCKED_KEYS) is True
    assert ("database.dsn" in _LOCKED_KEYS) is True


def test_locked_set_includes_model_namespace() -> None:
    """The model registry namespace is locked — endpoints are startup-baked.

    The generator / embedder / reranker / judge / guard / visual_caption
    clients are constructed in ``app.py:_build_query_service`` (and the
    swap-controller sidecar) and cannot be hot-swapped via the overlay.
    The registry is a single ``models`` namespace; locking the parent
    prefix rejects every ``models.<logical_name>.*`` path.
    """
    assert "models" in _LOCKED_KEYS
    # A representative sub-key (e.g. generator's served model endpoint)
    # is also locked because the entire ``models.`` namespace is locked.
    assert classify_key("models.generator.endpoint") == "locked"
