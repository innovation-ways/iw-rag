"""Sanity gate: AccessContext is immutable and its groups are a tuple.

This is a security-shaped invariant: ``principal_groups`` must not be a list a
downstream caller could append to (which would widen entitlements).
"""

from __future__ import annotations

import dataclasses

import pytest

from iw_rag.domain.acl import AccessContext


def test_is_frozen() -> None:
    ctx = AccessContext(principal_id="u1", principal_groups=("finance",))
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
        ctx.principal_id = "u2"  # type: ignore[misc]


def test_groups_are_an_immutable_tuple() -> None:
    ctx = AccessContext(principal_id="u1", principal_groups=("finance", "all-staff"))
    assert isinstance(ctx.principal_groups, tuple)
    assert not hasattr(ctx.principal_groups, "append")


def test_defaults() -> None:
    ctx = AccessContext(principal_id="u1", principal_groups=())
    assert ctx.is_operator is False
    assert ctx.request_id == ""
