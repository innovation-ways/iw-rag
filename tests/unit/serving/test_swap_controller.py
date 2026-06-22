"""Unit tests for the F-00094 swap-controller.

TDD RED→GREEN (S02): these tests were authored before
``src/iw_rag/serving/`` existed.  They drive the implementation of the
catalog loader, the activation state machine, and the swap-controller
client.  All llama-swap HTTP calls are MOCKED — no live GPU/containers.

Coverage map (mirrors F-00094 design §Acceptance Criteria + Boundary
Behavior + this step's prompt):

  * AC1: state machine ``idle → swapping → ready`` when a mocked
    ``/v1/models`` returns 200.
  * AC2: ``loading → failed`` on a mocked timeout; the controller MUST
    never report ``ready`` before a 200.
  * Boundary row 1: poll budget is exactly ``health_timeout_s``.
  * Boundary row 2: unknown ``model_id`` → typed ``NotFoundError``,
    no swap attempted.
  * Boundary row 3: ``catalog_with_state()`` annotates ``unreachable``
    when the client raises a connection error (no crash).
  * Idempotent re-activation of the already-active+ready model is a
    fast no-op returning ``ready``.
  * ``swap_controller_client`` propagates ``LlamaSwapUnavailable`` on
    connection error.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from iw_rag.domain.errors import NotFoundError
from iw_rag.serving.catalog import load_catalog
from iw_rag.serving.llama_swap_client import LlamaSwapUnavailable, ModelState
from iw_rag.serving.swap_controller import (
    ActivationState,
    SwapController,
)

# ---------------------------------------------------------------------------
# Fixtures — minimal valid catalog
# ---------------------------------------------------------------------------


def _valid_catalog_dict() -> dict[str, object]:
    """Return a minimal valid model_catalog payload (mirrors S01 yaml)."""
    return {
        "models": {
            "qwen3-7b-awq": {
                "id": "qwen3-7b-awq",
                "served_name": "Qwen/Qwen3-7B-AWQ",
                "role": "generation",
                "quantization": "awq",
                "expected_vram_gb": 5.0,
                "max_context": 16384,
                "vllm_flags": [
                    "--quantization awq",
                    "--kv-cache-dtype fp8",
                ],
            },
            "qwen3-14b-awq": {
                "id": "qwen3-14b-awq",
                "served_name": "Qwen/Qwen3-14B-AWQ",
                "role": "generation",
                "quantization": "awq",
                "expected_vram_gb": 9.0,
                "max_context": 16384,
                "vllm_flags": ["--quantization awq"],
            },
        }
    }


def _write_catalog(tmp_path, payload: dict[str, object] | None) -> str:
    """Write a catalog YAML file under tmp_path; return its path."""
    p = tmp_path / "model_catalog.yaml"
    if payload is not None:
        p.write_text(yaml.safe_dump(payload))
    return str(p)


# ---------------------------------------------------------------------------
# 1. SwapController state machine — mocked llama-swap client
#    (load_catalog / Catalog.get tests live in tests/unit/serving/test_catalog.py)
# ---------------------------------------------------------------------------


def _stub_client(
    *,
    list_models_payload: list[ModelState] | None = None,
    health_returns: list[bool] | None = None,
    activate_raises: Exception | None = None,
) -> MagicMock:
    """Build a MagicMock that quacks like ``LlamaSwapClient``."""
    client = MagicMock()
    client.list_models = AsyncMock(
        return_value=list_models_payload if list_models_payload is not None else []
    )
    if health_returns is None:
        health_returns = [True]
    # Use a callable that pops from a deque padded with the last value
    # so the iterator cannot be exhausted (AsyncMock raises
    # StopAsyncIteration once the side_effect list runs out, and
    # IndexError on an empty deque).  Pad generously to outlast the
    # poll loop.
    last = health_returns[-1]
    queue: deque[bool] = deque(list(health_returns) + [last] * 1_000_000)

    async def _next_health(model_id: str) -> bool:
        if not queue:
            return last
        return queue.popleft()

    client.model_health = AsyncMock(side_effect=_next_health)
    if activate_raises is None:
        client.activate = AsyncMock(return_value=None)
    else:
        client.activate = AsyncMock(side_effect=activate_raises)
    return client


def _build_controller(
    client: MagicMock,
    *,
    health_timeout_s: int = 300,
    poll_interval_s: float = 0.0,
) -> SwapController:
    """Build a controller with a real catalog + stubbed client."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(_valid_catalog_dict(), fh)
        path = fh.name
    cat = load_catalog(path)
    return SwapController(
        catalog=cat,
        client=client,  # type: ignore[arg-type]
        health_timeout_s=health_timeout_s,
        poll_interval_s=poll_interval_s,
    )


@pytest.mark.asyncio
async def test_state_machine_idle_to_swapping_to_ready() -> None:
    """AC1: on a successful health poll, status transitions to ``ready``."""
    client = _stub_client(health_returns=[True])
    ctrl = _build_controller(client)

    s0 = ctrl.status()
    assert s0.state == ActivationState.IDLE
    assert s0.active_model_id is None

    s1 = await ctrl.activate("qwen3-7b-awq")
    assert s1.state == ActivationState.READY
    assert s1.active_model_id == "qwen3-7b-awq"

    client.activate.assert_awaited_once_with("qwen3-7b-awq")
    client.model_health.assert_awaited()


@pytest.mark.asyncio
async def test_activate_raises_notfound_for_unknown_id() -> None:
    """Boundary row 2: unknown model id → ``NotFoundError``; no swap attempted."""
    client = _stub_client()
    ctrl = _build_controller(client)
    with pytest.raises(NotFoundError, match="Unknown catalog model id 'nope'"):
        await ctrl.activate("nope")
    client.activate.assert_not_awaited()
    client.model_health.assert_not_awaited()


@pytest.mark.asyncio
async def test_activate_idempotent_when_already_ready() -> None:
    """Activating the already-active+ready model is a fast no-op."""
    client = _stub_client(health_returns=[True])
    ctrl = _build_controller(client)
    await ctrl.activate("qwen3-7b-awq")
    client.activate.reset_mock()
    client.model_health.reset_mock()
    s = await ctrl.activate("qwen3-7b-awq")
    assert s.state == ActivationState.READY
    assert s.active_model_id == "qwen3-7b-awq"
    client.activate.assert_not_awaited()
    client.model_health.assert_not_awaited()


@pytest.mark.asyncio
async def test_activate_transitions_to_failed_on_timeout() -> None:
    """AC2: never report ``ready`` on timeout; transitions to ``failed``."""
    client = _stub_client(health_returns=[False] * 10_000)
    ctrl = _build_controller(client, health_timeout_s=1, poll_interval_s=0.0)
    s = await ctrl.activate("qwen3-7b-awq")
    assert s.state == ActivationState.FAILED
    assert s.failure_reason == "timeout"
    assert s.active_model_id is None


@pytest.mark.asyncio
async def test_activate_transitions_to_failed_on_client_error() -> None:
    """A client connection error surfaces a typed ``LlamaSwapUnavailable``."""
    client = _stub_client(activate_raises=LlamaSwapUnavailable("connect refused"))
    ctrl = _build_controller(client, health_timeout_s=10, poll_interval_s=0.0)
    with pytest.raises(LlamaSwapUnavailable, match="connect refused"):
        await ctrl.activate("qwen3-7b-awq")
    s = ctrl.status()
    assert s.state == ActivationState.FAILED


# ---------------------------------------------------------------------------
# 3. catalog_with_state — annotation contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_catalog_with_state_marks_active_model() -> None:
    """The active model is annotated ``active``; others are ``available``."""
    client = _stub_client(
        list_models_payload=[
            ModelState(id="qwen3-14b-awq", state="available"),
            ModelState(id="qwen3-7b-awq", state="available"),
        ],
        health_returns=[True],
    )
    ctrl = _build_controller(client)
    await ctrl.activate("qwen3-7b-awq")
    rows = await ctrl.catalog_with_state()
    by_id = {r.id: r for r in rows}
    assert by_id["qwen3-7b-awq"].state == "active"
    assert by_id["qwen3-14b-awq"].state == "available"


@pytest.mark.asyncio
async def test_catalog_with_state_marks_unreachable_on_client_error() -> None:
    """Boundary row 3: llama-swap down → every model ``unreachable``, no crash."""
    client = MagicMock()
    client.list_models = AsyncMock(side_effect=LlamaSwapUnavailable("nope"))
    client.model_health = AsyncMock(return_value=False)
    client.activate = AsyncMock(return_value=None)
    ctrl = _build_controller(client)
    rows = await ctrl.catalog_with_state()
    assert rows
    for row in rows:
        assert row.state == "unreachable"


# ---------------------------------------------------------------------------
# 4. swap_controller_client — connection error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_swap_controller_client_maps_connection_error_to_typed_exception() -> None:
    """A connection error to the sidecar raises ``LlamaSwapUnavailable``."""
    from iw_rag.serving.swap_controller_client import SwapControllerClient

    client = SwapControllerClient(base_url="http://swap-controller:8090", timeout_s=1.0)

    async def _boom(*args, **kwargs):
        import httpx as _httpx

        raise _httpx.ConnectError("connection refused")

    with patch("httpx.AsyncClient.get", new=_boom), pytest.raises(LlamaSwapUnavailable):
        await client.get_catalog()


# ---------------------------------------------------------------------------
# 5. S05 — Invariant 1: Infinity is never in the controller's activate path
# ---------------------------------------------------------------------------
# The Infinity embedder/reranker is ALWAYS resident.  The controller must
# never call the llama-swap client with model_id="infinity" (or any model
# id that is not a catalog entry).  We assert the controller raises a
# typed NotFoundError if asked, and never reaches the client.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_activate_raises_notfound_for_infinity_id() -> None:
    """Invariant 1: activating ``infinity`` → ``NotFoundError``; no swap attempted.

    The Infinity entry lives in ``config/llama-swap.yaml`` as a persistent
    entry (outside the exclusive ``generators`` group).  The catalog only
    carries generation-role models; ``infinity`` is therefore not a valid
    catalog id and the controller MUST reject it before reaching the
    llama-swap client.
    """
    client = _stub_client()
    ctrl = _build_controller(client)
    with pytest.raises(NotFoundError, match="Unknown catalog model id 'infinity'"):
        await ctrl.activate("infinity")
    client.activate.assert_not_awaited()
    client.model_health.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_never_lists_infinity_in_catalog_with_state() -> None:
    """Invariant 1: ``catalog_with_state`` never surfaces ``infinity`` as a row.

    The catalog only carries generation-role models; Infinity is not a
    catalog entry and is never in the per-model state listing.
    """
    client = _stub_client(
        list_models_payload=[
            ModelState(id="qwen3-7b-awq", state="available"),
            ModelState(id="infinity", state="available"),  # would be in llama-swap but not catalog
        ],
        health_returns=[True],
    )
    ctrl = _build_controller(client)
    rows = await ctrl.catalog_with_state()
    ids = {r.id for r in rows}
    assert "infinity" not in ids, "Infinity is not a catalog entry and must not appear in rows"
    assert "qwen3-7b-awq" in ids


# ---------------------------------------------------------------------------
# 6. S05 — Patched clock: timeout test that does not depend on real wall time
# ---------------------------------------------------------------------------
# The prompt requires: "patch asyncio.sleep/the clock so the timeout test
# is fast."  The S02 test uses ``poll_interval_s=0.0`` to avoid real sleep,
# but it still relies on ``time.monotonic()`` advancing naturally (the
# deadline is real time + health_timeout_s).  Below we patch the clock so
# the deadline can be tripped deterministically.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_activate_failed_on_timeout_with_patched_clock() -> None:
    """AC2 / Boundary row 1: never ``ready``; transitions to ``failed(timeout)``.

    Patches ``time.monotonic`` in the controller module so the deadline
    trips on the first poll, regardless of how many polls the loop would
    otherwise run.  No real sleep, no real wait — fully deterministic.
    """
    from iw_rag.serving import swap_controller as swap_controller_mod

    client = _stub_client(health_returns=[False] * 10_000)
    ctrl = _build_controller(client, health_timeout_s=1, poll_interval_s=0.0)

    # The poll loop is:
    #   deadline = time.monotonic() + health_timeout_s   (once)
    #   while True:
    #     ok = await self._client.model_health(model_id)
    #     if time.monotonic() >= deadline: return False
    #     await asyncio.sleep(self._poll_interval_s)
    #
    # Patching ``time.monotonic`` to advance the clock on the *first*
    # read of the deadline check trips the loop after the first poll.
    # We use a side_effect list — the first call (sets started_at)
    # returns 0.0; every subsequent call (including the deadline
    # computation and the per-iter check) returns 5.0, so the deadline
    # becomes 5.0 + 1 = 6.0 ... wait, that's past 5.0.  We flip it:
    # return 0.0 the first time, then always 5.0 EXCEPT once, where
    # we need deadline < now.  Cleanest: arrange so the first call is
    # the deadline set (so deadline = 0 + 1 = 1.0) and every later
    # call returns 5.0.
    #
    # The controller calls monotonic twice before the loop: once for
    # started_at (line 228) and once for the deadline (line 267).  So:
    # call 1 -> 0.0 (started_at); call 2 -> 0.0 (deadline = 1.0);
    # call 3+ -> 5.0 (5.0 >= 1.0 → exit).
    monotonic_values = iter([0.0, 0.0, 5.0, 5.0, 5.0, 5.0])

    def _fast_clock() -> float:
        return next(monotonic_values)

    with patch.object(swap_controller_mod.time, "monotonic", side_effect=_fast_clock):
        s = await ctrl.activate("qwen3-7b-awq")

    assert s.state == ActivationState.FAILED
    assert s.failure_reason == "timeout"
    assert s.active_model_id is None
    client.activate.assert_awaited_once_with("qwen3-7b-awq")
    assert client.model_health.await_count >= 1


@pytest.mark.asyncio
async def test_poll_loop_handles_transient_connection_error() -> None:
    """A transient ``LlamaSwapUnavailable`` on a health poll does NOT fail the activation.

    The controller treats transient connection errors on health polls as
    "not ready yet" and keeps polling until the budget elapses (which
    trips the timeout branch).  This is the resilience contract: a single
    bad poll does not abort the activation.
    """
    from collections import deque

    client = MagicMock()
    client.list_models = AsyncMock(return_value=[])
    client.activate = AsyncMock(return_value=None)

    queue: deque[object] = deque(
        [
            LlamaSwapUnavailable("transient"),
            LlamaSwapUnavailable("transient"),
            True,  # third poll succeeds
        ]
    )

    async def _health(model_id: str) -> bool:  # noqa: ARG001
        v = queue.popleft()
        if isinstance(v, Exception):
            raise v
        return bool(v)

    client.model_health = AsyncMock(side_effect=_health)
    ctrl = _build_controller(client, health_timeout_s=300, poll_interval_s=0.0)
    s = await ctrl.activate("qwen3-7b-awq")
    assert s.state == ActivationState.READY
    assert s.active_model_id == "qwen3-7b-awq"


# ---------------------------------------------------------------------------
# 7. S05 — catalog_with_state: every row is one of the documented states
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_catalog_with_state_every_row_has_valid_state() -> None:
    """Every row in ``catalog_with_state`` carries a documented state value."""
    client = _stub_client(
        list_models_payload=[ModelState(id="qwen3-7b-awq", state="available")],
        health_returns=[True],
    )
    ctrl = _build_controller(client)
    rows = await ctrl.catalog_with_state()
    valid_states = {"available", "loading", "active", "unreachable"}
    for row in rows:
        assert row.state in valid_states, f"unexpected state {row.state!r}"


@pytest.mark.asyncio
async def test_catalog_with_state_active_id_is_consistent_with_status() -> None:
    """The active model id in ``status()`` is the same as the row marked ``active``."""
    client = _stub_client(health_returns=[True])
    ctrl = _build_controller(client)
    await ctrl.activate("qwen3-14b-awq")
    rows = await ctrl.catalog_with_state()
    active_rows = [r for r in rows if r.state == "active"]
    assert len(active_rows) == 1
    assert active_rows[0].id == "qwen3-14b-awq"
    assert ctrl.status().active_model_id == "qwen3-14b-awq"
