"""Unit tests for the F-00096 S08 Scoreboard HTTP endpoint.

The S08 route is a **thin HTTP facade** over the S05
``ScoreboardService`` (``eval/benchmark/scoreboard.py``).  These tests
inject a fake service via ``app.state.scoreboard_service`` so no live
DB / no live ranking is reached — the route is exercised end-to-end
against the typed fake.

Covers:

* ``GET /benchmark/scoreboard?board=faithfulness`` returns a
  :class:`ScoreboardResponse` with entries in the order the S05
  service emits (AC6: descending faithfulness) and carries the
  ``model_config`` + ``created_at`` per row.
* The route forwards the exact ``board`` query string to the service
  (one of the six supported boards; the route does not invent a
  board name).
* **Empty report history** → 200 with ``entries == []`` (Boundary row
  *"Empty report history → explicit empty state, not an error"*).
* The route is open to any authenticated principal (mirrors
  :func:`list_benchmark_reports`).
* When the service is not wired onto ``app.state``, the route returns
  a typed 4xx (``ValidationError`` → 400) instead of a 5xx.
* The :class:`ScoreboardEntryOut` schema and the
  :class:`ScoreboardResponse` schema carry the field set the S05
  report defines (``report_id``, ``rank``, ``model_config``,
  ``created_at``, ``metric_value``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator

# ---------------------------------------------------------------------------
# Fake ScoreboardService — duck-types ScoreboardService.score()
# ---------------------------------------------------------------------------


@dataclass
class _FakeScoreboardEntry:
    """A canned row the fake returns inside a :class:`_FakeScoreboard`."""

    report_id: str
    model_config: dict[str, Any]
    created_at: str
    metric_value: float | None


@dataclass
class _FakeScoreboard:
    """A canned :class:`Scoreboard`-like return the fake yields."""

    faithfulness: list[_FakeScoreboardEntry]
    ndcg_at_5: list[_FakeScoreboardEntry]
    tok_per_s: list[_FakeScoreboardEntry]
    latency: list[_FakeScoreboardEntry]
    vram_efficiency: list[_FakeScoreboardEntry]
    composite: list[_FakeScoreboardEntry]


class FakeScoreboardService:
    """In-process fake of :class:`ScoreboardService`.

    Tests configure the per-board entry lists via the per-board
    attributes; the ``score()`` coroutine returns the assembled
    :class:`_FakeScoreboard`.  Every call records the exact
    invocation so tests can assert the route forwards the
    ``board`` query string verbatim.
    """

    def __init__(self) -> None:
        self.faithfulness: list[_FakeScoreboardEntry] = []
        self.ndcg_at_5: list[_FakeScoreboardEntry] = []
        self.tok_per_s: list[_FakeScoreboardEntry] = []
        self.latency: list[_FakeScoreboardEntry] = []
        self.vram_efficiency: list[_FakeScoreboardEntry] = []
        self.composite: list[_FakeScoreboardEntry] = []
        self.score_calls: int = 0

    async def score(self) -> _FakeScoreboard:
        """Return a canned :class:`_FakeScoreboard`.

        Each call increments :attr:`score_calls` so tests can assert
        the route called the service exactly once per request.
        """
        self.score_calls += 1
        return _FakeScoreboard(
            faithfulness=list(self.faithfulness),
            ndcg_at_5=list(self.ndcg_at_5),
            tok_per_s=list(self.tok_per_s),
            latency=list(self.latency),
            vram_efficiency=list(self.vram_efficiency),
            composite=list(self.composite),
        )


# ---------------------------------------------------------------------------
# Stub query service — satisfies create_app's `query_service` seam
# ---------------------------------------------------------------------------


class _StubQueryService:
    """Minimal query service stub for the create_app composition root."""

    async def answer(self, query: str, ctx: Any) -> Any:  # pragma: no cover - never called
        raise RuntimeError("not used in S08 scoreboard tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_client(
    *,
    fake: FakeScoreboardService | None,
) -> TestClient:
    """Build a TestClient with a static-token authenticator + a fake service.

    Args:
        fake: The fake :class:`ScoreboardService` to inject on
            ``app.state.scoreboard_service`` (``None`` for the
            unwired-deployment test, which expects a typed 4xx).
    """
    auth = StaticTokenAuthenticator.from_config(
        {"user-token": ["finance"], "eval-token": ["eval-operator"]},
        operator_tokens=[],
    )
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
        scoreboard_service=fake,  # type: ignore[arg-type]
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /benchmark/scoreboard — happy path (AC6)
# ---------------------------------------------------------------------------


def test_get_scoreboard_faithfulness_returns_entries_in_descending_order() -> None:
    """``GET /benchmark/scoreboard?board=faithfulness`` returns a 200 response.

    The response body carries a :class:`ScoreboardResponse` with the
    entries the S05 service emitted (descending faithfulness, per AC6)
    plus the ``model_config`` + ``created_at`` per row.

    The route is a thin HTTP facade: the ordering is computed by the
    S05 service, the route only serialises — the test proves the
    service's order is preserved verbatim.
    """
    fake = FakeScoreboardService()
    fake.faithfulness = [
        _FakeScoreboardEntry(
            report_id="r-high",
            model_config={"generator": "qwen3-14b"},
            created_at="2026-06-02T00:00:00Z",
            metric_value=0.95,
        ),
        _FakeScoreboardEntry(
            report_id="r-mid",
            model_config={"generator": "qwen3-10b"},
            created_at="2026-06-03T00:00:00Z",
            metric_value=0.78,
        ),
        _FakeScoreboardEntry(
            report_id="r-low",
            model_config={"generator": "qwen3-7b"},
            created_at="2026-06-01T00:00:00Z",
            metric_value=0.60,
        ),
    ]
    client = _build_client(fake=fake)

    resp = client.get(
        "/benchmark/scoreboard?board=faithfulness",
        headers={"authorization": "Bearer user-token"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["board"] == "faithfulness"
    entries = body["entries"]
    assert len(entries) == 3
    # The S05 service is the authority on ordering — the route must
    # preserve it verbatim (AC6: descending faithfulness).
    assert [e["report_id"] for e in entries] == ["r-high", "r-mid", "r-low"]
    assert [e["metric_value"] for e in entries] == pytest.approx([0.95, 0.78, 0.60])
    # Each row carries model_config + created_at (AC6 names both).
    assert entries[0]["config_snapshot"] == {"generator": "qwen3-14b"}
    assert entries[0]["created_at"] == "2026-06-02T00:00:00Z"
    # 1-based rank is assigned by the route in the order the service emitted.
    assert [e["rank"] for e in entries] == [1, 2, 3]
    # The service was called exactly once.
    assert fake.score_calls == 1


# ---------------------------------------------------------------------------
# GET /benchmark/scoreboard — board selection is forwarded verbatim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "board_name",
    ["faithfulness", "ndcg_at_5", "tok_per_s", "latency", "vram_efficiency", "composite"],
)
def test_get_scoreboard_forwards_each_supported_board_to_service(
    board_name: str,
) -> None:
    """The route forwards the ``board`` query param verbatim to the service.

    For each of the six boards the S05 service emits, the route picks
    the matching per-board list and serialises it; we verify the route
    selected the right list by checking the response body matches the
    canned per-board entry list.
    """
    fake = FakeScoreboardService()
    entry = _FakeScoreboardEntry(
        report_id=f"r-{board_name}",
        model_config={"generator": "qwen3-14b"},
        created_at="2026-06-02T00:00:00Z",
        metric_value=0.5,
    )
    setattr(fake, board_name, [entry])
    client = _build_client(fake=fake)

    resp = client.get(
        f"/benchmark/scoreboard?board={board_name}",
        headers={"authorization": "Bearer user-token"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["board"] == board_name
    assert len(body["entries"]) == 1
    assert body["entries"][0]["report_id"] == f"r-{board_name}"
    # The service was called exactly once.
    assert fake.score_calls == 1


# ---------------------------------------------------------------------------
# GET /benchmark/scoreboard — empty report history (Boundary row)
# ---------------------------------------------------------------------------


def test_get_scoreboard_empty_history_returns_200_with_empty_entries() -> None:
    """0 persisted reports → 200 with ``entries == []`` (Boundary row).

    The route MUST return an explicit empty state — NOT a 404 / 500
    (Boundary row *"Empty report history → explicit empty state, not
    an error"*).  The scoreboard service itself returns empty per-
    board lists when the history is empty; the route surfaces them
    verbatim.
    """
    fake = FakeScoreboardService()  # every board is []
    client = _build_client(fake=fake)

    resp = client.get(
        "/benchmark/scoreboard?board=faithfulness",
        headers={"authorization": "Bearer user-token"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["board"] == "faithfulness"
    assert body["entries"] == []
    # The service was still called (the route is the entry point).
    assert fake.score_calls == 1


# ---------------------------------------------------------------------------
# GET /benchmark/scoreboard — auth keystone (open to any authenticated principal)
# ---------------------------------------------------------------------------


def test_get_scoreboard_does_not_require_operator() -> None:
    """``GET /benchmark/scoreboard`` is open to any authenticated principal.

    Mirrors :func:`list_benchmark_reports` — the scoreboard reveals no
    sensitive content (ranked metric values, public model config
    snapshots).  A non-operator principal gets a 200.
    """
    fake = FakeScoreboardService()
    client = _build_client(fake=fake)

    resp = client.get(
        "/benchmark/scoreboard?board=faithfulness",
        headers={"authorization": "Bearer user-token"},
    )

    assert resp.status_code == 200


def test_get_scoreboard_missing_token_is_401() -> None:
    """No bearer token → 401 (the auth keystone still applies)."""
    fake = FakeScoreboardService()
    client = _build_client(fake=fake)

    resp = client.get("/benchmark/scoreboard?board=faithfulness")

    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /benchmark/scoreboard — unwired deployment (typed 4xx, not 5xx)
# ---------------------------------------------------------------------------


def test_get_scoreboard_unwired_service_returns_400() -> None:
    """When ``scoreboard_service`` is not wired, GET returns 400 (not 5xx).

    Mirrors the ``_benchmark_service`` / ``_swap_controller_client`` /
    ``_config_service`` seams — an unwired deployment surfaces a typed
    :class:`ValidationError` that ``_ERROR_STATUS`` maps to 400.
    """
    auth = StaticTokenAuthenticator.from_config({"u": ["u"]}, [])
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
    )
    client = TestClient(app)
    resp = client.get(
        "/benchmark/scoreboard?board=faithfulness",
        headers={"authorization": "Bearer u"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Schemas — ScoreboardResponse / ScoreboardEntryOut expose the documented fields
# ---------------------------------------------------------------------------


def test_scoreboard_entry_out_schema_field_set() -> None:
    """``ScoreboardEntryOut`` exposes the documented S05-aligned field set."""
    from iw_rag.api.schemas import ScoreboardEntryOut

    fields = set(ScoreboardEntryOut.model_fields.keys())
    # ``seq`` + ``display_id`` carry the derived sequential execution
    # number (BENCH-000001 …) so the console shows a human-readable run id
    # per leaderboard row instead of the random uuid.
    expected = {
        "report_id",
        "seq",
        "display_id",
        "rank",
        "config_snapshot",
        # Human dataset name resolved from eval_datasets — the board renders it
        # instead of the opaque uuid buried in config_snapshot.
        "dataset_name",
        "created_at",
        "metric_value",
    }
    assert fields == expected, (
        f"ScoreboardEntryOut must expose {expected}; got {fields}. "
        f"Adding/removing fields breaks the S05 → API contract."
    )


def test_scoreboard_response_schema_field_set() -> None:
    """``ScoreboardResponse`` exposes the documented field set."""
    from iw_rag.api.schemas import ScoreboardResponse

    fields = set(ScoreboardResponse.model_fields.keys())
    expected = {"board", "entries"}
    assert fields == expected, f"ScoreboardResponse must expose {expected}; got {fields}."


# ---------------------------------------------------------------------------
# App wiring: scoreboard_service is injectable on app.state
# ---------------------------------------------------------------------------


def test_create_app_wires_scoreboard_service_to_state() -> None:
    """``create_app(..., scoreboard_service=fake)`` places the service on ``app.state``."""
    auth = StaticTokenAuthenticator.from_config({"u": ["u"]}, [])
    fake = FakeScoreboardService()
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
        scoreboard_service=fake,  # type: ignore[arg-type]
    )
    assert app.state.scoreboard_service is fake
    paths = {r.path for r in app.routes}  # type: ignore[attr-defined]
    assert "/benchmark/scoreboard" in paths
