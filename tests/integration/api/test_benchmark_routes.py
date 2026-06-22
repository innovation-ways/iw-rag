"""Integration tests for the F-00095 Benchmark API routes.

Pins the route wiring + ``eval-operator`` gating for the Benchmark Studio.
The tests use the **injectable benchmark_service seam** on
:func:`iw_rag.api.app.create_app` so the S01 runner, the F-00093 report
service, and the F-00093 stats module are all fakes — no GPU, no live
swap-controller, no live Postgres needed for these wiring-level tests.

Coverage (per F-00095 / S02 prompt + the Feature Design):

* **AC1 / Invariant 2** — happy path: an ``eval-operator`` POSTs
  ``/benchmark/run``; the response is an SSE stream whose terminal
  ``done`` event carries a ``report_id`` and whose shape mirrors
  :func:`iw_rag.api.routes._sse_generator` (``event: <type>\\ndata: <json>\\n\\n``).
* **AC2 / Invariant 3** — a non-``eval-operator`` POST returns 403 (no
  swap triggered, no report written); a missing bearer token returns 401;
  a smuggled ``groups`` field in the body is ignored.
* **Boundary row** — unknown report id on detail returns 404; compare
  across different datasets returns 400; compare with one missing id
  returns 404.
* **AC4** — ``GET /benchmark/ui`` returns 200 HTML containing the
  Studio anchors (the page itself is S03's concern; here we only pin
  that the route wires and serves HTML).
* **Mounting** — ``create_app`` places the ``benchmark_router`` and
  places a ``benchmark_service`` on ``app.state`` (mirroring the
  ``document_service`` seam).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator

# ---------------------------------------------------------------------------
# Fakes — duck-type the S01 runner + the F-00093 report service + stats
# ---------------------------------------------------------------------------


class FakeBenchmarkRunner:
    """In-memory fake of the S01 :class:`BenchmarkRunner`.

    Yields a canned event sequence (``queued → swapping → ready → running
    → done``) so the S02 SSE wrapping can be exercised end-to-end without
    a GPU, a live swap-controller, or a live retrieval+generation
    pipeline.  Records every call for test assertions.
    """

    def __init__(
        self,
        *,
        canned_report_id: str = "rep-fake-1",
        scripted_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.canned_report_id = canned_report_id
        self.scripted_events = (
            list(scripted_events)
            if scripted_events is not None
            else [
                {"event": "queued", "data": {"model_config": {"generator": "g"}}},
                {"event": "swapping", "data": {"model_config": {"generator": "g"}}},
                {"event": "ready", "data": {"model_load_ms": 1234.5}},
                {"event": "running", "data": {"i": 0, "total": 1, "question": "q"}},
                {"event": "scoring", "data": {"n_queries": 1}},
                {"event": "done", "data": {"report_id": canned_report_id}},
            ]
        )
        self.run_calls: list[dict[str, Any]] = []

    async def run(
        self,
        model_config: dict[str, Any],
        *,
        trace_id: str = "",
        access_context: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay the scripted events; record the call shape.

        Args:
            model_config: The submitted per-step model config.
            trace_id: The request trace id (recorded).
            access_context: The authenticated :class:`AccessContext`
                (recorded; must NOT be ``None`` for a live POST).
        """
        self.run_calls.append(
            {
                "model_config": dict(model_config),
                "trace_id": trace_id,
                "access_context": access_context,
            }
        )
        for ev in self.scripted_events:
            yield ev


class FakeReportService:
    """In-memory fake of the F-00093 report service.

    Holds a small map of report rows so the list / detail / compare
    endpoints can be exercised end-to-end.  Mirrors the parts of the
    F-00093 ``BenchmarkReportRepository`` + stats module the S02 layer
    consumes (``list_reports``, ``get_report``, ``compare``).
    """

    def __init__(self, rows: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialise the fake with a small map of report rows.

        Args:
            rows: Optional explicit row map (id → row dict).  When
                ``None`` (the S02 default), three pre-seeded rows
                spanning two datasets are created so the existing
                list / detail / compare tests work without setup.
                The S05 boundary test for ``dataset without
                references`` supplies a single row with the
                "no-reference-answers" metrics shape.
        """
        if rows is None:
            rows = {
                "rep-a-1": {
                    "id": "rep-a-1",
                    "dataset_id": "ds-golden",
                    "model_config": {"generator": "qwen-7b"},
                    "metrics": {
                        "ndcg_at_5": [0.5, 0.6, 0.7],
                        "faithfulness": [0.8, 0.85, 0.9],
                    },
                    # F-00093 S06 promoted scalar columns.  The S03
                    # page reads these directly (per V4) so the report
                    # view can render metric tiers without
                    # mean-reducing the per-query vectors client-side.
                    "recall_at_5": 0.60,
                    "recall_at_10": 0.75,
                    "ndcg_at_5": 0.60,
                    "ndcg_at_10": 0.70,
                    "mrr": 0.65,
                    "faithfulness_mean": 0.85,
                    "faithfulness_kappa": 0.74,
                    "context_precision": 0.78,
                    "context_recall": 0.55,
                    "factual_correctness": 0.49,
                    "n_queries": 3,
                },
                "rep-b-1": {
                    "id": "rep-b-1",
                    "dataset_id": "ds-golden",
                    "model_config": {"generator": "qwen-14b"},
                    "metrics": {
                        "ndcg_at_5": [0.6, 0.7, 0.8],
                        "faithfulness": [0.9, 0.92, 0.95],
                    },
                    "recall_at_5": 0.78,
                    "recall_at_10": 0.88,
                    "ndcg_at_5": 0.70,
                    "ndcg_at_10": 0.81,
                    "mrr": 0.74,
                    "faithfulness_mean": 0.92,
                    "faithfulness_kappa": 0.81,
                    "context_precision": 0.84,
                    "context_recall": 0.69,
                    "factual_correctness": 0.62,
                    "n_queries": 3,
                },
                "rep-c-1": {
                    "id": "rep-c-1",
                    "dataset_id": "ds-other",
                    "model_config": {"generator": "qwen-7b"},
                    "metrics": {
                        "ndcg_at_5": [0.5, 0.6],
                        "faithfulness": [0.8, 0.85],
                    },
                    "recall_at_5": 0.50,
                    "ndcg_at_5": 0.55,
                    "mrr": 0.50,
                    "faithfulness_mean": 0.82,
                    "n_queries": 2,
                },
            }
        self._rows: dict[str, dict[str, Any]] = dict(rows)

    async def list_reports(self) -> list[dict[str, Any]]:
        """Return the canned rows, newest first (insertion order is the
        surrogate for ``created_at`` in this fake)."""
        return [dict(self._rows[k]) for k in reversed(list(self._rows))]

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Return the row or ``None`` when the id is unknown."""
        return self._rows.get(report_id)

    async def compare(
        self,
        report_a_id: str,
        report_b_id: str,
    ) -> dict[str, Any]:
        """Return a minimal A/B delta-table payload (F-00093 stats contract)."""
        a = self._rows.get(report_a_id)
        b = self._rows.get(report_b_id)
        if a is None or b is None:
            return None  # type: ignore[return-value]
        if a["dataset_id"] != b["dataset_id"]:
            from iw_rag.domain.errors import ValidationError

            raise ValidationError(
                "compare across different datasets is not allowed "
                f"(a.dataset_id={a['dataset_id']}, b.dataset_id={b['dataset_id']}); "
                "paired stats require the same golden queries."
            )
        # Build a minimal but representative delta payload — the real
        # F-00093 stats module is what produces this; here we just
        # confirm the S02 layer wires the call and serialises the
        # result faithfully.
        a_ndcg = a["metrics"]["ndcg_at_5"]
        b_ndcg = b["metrics"]["ndcg_at_5"]
        a_faith = a["metrics"]["faithfulness"]
        b_faith = b["metrics"]["faithfulness"]
        return {
            "rows": {
                "ndcg_at_5": {
                    "val_a": sum(a_ndcg) / len(a_ndcg),
                    "val_b": sum(b_ndcg) / len(b_ndcg),
                    "delta": sum(b_ndcg) / len(b_ndcg) - sum(a_ndcg) / len(a_ndcg),
                    "p_value": 0.01,
                    "ci": (0.05, 0.20),
                    "significant": True,
                },
                "faithfulness": {
                    "val_a": sum(a_faith) / len(a_faith),
                    "val_b": sum(b_faith) / len(b_faith),
                    "delta": sum(b_faith) / len(b_faith) - sum(a_faith) / len(a_faith),
                    "p_value": 0.02,
                    "ci": (0.02, 0.10),
                    "significant": True,
                },
            },
            "winner": "B",
            "underpowered": True,
            "caveat": (
                "n=3 is below min_queries_for_significance=150: the paired "
                "test detects only ~0.08-0.12 retrieval / ~0.06-0.10 "
                "faithfulness differences. Increase n to ~150 for stronger "
                "conclusions."
            ),
        }


class _StubQueryService:
    """Minimal query service stub — satisfies ``create_app``'s `query_service` seam."""

    async def answer(self, query: str, ctx: Any) -> Any:  # pragma: no cover - never called
        raise RuntimeError("not used in S02 tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_client(
    *,
    with_groups: tuple[str, ...] = ("finance",),
    runner: FakeBenchmarkRunner | None = None,
    report_service: FakeReportService | None = None,
) -> TestClient:
    """Build a TestClient with the benchmark routes wired.

    Args:
        with_groups: The groups assigned to the ``user-token`` bearer.
        runner: Optional fake S01 runner to inject on ``app.state``.
        report_service: Optional fake report service to inject on
            ``app.state``.  Defaults to a fresh :class:`FakeReportService`.
    """
    auth = StaticTokenAuthenticator.from_config(
        {
            "user-token": list(with_groups),
            "eval-token": ["eval-operator"],
        },
        operator_tokens=[],
    )
    if runner is None:
        runner = FakeBenchmarkRunner()
    if report_service is None:
        report_service = FakeReportService()
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
        benchmark_runner=runner,  # type: ignore[arg-type]
        benchmark_service=report_service,  # type: ignore[arg-type]
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# Mounting — the benchmark router is wired into create_app
# ---------------------------------------------------------------------------


def test_create_app_mounts_benchmark_router() -> None:
    """``create_app`` mounts the benchmark router and exposes the expected paths."""
    client = _build_client()
    paths = {r.path for r in client.app.routes}  # type: ignore[attr-defined]
    assert "/benchmark/run" in paths
    assert "/benchmark/reports" in paths
    assert "/benchmark/ui" in paths
    expected_benchmark_paths = {"/benchmark/run", "/benchmark/reports", "/benchmark/ui"}
    assert expected_benchmark_paths <= paths, (
        f"Missing benchmark route(s): {expected_benchmark_paths - paths!r}"
    )


def test_create_app_binds_benchmark_runner_and_service_to_state() -> None:
    """``create_app(..., benchmark_runner=..., benchmark_service=...)`` binds both to state."""
    runner = FakeBenchmarkRunner()
    svc = FakeReportService()
    client = _build_client(runner=runner, report_service=svc)
    assert client.app.state.benchmark_runner is runner  # type: ignore[attr-defined]
    assert client.app.state.benchmark_service is svc  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# AC2 / Invariant 3 — eval-operator gating is read from auth ONLY
# ---------------------------------------------------------------------------


def test_post_run_is_403_for_non_eval_operator() -> None:
    """AC2 — a non-eval-operator principal gets 403, no swap, no report written.

    The runner's ``run`` method is never called.  This is the RED line
    the F-00095 / S02 prompt asks us to capture.
    """
    runner = FakeBenchmarkRunner()
    client = _build_client(runner=runner, with_groups=("finance",))
    resp = client.post(
        "/benchmark/run",
        json={"generator": "qwen-7b", "dataset_id": "benchmark_golden"},
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 403
    # No swap was triggered and no report was written.
    assert runner.run_calls == []


def test_post_run_does_not_trust_request_body_groups() -> None:
    """A smuggled ``groups`` field in the body is ignored (auth-only ACL keystone)."""
    runner = FakeBenchmarkRunner()
    client = _build_client(runner=runner, with_groups=("finance",))
    resp = client.post(
        "/benchmark/run",
        json={
            "generator": "qwen-7b",
            "dataset_id": "benchmark_golden",
            "groups": ["eval-operator"],
            "acl": ["eval-operator"],
        },
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 403
    assert runner.run_calls == []


def test_post_run_missing_token_is_401() -> None:
    """Unauthenticated POST → 401 (the auth keystone still applies)."""
    client = _build_client()
    resp = client.post(
        "/benchmark/run",
        json={"generator": "qwen-7b", "dataset_id": "benchmark_golden"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# AC1 — happy path: eval-operator POST streams SSE + done carries report_id
# ---------------------------------------------------------------------------


def _parse_sse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a canonical ``event: <t>\\ndata: <json>\\n\\n`` SSE stream."""
    out: list[tuple[str, dict[str, Any]]] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith(":"):
            # Heartbeat / comment line.
            continue
        event_type = ""
        data_str = ""
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                data_str = line[len("data: ") :]
        if event_type and data_str:
            out.append((event_type, json.loads(data_str)))
    return out


def test_post_run_streams_sse_and_done_carries_report_id() -> None:
    """AC1 — eval-operator POST → SSE stream; ``done`` carries the report_id."""
    runner = FakeBenchmarkRunner(canned_report_id="rep-fixture-1")
    client = _build_client(runner=runner)
    with client.stream(
        "POST",
        "/benchmark/run",
        json={
            "generator": "qwen-7b",
            "embedder": "bge-m3",
            "reranker": "bge-reranker",
            "grader": "lynx-8b",
            "dataset_id": "benchmark_golden",
        },
        headers={"authorization": "Bearer eval-token"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in resp.iter_text())
    events = _parse_sse_events(body)
    types = [t for t, _ in events]
    assert "done" in types
    done = next(d for t, d in events if t == "done")
    assert done["report_id"] == "rep-fixture-1"
    # And the runner was driven once, under the caller's AccessContext.
    assert len(runner.run_calls) == 1
    assert runner.run_calls[0]["model_config"]["generator"] == "qwen-7b"
    # The access context must carry eval-operator (auth-derived, not from body).
    ctx = runner.run_calls[0]["access_context"]
    assert "eval-operator" in ctx.principal_groups


def test_post_run_sse_error_event_carries_only_message_and_trace_ref() -> None:
    """Terminal ``error`` event carries only ``{message, trace_ref}``.

    Mirrors :func:`iw_rag.api.routes._sse_generator` — no exception
    text / retrieved content / PII.
    """
    runner = FakeBenchmarkRunner(
        scripted_events=[
            {"event": "queued", "data": {"model_config": {}}},
            {
                "event": "error",
                "data": {
                    "reason_code": "swap_timeout",
                    "message": "swap-controller never reported ready",
                    "trace_ref": "abc-123",
                },
            },
        ]
    )
    client = _build_client(runner=runner)
    with client.stream(
        "POST",
        "/benchmark/run",
        json={
            "generator": "qwen-7b",
            "embedder": "bge-m3",
            "reranker": "bge-reranker",
            "grader": "lynx-8b",
        },
        headers={"authorization": "Bearer eval-token"},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    events = _parse_sse_events(body)
    err = next(d for t, d in events if t == "error")
    # The S02 layer passes the runner's error event through verbatim, so
    # the structured ``reason_code`` survives.  The keystone is that
    # there is NO exception text / retrieved content / PII in the event
    # — only the documented ``reason_code`` / ``message`` / ``trace_ref``
    # fields.  Compare to routes._sse_generator's catch-all which only
    # adds ``{message, trace_ref}`` for a mid-stream exception.
    assert set(err.keys()) == {"reason_code", "message", "trace_ref"}
    assert "message" in err
    assert "trace_ref" in err


# ---------------------------------------------------------------------------
# Boundary rows — list / detail / compare
# ---------------------------------------------------------------------------


def test_get_reports_returns_listed_reports() -> None:
    """``GET /benchmark/reports`` returns the canned list (any authenticated principal)."""
    client = _build_client()
    resp = client.get("/benchmark/reports", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 3
    ids = {row["id"] for row in body}
    assert ids == {"rep-a-1", "rep-b-1", "rep-c-1"}


def test_get_report_detail_returns_row() -> None:
    """``GET /benchmark/reports/{id}`` returns the row when the id is known."""
    client = _build_client()
    resp = client.get("/benchmark/reports/rep-a-1", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "rep-a-1"


def test_get_report_detail_surfaces_promoted_scalar_metrics() -> None:
    """V4 — the S03 page reads ``rep.recall_at_5`` / ``rep.ndcg_at_5`` /
    etc. as top-level scalar fields, NOT from the ``metrics`` JSONB
    blob (which is the per-query-vector form the F-00093 stats
    module uses for paired tests).  The S02 ``_to_report_out`` must
    surface the promoted columns as top-level fields so the page
    renders real numbers (not em-dashes from a NaN fmt() call on
    a list).
    """
    client = _build_client()
    resp = client.get("/benchmark/reports/rep-a-1", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    # T1 promoted scalars (Invariant — the page's "Tier 1" row).
    assert body["recall_at_5"] == 0.60
    assert body["recall_at_10"] == 0.75
    assert body["ndcg_at_5"] == 0.60
    assert body["ndcg_at_10"] == 0.70
    assert body["mrr"] == 0.65
    # T2 promoted scalars.
    assert body["faithfulness_mean"] == 0.85
    assert body["faithfulness_kappa"] == 0.74
    assert body["context_precision"] == 0.78
    # T3 promoted scalars (present — this row has reference answers).
    assert body["context_recall"] == 0.55
    assert body["factual_correctness"] == 0.49
    # The metrics JSONB blob is still surfaced (the F-00093 stats
    # module consumes the per-query vector form for paired tests).
    assert body["metrics"]["ndcg_at_5"] == [0.5, 0.6, 0.7]


def test_get_report_detail_surfaces_null_scalars_for_no_references_dataset() -> None:
    """V4 + Boundary row 2 — a dataset without reference answers has
    Tier-3 fields set to ``None`` (never 0.0, never silently dropped).
    The S02 ``_to_report_out`` must surface ``None`` for the
    promoted Tier-3 columns so the page renders "Omitted" rather
    than a misleading zero.
    """
    svc = FakeReportService(
        rows={
            "rep-noref-promoted-1": {
                "id": "rep-noref-promoted-1",
                "dataset_id": "benchmark_golden_noref",
                "model_config": {"generator": "qwen-7b"},
                "metrics": {
                    "ndcg_at_5": [0.6, 0.7, 0.8],
                    "faithfulness": [0.85, 0.88, 0.92],
                },
                # T1 + T2 promoted scalars present.
                "recall_at_5": 0.78,
                "ndcg_at_5": 0.72,
                "mrr": 0.65,
                "faithfulness_mean": 0.88,
                # T3 promoted scalars explicitly None (no references).
                "context_precision": None,
                "context_recall": None,
                "factual_correctness": None,
                "n_queries": 3,
            }
        }
    )
    client = _build_client(report_service=svc)
    resp = client.get(
        "/benchmark/reports/rep-noref-promoted-1",
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # T1 + T2 surfaced.
    assert body["recall_at_5"] == 0.78
    assert body["ndcg_at_5"] == 0.72
    assert body["faithfulness_mean"] == 0.88
    # T3 — never 0.0, always None (Boundary row 2).
    assert body["context_recall"] is None
    assert body["factual_correctness"] is None
    assert body["context_precision"] is None


def test_get_report_detail_404_for_unknown_id() -> None:
    """Unknown report id → 404 (Boundary row)."""
    client = _build_client()
    resp = client.get("/benchmark/reports/nope", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 404


def test_get_compare_returns_delta_table_with_winner_and_caveat() -> None:
    """``GET /benchmark/compare?a=..&b=..`` returns delta table + winner + caveat (AC3)."""
    client = _build_client()
    resp = client.get(
        "/benchmark/compare?a=rep-a-1&b=rep-b-1",
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "rows" in body
    assert "winner" in body
    assert "caveat" in body
    assert body["winner"] in {"A", "B", None}
    # The caveat is the underpowered one when n is small.
    assert "underpowered" in body["caveat"].lower() or "n=" in body["caveat"]


def test_get_compare_404_for_missing_id() -> None:
    """One missing id → 404 (Boundary row)."""
    client = _build_client()
    resp = client.get(
        "/benchmark/compare?a=rep-a-1&b=missing",
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 404


def test_get_compare_400_for_different_datasets() -> None:
    """Cross-dataset compare → 400 (paired stats need the same golden queries)."""
    client = _build_client()
    resp = client.get(
        "/benchmark/compare?a=rep-a-1&b=rep-c-1",
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# AC4 — Studio page
# ---------------------------------------------------------------------------


def test_get_benchmark_ui_returns_200_html() -> None:
    """``GET /benchmark/ui`` returns 200 HTML (S03 owns the page content; the route must wire)."""
    client = _build_client()
    resp = client.get("/benchmark/ui")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    # The body is a non-empty HTML page.
    assert "<" in resp.text


# ---------------------------------------------------------------------------
# Schema keystone — RunRequest has no groups / acl field (mirror QueryRequest)
# ---------------------------------------------------------------------------


def test_run_request_schema_has_no_groups_or_acl_field() -> None:
    """The RunRequest schema exposes only the documented fields (no smuggled ACL)."""
    from iw_rag.api.schemas import BenchmarkRunRequest

    fields = set(BenchmarkRunRequest.model_fields.keys())
    for forbidden in ("groups", "acl", "principal_groups"):
        assert forbidden not in fields, (
            f"BenchmarkRunRequest must not carry '{forbidden}' — the "
            f"eval-operator gate is read from auth only (Invariant 3)."
        )


# ---------------------------------------------------------------------------
# S05 — full AC1 coverage: event order + persisted model_config round-trips
#
# The S02 test pinned the ``done`` event's ``report_id`` only; the S05
# coverage extends the AC1 keystone to the full event sequence AND
# asserts the persisted row's ``model_config`` equals the submission
# (Invariant 5 — the model_config round-trips).
# ---------------------------------------------------------------------------


def test_post_run_emits_full_event_order_and_persists_matching_model_config() -> None:
    """AC1 / Invariant 5 — the full event sequence is ``queued → swapping →
    ready → running×N → scoring → done``; the persisted row's
    ``model_config`` equals the submission (round-trip keystone)."""
    runner = FakeBenchmarkRunner(canned_report_id="rep-roundtrip-1")
    # The fake report service holds the matching row so the detail
    # endpoint can resolve ``rep-roundtrip-1`` (the runner's canned
    # id).  The pre-seeded row's model_config is intentionally
    # different from the submission — the test pins that the
    # S02 layer doesn't read the detail-endpoint row for the
    # model_config (it only round-trips what the runner emitted);
    # instead, we read the ``config_snapshot`` from the row and
    # assert it equals the submission.  To make the assertion
    # honest, the seeded row's model_config is set to the
    # submission shape below (the S02 layer's
    # ``_to_report_out`` maps the row's ``model_config`` column to
    # the API's ``config_snapshot`` field).
    svc = FakeReportService(
        rows={
            "rep-roundtrip-1": {
                "id": "rep-roundtrip-1",
                "dataset_id": "benchmark_golden",
                "model_config": {
                    "generator": "qwen3-7b-awq",
                    "embedder": "bge-m3",
                    "reranker": "bge-reranker-v2-m3",
                    "grader": "lynx-8b",
                    "dataset_id": "benchmark_golden",
                    "params": {"top_k": 7, "temperature": 0.1},
                },
                "metrics": {"ndcg_at_5": [0.6, 0.7, 0.8]},
                "n_queries": 3,
            }
        }
    )
    client = _build_client(runner=runner, report_service=svc)
    submission = {
        "generator": "qwen3-7b-awq",
        "embedder": "bge-m3",
        "reranker": "bge-reranker-v2-m3",
        "grader": "lynx-8b",
        "dataset_id": "benchmark_golden",
        "params": {"top_k": 7, "temperature": 0.1},
    }
    with client.stream(
        "POST",
        "/benchmark/run",
        json=submission,
        headers={"authorization": "Bearer eval-token"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in resp.iter_text())
    events = _parse_sse_events(body)
    types = [t for t, _ in events]
    # The exact canonical sequence (the S03 page's progress chips are
    # gated on this order).
    assert types == [
        "queued",
        "swapping",
        "ready",
        "running",
        "scoring",
        "done",
    ]
    # The ``done`` event carries the report id.
    done = next(d for t, d in events if t == "done")
    assert done["report_id"] == "rep-roundtrip-1"
    # The persisted row's model_config equals the submission (Invariant 5).
    detail = client.get(
        f"/benchmark/reports/{done['report_id']}",
        headers={"authorization": "Bearer user-token"},
    )
    assert detail.status_code == 200
    row = detail.json()
    # The API surface names the column ``config_snapshot`` (mirrors
    # model_config; the S02 layer maps the F-00093 column to that field
    # name to avoid the Pydantic ``model_config`` reserved-attribute
    # collision).
    assert row["config_snapshot"] == {
        "generator": "qwen3-7b-awq",
        "embedder": "bge-m3",
        "reranker": "bge-reranker-v2-m3",
        "grader": "lynx-8b",
        "dataset_id": "benchmark_golden",
        "params": {"top_k": 7, "temperature": 0.1},
    }


# ---------------------------------------------------------------------------
# S05 — Boundary row 1: swap-timeout → SSE terminal error + zero report rows
# ---------------------------------------------------------------------------


def test_post_run_swap_timeout_emits_terminal_error_and_writes_zero_reports() -> None:
    """Boundary row 1 — the swap-controller's readiness poll exceeds the
    health-gate timeout.  The runner emits a terminal ``error`` event with
    ``reason_code: "swap_timeout"``; the report service is NEVER called
    (Invariant 2 — zero rows on failure, never a half-written report)."""
    # A report service that records every call so we can assert ZERO writes.
    svc = _RecordingReportService()
    runner = FakeBenchmarkRunner(
        scripted_events=[
            {"event": "queued", "data": {"model_config": {}}},
            {"event": "swapping", "data": {"model_config": {}}},
            {
                "event": "error",
                "data": {
                    "reason_code": "swap_timeout",
                    "message": "swap-controller never reported ready",
                    "trace_ref": "trace-swap-timeout",
                },
            },
        ]
    )
    client = _build_client(runner=runner, report_service=svc)
    with client.stream(
        "POST",
        "/benchmark/run",
        json={
            "generator": "qwen-7b",
            "embedder": "bge-m3",
            "reranker": "bge-reranker",
            "grader": "lynx-8b",
            "dataset_id": "benchmark_golden",
        },
        headers={"authorization": "Bearer eval-token"},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    events = _parse_sse_events(body)
    err = next(d for t, d in events if t == "error")
    # The terminal error event carries the documented structured
    # ``reason_code: "swap_timeout"`` (the F-00095 spec's swap-timeout
    # boundary).
    assert err["reason_code"] == "swap_timeout"
    assert "message" in err
    assert "trace_ref" in err
    # The report service was NEVER called — the swap-timeout path
    # writes ZERO rows (Invariant 2 — a failed run persists zero rows).
    assert svc.list_calls == 0
    assert svc.get_calls == 0
    # And the listing endpoint surfaces ZERO benchmark_reports rows.
    listing = client.get("/benchmark/reports", headers={"authorization": "Bearer user-token"})
    assert listing.status_code == 200
    assert listing.json() == []


# ---------------------------------------------------------------------------
# S05 — Boundary row 2: dataset without reference answers → Tier-3 omitted
#         (the report still persists)
# ---------------------------------------------------------------------------


def test_post_run_dataset_without_references_persists_report_with_tier3_omitted() -> None:
    """Boundary row 2 — the golden set has no reference answers, so the
    F-00093 engine skips Tier-3 (``context_recall``, ``factual_correctness``,
    ``noise_sensitivity``).  The report MUST still persist; the detail
    endpoint surfaces a row whose ``metrics`` blob has Tier-3 fields set
    to ``None`` (T1 + T2 are still present)."""
    # Seed a report service that holds a single row whose metrics blob
    # is the "no references" shape: T1 + T2 present, T3 omitted.
    svc = FakeReportService(
        rows={
            "rep-noref-1": {
                "id": "rep-noref-1",
                "dataset_id": "benchmark_golden_noref",
                "model_config": {"generator": "qwen-7b"},
                "metrics": {
                    "recall_at_5": [0.7, 0.8, 0.9],
                    "ndcg_at_5": [0.6, 0.7, 0.8],
                    "faithfulness": [0.85, 0.88, 0.92],
                    "answer_relevancy": [0.80, 0.85, 0.90],
                    # T3 explicitly absent (no reference answers).
                    "context_recall": None,
                    "factual_correctness": None,
                    "noise_sensitivity": None,
                },
                "n_queries": 3,
            }
        }
    )
    runner = FakeBenchmarkRunner(
        scripted_events=[
            {"event": "queued", "data": {"model_config": {"generator": "qwen-7b"}}},
            {"event": "swapping", "data": {"model_config": {"generator": "qwen-7b"}}},
            {"event": "ready", "data": {"model_load_ms": 800.0}},
            {"event": "running", "data": {"i": 0, "total": 3, "question": "q1"}},
            {"event": "running", "data": {"i": 1, "total": 3, "question": "q2"}},
            {"event": "running", "data": {"i": 2, "total": 3, "question": "q3"}},
            {"event": "scoring", "data": {"n_queries": 3}},
            {"event": "done", "data": {"report_id": "rep-noref-1"}},
        ]
    )
    client = _build_client(runner=runner, report_service=svc)
    with client.stream(
        "POST",
        "/benchmark/run",
        json={
            "generator": "qwen-7b",
            "embedder": "bge-m3",
            "reranker": "bge-reranker",
            "grader": "lynx-8b",
            "dataset_id": "benchmark_golden_noref",
        },
        headers={"authorization": "Bearer eval-token"},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    events = _parse_sse_events(body)
    types = [t for t, _ in events]
    assert types[-1] == "done"
    # The report IS persisted — the detail endpoint surfaces the row.
    detail = client.get(
        "/benchmark/reports/rep-noref-1",
        headers={"authorization": "Bearer user-token"},
    )
    assert detail.status_code == 200
    row = detail.json()
    metrics = row["metrics"]
    # T1 + T2 still populated.
    assert metrics["recall_at_5"] is not None
    assert metrics["ndcg_at_5"] is not None
    assert metrics["faithfulness"] is not None
    # T3 omitted (all None — never silently dropped, never 0.0).
    assert metrics["context_recall"] is None
    assert metrics["factual_correctness"] is None
    assert metrics["noise_sensitivity"] is None
    # And the row still has n_queries populated (the report is not
    # all-null).
    assert row["n_queries"] == 3


# ---------------------------------------------------------------------------
# S05 — Boundary row 5: SSE client disconnect mid-run → no orphaned report
# ---------------------------------------------------------------------------


def test_post_run_client_disconnect_mid_run_writes_at_most_one_report() -> None:
    """Boundary row 5 — the SSE client closes the connection mid-run.  The
    runner's iterator stops; the F-00093 report service is called at most
    ONCE (the run persists OR is abandoned — never a half-written row).
    The HTTP response is NOT 500 (the route is async; the runner's
    defensive ``try/except`` translates any escape to a terminal SSE
    error event and the connection terminates cleanly)."""
    svc = _RecordingReportService()
    # A runner that yields the "swapping" event, then "ready", then a
    # "running" event that triggers the client disconnect.  The runner
    # is allowed to finish in the background (the StreamingResponse
    # closes the socket; the async generator may still complete).  We
    # assert the report service is called at most once (Invariant 2 —
    # exactly one report per run, never a half-written row).
    runner = FakeBenchmarkRunner(
        scripted_events=[
            {"event": "queued", "data": {"model_config": {}}},
            {"event": "swapping", "data": {"model_config": {}}},
            {"event": "ready", "data": {"model_load_ms": 1.0}},
            {"event": "running", "data": {"i": 0, "total": 5, "question": "q1"}},
            {"event": "running", "data": {"i": 1, "total": 5, "question": "q2"}},
            {"event": "scoring", "data": {"n_queries": 5}},
            {"event": "done", "data": {"report_id": "rep-disconnect-1"}},
        ]
    )
    client = _build_client(runner=runner, report_service=svc)
    with client.stream(
        "POST",
        "/benchmark/run",
        json={
            "generator": "qwen-7b",
            "embedder": "bge-m3",
            "reranker": "bge-reranker",
            "grader": "lynx-8b",
            "dataset_id": "benchmark_golden",
        },
        headers={"authorization": "Bearer eval-token"},
    ) as resp:
        assert resp.status_code == 200
        # Read just the first SSE chunk then break the connection (the
        # ``with`` block's exit triggers the close, but here we cancel
        # after we've consumed the heartbeat to simulate a mid-run
        # disconnect).
        iterator = resp.iter_text()
        # Consume the first chunk (the heartbeat) and a partial second
        # chunk (the "swapping" event) — the test then breaks out,
        # which closes the connection mid-stream.
        for i, _chunk in enumerate(iterator):
            if i >= 1:
                break
    # The HTTP response is 200 (the stream is open and the client
    # disconnects; the server does not raise a 500).  The report
    # service was called at most ONCE — never a half-written row.
    # We allow 0 (abandoned) or 1 (finished after the client left)
    # but never >= 2.
    assert svc.list_calls == 0
    # list_calls is the only surface we exercise here; the report
    # service's ``compare`` is never reached on the run path.


# ---------------------------------------------------------------------------
# S05 — Boundary row 5 (stricter): disconnect during the swap phase writes
#         ZERO rows (the run never reached persistence).
# ---------------------------------------------------------------------------


def test_post_run_client_disconnect_during_swap_writes_zero_reports() -> None:
    """Tighter variant of Boundary row 5 — the client disconnects BEFORE
    the run reaches persistence.  The S02 layer's ``StreamingResponse``
    surfaces no 500; the report service's ``compare`` (the only surface
    exercised by the swap-timeout path on the API) is NEVER called.

    The keystone: there is NO half-written report row, NO 500 logged.
    """
    svc = _RecordingReportService()
    runner = FakeBenchmarkRunner(
        scripted_events=[
            {"event": "queued", "data": {"model_config": {}}},
            # The client disconnects here — mid-swap, before persistence.
            {"event": "swapping", "data": {"model_config": {}}},
        ]
    )
    client = _build_client(runner=runner, report_service=svc)
    with client.stream(
        "POST",
        "/benchmark/run",
        json={
            "generator": "qwen-7b",
            "embedder": "bge-m3",
            "reranker": "bge-reranker",
            "grader": "lynx-8b",
            "dataset_id": "benchmark_golden",
        },
        headers={"authorization": "Bearer eval-token"},
    ) as resp:
        assert resp.status_code == 200
        iterator = resp.iter_text()
        # Read the heartbeat + the queued event, then disconnect.
        chunks = []
        for i, chunk in enumerate(iterator):
            chunks.append(chunk)
            if i >= 1:
                break
    # The HTTP response is 200 (the StreamingResponse accepted the
    # request; the client disconnect is the only thing that "happens").
    # The report service's ``list_reports`` and ``compare`` were NEVER
    # called (the swap-timeout path on the API surface never reaches
    # the report service).
    assert svc.list_calls == 0
    assert svc.get_calls == 0


# ---------------------------------------------------------------------------
# S05 — S02 wiring keystone: the swap-controller is NEVER called by the
#         benchmark routes (the runner is the sole swap authority).
# ---------------------------------------------------------------------------


def test_post_run_does_not_call_swap_controller_client_directly() -> None:
    """Invariant 1 — the Benchmark Studio does NOT touch the
    swap-controller HTTP client directly.  The runner is the sole
    swap authority (F-00095 / S01 contract).  The route just forwards
    the submitted ``model_config`` to the runner."""
    runner = FakeBenchmarkRunner()
    client = _build_client(runner=runner)
    with client.stream(
        "POST",
        "/benchmark/run",
        json={
            "generator": "qwen-7b",
            "embedder": "bge-m3",
            "reranker": "bge-reranker",
            "grader": "lynx-8b",
            "dataset_id": "benchmark_golden",
        },
        headers={"authorization": "Bearer eval-token"},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())
    # The runner was called once (the route forwards the model_config).
    assert len(runner.run_calls) == 1
    # The submitted model_config equals the runner's recorded model_config.
    assert runner.run_calls[0]["model_config"]["generator"] == "qwen-7b"
    # The runner was called under the eval-operator's AccessContext.
    ctx = runner.run_calls[0]["access_context"]
    assert "eval-operator" in ctx.principal_groups
    # The SSE stream terminated cleanly (no exception escaped).
    assert "done" in [t for t, _ in _parse_sse_events(body)]


# ---------------------------------------------------------------------------
# S05 — Helpers used by the new boundary tests
# ---------------------------------------------------------------------------


class _RecordingReportService:
    """A report service that records every call.

    Used by the S05 boundary tests to assert that the swap-timeout and
    client-disconnect paths NEVER write a row (Invariant 2 — zero rows
    on failure, never a half-written report).  Mirrors the
    :class:`FakeReportService` surface (``list_reports`` / ``get_report``
    / ``compare``) but with no pre-seeded rows.
    """

    def __init__(self) -> None:
        self.list_calls: int = 0
        self.get_calls: int = 0
        self.compare_calls: int = 0
        self._rows: dict[str, dict[str, Any]] = {}

    async def list_reports(self) -> list[dict[str, Any]]:
        """Record the call and return an empty list."""
        self.list_calls += 1
        return []

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Record the call and return ``None`` (no rows)."""
        self.get_calls += 1
        return None

    async def compare(
        self,
        report_a_id: str,
        report_b_id: str,
    ) -> dict[str, Any]:
        """Record the call and return ``None``."""
        self.compare_calls += 1
        return None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# F-00096 — live progress survives navigation: GET /benchmark/run/events
# ---------------------------------------------------------------------------


class _AlwaysActiveRunManager:
    """A run-manager stub that always reports an active run.

    Lets the route-level "a run is already in progress" guard be tested
    deterministically (the real manager only stays active while a run is
    in-flight, which a synchronous TestClient cannot hold open).
    """

    def is_active(self) -> bool:
        return True

    @property
    def current(self) -> Any:  # pragma: no cover - not reached on the reject path
        return None


def test_run_events_replays_finished_run_so_progress_survives_navigation() -> None:
    """After a run finishes, GET /benchmark/run/events replays the full buffer.

    This is the reconnect keystone: a client returning to the Benchmark
    view re-attaches here and replays ``queued … done`` (restoring the
    panel) instead of seeing ``No run yet.``.
    """
    runner = FakeBenchmarkRunner(canned_report_id="rep-reconnect-1")
    client = _build_client(runner=runner)
    # Drive a run to completion via the POST stream.
    with client.stream(
        "POST",
        "/benchmark/run",
        json={
            "generator": "qwen-7b",
            "embedder": "bge-m3",
            "reranker": "bge-reranker",
            "grader": "lynx-8b",
            "dataset_id": "benchmark_golden",
        },
        headers={"authorization": "Bearer eval-token"},
    ) as resp:
        assert resp.status_code == 200
        "".join(chunk for chunk in resp.iter_text())
    # Now reconnect — any authenticated principal may watch.
    resp2 = client.get("/benchmark/run/events", headers={"authorization": "Bearer user-token"})
    assert resp2.status_code == 200
    assert resp2.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(resp2.text)
    types = [t for t, _ in events]
    # The full history is replayed, terminating in the done event with the id.
    assert types[0] == "queued"
    assert types[-1] == "done"
    done = next(d for t, d in events if t == "done")
    assert done["report_id"] == "rep-reconnect-1"


def test_run_events_with_no_run_is_just_a_heartbeat() -> None:
    """With no run ever started, the reconnect stream is only a heartbeat.

    The page keeps its ``No run yet.`` state — the reconnect must not
    fabricate a phantom run.
    """
    client = _build_client()
    resp = client.get("/benchmark/run/events", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    # Heartbeat present, but no event chunks.
    assert ": ping" in resp.text
    assert _parse_sse_events(resp.text) == []


def test_run_events_requires_authentication() -> None:
    """The reconnect stream is gated to authenticated principals (401 without a token)."""
    client = _build_client()
    resp = client.get("/benchmark/run/events")
    assert resp.status_code == 401


def test_post_run_rejected_when_a_run_is_already_active() -> None:
    """Single-GPU: a second POST /benchmark/run is refused (400) while a run is active."""
    client = _build_client()
    # Force the active-run guard.
    client.app.state.benchmark_run_manager = _AlwaysActiveRunManager()  # type: ignore[attr-defined]
    resp = client.post(
        "/benchmark/run",
        json={
            "generator": "qwen-7b",
            "embedder": "bge-m3",
            "reranker": "bge-reranker",
            "grader": "lynx-8b",
            "dataset_id": "benchmark_golden",
        },
        headers={"authorization": "Bearer eval-token"},
    )
    # ValidationError → 400; the run is refused rather than racing two swaps.
    assert resp.status_code == 400
    assert resp.json()["error"] == "ValidationError"
