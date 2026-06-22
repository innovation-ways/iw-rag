"""Headline experiment #9 (agentic-loop trajectory quality) over a golden set — LIVE.

Runs the live LangGraph loop **k times per query** (agents chain randomness, so a
single run is noisy) and scores each trajectory against the four structural
invariants (I1-I4, label-free) plus optional label-based route/disposition
accuracy:

    route → grade_documents → generate → grade_generation → output_guard → done
                                                          ↘ consume_retry ↺
                                                          ↘ fail_closed (refuse)

The structural invariants are deterministic Python — no judge is wired, so this
never contends with the guard/generator models for VRAM (same economics as
run_loop_eval.py).  It persists one append-only ``eval_runs`` row
(``agent_loop_trajectory``) + one ``eval_results`` row.

Run (live stack up; the loop is the system under test)::

    set -a; . ./.env; set +a
    IW_RAG_TRAJ_SAMPLES=3 uv run --group ingest python scripts/run_trajectory_eval.py \
        eval/golden/demo_golden.yaml

The sample count k defaults to 3 and is recorded in the run config (no silent
caps).  Optional per-query ``expected_route`` / ``expected_disposition`` golden
fields drive route/disposition accuracy when present; they are back-compatible
(absent → label-free structural scoring only).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from iw_rag.api.app import _build_query_service
from iw_rag.domain.acl import AccessContext
from iw_rag.eval.experiments import run_experiment_9_trajectory
from iw_rag.eval.repository import EvalRepository, GoldenQuery
from iw_rag.settings import get_settings

_DEFAULT_GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden" / "demo_golden.yaml"
_DEFAULT_GROUPS = ("finance", "hr", "public")
_DEFAULT_SAMPLES = 3


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S603,S607
    except Exception:  # noqa: BLE001
        return None


def _samples_per_query() -> int:
    """Resolve k (samples per query) from the env knob; default 3."""
    raw = os.environ.get("IW_RAG_TRAJ_SAMPLES", "")
    try:
        k = int(raw) if raw else _DEFAULT_SAMPLES
    except ValueError:
        k = _DEFAULT_SAMPLES
    return max(1, k)


def _outcome_to_sample(outcome: Any) -> dict[str, Any]:
    """Map a QueryOutcome's diagnostics to the per-sample state dict exp #9 consumes."""
    diag = outcome.diagnostics or {}
    return {
        "trajectory": diag.get("trajectory") or [],
        "route": diag.get("route"),
        "retries": diag.get("retries", 0),
        "max_retries": diag.get("max_retries", 0),
        "disposition": diag.get("disposition"),
        "grounded": bool(getattr(outcome.answer, "grounded", False)),
    }


async def main(spec: dict[str, Any], golden_name: str) -> None:
    cfg = get_settings()
    service, database, _llm = _build_query_service(cfg)

    groups = tuple(spec.get("principal_groups") or _DEFAULT_GROUPS)
    ctx = AccessContext(principal_id="eval", principal_groups=groups)
    k = _samples_per_query()

    queries: list[GoldenQuery] = []
    expected_routes: dict[str, str] = {}
    expected_dispositions: dict[str, str] = {}
    for q in spec["queries"]:
        question = q["question"]
        queries.append(
            GoldenQuery(
                question=question,
                query_type=q.get("query_type"),
                # id left None — demo golden queries aren't persisted EvalQuery rows.
            )
        )
        if q.get("expected_route"):
            expected_routes[question] = q["expected_route"]
        if q.get("expected_disposition"):
            expected_dispositions[question] = q["expected_disposition"]

    print(f"Golden: {golden_name} | principal groups: {groups} | samples/query k={k}")
    print(
        f"Experiment #9 (trajectory quality) over {len(queries)} golden queries "
        f"(guards={'on' if cfg.guardrails.enabled else 'off'}, "
        f"labeled_routes={len(expected_routes)}, "
        f"labeled_dispositions={len(expected_dispositions)})\n"
    )

    # Run the live loop k times per query; collect per-sample state dicts.
    samples: dict[str, list[dict[str, Any]]] = {}
    try:
        for q in queries:
            runs: list[dict[str, Any]] = []
            for _ in range(k):
                outcome = await service.answer(q.question, ctx, loop=True)
                runs.append(_outcome_to_sample(outcome))
            samples[q.question] = runs
            dispositions = [s["disposition"] for s in runs]
            routes = [s["route"] for s in runs]
            print(f"  {q.question[:50]:50}  route={routes[0]!s:13} dispositions={dispositions}")

        async with database.session() as session:
            repo = EvalRepository(session)
            report = await run_experiment_9_trajectory(
                repo,
                queries=queries,
                samples=samples,
                samples_per_query=k,
                expected_routes=expected_routes or None,
                expected_dispositions=expected_dispositions or None,
                git_sha=_git_sha(),
                dataset_id=None,  # demo set is not a persisted EvalDataset row
            )
            await session.commit()
    finally:
        await database.dispose()

    print("\n=== Experiment #9 — agentic-loop trajectory quality (demo corpus) ===")
    print(f"  run id:                 {report.run_id}")
    print(f"  trajectory validity:    {report.trajectory_validity_rate:.3f}")
    print(f"  route accuracy:         {report.route_accuracy:.3f}")
    da = report.disposition_accuracy
    print(f"  disposition accuracy:   {'n/a (no labels)' if da is None else f'{da:.3f}'}")
    print(f"  mean retries:           {report.mean_retries:.3f}")
    print(f"  disposition stability:  {report.disposition_stability:.3f}")
    print(f"  route stability:        {report.route_stability:.3f}")
    print(f"  n_queries={report.n_queries}  samples_per_query={report.samples_per_query}")
    print("  per-invariant pass rates:")
    for name, rate in report.invariant_pass_rates.items():
        print(f"    {name:28} {rate:.3f}")
    print(
        "\n  NOTE: commit trajectory_validity_rate + route_accuracy into "
        "tests/eval/baselines/phase0.json to tighten the CI gate from this live run."
    )


if __name__ == "__main__":
    _path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_GOLDEN
    _spec = yaml.safe_load(_path.read_text())
    asyncio.run(main(_spec, _path.name))
