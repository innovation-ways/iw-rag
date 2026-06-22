"""run_eval.py — CLI entrypoint for offline headline experiments.

Runs one or more headline experiments (CONVENTIONS §9) and emits a structured
report.  Designed for the two-clocks pattern: heavy eval work runs offline
(batch), not on the hot path.

Usage::

    uv run python scripts/run_eval.py --experiment retrieval
    uv run python scripts/run_eval.py --experiment loop
    uv run python scripts/run_eval.py --experiment guardrails

    # With git SHA for reproducibility:
    uv run python scripts/run_eval.py --experiment guardrails --git-sha $(git rev-parse HEAD)

Available experiments:

- ``retrieval``  — headline experiment #1 (dense / hybrid / hybrid+reranker
  ablation, judge-free retrieval metrics).  **Live run required** (real spine
  + golden set).
- ``loop``       — headline experiment #4 (agentic loop ON vs OFF).  **Live
  run required** (real graph + judge).
- ``guardrails`` — headline experiment #5 (guardrail efficacy / red-team:
  three demos + 5a–5e sub-experiments + four red-team metrics + guard-quality
  table).  Harness gate runs with mocked models; live promptfoo/garak/PyRIT
  runs are a deferred heavy run (see PHASE-3-GATE.md).

References:
- task-3.5-guardrail-eval §4 (register experiment #5, ``--experiment guardrails``
  entrypoint)
- lld-eval-observability §6 (the eight headline experiments)
- CONVENTIONS §9 (canonical experiment numbering)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured :class:`~argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description=(
            "Run a headline experiment and emit a structured report. "
            "Heavy live runs (real models / promptfoo) are opt-in "
            "(see PHASE-3-GATE.md for the deferred list)."
        ),
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["retrieval", "loop", "guardrails"],
        help=("Which headline experiment to run.  retrieval=exp#1, loop=exp#4, guardrails=exp#5."),
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help="Git SHA recorded on the eval run row (optional).",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Eval dataset id to link the run to (optional).",
    )
    parser.add_argument(
        "--output",
        default="-",
        help=("Output file path for the JSON report.  Use '-' (default) for stdout."),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help=(
            "Opt in to the live run (real models, real DB, promptfoo / garak). "
            "Without this flag the guardrails experiment runs the harness gate "
            "(mocked models)."
        ),
    )
    return parser


async def _run_guardrails(
    git_sha: str | None,
    dataset_id: str | None,
    live: bool,
) -> dict[str, Any]:
    """Run headline experiment #5 (guardrail efficacy / red-team).

    Args:
        git_sha: Code revision under test.
        dataset_id: Eval dataset id (optional).
        live: When True, run against real models + DB.  When False (default),
            run the harness gate with mocked models.

    Returns:
        JSON-serialisable report dict.
    """
    from unittest.mock import AsyncMock, MagicMock

    from iw_rag.eval.experiments import GuardrailReport, run_experiment_5_guardrails

    if live:
        # Live path — requires DB, real models, full stack up.
        # Import DB + repo here so the mocked path never touches them.
        from iw_rag.settings import get_settings
        from iw_rag.storage.db import async_session_maker

        settings = get_settings()
        _ = settings  # used for DB config; real session wired below
        async with async_session_maker() as session:
            from iw_rag.eval.repository import EvalRepository

            repo = EvalRepository(session)
            report = await run_experiment_5_guardrails(
                repo,
                git_sha=git_sha,
                dataset_id=dataset_id,
            )
            await session.commit()
    else:
        # Harness gate — mocked repo; no DB required.
        mock_repo: Any = MagicMock()
        mock_repo.create_run = AsyncMock(return_value="local-run-id-exp5")
        mock_repo.add_results = AsyncMock(return_value=1)
        report = await run_experiment_5_guardrails(
            mock_repo,
            git_sha=git_sha,
            dataset_id=dataset_id,
        )

    assert isinstance(report, GuardrailReport)
    return {
        "experiment": "exp5_guardrails",
        "run_id": report.run_id,
        "demo_acl_leak_prevented": report.demo_acl_leak_prevented,
        "demo_injection_caught": report.demo_injection_caught,
        "demo_fail_closed_pass": report.demo_fail_closed_pass,
        "metrics": {
            "acl_leak_rate": report.acl_leak_rate,
            "asr": report.asr,
            "catch_rate": report.catch_rate,
            "fpr": report.fpr,
            "fail_closed_rate": report.fail_closed_rate,
        },
        "guard_quality": (
            {
                "injection_fpr": report.guard_quality.injection_fpr,
                "injection_fnr": report.guard_quality.injection_fnr,
                "groundedness_accuracy": report.guard_quality.groundedness_accuracy,
                "pii_recall": report.guard_quality.pii_recall,
                "pii_precision": report.guard_quality.pii_precision,
                "acl_leak_rate": report.guard_quality.acl_leak_rate,
                "attack_catch_rate": report.guard_quality.attack_catch_rate,
                "latency_overhead_ms": report.guard_quality.latency_overhead_ms,
            }
            if report.guard_quality is not None
            else None
        ),
        "sub_experiments": [
            {
                "name": s.name,
                "asr": s.asr,
                "catch_rate": s.catch_rate,
                "fpr": s.fpr,
                "fail_closed_rate": s.fail_closed_rate,
                "notes": s.notes,
            }
            for s in report.sub_experiments
        ],
        "notes": report.notes,
        "live": live,
        "deferred": (
            [
                "5a Qwen3-14B ASR reduction (spotlighting ablation, real model required)",
                "live promptfoo run against /query (real stack + real models)",
                "garak latentinjection deep-probe pass",
                "PyRIT 5e multi-turn Crescendo/TAP/PAIR (per-release, not per-PR)",
                "GitHub Action CI wiring (promptfoo/promptfoo-action per PR)",
            ]
            if not live
            else []
        ),
    }


async def _run_retrieval(
    git_sha: str | None,
    dataset_id: str | None,
) -> dict[str, Any]:
    """Stub for headline experiment #1 (retrieval ablation).

    Args:
        git_sha: Code revision under test.
        dataset_id: Eval dataset id (optional).

    Returns:
        JSON-serialisable stub dict (live run deferred).
    """
    return {
        "experiment": "exp1_retrieval_ablation",
        "note": (
            "Live run required: real spine + golden set. "
            "Use run_experiment() from iw_rag.eval.experiments with a real repo + retrieve."
        ),
        "git_sha": git_sha,
        "dataset_id": dataset_id,
    }


async def _run_loop(
    git_sha: str | None,
    dataset_id: str | None,
) -> dict[str, Any]:
    """Stub for headline experiment #4 (agentic loop ON vs OFF).

    Args:
        git_sha: Code revision under test.
        dataset_id: Eval dataset id (optional).

    Returns:
        JSON-serialisable stub dict (live run deferred).
    """
    return {
        "experiment": "exp4_loop_on_off",
        "note": (
            "Live run required: real graph + faithfulness judge + golden set. "
            "Use run_loop_on_off() from iw_rag.eval.experiments with a real repo."
        ),
        "git_sha": git_sha,
        "dataset_id": dataset_id,
    }


async def _main(args: argparse.Namespace) -> int:
    """Main async entrypoint.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = success).
    """
    report: dict[str, Any]

    if args.experiment == "guardrails":
        report = await _run_guardrails(
            git_sha=args.git_sha,
            dataset_id=args.dataset_id,
            live=args.live,
        )
    elif args.experiment == "retrieval":
        report = await _run_retrieval(
            git_sha=args.git_sha,
            dataset_id=args.dataset_id,
        )
    elif args.experiment == "loop":
        report = await _run_loop(
            git_sha=args.git_sha,
            dataset_id=args.dataset_id,
        )
    else:
        print(f"Unknown experiment: {args.experiment}", file=sys.stderr)
        return 1

    output_json = json.dumps(report, indent=2)
    _write_report(output_json, args.output)
    return 0


def _write_report(output_json: str, output_path: str) -> None:
    """Write the JSON report to stdout or a file (synchronous I/O helper).

    Extracted from the async ``_main`` so the blocking I/O (``Path.write_text``)
    lives in a synchronous function — avoids ASYNC240 (trio/anyio path rule).

    Args:
        output_json: JSON-serialised report string.
        output_path: File path or ``"-"`` for stdout.
    """
    if output_path == "-":
        print(output_json)
    else:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json)
        print(f"Report written to {output_path}", file=sys.stderr)


def main() -> None:
    """CLI entry point — parse args, run, exit.

    This is the function called by ``uv run python scripts/run_eval.py``.
    """
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
