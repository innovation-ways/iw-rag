"""Headline experiment #4 (agentic loop ON vs OFF) over the demo golden set — LIVE.

Runs the same golden set twice against the live stack:

- **loop ON**  — the full LangGraph agent loop (route → retrieve → sanitize →
  grade_documents → generate → grade_generation → output_guard → done/fail_closed).
- **loop OFF** — the Phase-0 path (spine.retrieve → assembler.generate) via the
  ``loop=False`` escape hatch, i.e. a single pass with no self-correction.

Retrieval is identical across arms (the loop changes generation/grading/retries,
not retrieval), so the headline signal is the **budget-exhaustion / fail-closed
rate** and the **grounded-answer rate** — does the self-correcting loop refuse
ungrounded answers that the single pass would emit?

Run (live stack up; guards may be on or off — this experiment is about the loop,
not the guards)::

    set -a; . ./.env; set +a
    uv run --group ingest python scripts/run_loop_eval.py

Judge-free by default (no faithfulness judge wired) so it never contends with the
guard models for VRAM. Persists two append-only ``eval_runs`` rows
(``agent_loop_on`` / ``agent_loop_off``). Companion to run_golden_eval.py (#1)
and run_faithfulness_eval.py (#3).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text

from iw_rag.api.app import _build_query_service
from iw_rag.domain.acl import AccessContext
from iw_rag.eval.experiments import run_loop_on_off
from iw_rag.eval.repository import EvalRepository, GoldenQuery
from iw_rag.settings import get_settings

_DEFAULT_GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden" / "demo_golden.yaml"
_DEFAULT_GROUPS = ("finance", "hr", "public")


async def _resolve_relevant(db: Any, source: str | None, keyword: str | None) -> list[str]:
    """Resolve (source_uri, keyword) → the matching live chunk ids (str)."""
    if not source or not keyword:
        return []
    sql = text(
        "SELECT c.id FROM iw_rag.chunks c JOIN iw_rag.documents d ON d.id = c.document_id "
        "WHERE d.source_uri = :src AND c.content ILIKE :kw"
    )
    async with db.session() as session:
        rows = await session.execute(sql, {"src": source, "kw": f"%{keyword}%"})
        return [str(r[0]) for r in rows.all()]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S603,S607
    except Exception:  # noqa: BLE001
        return None


def _outcome_to_state(outcome: Any) -> dict[str, Any]:
    """Map a QueryOutcome to the terminal-state dict exp #4 consumes."""
    diag = outcome.diagnostics or {}
    grounded = bool(getattr(outcome.answer, "grounded", False))
    disposition = diag.get("disposition") or ("answered" if grounded else "refused")
    return {
        "disposition": disposition,
        "answer": outcome.answer.answer if grounded else None,
        "refusal": None if grounded else outcome.answer.answer,
        "grounded": grounded,
    }


async def main(spec: dict, golden_name: str) -> None:
    cfg = get_settings()
    service, database, _llm = _build_query_service(cfg)

    groups = tuple(spec.get("principal_groups") or _DEFAULT_GROUPS)
    ctx = AccessContext(principal_id="eval", principal_groups=groups)
    print(f"Golden: {golden_name} | principal groups: {groups}")
    queries: list[GoldenQuery] = []
    for q in spec["queries"]:
        rel = await _resolve_relevant(database, q.get("source"), q.get("keyword"))
        queries.append(
            GoldenQuery(
                question=q["question"],
                relevant_chunk_ids=rel,
                query_type=q.get("query_type"),
                # id left None — demo golden queries aren't persisted EvalQuery
                # rows, so eval_results.query_id stays NULL.
            )
        )

    print(
        f"Experiment #4 (loop ON vs OFF) over {len(queries)} golden queries "
        f"(guards={'on' if cfg.guardrails.enabled else 'off'})\n"
    )

    # Pre-compute per-question caches (state_fn is a sync callable in the harness).
    retr_cache: dict[str, Any] = {}
    on_state: dict[str, dict[str, Any]] = {}
    off_state: dict[str, dict[str, Any]] = {}
    try:
        for q in queries:
            retr_cache[q.question] = await service._spine.retrieve(q.question, ctx)
            on = await service.answer(q.question, ctx, loop=True)
            off = await service.answer(q.question, ctx, loop=False)
            on_state[q.question] = _outcome_to_state(on)
            off_state[q.question] = _outcome_to_state(off)
            print(
                f"  {q.question[:54]:54}  on={on_state[q.question]['disposition']:8} "
                f"off={off_state[q.question]['disposition']}"
            )

        async def on_retrieve(question: str) -> Any:
            return retr_cache[question]

        async def off_retrieve(question: str) -> Any:
            return retr_cache[question]

        async with database.session() as session:
            repo = EvalRepository(session)
            cmp = await run_loop_on_off(
                repo,
                queries=queries,
                on_retrieve=on_retrieve,
                off_retrieve=off_retrieve,
                on_state_fn=lambda question: on_state[question],
                off_state_fn=lambda question: off_state[question],
                faithfulness=None,  # judge-free (no VRAM contention with guards)
                dataset_id=None,  # demo set is not a persisted EvalDataset row
                git_sha=_git_sha(),
            )
            await session.commit()
    finally:
        await database.dispose()

    on_grounded = sum(1 for s in on_state.values() if s["grounded"])
    off_grounded = sum(1 for s in off_state.values() if s["grounded"])
    n = len(queries)
    on_r10, off_r10 = cmp.on_mean_recall.get(10), cmp.off_mean_recall.get(10)
    print("\n=== Experiment #4 — agentic loop ON vs OFF (demo corpus) ===")
    print(f"  run ids:           on={cmp.on_run_id}  off={cmp.off_run_id}")
    print(f"  recall@10:         on={on_r10:.3f}  off={off_r10:.3f}")
    print(f"  MRR:               on={cmp.on_mean_mrr:.3f}  off={cmp.off_mean_mrr:.3f}")
    print(
        f"  exhaustion (refused) rate: "
        f"on={cmp.on_exhaustion_rate:.3f}  off={cmp.off_exhaustion_rate:.3f}"
    )
    print(f"  grounded-answer rate:      on={on_grounded}/{n}  off={off_grounded}/{n}")
    print(f"  scored={cmp.n_scored}  zero_gold={cmp.n_zero_gold}")


if __name__ == "__main__":
    _path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_GOLDEN
    _spec = yaml.safe_load(_path.read_text())
    asyncio.run(main(_spec, _path.name))
