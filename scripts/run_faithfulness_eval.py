"""Headline experiment #3 (judged faithfulness) over the demo golden set — LIVE.

For each gold query: retrieve (live hybrid+rerank spine) -> generate a cited
answer (Qwen3-14B) -> judge faithfulness with Lynx-8B (different model family —
no self-preference). Reports per-query PASS/FAIL + mean faithfulness.

Run (live stack up + a served faithfulness judge at IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT)::

    uv run --group ingest python scripts/run_faithfulness_eval.py

Two-clocks: the judge is offline/eval — it need not co-reside with the hot path.
This is the judged complement to scripts/run_golden_eval.py (judge-free retrieval).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from iw_rag.domain.acl import AccessContext
from iw_rag.eval.judges import FaithfulnessJudge
from iw_rag.generation.answer import AnswerAssembler
from iw_rag.generation.llm import GenerationParams, VllmClient
from iw_rag.generation.prompts import CitationPrompt
from iw_rag.retrieval.dense import DenseRetriever
from iw_rag.retrieval.encode import InfinityQueryEncoder
from iw_rag.retrieval.rerank import InfinityReranker
from iw_rag.retrieval.sparse import SparseRetriever
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine
from iw_rag.settings import get_settings
from iw_rag.storage.db import Database

_DEFAULT_GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden" / "demo_golden.yaml"
_DEFAULT_GROUPS = ("finance", "hr", "public")


async def _phase_a_generate(spec: dict, ctx: AccessContext) -> list[dict]:
    """Retrieve + generate (needs the generator + infinity) → (question, context, answer) triples.

    Phase A of the two-phase run: this and the judge never co-reside in VRAM
    (generator ~19 GB + Lynx-8B bf16 ~15 GB + infinity ~4 GB > 32 GB), so the
    generation half is dumped to disk and the judge half reads it back after the
    stack is swapped.
    """
    cfg = get_settings()
    db = Database(cfg.database)
    gen = cfg.models.get(cfg.generation.model)
    emb = cfg.models.get(cfg.retrieval.embed_model)
    rer = cfg.models.get(cfg.retrieval.rerank_model)
    spine = RetrievalSpine(
        db=db,
        encoder=InfinityQueryEncoder(emb.endpoint, emb.served_name),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=InfinityReranker(rer.endpoint, rer.served_name),
        cfg=RetrievalConfig(),
    )
    assembler = AnswerAssembler(
        llm=VllmClient(gen.endpoint),
        prompt=CitationPrompt(),
        params=GenerationParams(
            model=gen.served_name,
            temperature=cfg.generation.temperature,
            max_tokens=cfg.generation.max_tokens,
        ),
        cite_required=cfg.generation.cite_required,
    )
    queries = [q for q in spec["queries"] if q.get("source")]  # gold (answerable) only
    triples: list[dict] = []
    try:
        for q in queries:
            question = q["question"]
            result = await spine.retrieve(question, ctx)
            generated = await assembler.generate(question, result)
            context = "\n\n".join(c.content for c in result.chunks)
            triples.append({"question": question, "context": context, "answer": generated.answer})
            print(f"  generated: {question[:70]}")
    finally:
        await db.dispose()
    return triples


async def _phase_b_judge(triples: list[dict]) -> dict:
    """Judge cached triples with the faithfulness judge (needs only Lynx at :8002)."""
    cfg = get_settings()
    gen = cfg.models.get(cfg.generation.model)
    judge_cfg = cfg.models.get(cfg.eval.judges.faithfulness_model)
    judge = FaithfulnessJudge(
        judge_client=VllmClient(judge_cfg.endpoint),
        generator_model=gen.served_name,
        judge_model=judge_cfg.served_name,
    )
    print(f"Judging {len(triples)} answers (judge={judge_cfg.served_name})\n")
    scores: list[float] = []
    passes: list[bool] = []
    rows: list[dict] = []
    for t in triples:
        verdict = await judge.judge(t["question"], t["context"], t["answer"])
        scores.append(verdict.score)
        passes.append(verdict.passed)
        rows.append({"question": t["question"], "score": verdict.score, "passed": verdict.passed})
        mark = "PASS" if verdict.passed else "FAIL"
        print(f"  [{mark}] {t['question']}")
    mean = sum(scores) / len(scores) if scores else 0.0
    n_pass = sum(1 for p in passes if p)
    print(f"\nMean faithfulness (Lynx-8B): {mean:.3f}  ({n_pass}/{len(passes)} PASS by verdict)")
    return {
        "experiment": "exp3_faithfulness",
        "judge": judge_cfg.served_name,
        "n": len(rows),
        "mean_faithfulness": mean,
        "n_pass": n_pass,
        "per_query": rows,
    }


def _write_json(path: str, obj: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment #3 — judged faithfulness (two-phase).")
    ap.add_argument("golden", nargs="?", default=str(_DEFAULT_GOLDEN))
    ap.add_argument("--dump", help="Phase A: write (q,context,answer) triples here, skip judging")
    ap.add_argument("--judge-from", help="Phase B: read triples from here and judge only")
    ap.add_argument(
        "--report",
        default="eval/reports/benchmark_faithfulness.json",
        help="Phase B summary output",
    )
    args = ap.parse_args()

    if args.judge_from:
        triples = json.loads(Path(args.judge_from).read_text())
        report = asyncio.run(_phase_b_judge(triples))
        _write_json(args.report, report)
        print(f"Wrote {args.report}")
        return

    spec = yaml.safe_load(Path(args.golden).read_text())
    groups = tuple(spec.get("principal_groups") or _DEFAULT_GROUPS)
    ctx = AccessContext(principal_id="eval", principal_groups=groups)
    print(f"Golden: {Path(args.golden).name} | principal groups: {groups}")
    triples = asyncio.run(_phase_a_generate(spec, ctx))

    if args.dump:
        _write_json(args.dump, triples)
        print(f"\nPhase A done: wrote {len(triples)} triples → {args.dump}")
        print("Now swap the stack (generator down → Lynx up) and run with --judge-from.")
        return

    # Interleaved (single-stack) fallback — only viable if generator + judge co-reside.
    report = asyncio.run(_phase_b_judge(triples))
    _write_json(args.report, report)


if __name__ == "__main__":
    main()
