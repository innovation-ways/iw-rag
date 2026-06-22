"""Headline experiment #7 (visual) — ColVision late-interaction vs parse-then-caption.

Three-phase to keep the two visual models off the same card at once:

  --phase caption    : caption each page with a VLM (Qwen2.5-VL via vLLM at
                       IW_RAG_VL_ENDPOINT), embed caption+queries via infinity
                       BGE-M3, rank pages by cosine → /tmp/visual_caption.json
  --phase colvision  : encode each page image + query with ColQwen2.5 (colpali,
                       in-process), rank pages by MaxSim → /tmp/visual_colvision.json
  --phase score      : load both rankings + eval/golden/visual_golden.yaml and run
                       run_experiment_7_visual (recall/nDCG, gate)

Each page's key fact is visible ONLY in the rendered image (data/visual_corpus) —
absent from text — so the text spine cannot retrieve it; this isolates whether the
visual win comes from late-interaction (ColVision) or from VLM captions.

VRAM: run `caption` with the VL served (guards/generator dropped), then stop the VL
and run `colvision` (ColQwen ~7GB in-process), then `score` (no GPU). See
eval/reports/visual_experiment.md.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

import yaml

_PAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "visual_corpus" / "pages"
_GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden" / "visual_golden.yaml"
_CAP_JSON = "/tmp/visual_caption.json"
_COLV_JSON = "/tmp/visual_colvision.json"
_INFINITY = "http://localhost:7997"
_VL_ENDPOINT = "http://localhost:8001/v1"
_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
_CAPTION_PROMPT = (
    "Transcribe every label, number, and value visible in this figure as a dense "
    "plain-text caption. Include each category and its exact value."
)


def _pages() -> list[str]:
    return [p.name for p in sorted(_PAGES_DIR.glob("*.png"))]


def _queries() -> list[dict]:
    return yaml.safe_load(_GOLDEN.read_text())["queries"]


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Phase: caption (VLM → BGE-M3)                                                 #
# --------------------------------------------------------------------------- #
def _embed(texts: list[str]) -> list[list[float]]:
    import httpx

    resp = httpx.post(
        f"{_INFINITY}/embeddings",
        json={"model": "BAAI/bge-m3", "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


def _cos(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def phase_caption() -> None:
    import httpx

    pages = _pages()
    captions: dict[str, str] = {}
    for name in pages:
        data_url = f"data:image/png;base64,{_b64(_PAGES_DIR / name)}"
        body = {
            "model": _VL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": _CAPTION_PROMPT},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 300,
        }
        r = httpx.post(f"{_VL_ENDPOINT}/chat/completions", json=body, timeout=180.0)
        r.raise_for_status()
        cap = r.json()["choices"][0]["message"]["content"]
        captions[name] = cap
        print(f"  captioned {name}: {cap[:80]!r}")

    queries = _queries()
    cap_vecs = _embed([captions[p] for p in pages])
    q_vecs = _embed([q["question"] for q in queries])
    rankings: list[dict] = []
    for q, qv in zip(queries, q_vecs, strict=True):
        scored = sorted(
            ((p, _cos(qv, cv)) for p, cv in zip(pages, cap_vecs, strict=True)),
            key=lambda t: t[1],
            reverse=True,
        )
        rankings.append({"question": q["question"], "ranked_pages": [p for p, _ in scored]})
    Path(_CAP_JSON).write_text(json.dumps({"captions": captions, "rankings": rankings}, indent=2))
    print(f"Wrote {_CAP_JSON}")


# --------------------------------------------------------------------------- #
# Phase: colvision (ColQwen2.5 late-interaction, in-process)                    #
# --------------------------------------------------------------------------- #
def phase_colvision() -> None:
    import torch
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from PIL import Image

    model_name = "vidore/colqwen2.5-v0.2"
    print(f"loading {model_name} (in-process)...")
    model = ColQwen2_5.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(model_name)

    pages = _pages()
    images = [Image.open(_PAGES_DIR / p).convert("RGB") for p in pages]
    queries = _queries()
    q_texts = [q["question"] for q in queries]

    with torch.no_grad():
        page_batch = processor.process_images(images).to(model.device)
        page_embs = model(**page_batch)
        q_batch = processor.process_queries(q_texts).to(model.device)
        q_embs = model(**q_batch)
        # MaxSim late-interaction scores: [n_queries, n_pages]
        scores = processor.score_multi_vector(q_embs, page_embs)

    rankings: list[dict] = []
    for i, q in enumerate(queries):
        order = sorted(range(len(pages)), key=lambda j: float(scores[i][j]), reverse=True)
        rankings.append(
            {
                "question": q["question"],
                "ranked_pages": [pages[j] for j in order],
                "scores": {pages[j]: round(float(scores[i][j]), 3) for j in order},
            }
        )
    Path(_COLV_JSON).write_text(json.dumps({"rankings": rankings}, indent=2))
    print(f"Wrote {_COLV_JSON}")


# --------------------------------------------------------------------------- #
# Phase: score (run the experiment #7 harness)                                 #
# --------------------------------------------------------------------------- #
async def phase_score() -> None:
    from iw_rag.domain.models import RetrievalResult, RetrievedChunk

    # import the harness (and its types) lazily so the other phases don't need them
    from iw_rag.eval.experiments import run_experiment_7_visual
    from iw_rag.eval.repository import EvalRepository, GoldenQuery
    from iw_rag.settings import get_settings
    from iw_rag.storage.db import Database

    queries = _queries()
    cap = _read_json(_CAP_JSON)["rankings"]
    colv = _read_json(_COLV_JSON)["rankings"]
    cap_by_q = {r["question"]: r["ranked_pages"] for r in cap}
    colv_by_q = {r["question"]: r["ranked_pages"] for r in colv}

    # gold = the page that holds each query's answer (chunk_id == page filename)
    golden = [
        GoldenQuery(question=q["question"], relevant_chunk_ids=[q["page"]], query_type="visual")
        for q in queries
    ]

    def _result(ranked: list[str]) -> RetrievalResult:
        n = len(ranked)
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    chunk_id=p,
                    document_id="",
                    content=p,
                    score=(n - i) / n,
                    source_uri=f"visual://{p}",
                    metadata={},
                    retriever="visual",
                    acl_applied=True,
                )
                for i, p in enumerate(ranked)
            ],
            retriever="visual",
            acl_applied=True,
        )

    async def caption_retrieve(question: str) -> RetrievalResult:
        return _result(cap_by_q[question])

    async def colvision_retrieve(question: str) -> RetrievalResult:
        return _result(colv_by_q[question])

    cfg = get_settings()
    db = Database(cfg.database)
    try:
        async with db.session() as session:
            repo = EvalRepository(session)
            report = await run_experiment_7_visual(
                repo,
                queries=golden,
                colvision_retrieve=colvision_retrieve,
                caption_retrieve=caption_retrieve,
                config={"corpus": "visual_corpus", "n_pages": len(_pages())},
            )
            await session.commit()
    finally:
        await db.dispose()

    def _g(d: dict[int, float], k: int) -> float:
        return d.get(k, 0.0)

    print("\n=== Experiment #7 — ColVision vs parse-then-caption (visual corpus) ===")
    print(f"{'arm':<18}{'recall@5':>10}{'recall@10':>11}{'nDCG@5':>9}{'nDCG@10':>9}")
    print(
        f"{'colvision':<18}{_g(report.colvision_mean_recall, 5):>10.3f}"
        f"{_g(report.colvision_mean_recall, 10):>11.3f}"
        f"{_g(report.colvision_mean_ndcg, 5):>9.3f}{_g(report.colvision_mean_ndcg, 10):>9.3f}"
    )
    print(
        f"{'caption':<18}{_g(report.caption_mean_recall, 5):>10.3f}"
        f"{_g(report.caption_mean_recall, 10):>11.3f}"
        f"{_g(report.caption_mean_ndcg, 5):>9.3f}{_g(report.caption_mean_ndcg, 10):>9.3f}"
    )
    print(f"\ncolvision_wins (nDCG@5): {report.colvision_wins}")
    if report.notes:
        print(f"notes: {report.notes}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["caption", "colvision", "score"])
    args = ap.parse_args()
    if args.phase == "caption":
        phase_caption()
    elif args.phase == "colvision":
        phase_colvision()
    else:
        asyncio.run(phase_score())


if __name__ == "__main__":
    main()
