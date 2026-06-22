"""Experiment #8 — long-context (native 32K) vs RAG, needle-recovery probe — LIVE.

The canonical harness (run_experiment_8_long_context) scores *retrieval* recall/nDCG;
the genuinely interesting long-context question is a **generation** one — "does a
native-32K whole-doc pass recover a needle that top-k RAG misses?" — so this probe
measures that directly and deterministically (needle string present in the answer or
not; no judge).

Two arms over the long needle doc (``data/longdoc_corpus`` — ~21K tokens, 141 chunks,
one buried needle):
  - long_context : stuff the WHOLE document into the native 32K window → generate.
  - rag          : retrieve top-k chunks (hybrid spine) → generate.

Metric: needle recovered (answer contains the planted passphrase), latency, and the
context size each arm fed the generator.

Requires the generator served at NATIVE 32K (``--max-model-len 32768``). The default
stack runs 16384, so restart vLLM at 32K first (drop the guards for the KV VRAM):

    docker stop iw-rag-vllm-1 iw-rag-vllm-guard injguard
    docker run -d --name iw-rag-vllm32k --gpus all --ipc=host -p 8001:8001 \
      -e VLLM_ATTENTION_BACKEND=TRITON_ATTN -v iw-rag_hf-cache:/root/.cache/huggingface \
      vllm/vllm-openai:v0.11.0 --model Qwen/Qwen3-14B-AWQ --quantization awq \
      --kv-cache-dtype fp8 --max-model-len 32768 --gpu-memory-utilization 0.9 \
      --enable-prefix-caching --port 8001
    uv run --group ingest python scripts/run_longcontext_eval.py
    # restore: docker rm -f iw-rag-vllm32k; docker start iw-rag-vllm-1 iw-rag-vllm-guard injguard
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from iw_rag.domain.acl import AccessContext
from iw_rag.generation.llm import GenerationParams, VllmClient
from iw_rag.retrieval.dense import DenseRetriever
from iw_rag.retrieval.encode import InfinityQueryEncoder
from iw_rag.retrieval.rerank import InfinityReranker
from iw_rag.retrieval.sparse import SparseRetriever
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine
from iw_rag.settings import get_settings
from iw_rag.storage.db import Database

_DOC = Path(__file__).resolve().parent.parent / "data" / "longdoc_corpus" / "ops" / "handbook.md"
_NEEDLE_ANSWER = "mauve-otter-49-banister"
_QUESTION = "What is the recovery passphrase for the archive vault?"
_CTX = AccessContext(principal_id="eval", principal_groups=("public",))
_OUT = "eval/reports/longcontext_needle.json"

_LC_SYSTEM = (
    "You answer strictly from the supplied document. If the document contains the "
    "answer, quote the exact value."
)


def _toks(text: str) -> int:
    return int(len(text.split()) / 0.75)


def _write_report(report: dict) -> None:
    Path(_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(_OUT).write_text(json.dumps(report, indent=2))


async def main() -> None:
    cfg = get_settings()
    db = Database(cfg.database)
    gen = cfg.models.get(cfg.generation.model)
    emb = cfg.models.get(cfg.retrieval.embed_model)
    rer = cfg.models.get(cfg.retrieval.rerank_model)
    llm = VllmClient(gen.endpoint)
    params = GenerationParams(model=gen.served_name, temperature=0.0, max_tokens=256)

    doc_text = _DOC.read_text()

    spine = RetrievalSpine(
        db=db,
        encoder=InfinityQueryEncoder(emb.endpoint, emb.served_name),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=InfinityReranker(rer.endpoint, rer.served_name),
        cfg=RetrievalConfig(),
    )
    try:
        # --- Arm A: long_context — stuff the whole doc into the 32K window ---
        t0 = time.monotonic()
        lc_messages = [
            {"role": "system", "content": _LC_SYSTEM},
            {"role": "user", "content": f"{doc_text}\n\nQuestion: {_QUESTION} /no_think"},
        ]
        lc_resp = await llm.complete(lc_messages, params)
        lc_ms = (time.monotonic() - t0) * 1000.0
        lc_answer = lc_resp.text
        lc_found = _NEEDLE_ANSWER.lower() in lc_answer.lower()

        # --- Arm B: rag — retrieve top-k chunks then generate (bounded context) ---
        # Use the bare reranked chunks (NOT the spine's parent-expansion, which would
        # balloon to >200K tokens on this long-doc corpus) — i.e. a standard RAG window.
        t0 = time.monotonic()
        result = await spine.retrieve(_QUESTION, _CTX)
        top = result.chunks[:8]
        # did retrieval even surface the needle chunk?
        needle_retrieved = any(_NEEDLE_ANSWER.lower() in c.content.lower() for c in top)
        rag_ctx = "\n\n".join(f"[{i + 1}] {c.content[:800]}" for i, c in enumerate(top))
        rag_messages = [
            {"role": "system", "content": _LC_SYSTEM},
            {"role": "user", "content": f"{rag_ctx}\n\nQuestion: {_QUESTION} /no_think"},
        ]
        rag_resp = await llm.complete(rag_messages, params)
        rag_ms = (time.monotonic() - t0) * 1000.0
        rag_answer = rag_resp.text
        rag_found = _NEEDLE_ANSWER.lower() in rag_answer.lower()
        rag_ctx_tokens = _toks(rag_ctx)
    finally:
        await db.dispose()

    report = {
        "experiment": "exp8_longcontext_needle",
        "question": _QUESTION,
        "needle": _NEEDLE_ANSWER,
        "doc_tokens_est": _toks(doc_text),
        "long_context": {
            "context_tokens_est": _toks(doc_text),
            "needle_recovered": lc_found,
            "latency_ms": round(lc_ms, 1),
            "answer": lc_answer[:200],
        },
        "rag": {
            "retrieved_chunks": len(result.chunks),
            "context_tokens_est": rag_ctx_tokens,
            "needle_in_retrieved_chunks": needle_retrieved,
            "needle_recovered": rag_found,
            "latency_ms": round(rag_ms, 1),
            "answer": rag_answer[:200],
        },
    }
    _write_report(report)

    lc_ctx = report["long_context"]["context_tokens_est"]
    print("=== Experiment #8 — native-32K long-context vs RAG (needle recovery) ===")
    print(f"doc ≈ {report['doc_tokens_est']} tokens · needle = {_NEEDLE_ANSWER!r}\n")
    print(f"  long_context: recovered={lc_found}  ctx≈{lc_ctx}tok latency={lc_ms:.0f}ms")
    print(f"     answer: {lc_answer[:120]!r}")
    print(
        f"  rag (top-k):  recovered={rag_found}  needle_in_chunks={needle_retrieved}  "
        f"ctx≈{rag_ctx_tokens}tok latency={rag_ms:.0f}ms"
    )
    print(f"     answer: {rag_answer[:120]!r}")
    print(f"\nWrote {_OUT}")


if __name__ == "__main__":
    asyncio.run(main())
