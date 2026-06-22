"""GPU-free model stub for the iw-rag E2E browser-verification stack.

A single FastAPI app that impersonates the three model backends the /query
hot path calls, so a query produces a grounded, cited answer with NO GPU:

  * generator (vLLM OpenAI-compatible) — ``GET /v1/models``,
    ``POST /v1/chat/completions``. Branches on the request's ``guided_json``
    schema so the Self-RAG loop's structured grader calls (router / ISREL /
    ISSUP / ISUSE) get the verdicts that drive the graph to a grounded,
    answered disposition; the plain answer call (no ``guided_json``) returns
    text carrying a ``[1]`` citation marker.
  * embedder (infinity) — ``POST /embeddings`` returns the same constant
    1024-vector for every input, matching the vector ``e2e_seed.py`` writes
    on the seeded chunk so cosine distance is 0 and retrieval surfaces it.
  * reranker (infinity) — ``POST /rerank`` echoes the input order with
    descending scores. ``GET /health`` → 200.

Wire contracts are taken verbatim from the query pipeline (llm.py / encode.py
/ rerank.py / nodes.py / agent/prompts.py). This server has no iw_rag imports
so it can run as a tiny standalone process inside the E2E api image.
"""

from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# The constant query/chunk embedding. MUST equal E2E_SEED_VECTOR in
# scripts/e2e_seed.py so the seeded chunk and every query embed identically
# (cosine distance 0 → the chunk is always retrieved). Dim = 1024 (bge-m3).
EMBED_DIM = 1024
CONST_EMBEDDING: list[float] = [0.1] * EMBED_DIM

# The canned answer. Quotes the seeded chunk's content closely and carries a
# ``[1]`` marker so AnswerAssembler resolves exactly one Citation → grounded.
ANSWER_TEXT = (
    "iw-rag is a fully-local, single-GPU, open-weight agentic RAG system "
    "with citations, ACL, and a fail-closed verification layer [1]."
)

app = FastAPI(title="iw-rag e2e model stub")


@app.get("/health")
async def health() -> JSONResponse:
    """Reranker / liveness health ping (rerank.py:142)."""
    return JSONResponse({"status": "ok"})


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """Generator health ping (llm.py:190) — body is unused by the client."""
    return JSONResponse({"object": "list", "data": [{"id": "e2e-stub", "object": "model"}]})


@app.post("/embeddings")
async def embeddings(req: Request) -> JSONResponse:
    """infinity embeddings (encode.py:57). Returns one constant 1024-vector
    per input; the client reads ``data[0].embedding``."""
    body = await req.json()
    raw = body.get("input", "")
    inputs = raw if isinstance(raw, list) else [raw]
    data = [
        {"object": "embedding", "index": i, "embedding": CONST_EMBEDDING}
        for i in range(len(inputs))
    ]
    return JSONResponse({"object": "list", "model": body.get("model", "e2e-stub"), "data": data})


@app.post("/rerank")
async def rerank(req: Request) -> JSONResponse:
    """infinity rerank (rerank.py:121). Echoes input order with descending
    scores; the client reads ``results[].index`` / ``relevance_score``."""
    body = await req.json()
    docs = body.get("documents", []) or []
    results = [{"index": i, "relevance_score": 1.0 / (i + 1)} for i in range(len(docs))]
    return JSONResponse({"results": results})


def _guided_schema_keys(body: dict[str, Any]) -> set[str]:
    """Return the property names of the request's ``guided_json`` schema (or
    empty when the call is unstructured — i.e. the plain answer call)."""
    guided = body.get("guided_json")
    if not isinstance(guided, dict):
        return set()
    props = guided.get("properties")
    if isinstance(props, dict):
        return set(props.keys())
    return set()


def _structured_verdict(keys: set[str], messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Map a structured grader call (identified by its guided_json schema) to
    the verdict that drives the loop to a grounded, answered outcome.

    Returns None when the call is the plain answer call (no guided_json)."""
    if "supported" in keys:  # ISSUP groundedness gate (prompts.py:219)
        return {"supported": True, "reason": "e2e stub: grounded"}
    if "answers" in keys:  # ISUSE usefulness gate (prompts.py:267)
        return {"answers": True, "reason": "e2e stub: useful"}
    if "grades" in keys:  # grade_documents / ISREL (DOC_GRADE_SCHEMA)
        # Count the numbered sources in the user message so the grades array
        # length matches; default to one relevant grade.
        n = 1
        for m in messages:
            if m.get("role") == "user":
                content = str(m.get("content", ""))
                marks = [ln for ln in content.splitlines() if ln.strip().startswith("[")]
                n = max(1, len(marks))
        return {"grades": [{"relevant": True} for _ in range(n)]}
    if "route" in keys:  # ROUTER_SCHEMA
        return {"route": "retrieve", "query_type": "factual"}
    # Unknown structured schema — emit an empty object; tolerant parsers in
    # nodes.py default to the safe/keep branch.
    return {}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request) -> JSONResponse:
    """vLLM OpenAI-compatible chat completion (llm.py:131).

    Non-streaming (the API synthesises SSE itself). Branches:
      * ``guided_json`` present → structured grader verdict (JSON content).
      * otherwise → the plain answer call → ANSWER_TEXT (carries ``[1]``).
    """
    body = await req.json()
    messages = body.get("messages", []) or []
    keys = _guided_schema_keys(body)

    if keys or body.get("guided_json") is not None:
        verdict = _structured_verdict(keys, messages)
        import json as _json

        content = _json.dumps(verdict)
    else:
        content = ANSWER_TEXT

    return JSONResponse(
        {
            "id": "e2e-stub-cmpl",
            "object": "chat.completion",
            "model": body.get("model", "e2e-stub"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "tool_calls": []},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


# ---------------------------------------------------------------------------
# Swap-controller catalog (F-00096 Benchmark Runner / V3).
#
# The Operator Console's Benchmark Runner pickers + VRAM bar are fed by
# ``GET /models`` (models_routes.py), which proxies to the swap-controller
# sidecar's ``GET /catalog`` / ``GET /catalog/status`` / ``POST /catalog/activate``
# (serving/swap_controller_client.py). The GPU-free e2e stack has no real
# swap-controller, so this stub serves the static generator catalog
# (mirrors config/model_catalog.yaml) — enough for V3 to populate the pickers
# and drive the VRAM budget bar RED when an over-budget model is picked.
# No model is ever resident (active_model_id = null); activate is a no-op ack.
# ---------------------------------------------------------------------------

# Mirrors config/model_catalog.yaml (id / served_name / quantization /
# expected_vram_gb / max_context), PLUS one synthetic over-budget candidate
# (qwen3-72b-awq, 40 GB) that is NOT in the real catalog. The production
# catalog only lists models that fit the 32 GB GPU, so nothing there trips the
# VRAM guard; V3 needs a candidate whose expected_vram_gb + the always-resident
# 4 GB Infinity exceeds the 32 GB ceiling (fits = vram + 4 <= 32, so > 28 GB is
# over budget) to verify the VRAM bar's RED zone + the Run-disabled branch
# (AC1). Kept inline so the stub stays import-free.
_CATALOG_MODELS: list[dict[str, Any]] = [
    {
        "id": "qwen3-7b-awq",
        "served_name": "Qwen/Qwen3-7B-AWQ",
        "quantization": "awq",
        "expected_vram_gb": 5.0,
        "max_context": 16384,
        "state": "available",
    },
    {
        "id": "qwen3-14b-awq",
        "served_name": "Qwen/Qwen3-14B-AWQ",
        "quantization": "awq",
        "expected_vram_gb": 9.0,
        "max_context": 16384,
        "state": "available",
    },
    {
        "id": "qwen3-32b-awq",
        "served_name": "Qwen/Qwen3-32B-AWQ",
        "quantization": "awq",
        "expected_vram_gb": 21.0,
        "max_context": 16384,
        "state": "available",
    },
    {
        # Synthetic e2e-only over-budget candidate: 40 + 4 (Infinity) = 44 GB
        # > 32 GB ceiling → fits=False. Exercises V3's VRAM-bar RED zone +
        # Run-disabled branch (AC1). Not a real catalog entry (a 72B AWQ does
        # not fit the single 32 GB GPU).
        "id": "qwen3-72b-awq",
        "served_name": "Qwen/Qwen3-72B-AWQ",
        "quantization": "awq",
        "expected_vram_gb": 40.0,
        "max_context": 16384,
        "state": "available",
    },
]


@app.get("/catalog")
async def catalog() -> JSONResponse:
    """Swap-controller catalog (swap_controller_client.get_catalog → GET /catalog).

    Returns every generator candidate annotated with state. No model is
    resident in the e2e stack, so all are ``available`` and none ``active``."""
    return JSONResponse({"models": _CATALOG_MODELS})


@app.get("/catalog/status")
async def catalog_status() -> JSONResponse:
    """Swap-controller status (swap_controller_client.get_status → GET
    /catalog/status). No model is resident in the GPU-free stack."""
    return JSONResponse({"active_model_id": None})


@app.post("/catalog/activate")
async def catalog_activate(req: Request) -> JSONResponse:
    """Swap-controller activate (swap_controller_client.activate → POST
    /catalog/activate). A no-op ack — the stub never loads weights; it just
    reports the requested model as ready so the activate flow is exercisable."""
    body = await req.json()
    model_id = str(body.get("model_id", ""))
    known = any(m["id"] == model_id for m in _CATALOG_MODELS)
    if not known:
        return JSONResponse(
            {
                "state": "error",
                "active_model_id": None,
                "failure_reason": f"unknown model {model_id}",
            }
        )
    return JSONResponse({"state": "ready", "active_model_id": model_id, "failure_reason": None})


if __name__ == "__main__":
    port = int(os.environ.get("E2E_MODEL_STUB_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
