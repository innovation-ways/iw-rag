"""HTTP routes behind the Learn section's interactive labs.

Read-only and stateless: each lab endpoint runs a real pipeline component over a
bundled sample document, never over user data, so it needs no access filtering.  It
is still authenticated like the rest of the API, because the console is one
sign-in.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from iw_rag.api.routes import AccessCtx
from iw_rag.learn.chunking_lab import MAX_LEAF_CHARS, MIN_LEAF_CHARS, run_chunking_lab
from iw_rag.learn.guards_lab import run_guards_lab
from iw_rag.learn.guards_scenarios import get_scenario
from iw_rag.learn.hybrid_lab import (
    MAX_RRF_K,
    MAX_WEIGHT,
    get_principal,
    get_query,
    run_hybrid_lab,
)
from iw_rag.learn.rerank_lab import (
    MAX_BUDGET,
    MAX_TOP_K,
    MIN_BUDGET,
    MIN_TOP_K,
    run_rerank_lab,
)
from iw_rag.learn.rerank_lab import get_query as get_rerank_query
from iw_rag.learn.samples import get_sample

learn_router = APIRouter()


@learn_router.get("/learn/lab/chunking")
async def chunking_lab(
    ctx: AccessCtx,
    sample: Annotated[str, Query(max_length=40)],
    leaf: Annotated[int, Query(ge=MIN_LEAF_CHARS, le=MAX_LEAF_CHARS)] = 500,
) -> dict[str, Any]:
    """Run the real chunker over a bundled sample at the requested leaf size.

    Args:
        ctx: The authenticated principal (unused beyond requiring sign-in).
        sample: Id of a bundled sample document.
        leaf: Target leaf size in characters.

    Returns:
        The leaves, parents, per-probe verdicts and summary numbers.

    Raises:
        HTTPException: 404 when ``sample`` is not a bundled sample.
    """
    doc = get_sample(sample)
    if doc is None:
        raise HTTPException(status_code=404, detail="Unknown sample")
    return asdict(run_chunking_lab(doc, leaf))


@learn_router.get("/learn/lab/hybrid")
async def hybrid_lab(
    ctx: AccessCtx,
    query: Annotated[str, Query(max_length=40)],
    principal: Annotated[str, Query(max_length=40)],
    w_dense: Annotated[float, Query(ge=0, le=MAX_WEIGHT)] = 1.0,
    w_sparse: Annotated[float, Query(ge=0, le=MAX_WEIGHT)] = 1.0,
    k: Annotated[int, Query(ge=1, le=MAX_RRF_K)] = 60,
) -> dict[str, Any]:
    """Replay a recorded real search run, re-fusing with the requested RRF settings.

    Args:
        ctx: The authenticated principal (unused beyond requiring sign-in).
        query: Id of a lab question.
        principal: Id of the lab user to replay as.
        w_dense: RRF weight of the dense arm (0 switches it off).
        w_sparse: RRF weight of the BM25 arm (0 switches it off).
        k: The RRF constant.

    Returns:
        The top hits of each stage and a verdict on the answer chunk.

    Raises:
        HTTPException: 404 for an unknown question or user.
    """
    lab_query, lab_principal = get_query(query), get_principal(principal)
    if lab_query is None or lab_principal is None:
        raise HTTPException(status_code=404, detail="Unknown question or user")
    result = run_hybrid_lab(lab_query, lab_principal, w_dense=w_dense, w_sparse=w_sparse, rrf_k=k)
    return asdict(result)


@learn_router.get("/learn/lab/guards")
async def guards_lab(
    ctx: AccessCtx,
    scenario: Annotated[str, Query(max_length=40)],
) -> dict[str, Any]:
    """Replay a recorded run of the production graph as explained checkpoint steps.

    Args:
        ctx: The authenticated principal (unused beyond requiring sign-in).
        scenario: Id of a bundled scenario.

    Returns:
        The explained steps, the final outcome and what in the run was real or scripted.

    Raises:
        HTTPException: 404 when ``scenario`` is not a bundled scenario.
    """
    chosen = get_scenario(scenario)
    if chosen is None:
        raise HTTPException(status_code=404, detail="Unknown scenario")
    return asdict(run_guards_lab(chosen))


@learn_router.get("/learn/lab/rerank")
async def rerank_lab(
    ctx: AccessCtx,
    query: Annotated[str, Query(max_length=40)],
    top_k: Annotated[int, Query(ge=MIN_TOP_K, le=MAX_TOP_K)] = 8,
    reranker: Annotated[str, Query(pattern="^(on|off)$")] = "on",
    expansion: Annotated[str, Query(pattern="^(parent|leaf)$")] = "parent",
    budget: Annotated[int, Query(ge=MIN_BUDGET, le=MAX_BUDGET)] = 4000,
    sections: Annotated[str, Query(pattern="^(on|off)$")] = "on",
) -> dict[str, Any]:
    """Replay a recorded real spine run under the requested rerank and expansion settings.

    Args:
        ctx: The authenticated principal (unused beyond requiring sign-in).
        query: Id of a lab question.
        top_k: How many chunks to keep after reranking.
        reranker: ``on``, or ``off`` to replay the spine's fallback when the reranker is down.
        expansion: ``parent`` (small-to-big) or ``leaf`` (no expansion).
        budget: The per-candidate expansion budget in characters.
        sections: ``on``, or ``off`` to switch off ``retrieval.section_hits`` (sections cannot
            match on their own).

    Returns:
        The fused and reranked lists, what the generator receives and the facts about it.

    Raises:
        HTTPException: 404 for an unknown question.
    """
    lab_query = get_rerank_query(query)
    if lab_query is None:
        raise HTTPException(status_code=404, detail="Unknown question")
    result = run_rerank_lab(
        lab_query,
        top_k=top_k,
        reranker_on=reranker == "on",
        expansion=expansion,
        budget_chars=budget,
        sections=sections == "on",
    )
    return asdict(result)


__all__ = ["learn_router"]
