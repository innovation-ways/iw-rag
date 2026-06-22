"""Deep health probe — DB + extensions, vLLM, infinity (embed + rerank).

The readiness check the API's ``/health?deep=true`` reuses (lld-api §3). Prints a
JSON status naming each dependency; exits non-zero if any check fails. Never
crashes on an unreachable dependency — it reports ``degraded`` with the failure.

Run: ``uv run python scripts/health_probe.py`` (needs the .env / IW_RAG__* vars).
"""

from __future__ import annotations

import asyncio
import json

import httpx

from iw_rag.settings import get_settings
from iw_rag.storage.db import Database


async def _check_db() -> dict[str, object]:
    settings = get_settings()
    db = Database(settings.database)
    try:
        status = await db.health()
        return {"ok": status.ok, "reachable": status.reachable, "detail": status.detail}
    finally:
        await db.dispose()


async def _check_vllm(client: httpx.AsyncClient, endpoint: str) -> dict[str, object]:
    try:
        resp = await client.get(f"{endpoint.rstrip('/')}/models")
        return {"ok": resp.status_code == 200, "detail": f"/models → {resp.status_code}"}
    except Exception as exc:  # noqa: BLE001 — probe must not raise
        return {"ok": False, "detail": f"unreachable: {exc}"}


async def _check_embedder(
    client: httpx.AsyncClient, endpoint: str, model: str
) -> dict[str, object]:
    try:
        resp = await client.post(
            f"{endpoint.rstrip('/')}/embeddings",
            json={"model": model, "input": "health probe"},
        )
        data = resp.json()
        dims = len(data["data"][0]["embedding"])
        return {"ok": dims == 1024, "detail": f"embedding dims={dims}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"unreachable: {exc}"}


async def _check_reranker(
    client: httpx.AsyncClient, endpoint: str, model: str
) -> dict[str, object]:
    try:
        resp = await client.post(
            f"{endpoint.rstrip('/')}/rerank",
            json={
                "model": model,
                "query": "invoice total",
                "documents": ["the total due is 100", "weather forecast"],
            },
        )
        data = resp.json()
        results = data.get("results", data)
        return {"ok": bool(results), "detail": f"reranked {len(results)} docs"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"unreachable: {exc}"}


async def main() -> int:
    """Run all dependency checks and print a JSON status."""
    settings = get_settings()
    gen = settings.models.get("generator")
    emb = settings.models.get("embedder")
    rer = settings.models.get("reranker")

    checks: dict[str, dict[str, object]] = {"database": await _check_db()}
    async with httpx.AsyncClient(timeout=15) as client:
        checks["vllm"] = await _check_vllm(client, gen.endpoint)
        checks["embedder"] = await _check_embedder(client, emb.endpoint, emb.served_name)
        checks["reranker"] = await _check_reranker(client, rer.endpoint, rer.served_name)

    overall_ok = all(c["ok"] for c in checks.values())
    print(json.dumps({"status": "ok" if overall_ok else "degraded", "checks": checks}, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
