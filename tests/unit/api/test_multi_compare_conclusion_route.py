"""Unit tests for the N-way compare conclusion route + its cache.

``GET /benchmark/compare/multi/conclusion`` computes the comparison
payload, then has the LOCAL judge write a grounded conclusion (cached per
comparison identity so the PDF reuses it; ``refresh=true`` regenerates).
These tests drive the route function directly with a fake request +
service and a patched conclusion factory — no app/DB/LLM needed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from iw_rag.api import benchmark_routes as br
from iw_rag.domain.errors import NotFoundError


def _service_result() -> dict[str, Any]:
    """A minimal service ``compare_multi`` payload (mappable by the route)."""
    return {
        "dataset_id": "ds",
        "baseline_id": "a",
        "n_queries": 3,
        "underpowered": True,
        "caveat": "n=3 < 150",
        "metrics": ["ndcg_at_5"],
        "runs": [
            {
                "id": "a",
                "display_id": "BENCH-1",
                "label": "qwen",
                "is_baseline": True,
                "color": "#2DD4BF",
                "config_snapshot": {"generator": "qwen"},
                "dataset_name": "benchmark_golden",
                "created_at": "2026-06-11",
                "n_queries": 3,
                "means": {"ndcg_at_5": 0.5},
                "delta": None,
            },
            {
                "id": "b",
                "display_id": "BENCH-2",
                "label": "mistral",
                "is_baseline": False,
                "color": "#a78bfa",
                "config_snapshot": {"generator": "mistral"},
                "dataset_name": "benchmark_golden",
                "created_at": "2026-06-12",
                "n_queries": 3,
                "means": {"ndcg_at_5": 0.6},
                "delta": None,
            },
        ],
        "items": [],
    }


class _FakeService:
    def __init__(self, result: dict[str, Any] | None) -> None:
        self._result = result

    async def compare_multi(self, ids: list[str], baseline_id: str | None = None) -> Any:
        return self._result


_SENTINEL_SETTINGS = object()


def _request(service: Any, *, settings: Any = _SENTINEL_SETTINGS) -> Any:
    state = SimpleNamespace(benchmark_service=service, settings=settings)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _patch_writer(counter: list[int], text: str | None = "GEN"):
    """Patch the conclusion factory to a counting fake (no LLM)."""

    def _factory(_settings: Any) -> Any:
        async def _fn(_payload: dict[str, Any]) -> str | None:
            counter.append(1)
            return text

        return _fn

    return patch("iw_rag.api.benchmark_composition.build_compare_conclusion_fn", _factory)


def test_conclusion_generates_and_caches() -> None:
    req = _request(_FakeService(_service_result()))
    calls: list[int] = []
    with _patch_writer(calls):
        out1 = asyncio.run(br.compare_benchmark_reports_multi_conclusion(req, None, ids="a,b"))
        out2 = asyncio.run(br.compare_benchmark_reports_multi_conclusion(req, None, ids="a,b"))
    assert out1 == {"conclusion": "GEN"}
    assert out2 == {"conclusion": "GEN"}
    # Second call served from cache → the writer ran only once.
    assert len(calls) == 1


def test_conclusion_refresh_regenerates() -> None:
    req = _request(_FakeService(_service_result()))
    calls: list[int] = []
    with _patch_writer(calls):
        asyncio.run(br.compare_benchmark_reports_multi_conclusion(req, None, ids="a,b"))
        asyncio.run(
            br.compare_benchmark_reports_multi_conclusion(req, None, ids="a,b", refresh=True)
        )
    assert len(calls) == 2


def test_conclusion_cache_key_is_order_independent() -> None:
    req = _request(_FakeService(_service_result()))
    calls: list[int] = []
    with _patch_writer(calls):
        asyncio.run(br.compare_benchmark_reports_multi_conclusion(req, None, ids="a,b"))
        # Same run set, different id order → same cache entry, no regeneration.
        asyncio.run(br.compare_benchmark_reports_multi_conclusion(req, None, ids="b,a"))
    assert len(calls) == 1


def test_conclusion_none_when_settings_absent() -> None:
    req = _request(_FakeService(_service_result()), settings=None)
    out = asyncio.run(br.compare_benchmark_reports_multi_conclusion(req, None, ids="a,b"))
    assert out == {"conclusion": None}


def test_conclusion_none_when_writer_unavailable() -> None:
    req = _request(_FakeService(_service_result()))
    with patch("iw_rag.api.benchmark_composition.build_compare_conclusion_fn", lambda _s: None):
        out = asyncio.run(br.compare_benchmark_reports_multi_conclusion(req, None, ids="a,b"))
    assert out == {"conclusion": None}


def test_conclusion_404_when_id_unknown() -> None:
    req = _request(_FakeService(None))  # service returns None → unknown id
    with pytest.raises(NotFoundError):
        asyncio.run(br.compare_benchmark_reports_multi_conclusion(req, None, ids="a,b"))


def test_cache_key_helper_dedupes_and_sorts() -> None:
    assert br._compare_cache_key(["b", "a", "a"], "a") == "a|b::a"
    assert br._compare_cache_key(["a", "b"], "a") == br._compare_cache_key(["b", "a"], "a")
