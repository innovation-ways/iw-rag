"""Unit tests for the N-way comparison conclusion writer.

``CompareConclusionWriter`` has a LOCAL judge write a grounded
conclusion+reasoning from an already-computed multi-compare payload.
These tests pin the deterministic prompt (real numbers only, baseline
marked, deltas + significance) and the writer's degrade-to-None contract
— with a fake chat client so no network/LLM is touched.
"""

from __future__ import annotations

import asyncio
from typing import Any

from iw_rag.eval.benchmark.conclusion import CompareConclusionWriter, build_compare_user_prompt


def _payload() -> dict[str, Any]:
    return {
        "dataset_id": "ds",
        "baseline_id": "a",
        "n_queries": 6,
        "underpowered": True,
        "caveat": "n=6 below min_queries_for_significance=150",
        "metrics": ["ndcg_at_5", "faithfulness"],
        "runs": [
            {
                "id": "a",
                "display_id": "BENCH-1",
                "is_baseline": True,
                "config_snapshot": {"generator": "qwen3-14b", "embedder": "bge-m3"},
                "means": {"ndcg_at_5": 0.60, "faithfulness": 0.80},
                "deltas": [],
            },
            {
                "id": "b",
                "display_id": "BENCH-2",
                "is_baseline": False,
                "config_snapshot": {"generator": "mistral-small"},
                "means": {"ndcg_at_5": 0.66, "faithfulness": 0.84},
                "deltas": [
                    {
                        "metric": "ndcg_at_5",
                        "delta": 0.06,
                        "p_corrected": 0.30,
                        "significant": False,
                    },
                    {
                        "metric": "faithfulness",
                        "delta": 0.04,
                        "p_corrected": 0.04,
                        "significant": True,
                    },
                ],
            },
        ],
    }


def test_compare_prompt_marks_baseline_and_modules() -> None:
    """Baseline run is marked '(BASELINE)' and precedes the comparison run in the prompt."""
    prompt = build_compare_user_prompt(_payload())
    assert "BENCH-1 (BASELINE)" in prompt
    assert "modules: qwen3-14b / bge-m3" in prompt
    assert "BENCH-2" in prompt
    # The baseline run section must appear before the non-baseline run section.
    assert prompt.index("BENCH-1 (BASELINE)") < prompt.index("BENCH-2")


def test_compare_prompt_includes_means_and_signed_deltas() -> None:
    """Means use 3-decimal format; deltas carry explicit sign; significance is labelled."""
    prompt = build_compare_user_prompt(_payload())
    assert "ndcg_at_5: 0.600" in prompt  # baseline mean
    assert "ndcg_at_5: +0.060" in prompt  # signed delta vs baseline
    assert "faithfulness: +0.040" in prompt
    # Significance is surfaced per delta.
    assert "significant" in prompt
    assert "not significant" in prompt
    # The ndcg_at_5 baseline mean appears before its delta entry.
    assert prompt.index("ndcg_at_5: 0.600") < prompt.index("ndcg_at_5: +0.060")


def test_compare_prompt_surfaces_underpowered_caveat() -> None:
    """Underpowered caveat is rendered as a 'CAVEAT' block containing the min-queries threshold."""
    prompt = build_compare_user_prompt(_payload())
    assert "CAVEAT" in prompt
    assert "150" in prompt
    # The caveat block must contain the threshold in the same region of the prompt.
    assert prompt.index("CAVEAT") < prompt.index("150")


def test_compare_prompt_omits_deltas_for_baseline() -> None:
    """The baseline run has no 'vs baseline' delta block."""
    prompt = build_compare_user_prompt(_payload())
    # The baseline section appears before the run section; count delta blocks.
    assert prompt.count("vs baseline") == 1  # only the one non-baseline run


def test_writer_returns_text_from_fake_client() -> None:
    class _Resp:
        text = "  Mistral wins on faithfulness; nDCG gain is not significant.  "

    class _Client:
        async def complete(self, messages: Any, params: Any) -> Any:
            # The judge must be given system + user messages.
            assert messages[0]["role"] == "system"
            assert "comparison" in messages[0]["content"].lower()
            return _Resp()

    writer = CompareConclusionWriter(chat_client=_Client(), model="judge")
    out = asyncio.run(writer.write(_payload()))
    assert out == "Mistral wins on faithfulness; nDCG gain is not significant."


def test_writer_degrades_to_none_on_error() -> None:
    class _Boom:
        async def complete(self, messages: Any, params: Any) -> Any:
            raise RuntimeError("judge endpoint down")

    writer = CompareConclusionWriter(chat_client=_Boom(), model="judge")
    assert asyncio.run(writer.write(_payload())) is None


def test_writer_returns_none_on_empty_text() -> None:
    class _Resp:
        text = "   "

    class _Client:
        async def complete(self, messages: Any, params: Any) -> Any:
            return _Resp()

    writer = CompareConclusionWriter(chat_client=_Client(), model="judge")
    assert asyncio.run(writer.write(_payload())) is None


def test_conclusion_module_exposes_compare_writer() -> None:
    """Module __all__ exposes both the compare writer class and the prompt builder."""
    from iw_rag.eval.benchmark import conclusion as mod

    assert "CompareConclusionWriter" in mod.__all__
    assert "build_compare_user_prompt" in mod.__all__
    # Exactly these two compare-specific symbols must be in __all__ (not zero, not buried).
    compare_symbols = [s for s in mod.__all__ if "compare" in s.lower()]
    assert len(compare_symbols) == 2
