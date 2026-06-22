"""Unit tests for the run-time conclusion (deterministic comparison + prompt).

The LLM only writes prose; the numbers it is given MUST be computed
deterministically (so the conclusion never hallucinates a figure).  These
pin the comparison math + that the real deltas reach the prompt.
"""

from __future__ import annotations

from iw_rag.eval.benchmark.conclusion import (
    build_basis,
    build_user_prompt,
    summarize_report,
)


def _row(gen: str, faith: float, vram: float, e2e: float) -> dict[str, object]:
    return {
        "report_label": gen,
        "model_config": {"generator": gen},
        "faithfulness_mean": faith,
        "ndcg_at_5": 0.5,
        "latency": {"e2e_p99_s": e2e, "per_stage_ms": {"retrieve_generate_ms": 1000.0}},
        "resource": {"gpu": {"vram_peak_mb": vram, "vram_incremental_mb": vram - 1000}},
    }


def test_summarize_report_extracts_headline_numbers() -> None:
    s = summarize_report(_row("qwen3-14b", 0.80, 24000, 2.0))
    assert s["generator"] == "qwen3-14b"
    assert s["faithfulness"] == 0.80
    assert s["vram_peak_mb"] == 24000
    assert s["e2e_p99_s"] == 2.0


def test_build_basis_computes_signed_deltas_vs_best_comparable() -> None:
    """Deltas are this-run vs the best (highest-faithfulness, different-gen) run."""
    this = summarize_report(_row("qwen3-14b", 0.80, 24000, 2.0))
    comparables = [
        summarize_report(_row("qwen3-8b", 0.70, 16000, 1.6)),  # the baseline
        summarize_report(_row("qwen3-8b-alt", 0.60, 15000, 1.5)),
    ]
    basis = build_basis(this, comparables)
    assert basis["baseline"]["generator"] == "qwen3-8b"  # highest-faith comparable
    d = basis["deltas"]
    # +0.10 faithfulness over 0.70 = +14.3%
    assert round(d["faithfulness_pct"], 1) == 14.3
    # 24000 vs 16000 VRAM = +50%
    assert round(d["vram_pct"], 1) == 50.0
    # 2.0 vs 1.6 e2e = +25% (slower)
    assert round(d["e2e_pct"], 1) == 25.0


def test_build_user_prompt_carries_the_real_numbers() -> None:
    this = summarize_report(_row("qwen3-14b", 0.80, 24000, 2.0))
    comparables = [summarize_report(_row("qwen3-8b", 0.70, 16000, 1.6))]
    prompt = build_user_prompt(build_basis(this, comparables))
    lines = prompt.splitlines()
    # The deterministic deltas reach the prompt as whole rendered lines — pin
    # the exact line (label + computed pct + sign hint) rather than a loose
    # substring, so a mis-labelled, mis-signed, or mis-rounded figure fails.
    faith_line = next(ln for ln in lines if "faithfulness:" in ln and "%" in ln)
    vram_line = next(ln for ln in lines if ln.strip().startswith("VRAM:"))
    assert faith_line == "  faithfulness: +14.3%"
    # The VRAM delta line also carries the sign meaning the LLM is told.
    assert vram_line == "  VRAM: +50.0% (positive = more memory)"


def test_build_basis_no_comparable_yields_empty_deltas() -> None:
    this = summarize_report(_row("qwen3-14b", 0.80, 24000, 2.0))
    basis = build_basis(this, [])
    assert basis["baseline"] is None
    assert basis["deltas"] == {}
    assert "on its own merits" in build_user_prompt(basis)
