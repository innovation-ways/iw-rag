"""Unit tests for the multi-run Compare PDF HTML builder.

:func:`iw_rag.api.benchmark_compare_pdf.build_compare_html` turns the
``BenchmarkMultiCompareResponse`` payload into a self-contained, branded,
print-optimised HTML document.  These tests pin the structural contract:
the run legend + per-run palette colours, the aggregate bars, the
baseline-relative delta tables, the per-item heatmaps, the answers
appendix, and the local-only invariant (no external CDN/fonts).
"""

from __future__ import annotations

from typing import Any

from iw_rag.api.benchmark_compare_pdf import build_compare_html


def _payload(*, with_samples: bool = True) -> dict[str, Any]:
    samples_a = [{"answer": "Ans-A", "contexts": ["ctx-1"]}] if with_samples else [None]
    return {
        "dataset_id": "ds-1",
        "baseline_id": "a",
        "n_queries": 1,
        "underpowered": True,
        "caveat": "n=1 below min_queries_for_significance=150",
        "metrics": ["ndcg_at_5", "faithfulness"],
        "runs": [
            {
                "id": "a",
                "display_id": "BENCH-a",
                "is_baseline": True,
                "color": "#2DD4BF",
                "config_snapshot": {"generator": "gen-a", "embedder": "bge"},
                "dataset_name": "benchmark_golden",
                "means": {"ndcg_at_5": 0.5, "faithfulness": 0.8},
                "deltas": [],
            },
            {
                "id": "b",
                "display_id": "BENCH-b",
                "is_baseline": False,
                "color": "#a78bfa",
                "config_snapshot": {"generator": "gen-b", "embedder": "bge"},
                "dataset_name": "benchmark_golden",
                "means": {"ndcg_at_5": 0.7, "faithfulness": 0.9},
                "deltas": [
                    {
                        "metric": "ndcg_at_5",
                        "val_a": 0.5,
                        "val_b": 0.7,
                        "delta": 0.2,
                        "p_value": 0.01,
                        "p_corrected": 0.02,
                        "ci": (0.01, 0.2),
                        "significant": True,
                    }
                ],
            },
        ],
        "items": [
            {
                "index": 0,
                "question": "What is X?",
                "values": {"ndcg_at_5": [0.5, 0.7], "faithfulness": [0.8, 0.9]},
                "samples": samples_a + [None],
            }
        ],
    }


def test_build_compare_html_is_a_full_document() -> None:
    html = build_compare_html(_payload())
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "</html>" in html
    assert "<title>Benchmark Comparison</title>" in html


def test_build_compare_html_has_legend_with_run_colours() -> None:
    html = build_compare_html(_payload())
    assert "legend" in html
    assert "#2DD4BF" in html  # baseline run colour
    assert "#a78bfa" in html  # second run colour
    assert "BENCH-a" in html and "BENCH-b" in html
    assert "baseline" in html


def test_build_compare_html_has_aggregate_and_delta_sections() -> None:
    html = build_compare_html(_payload())
    assert "Aggregate" in html
    assert "Baseline-relative deltas" in html
    # delta sign rendered with a leading + for positive deltas
    assert "+0.200" in html


def test_build_compare_html_has_heatmaps_and_answers() -> None:
    html = build_compare_html(_payload())
    assert "heatmap" in html
    assert "Per-item heatmaps" in html
    assert "Answers per item" in html
    assert "Ans-A" in html
    assert "What is X?" in html


def test_build_compare_html_omits_answers_when_no_samples() -> None:
    html = build_compare_html(_payload(with_samples=False))
    assert "Answers per item" not in html
    # The heatmap + aggregate sections still render.
    assert "Per-item heatmaps" in html


def test_build_compare_html_renders_underpowered_caveat() -> None:
    html = build_compare_html(_payload())
    assert "Underpowered" in html
    assert "150" in html


def test_build_compare_html_no_external_cdn() -> None:
    html = build_compare_html(_payload()).lower()
    for forbidden in (
        "fonts.googleapis.com",
        "cdn.jsdelivr.net",
        "unpkg.com",
        "http://",
        "https://",
    ):
        assert forbidden not in html, f"PDF HTML must be local-only (found {forbidden})"


def test_build_compare_html_renders_conclusion_when_present() -> None:
    payload = _payload()
    payload["conclusion"] = "Mistral is the strongest overall: +0.06 nDCG@5 vs the baseline."
    html = build_compare_html(payload)
    assert "Conclusion &amp; reasoning" in html
    assert "Mistral is the strongest overall" in html


def test_build_compare_html_omits_conclusion_when_absent() -> None:
    html = build_compare_html(_payload())  # no "conclusion" key
    assert "Conclusion &amp; reasoning" not in html


def test_build_compare_html_has_glossary() -> None:
    html = build_compare_html(_payload())
    assert "Glossary" in html
    # metric definitions for the metrics shown
    assert "nDCG@5" in html and "Faithfulness" in html
    # how-to-read-the-comparison section
    assert "How to read this comparison" in html
    for term in ("Baseline", "p_corrected", "CI 95%", "Significant", "Heatmap", "Underpowered"):
        assert term in html, f"glossary missing '{term}'"


def test_build_compare_html_glossary_only_shows_present_metrics() -> None:
    payload = _payload()
    payload["metrics"] = ["ndcg_at_5"]  # faithfulness not shown
    html = build_compare_html(payload)
    # The metric glossary table should not define a metric not in the comparison.
    # (Faithfulness may still appear in prose elsewhere, so anchor on the unit line.)
    assert "Recall@5" not in html


def test_build_compare_html_handles_empty_items() -> None:
    payload = _payload()
    payload["items"] = []
    html = build_compare_html(payload)
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "Per-item heatmaps" in html
