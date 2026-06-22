"""Unit tests for the server-side benchmark PDF/HTML report builder.

``benchmark_pdf_report.build_report_html`` re-derives every figure in Python so
the ``GET /benchmark/reports/{id}/pdf`` route stays fully server-side and
deterministic (no JS engine beyond Chromium's print path).  These pin that the
real persisted numbers reach the HTML, that the value-class thresholds and the
unit formatters behave at their boundaries, and that missing data degrades to
``—`` instead of raising.
"""

from __future__ import annotations

from typing import Any

from iw_rag.api.benchmark_pdf_report import (
    _clean_label,
    _dataset_label,
    _f,
    _mb,
    _ms,
    _pick,
    _score_cell,
    _secs,
    _util_pct,
    build_report_html,
)


def _full_row() -> dict[str, Any]:
    """A populated ``benchmark_reports`` row covering every render branch."""
    return {
        "id": "abcdef1234567890",
        "display_id": "BR-0007",
        "report_label": "qwen3-14b nightly",
        "created_at": "2026-06-18T09:00:00Z",
        "n_queries": 50,
        "n_judged": 48,
        "dataset_id": "f6c10fa1-c229-4a18-a91d-5966b61bfebc",
        "dataset_name": "benchmark_golden",
        "dataset_version": "v1",
        "model_config": {
            "generator": "qwen3-14b",
            "embedder": "bge-m3",
            "reranker": "bge-reranker",
            "grader": "selene",
            "dataset_id": "golden",
            "params": {"top_k": 5, "temperature": 0.0, "max_tokens": 512},
        },
        "faithfulness_mean": 0.82,
        "ndcg_at_5": 0.71,
        "latency": {
            "e2e_p99_s": 2.4,
            "ttft_p99_s": 0.3,
            "tpot_p99_s": 0.02,
            "queue_time_p99_s": 0.5,
            "model_load_ms": 8200.0,
            "per_stage_ms": {"retrieve_generate_ms": 1300.0},
            "phase_ms": {
                "model_load_ms": 8200.0,
                "generate_ms": 1300.0,
                "faithfulness_ms": 600.0,
                "grading_ms": 700.0,
                "ragas_ms": 900.0,
            },
        },
        "resource": {
            "gpu": {
                "vram_peak_mb": 24576.0,
                "vram_incremental_mb": 15360.0,
                "util_peak_pct": 96,
                "device_name": "NVIDIA RTX 5090",
            }
        },
        "metrics": {
            "recall_at_5": [1.0, 0.0, 1.0, 1.0],
            "recall_at_10": [1.0, 1.0, 1.0, 1.0],
            "ndcg_at_10": [0.8, 0.6],
            "mrr": [0.5, 1.0],
            "context_precision": [0.9, 0.7],
            "context_recall": [0.6, 0.4],
            "factual_correctness": [0.55],
            "noise_sensitivity": [0.2, 0.4],
            "conclusion": {"text": "qwen3-14b leads on faithfulness at a 50% VRAM cost."},
        },
    }


def test_build_report_html_carries_headline_and_identity() -> None:
    html = build_report_html(_full_row())
    assert "<!doctype html>" in html.lower()
    # Identity + meta line.
    assert "Benchmark Report — BR-0007" in html
    assert "qwen3-14b nightly" in html
    assert "queries: 50" in html
    assert "judged: 48" in html
    # Headline stat cards render the promoted scalars at 3 dp — the bold card
    # value div is unique to the headline (tiers use <span class="v-…">).
    assert html.count('<div class="b">0.820</div>') == 1  # faithfulness_mean card
    assert html.count('<div class="b">0.710</div>') == 1  # ndcg_at_5 card


def test_build_report_html_shows_human_dataset_name_not_uuid() -> None:
    """The meta line + intro render the resolved ``name (version)``, never the
    opaque dataset uuid."""
    html = build_report_html(_full_row())
    assert "dataset: benchmark_golden (v1)" in html
    # The raw uuid must not leak into the rendered report.
    assert "f6c10fa1-c229-4a18-a91d-5966b61bfebc" not in html
    # The intro paragraph names the same human dataset.
    assert "“benchmark_golden”" in html


def test_clean_label_strips_trailing_uuid_only() -> None:
    # Legacy auto-label "<gen>@<uuid>" → drop the uuid suffix.
    assert _clean_label("qwen3-8b-awq@f6c10fa1-c229-4a18-a91d-5966b61bfebc") == "qwen3-8b-awq"
    # A human label is left untouched (no uuid suffix).
    assert _clean_label("qwen3-14b nightly") == "qwen3-14b nightly"
    # A non-uuid dataset suffix is preserved.
    assert _clean_label("qwen3-14b@my-set") == "qwen3-14b@my-set"


def test_dataset_label_prefers_name_then_falls_back_to_uuid() -> None:
    # name + version → "name (version)".
    assert _dataset_label({"dataset_name": "benchmark_golden", "dataset_version": "v1"}) == (
        "benchmark_golden (v1)"
    )
    # name without version → bare name.
    assert _dataset_label({"dataset_name": "benchmark_golden"}) == "benchmark_golden"
    # no resolved name → fall back to the raw id (legacy rows), config first.
    assert _dataset_label({"dataset_id": "f6c10fa1"}) == "f6c10fa1"
    assert _dataset_label({"model_config": {"dataset_id": "cfg-id"}}) == "cfg-id"
    # nothing at all → em-dash, never blank.
    assert _dataset_label({}) == "—"


def test_build_report_html_conclusion_block_present_when_text_set() -> None:
    html = build_report_html(_full_row())
    assert html.count("<h2>Conclusion</h2>") == 1
    assert "leads on faithfulness" in html


def test_build_report_html_mean_reduces_vector_metrics() -> None:
    """A metrics-blob vector with no promoted scalar is mean-reduced for the cell."""
    html = build_report_html(_full_row())
    # recall_at_5 = mean([1,0,1,1]) = 0.75 -> rendered at 3 dp as a "good" cell.
    # Pin the exact cell the renderer should emit, and that it reaches the HTML.
    assert _score_cell(0.75) == '<span class="v-good">0.750</span>'
    assert '<span class="v-good">0.750</span>' in html


def test_build_report_html_timings_waterfall_svg_and_total() -> None:
    html = build_report_html(_full_row())
    assert "<h2>Step timings</h2>" in html
    assert "<svg" in html
    # Each declared phase label reaches the waterfall.
    assert "Model load / swap" in html
    assert "Ragas metrics" in html
    # Total = 8200+1300+600+700+900 = 11700 ms -> formatted as "11.7 s", once.
    assert _ms(11700.0) == "11.7 s"
    assert html.count("11.7 s") == 1


def test_build_report_html_includes_metric_glossary() -> None:
    """The report ends with a glossary explaining every value + the methodology."""
    html = build_report_html(_full_row())
    assert html.count("<h2>Glossary — what each value means</h2>") == 1
    # A representative term from each tier is defined.
    for term in ("Recall@5", "nDCG@5", "MRR", "Faithfulness", "Noise sensitivity", "VRAM peak"):
        assert term in html
    # The methodology line states scoring is local (no hosted API).
    assert "never by a hosted API" in html


def test_full_row_has_no_missing_data_caveats() -> None:
    """A fully-populated row shows neither the GPU-missing nor the
    reference-required caveat (both sections have real data)."""
    html = build_report_html(_full_row())
    assert "No GPU telemetry was captured" not in html
    assert "Reference-required metrics were not computed" not in html


def test_missing_gpu_telemetry_renders_explanatory_note() -> None:
    """When GPU telemetry is absent, the report explains WHY rather than showing
    a bare column of em-dashes."""
    row = dict(_full_row())
    row["resource"] = {"gpu": {"available": False, "vram_peak_mb": None}}
    html = build_report_html(row)
    assert html.count("No GPU telemetry was captured for this run") == 1


def test_missing_reference_required_metrics_render_explanatory_note() -> None:
    """When the Ragas reference-required metrics are all absent, the report
    explains the blanks (faithfulness, being reference-free, is unaffected)."""
    row = dict(_full_row())
    row["metrics"] = {
        "context_recall": [None, None],
        "factual_correctness": [None],
        "noise_sensitivity": [None, None],
    }
    html = build_report_html(row)
    assert html.count("Reference-required metrics were not computed for this run") == 1


def test_build_report_html_degrades_gracefully_on_empty_row() -> None:
    """A near-empty row must render (em-dashes), never raise."""
    html = build_report_html({})
    assert "<!doctype html>" in html.lower()
    assert "—" in html  # missing metrics collapse to em-dash
    # No conclusion / no timings sections when the data is absent.
    assert "<h2>Conclusion</h2>" not in html
    assert "<h2>Step timings</h2>" not in html


def test_timings_falls_back_to_score_ms_shape() -> None:
    """When none of the named phases exist but legacy ``score_ms`` does, the
    3-row fallback renders (the named-phase list must stay empty to reach it)."""
    row = {"latency": {"phase_ms": {"score_ms": 500.0}}}
    html = build_report_html(row)
    assert html.count("Scoring (judging)") == 1
    assert "<svg" in html


# --- helper boundary behaviour ------------------------------------------------


def test_f_parses_numbers_and_rejects_nan_and_garbage() -> None:
    assert _f("3.5") == 3.5
    assert _f(None) is None
    assert _f("not-a-number") is None
    assert _f(float("nan")) is None


def test_pick_prefers_promoted_scalar_over_vector() -> None:
    rep = {"faithfulness_mean": 0.9, "metrics": {"faithfulness_mean": [0.1, 0.1]}}
    assert _pick(rep, "faithfulness_mean") == 0.9


def test_score_cell_classes_at_thresholds() -> None:
    # higher-better: >=0.75 good, >=0.50 mid, else low — pin the full cell.
    assert _score_cell(0.75) == '<span class="v-good">0.750</span>'
    assert _score_cell(0.50) == '<span class="v-mid">0.500</span>'
    assert _score_cell(0.49) == '<span class="v-low">0.490</span>'
    # lower-better inverts: <=0.30 good, <=0.60 mid, else low.
    assert _score_cell(0.30, lower=True) == '<span class="v-good">0.300</span>'
    assert _score_cell(0.61, lower=True) == '<span class="v-low">0.610</span>'
    # None -> the neutral em-dash cell.
    assert _score_cell(None) == '<span class="na">—</span>'


def test_unit_formatters_cross_their_boundaries() -> None:
    # _ms: >=1000 -> seconds, else ms.
    assert _ms(1500.0) == "1.5 s"
    assert _ms(800.0) == "800 ms"
    assert _ms(None) == "—"
    # _secs: <1 -> ms, else seconds at 2 dp.
    assert _secs(0.25) == "250 ms"
    assert _secs(2.0) == "2.00 s"
    assert _secs(None) == "—"
    # _mb: >=1024 -> GB, else MB.
    assert _mb(2048.0) == "2.0 GB"
    assert _mb(512.0) == "512 MB"
    assert _mb(None) == "—"
    # _util_pct: int percent or em-dash.
    assert _util_pct(96) == "96%"
    assert _util_pct(None) == "—"
