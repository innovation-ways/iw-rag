# ruff: noqa: E501
"""Standalone, branded HTML for the multi-run Compare "Generate Report" PDF.

:func:`build_compare_html` turns the
:class:`~iw_rag.api.schemas.BenchmarkMultiCompareResponse` payload (as a
plain dict — ``model_dump()``) into a self-contained, print-optimised
HTML document (Innovation Ways teal/ink branding, inline CSS, inline
coloured bars + heatmap cells) that the
``GET /benchmark/compare/multi/pdf`` route renders to PDF via headless
Chromium.  It mirrors the per-run colour identity from the live Compare
view: each run keeps the same palette colour across the legend, the
aggregate bars, the heatmaps, and the per-item answers.

Local-only (Invariant 1/8): no external fonts/CDNs; everything is inline.
It re-derives the figures in Python so the PDF is fully server-side and
needs no JS engine beyond Chromium's print path (mirrors
:mod:`iw_rag.api.benchmark_pdf_report`).
"""

from __future__ import annotations

import html as _html
import math
from typing import Any

_CSS = """
  :root { --teal:#0D9488; --ink:#181D29; --line:#cfd6e4; --text:#1b2030; --muted:#5b6478; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Helvetica Neue",Arial,sans-serif; color:var(--text); font-size:12px; line-height:1.5; }
  .page { padding:30px 34px; }
  header { display:flex; align-items:center; gap:12px; border-bottom:3px solid var(--teal); padding-bottom:12px; margin-bottom:16px; }
  .logo { font-weight:800; font-size:20px; color:var(--ink); }
  .logo .t { color:var(--teal); }
  .sub { color:var(--muted); font-size:12px; }
  .spacer { flex:1; }
  h1 { font-size:19px; margin:0 0 2px; color:var(--ink); }
  h2 { font-size:14px; margin:20px 0 8px; color:var(--ink); border-bottom:1px solid var(--line); padding-bottom:4px; }
  h3 { font-size:12.5px; margin:12px 0 4px; color:var(--ink); }
  .meta { color:var(--muted); font-size:11px; margin-bottom:12px; }
  .caveat { background:#fff7ed; border:1px solid #f59e0b; border-radius:8px; padding:10px 12px; margin:8px 0; font-size:11px; color:#7c4a03; }
  .concl { background:#f3f6fb; border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:0 0 8px; white-space:pre-wrap; font-size:11.5px; }
  table { width:100%; border-collapse:collapse; margin:6px 0 4px; }
  th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); font-size:11px; vertical-align:middle; }
  th { color:var(--muted); text-transform:uppercase; font-size:9.5px; letter-spacing:.5px; }
  td.num,th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .pos { color:#0D9488; font-weight:700; } .neg { color:#c0392b; font-weight:700; }
  .sw { display:inline-block; width:11px; height:11px; border-radius:3px; vertical-align:middle; margin-right:5px; }
  .chip { display:inline-block; font-size:9px; padding:1px 6px; border-radius:999px; border:1px solid var(--line); }
  .legend span.run { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:4px 10px; margin:0 8px 8px 0; font-size:11px; }
  .legend .mods { color:var(--muted); font-size:10px; }
  .bar-row { display:grid; grid-template-columns:120px 1fr 56px; align-items:center; gap:8px; padding:2px 0; }
  .bar-name { font-size:10px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .bar-track { height:11px; background:#e6ebf3; border-radius:6px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:6px; }
  .bar-val { font-size:10.5px; color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }
  table.heatmap td, table.heatmap th { border:1px solid var(--line); text-align:center; padding:4px 6px; font-size:10px; }
  table.heatmap th.q, table.heatmap td.q { text-align:left; max-width:300px; }
  .ans { border-left:3px solid var(--teal); padding:4px 10px; margin:6px 0; }
  .ans .h { font-size:10.5px; font-weight:700; margin-bottom:2px; }
  .ans .b { font-size:10.5px; white-space:pre-wrap; }
  .note { color:var(--muted); font-size:10.5px; margin:4px 0; }
  footer { margin-top:22px; border-top:1px solid var(--line); padding-top:8px; color:var(--muted); font-size:10px; }
  @page { size:A4 landscape; margin:12mm; }
"""

# Compact glossary labels — kept in sync with the console's IW_GLOSSARY so the
# PDF names each metric the same way the live view does.
_LABELS: dict[str, str] = {
    "ndcg_at_5": "nDCG@5",
    "recall_at_5": "Recall@5",
    "recall_at_10": "Recall@10",
    "faithfulness": "Faithfulness",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "factual_correctness": "Factual correctness",
    "noise_sensitivity": "Noise sensitivity",
}


def _e(s: Any) -> str:
    return _html.escape("" if s is None else str(s))


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _label(metric: str) -> str:
    return _LABELS.get(metric, metric)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` (or ``#rgb``) to an ``rgba(...)`` string."""
    h = (hex_color or "#0D9488").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        r, g, b = 13, 148, 136
    a = max(0.0, min(1.0, alpha))
    return f"rgba({r},{g},{b},{a:.3f})"


def _run_mods(run: dict[str, Any]) -> str:
    cfg = run.get("config_snapshot") or {}
    parts = [str(cfg[k]) for k in ("generator", "embedder", "reranker", "grader") if cfg.get(k)]
    return " / ".join(parts)


def _run_label(run: dict[str, Any]) -> str:
    return str(run.get("display_id") or ("#" + str(run.get("id") or "")[:8]))


def _legend(runs: list[dict[str, Any]]) -> str:
    chips = ""
    for run in runs:
        bl = '<span class="chip">baseline</span>' if run.get("is_baseline") else ""
        chips += (
            f'<span class="run"><span class="sw" style="background:{_e(run.get("color"))}"></span>'
            f"{_e(_run_label(run))} {bl} <span class='mods'>{_e(_run_mods(run))}</span></span>"
        )
    return f'<div class="legend">{chips}</div>'


def _aggregate(runs: list[dict[str, Any]], metrics: list[str]) -> str:
    body = ""
    for m in metrics:
        rows = ""
        for run in runs:
            v = _f((run.get("means") or {}).get(m))
            pct = 0.0 if v is None else max(0.0, min(1.0, v)) * 100.0
            val = "—" if v is None else f"{v:.3f}"
            rows += (
                f'<div class="bar-row"><div class="bar-name">{_e(_run_label(run))}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{_e(run.get("color"))}"></div></div>'
                f'<div class="bar-val">{_e(val)}</div></div>'
            )
        body += f"<h3>{_e(_label(m))}</h3>{rows}"
    return f"<h2>Aggregate — mean per metric</h2>{body}"


def _deltas(runs: list[dict[str, Any]], baseline_id: str) -> str:
    base_label = next((_run_label(r) for r in runs if r.get("id") == baseline_id), baseline_id)
    blocks = ""
    for run in runs:
        if run.get("is_baseline"):
            continue
        blocks += (
            f'<h3><span class="sw" style="background:{_e(run.get("color"))}"></span>'
            f"{_e(_run_label(run))} vs baseline</h3>"
        )
        rows = run.get("deltas") or []
        if not rows:
            blocks += '<p class="note">No shared per-query metrics with the baseline.</p>'
            continue
        body = ""
        for row in rows:
            delta = _f(row.get("delta"))
            dcls = "" if delta is None else ("pos" if delta >= 0 else "neg")
            dtxt = "—" if delta is None else (f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}")
            ci = row.get("ci")
            ci_txt = (
                f"[{_f(ci[0]):.3f}, {_f(ci[1]):.3f}]"
                if isinstance(ci, list | tuple) and len(ci) == 2 and _f(ci[0]) is not None
                else "—"
            )
            pc = _f(row.get("p_corrected"))
            sig = "✓" if row.get("significant") else "·"
            va, vb = _f(row.get("val_a")), _f(row.get("val_b"))
            body += (
                f"<tr><td>{_e(_label(str(row.get('metric'))))}</td>"
                f'<td class="num">{"—" if va is None else f"{va:.3f}"}</td>'
                f'<td class="num">{"—" if vb is None else f"{vb:.3f}"}</td>'
                f'<td class="num {dcls}">{_e(dtxt)}</td>'
                f'<td class="num">{"—" if pc is None else f"{pc:.4f}"}</td>'
                f'<td class="num">{_e(ci_txt)}</td><td class="num">{sig}</td></tr>'
            )
        blocks += (
            '<table><thead><tr><th>metric</th><th class="num">baseline</th><th class="num">this run</th>'
            '<th class="num">delta</th><th class="num">p_corr</th><th class="num">CI 95%</th><th class="num">sig</th></tr></thead>'
            f"<tbody>{body}</tbody></table>"
        )
    return (
        f"<h2>Baseline-relative deltas (vs {_e(base_label)})</h2>"
        '<p class="note">Delta = run − baseline (positive ⇒ the run beats the baseline). '
        "Paired Wilcoxon + BCa 95% CI, Bonferroni-Holm corrected.</p>" + blocks
    )


def _heatmaps(runs: list[dict[str, Any]], items: list[dict[str, Any]], metrics: list[str]) -> str:
    out = ""
    for m in metrics:
        head = '<table class="heatmap"><thead><tr><th class="q">item</th>'
        for run in runs:
            head += f'<th><span class="sw" style="background:{_e(run.get("color"))}"></span>{_e(_run_label(run))}</th>'
        head += "</tr></thead><tbody>"
        body = ""
        for it in items:
            q = it.get("question") or f"Item {int(it.get('index', 0)) + 1}"
            if len(q) > 80:
                q = q[:80] + "…"
            body += f'<tr><td class="q">{_e(q)}</td>'
            vals = (it.get("values") or {}).get(m) or []
            for ri, run in enumerate(runs):
                v = _f(vals[ri]) if ri < len(vals) else None
                if v is None:
                    body += '<td style="color:#9aa3b8">—</td>'
                else:
                    bg = _hex_to_rgba(str(run.get("color")), 0.12 + 0.78 * max(0.0, min(1.0, v)))
                    body += f'<td style="background:{bg}">{v:.2f}</td>'
            body += "</tr>"
        out += f"<h3>{_e(_label(m))}</h3>{head}{body}</tbody></table>"
    return f"<h2>Per-item heatmaps</h2>{out}"


def _answers(
    runs: list[dict[str, Any]], items: list[dict[str, Any]], *, max_chars: int = 600
) -> str:
    has_any = any(any(s for s in (it.get("samples") or [])) for it in items)
    if not has_any:
        return ""
    blocks = ""
    for it in items:
        samples = it.get("samples") or []
        if not any(samples):
            continue
        q = it.get("question") or f"Item {int(it.get('index', 0)) + 1}"
        runs_html = ""
        for ri, run in enumerate(runs):
            s = samples[ri] if ri < len(samples) else None
            if not s:
                continue
            ans = (s.get("answer") or "")[:max_chars]
            runs_html += (
                f'<div class="ans" style="border-left-color:{_e(run.get("color"))}">'
                f'<div class="h">{_e(_run_label(run))}</div>'
                f'<div class="b">{_e(ans) if ans else "<em>no answer recorded</em>"}</div></div>'
            )
        blocks += f"<h3>Item {int(it.get('index', 0)) + 1} — {_e(q)}</h3>{runs_html}"
    return f"<h2>Answers per item</h2>{blocks}"


def _conclusion(payload: dict[str, Any]) -> str:
    """Render the local-LLM conclusion + reasoning block (empty when absent)."""
    text = payload.get("conclusion")
    if not text:
        return ""
    return f'<h2>Conclusion &amp; reasoning</h2><div class="concl">{_e(text)}</div>'


# Per-metric definitions — condensed from the console's IW_GLOSSARY so the PDF
# explains every value shown in the comparison the same way the live view does.
_METRIC_DEFS: dict[str, tuple[str, str]] = {
    "ndcg_at_5": (
        "0–1 · higher is better",
        "Normalised Discounted Cumulative Gain over the top 5 retrieved passages — "
        "rewards ranking the most relevant evidence near the top. 1.0 = perfect ordering.",
    ),
    "recall_at_5": (
        "0–1 · higher is better",
        "Of the passages that should answer the question, the fraction that appeared "
        "in the top 5 retrieved results.",
    ),
    "recall_at_10": (
        "0–1 · higher is better",
        "Same as Recall@5, but counting the top 10 retrieved passages.",
    ),
    "faithfulness": (
        "0–1 · higher is better",
        "Of the claims the answer made, the fraction actually supported by the "
        "retrieved context. Low = hallucination. Scored locally by the Lynx-8B judge.",
    ),
    "context_precision": (
        "0–1 · higher is better",
        "Of the passages handed to the generator, the fraction that were genuinely "
        "relevant. Low = the context was padded with noise.",
    ),
    "context_recall": (
        "0–1 · higher is better",
        "Of the facts in the gold reference answer, the fraction covered by the "
        "retrieved context. Requires a dataset that ships reference answers.",
    ),
    "factual_correctness": (
        "0–1 · higher is better",
        "How factually aligned the generated answer is with the gold reference "
        "answer. Requires reference answers.",
    ),
    "noise_sensitivity": (
        "0–1 · LOWER is better",
        "How often the answer makes claims drawn from irrelevant retrieved passages. "
        "Lower = more robust to noisy context.",
    ),
}

# How-to-read definitions specific to the comparison (not single-run metrics).
_COMPARE_DEFS: list[tuple[str, str]] = [
    (
        "Baseline",
        "One run is chosen as the reference. Every other run is compared against it "
        "item-by-item; the baseline keeps its own colour and is marked in the legend.",
    ),
    (
        "Delta (vs baseline)",
        "this run − baseline for a metric. Positive (green) means the run beats the "
        "baseline on that metric; negative (red) means it is worse.",
    ),
    (
        "p_corrected",
        "The paired-test p-value after Bonferroni-Holm correction across the compared "
        "metrics. Smaller = less likely the difference is chance.",
    ),
    (
        "CI 95%",
        "The 95% bootstrap confidence interval of the mean difference. When it "
        "excludes 0, the direction of the delta is reliable.",
    ),
    (
        "Significant",
        "Marked only when p_corrected < 0.05 AND the 95% CI excludes 0 — both "
        "conditions are required before a difference is called real.",
    ),
    (
        "Heatmap",
        "One row per dataset item, one column per run. Each cell is tinted in the "
        "run's colour with depth proportional to the value (deeper = higher).",
    ),
    (
        "Run colours",
        "Each run is assigned a stable colour used across the legend, the aggregate "
        "bars, the heatmaps, and the per-item answers, so a run is recognisable "
        "everywhere in this report.",
    ),
    (
        "Underpowered",
        "With few queries the paired test detects only large differences. When shown, "
        "read small, non-significant deltas with caution.",
    ),
]


def _glossary(metrics: list[str]) -> str:
    """Render the metric + how-to-read glossary for the comparison PDF.

    Mirrors the single-report PDF's glossary so the comparison export is
    self-explanatory: a methodology note, a row per metric actually shown,
    and a 'how to read this comparison' section.
    """
    methodology = (
        "<p class='note'>All scores are computed locally on a single RTX 5090 — never "
        "by a hosted API. Retrieval metrics (recall, nDCG) are scored deterministically "
        "against the dataset's human relevance labels; faithfulness is judged by the "
        "local Lynx-8B model and the reference-required metrics by the local Ragas suite "
        "using a different judge family from the generator (self-preference de-bias). "
        "This comparison pairs every run on the SAME dataset and tests each run against "
        "the baseline with a paired Wilcoxon signed-rank test + BCa bootstrap CI, "
        "Bonferroni-Holm corrected across the metrics.</p>"
    )
    metric_rows = "".join(
        f"<tr><td><strong>{_e(_label(m))}</strong><div class='note'>{_e(_METRIC_DEFS[m][0])}</div></td>"
        f"<td>{_e(_METRIC_DEFS[m][1])}</td></tr>"
        for m in metrics
        if m in _METRIC_DEFS
    )
    compare_rows = "".join(
        f"<tr><td><strong>{_e(term)}</strong></td><td>{_e(meaning)}</td></tr>"
        for term, meaning in _COMPARE_DEFS
    )
    return (
        "<h2>Glossary — what each value means</h2>"
        f"{methodology}"
        "<h3 style='color:#0D9488'>Metrics</h3>"
        f"<table><tbody>{metric_rows}</tbody></table>"
        "<h3 style='color:#0D9488'>How to read this comparison</h3>"
        f"<table><tbody>{compare_rows}</tbody></table>"
    )


def build_compare_html(payload: dict[str, Any]) -> str:
    """Render a multi-compare payload into a self-contained, branded HTML report.

    Args:
        payload: The ``BenchmarkMultiCompareResponse.model_dump()`` dict —
            ``runs`` / ``items`` / ``metrics`` / ``baseline_id`` /
            ``n_queries`` / ``underpowered`` / ``caveat`` / ``dataset_id``.

    Returns:
        A complete HTML document ready for Chromium's ``page.pdf``.
    """
    runs: list[dict[str, Any]] = list(payload.get("runs") or [])
    items: list[dict[str, Any]] = list(payload.get("items") or [])
    metrics: list[str] = list(payload.get("metrics") or [])
    baseline_id = str(payload.get("baseline_id") or "")
    dataset = ""
    if runs:
        dataset = str(runs[0].get("dataset_name") or payload.get("dataset_id") or "")

    caveat_block = ""
    if payload.get("underpowered") or payload.get("caveat"):
        caveat_block = (
            f'<div class="caveat"><strong>Underpowered.</strong> '
            f"{_e(payload.get('caveat') or 'Small sample — read deltas with care.')}</div>"
        )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Benchmark Comparison</title>"
        f"<style>{_CSS}</style></head><body><div class='page'>"
        "<header><span class='logo'><span class='t'>Innovation</span> Ways</span>"
        "<span class='sub'>· Benchmark Comparison</span><span class='spacer'></span></header>"
        f"<h1>Benchmark Comparison — {len(runs)} runs</h1>"
        f"<div class='meta'>dataset: {_e(dataset or '—')} · queries: {_e(payload.get('n_queries', '?'))} · "
        f"baseline: {_e(next((_run_label(r) for r in runs if r.get('id') == baseline_id), baseline_id))}</div>"
        f"{_legend(runs)}"
        f"{caveat_block}"
        f"{_conclusion(payload)}"
        f"{_aggregate(runs, metrics)}"
        f"{_deltas(runs, baseline_id)}"
        f"{_heatmaps(runs, items, metrics)}"
        f"{_answers(runs, items)}"
        f"{_glossary(metrics)}"
        "<footer>Generated by the IW-RAG Benchmark Studio · fully local, single-GPU, open-weight · scores judged by local models (no hosted APIs).</footer>"
        "</div></body></html>"
    )


__all__ = ["build_compare_html"]
