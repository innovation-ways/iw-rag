# ruff: noqa: E501
"""Standalone, branded HTML for the Benchmark "Generate Report" PDF.

:func:`build_report_html` turns a persisted ``benchmark_reports`` row into a
self-contained, print-optimised HTML document (Innovation Ways teal/ink
branding, inline CSS, inline-SVG timing waterfall) that the
``GET /benchmark/reports/{id}/pdf`` route renders to PDF via headless
Chromium.  It deliberately re-derives the figures in Python (rather than
reusing the console's client-side renderer) so the PDF is fully server-side,
deterministic, and needs no JS engine beyond Chromium's print path.

Local-only (Invariant 1/8): no external fonts/CDNs; everything is inline.
"""

from __future__ import annotations

import html as _html
import math
import re
from typing import Any

# Legacy report labels were auto-generated as ``{generator}@{dataset_uuid}``
# (the run route had already resolved the dataset name to a uuid).  Strip a
# trailing ``@<uuid>`` so an already-persisted report still renders a clean
# label; new runs no longer embed the uuid (see runner._build_report_label).
_LABEL_UUID_SUFFIX = re.compile(
    r"@[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _clean_label(label: Any) -> str:
    """Drop a trailing ``@<uuid>`` from a legacy auto-generated report label."""
    return _LABEL_UUID_SUFFIX.sub("", str(label))


_CSS = """
  :root { --teal:#2DD4BF; --teal-deep:#0D9488; --ink:#181D29; --ink-2:#222631; --line:#cfd6e4; --text:#1b2030; --muted:#5b6478; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Helvetica Neue",Arial,sans-serif; color:var(--text); font-size:12px; line-height:1.5; }
  .page { padding:32px 36px; }
  header { display:flex; align-items:center; gap:12px; border-bottom:3px solid var(--teal-deep); padding-bottom:12px; margin-bottom:18px; }
  .logo { font-weight:800; font-size:20px; letter-spacing:.2px; color:var(--ink); }
  .logo .t { color:var(--teal-deep); }
  .sub { color:var(--muted); font-size:12px; }
  .spacer { flex:1; }
  h1 { font-size:19px; margin:0 0 2px; color:var(--ink); }
  h2 { font-size:14px; margin:20px 0 8px; color:var(--ink); border-bottom:1px solid var(--line); padding-bottom:4px; }
  .meta { color:var(--muted); font-size:11px; margin-bottom:14px; }
  .intro { background:#f3f6fb; border-left:3px solid var(--teal-deep); border-radius:8px; padding:12px 14px; margin:0 0 16px; }
  .concl { background:#f3f6fb; border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:0 0 8px; white-space:pre-wrap; }
  table { width:100%; border-collapse:collapse; margin:6px 0 4px; }
  th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); font-size:11px; vertical-align:middle; }
  th { color:var(--muted); text-transform:uppercase; font-size:9.5px; letter-spacing:.5px; }
  td.num,th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .v-good{color:var(--teal-deep);font-weight:700;} .v-mid{color:#b7791f;font-weight:700;} .v-low{color:#c0392b;font-weight:700;}
  .na{color:var(--muted);}
  .stats { display:flex; gap:10px; margin:4px 0 8px; flex-wrap:wrap; }
  .stat { flex:1; min-width:120px; background:var(--ink-2); color:#fff; border-radius:8px; padding:10px 12px; }
  .stat .b { font-size:20px; font-weight:800; color:var(--teal); }
  .stat .l { font-size:10px; opacity:.8; text-transform:uppercase; letter-spacing:.4px; }
  .stat .u { font-size:9.5px; opacity:.7; }
  .note { color:var(--muted); font-size:10.5px; margin:4px 0; }
  footer { margin-top:22px; border-top:1px solid var(--line); padding-top:8px; color:var(--muted); font-size:10px; }
  @page { size:A4; margin:14mm; }
"""


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _e(s: Any) -> str:
    return _html.escape("" if s is None else str(s))


def _pick(rep: dict[str, Any], *keys: str) -> float | None:
    """Promoted scalar first, else mean-reduce the metrics-blob vector."""
    blob = rep.get("metrics") or {}
    for k in keys:
        if rep.get(k) is not None:
            return _f(rep.get(k))
        v = blob.get(k)
        if isinstance(v, list):
            xs = [float(x) for x in v if isinstance(x, int | float) and not math.isnan(float(x))]
            if xs:
                return sum(xs) / len(xs)
    return None


def _score_cell(v: float | None, lower: bool = False) -> str:
    if v is None:
        return '<span class="na">—</span>'
    good = (v <= 0.30) if lower else (v >= 0.75)
    mid = (v <= 0.60) if lower else (v >= 0.50)
    cls = "v-good" if good else ("v-mid" if mid else "v-low")
    return f'<span class="{cls}">{v:.3f}</span>'


def _ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


def _secs(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"


def _mb(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v / 1024:.1f} GB" if v >= 1024 else f"{v:.0f} MB"


def _util_pct(v: Any) -> str:
    f = _f(v)
    return "—" if f is None else f"{int(f)}%"


def _dataset_label(rep: dict[str, Any]) -> str:
    """Human-readable dataset identity: ``name (version)``, uuid as fallback.

    The service resolves the opaque ``dataset_id`` uuid to the ``eval_datasets``
    ``dataset_name`` + ``dataset_version`` (e.g. ``benchmark_golden`` / ``v1``);
    legacy rows without that resolution fall back to whatever id is present so
    the line is never blank.
    """
    name = rep.get("dataset_name")
    if name:
        version = rep.get("dataset_version")
        return f"{name} ({version})" if version else str(name)
    cfg = rep.get("model_config") or rep.get("config_snapshot") or {}
    return str(cfg.get("dataset_id") or rep.get("dataset_id") or "—")


def _intro(rep: dict[str, Any]) -> str:
    c = rep.get("model_config") or rep.get("config_snapshot") or {}
    p = c.get("params") or {}
    nq = rep.get("n_queries", "?")
    return (
        f"This benchmark evaluated the {_e(c.get('generator') or 'configured generator')} generator — with "
        f"{_e(c.get('embedder') or 'an embedder')} embeddings and {_e(c.get('reranker') or 'a reranker')} reranking — over the "
        f"“{_e(rep.get('dataset_name') or c.get('dataset_id') or 'golden')}” set of {_e(nq)} questions on a single RTX 5090 (32 GB). "
        "Retrieval quality was scored deterministically against relevance labels; answer quality was judged locally "
        f"(Lynx-8B for faithfulness, the Ragas suite via {_e(c.get('grader') or 'the grader')} for context precision/recall, "
        "factual correctness and noise sensitivity). "
        f"Parameters: top-k={_e(p.get('top_k', '–'))}, temperature={_e(p.get('temperature', '–'))}, max-tokens={_e(p.get('max_tokens', '–'))}."
    )


_TIER_ROWS: list[tuple[str, bool, list[tuple[str, tuple[str, ...]]]]] = [
    (
        "Retrieval quality",
        False,
        [
            ("Recall@5", ("recall_at_5",)),
            ("Recall@10", ("recall_at_10",)),
            ("nDCG@5", ("ndcg_at_5",)),
            ("nDCG@10", ("ndcg_at_10",)),
            ("MRR", ("mrr",)),
        ],
    ),
    (
        "Answer quality — reference-free",
        False,
        [
            ("Faithfulness", ("faithfulness_mean", "faithfulness")),
            ("Context precision", ("context_precision",)),
        ],
    ),
    (
        "Answer quality — reference-required",
        False,
        [
            ("Context recall", ("context_recall",)),
            ("Factual correctness", ("factual_correctness",)),
            ("Noise sensitivity (lower better)", ("noise_sensitivity",)),
        ],
    ),
]

_PHASES = [
    ("model_load_ms", "Model load / swap"),
    ("generate_ms", "Retrieval + generation"),
    ("faithfulness_ms", "Faithfulness judge (Lynx)"),
    ("grading_ms", "Grading judge (Selene)"),
    ("ragas_ms", "Ragas metrics"),
]


def _timings_svg(rep: dict[str, Any]) -> str:
    pm = (rep.get("latency") or {}).get("phase_ms") or {}
    phases = [(label, _f(pm.get(key))) for key, label in _PHASES if pm.get(key) is not None]
    if not phases and pm.get("score_ms") is not None:
        phases = [
            ("Model load / swap", _f(pm.get("model_load_ms")) or 0.0),
            ("Retrieval + generation", _f(pm.get("generate_ms")) or 0.0),
            ("Scoring (judging)", _f(pm.get("score_ms")) or 0.0),
        ]
    if not phases:
        return ""
    total = sum(v or 0.0 for _, v in phases) or 1.0
    rows = ""
    y = 0
    rowh = 24
    barw = 360
    for label, ms in phases:
        pct = (ms or 0.0) / total * 100.0
        w = (ms or 0.0) / total * barw
        rows += (
            f'<text x="0" y="{y + 15}" font-size="11" fill="#1b2030">{_e(label)}</text>'
            f'<rect x="200" y="{y + 4}" width="{barw}" height="13" rx="6" fill="#e6ebf3"/>'
            f'<rect x="200" y="{y + 4}" width="{w:.1f}" height="13" rx="6" fill="#0D9488"/>'
            f'<text x="{200 + barw + 8}" y="{y + 15}" font-size="10.5" fill="#5b6478">{_e(_ms(ms))} · {pct:.0f}%</text>'
        )
        y += rowh
    return (
        f'<svg width="100%" viewBox="0 0 640 {y}" xmlns="http://www.w3.org/2000/svg" '
        f'style="max-width:640px">{rows}</svg>'
        f'<p class="note">Total ≈ {_e(_ms(total))}.</p>'
    )


# Glossary — mirrors the console's ``IW_GLOSSARY`` (benchmark_render.py) so the
# PDF and the live Console explain every value identically.  Each entry is
# ``(term, unit, meaning)``; grouped by the same tiers the report renders.
_GLOSSARY: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Retrieval quality — does the search surface the right evidence?",
        [
            (
                "Recall@5",
                "0–1 · higher is better",
                "Of the passages that SHOULD answer the question, the fraction that "
                "appeared in the top 5 retrieved results. Measures whether retrieval "
                "surfaces the right evidence at all.",
            ),
            (
                "Recall@10",
                "0–1 · higher is better",
                "Same as Recall@5, but counting the top 10 retrieved passages.",
            ),
            (
                "nDCG@5",
                "0–1 · higher is better",
                "Normalised Discounted Cumulative Gain over the top 5. Rewards ranking "
                "the most relevant passages near the top, not merely retrieving them. "
                "1.0 = perfect ordering.",
            ),
            (
                "nDCG@10",
                "0–1 · higher is better",
                "Ranking quality over the top 10 retrieved passages.",
            ),
            (
                "MRR",
                "0–1 · higher is better",
                "Mean Reciprocal Rank: 1 / (position of the first relevant passage), "
                "averaged over questions. 1.0 = the first hit was always relevant; "
                "0.5 = it was typically 2nd.",
            ),
        ],
    ),
    (
        "Answer quality — reference-free (no gold answer needed)",
        [
            (
                "Faithfulness",
                "0–1 · higher is better",
                "Of the claims the model made in its answer, the fraction actually "
                "supported by the retrieved context. Low = it invented things beyond "
                "its sources (hallucination). Scored locally by the Lynx-8B judge.",
            ),
            (
                "Context precision",
                "0–1 · higher is better",
                "Of the passages handed to the generator, the fraction that were "
                "genuinely relevant. Low = the context was padded with noise. Scored by "
                "the local Ragas suite.",
            ),
        ],
    ),
    (
        "Answer quality — reference-required (needs gold reference answers)",
        [
            (
                "Context recall",
                "0–1 · higher is better",
                "Of the facts in the gold reference answer, the fraction covered by the "
                "retrieved context. Requires a dataset that ships reference answers.",
            ),
            (
                "Factual correctness",
                "0–1 · higher is better",
                "How factually aligned the generated answer is with the gold reference "
                "answer. Requires reference answers.",
            ),
            (
                "Noise sensitivity",
                "0–1 · LOWER is better",
                "How often the answer makes claims drawn from irrelevant retrieved "
                "passages. Lower = more robust to noisy context. Requires reference "
                "answers.",
            ),
        ],
    ),
    (
        "Speed & latency",
        [
            (
                "Model load",
                "cold-swap time",
                "Wall-clock time to cold-swap and load the generator onto the GPU. "
                "Tracked in its own bucket and NEVER folded into per-query or "
                "end-to-end latency.",
            ),
            (
                "Retrieve + generate (per query)",
                "per query, mean",
                "Average wall-clock time for one query's retrieve + generate stage, "
                "measured by the app's own request timeline.",
            ),
            (
                "TTFT p99",
                "seconds · lower is better",
                "Time-To-First-Token, 99th percentile: how long until the first output "
                "token appears. Reported by the vLLM server.",
            ),
            (
                "TPOT p99",
                "seconds/token · lower is better",
                "Time-Per-Output-Token (inter-token latency), 99th percentile — the "
                "streaming speed once generation starts. Reported by vLLM.",
            ),
            (
                "End-to-end p99",
                "seconds · lower is better",
                "Full request latency at the 99th percentile, as seen by the vLLM "
                "server. The headline 'how slow is the worst case' number.",
            ),
            (
                "Queue p99",
                "seconds · lower is better",
                "Time a request waited in the vLLM queue before processing began, 99th percentile.",
            ),
        ],
    ),
    (
        "GPU & memory",
        [
            (
                "VRAM peak",
                "GPU memory",
                "Highest GPU memory in use at any point during the measured run — true "
                "device memory read from the driver (pynvml), not framework "
                "self-report.",
            ),
            (
                "VRAM footprint (incremental)",
                "GPU memory",
                "Peak minus the idle baseline — the resident model's own memory "
                "footprint for this run.",
            ),
            (
                "GPU util peak",
                "%",
                "Highest GPU compute utilisation sampled during the run.",
            ),
        ],
    ),
]


def _glossary_html() -> str:
    """Render the metric glossary — every value, its unit, and how to read it.

    Mirrors the console's ``IW_GLOSSARY`` so the PDF and live UI agree.
    """
    methodology = (
        "<p class='note'>All scores are computed locally on a single RTX 5090 — "
        "never by a hosted API. Retrieval metrics (recall, nDCG, MRR) are scored "
        "deterministically against the dataset's human relevance labels. Faithfulness "
        "is judged by the local Lynx-8B model; the reference-required metrics (context "
        "recall, factual correctness, noise sensitivity) and context precision by the "
        "local Ragas suite using a DIFFERENT judge family from the generator "
        "(self-preference de-bias). Latency percentiles come from the vLLM server's "
        "own <code>/metrics</code>; VRAM from the GPU driver (pynvml).</p>"
    )
    sections = ""
    for title, entries in _GLOSSARY:
        rows = "".join(
            f"<tr><td><strong>{_e(term)}</strong><div class='note'>{_e(unit)}</div></td>"
            f"<td>{_e(meaning)}</td></tr>"
            for term, unit, meaning in entries
        )
        sections += (
            f"<h3 style='font-size:12px;margin:12px 0 4px;color:#0D9488'>{_e(title)}</h3>"
            f"<table><tbody>{rows}</tbody></table>"
        )
    return f"<h2>Glossary — what each value means</h2>{methodology}{sections}"


def _answer_quality_note(rep: dict[str, Any]) -> str:
    """A caveat line when the reference-required answer-quality tier is empty.

    These metrics (context recall / factual correctness / noise sensitivity)
    are ``None`` when the Ragas judge produced no values for the run — e.g. the
    dataset ships no reference answers, or the judge endpoint was unavailable.
    Explains the blanks instead of leaving a column of em-dashes unexplained.
    """
    keys = ("context_recall", "factual_correctness", "noise_sensitivity")
    if any(_pick(rep, k) is not None for k in keys):
        return ""
    return (
        "<p class='note'>Reference-required metrics were not computed for this run — "
        "the local Ragas judge returned no values (the dataset ships no reference "
        "answers, or the judge endpoint was unavailable at compute time). "
        "Reference-free quality (faithfulness) above is unaffected.</p>"
    )


def _gpu_unavailable_note(gpu: dict[str, Any]) -> str:
    """A caveat line when no GPU telemetry was captured for the run.

    ``available: False`` (or every numeric field ``None``) means the benchmark
    ran without GPU device access (pynvml could not initialise), so VRAM /
    utilisation could not be sampled.  Explains the blanks rather than showing a
    table of bare em-dashes.
    """
    measured = any(
        _f(gpu.get(k)) is not None for k in ("vram_peak_mb", "vram_incremental_mb", "util_peak_pct")
    )
    if gpu.get("available") is not False and measured:
        return ""
    return (
        "<p class='note'>No GPU telemetry was captured for this run — the benchmark "
        "executed without GPU device access (the driver / pynvml could not be reached "
        "from the runner), so VRAM and utilisation could not be sampled. Quality and "
        "vLLM-reported latency above are unaffected.</p>"
    )


def build_report_html(rep: dict[str, Any]) -> str:
    """Render a persisted report row into a self-contained, branded HTML report.

    Args:
        rep: A ``benchmark_reports`` row dict (promoted columns + ``latency`` /
            ``resource`` / ``metrics`` JSONB + ``display_id`` from the service).

    Returns:
        A complete HTML document ready for Chromium's ``page.pdf``.
    """
    did = rep.get("display_id") or ("#" + str(rep.get("id", ""))[:8])
    cfg: dict[str, Any] = rep.get("model_config") or rep.get("config_snapshot") or {}
    lat: dict[str, Any] = rep.get("latency") or {}
    res: dict[str, Any] = rep.get("resource") or {}
    gpu: dict[str, Any] = res.get("gpu") or {}
    blob: dict[str, Any] = rep.get("metrics") or {}
    concl = blob.get("conclusion")
    concl_text = (
        concl.get("text")
        if isinstance(concl, dict)
        else (concl if isinstance(concl, str) else None)
    )

    # Headline stat cards.
    faith = _pick(rep, "faithfulness_mean", "faithfulness")
    ndcg = _pick(rep, "ndcg_at_5")
    e2e = _f(lat.get("e2e_p99_s"))
    vram = _f(gpu.get("vram_incremental_mb"))
    stats = ""
    for label, val, unit in [
        ("Faithfulness", f"{faith:.3f}" if faith is not None else "—", "0–1 · higher better"),
        ("nDCG@5", f"{ndcg:.3f}" if ndcg is not None else "—", "0–1 · higher better"),
        ("End-to-end p99", _secs(e2e), "lower better"),
        ("VRAM footprint", _mb(vram), "incremental"),
    ]:
        stats += f'<div class="stat"><div class="b">{_e(val)}</div><div class="l">{_e(label)}</div><div class="u">{_e(unit)}</div></div>'

    # Quality tiers.
    tiers = ""
    for title, _lo, rows in _TIER_ROWS:
        body = ""
        for name, keys in rows:
            lower = "lower" in name.lower()
            body += f'<tr><td>{_e(name)}</td><td class="num">{_score_cell(_pick(rep, *keys), lower)}</td></tr>'
        tiers += f"<h2>{_e(title)}</h2><table><tbody>{body}</tbody></table>"

    # Speed + GPU.
    ps = lat.get("per_stage_ms") or {}
    speed = (
        "<table><tbody>"
        f'<tr><td>Model load</td><td class="num">{_e(_ms(_f(lat.get("model_load_ms"))))}</td></tr>'
        f'<tr><td>Retrieve + generate (per query)</td><td class="num">{_e(_ms(_f(ps.get("retrieve_generate_ms"))))}</td></tr>'
        f'<tr><td>TTFT p99</td><td class="num">{_e(_secs(_f(lat.get("ttft_p99_s"))))}</td></tr>'
        f'<tr><td>TPOT p99</td><td class="num">{_e(_secs(_f(lat.get("tpot_p99_s"))))}</td></tr>'
        f'<tr><td>End-to-end p99</td><td class="num">{_e(_secs(_f(lat.get("e2e_p99_s"))))}</td></tr>'
        f'<tr><td>Queue p99</td><td class="num">{_e(_secs(_f(lat.get("queue_time_p99_s"))))}</td></tr>'
        "</tbody></table>"
    )
    gpu_html = (
        "<table><tbody>"
        f'<tr><td>VRAM peak</td><td class="num">{_e(_mb(_f(gpu.get("vram_peak_mb"))))}</td></tr>'
        f'<tr><td>VRAM footprint (incremental)</td><td class="num">{_e(_mb(_f(gpu.get("vram_incremental_mb"))))}</td></tr>'
        f'<tr><td>GPU util peak</td><td class="num">{_e(_util_pct(gpu.get("util_peak_pct")))}</td></tr>'
        f'<tr><td>Device</td><td class="num">{_e(gpu.get("device_name") or "—")}</td></tr>'
        "</tbody></table>"
    )

    timings = _timings_svg(rep)
    concl_block = (
        f'<h2>Conclusion</h2><div class="concl">{_e(concl_text)}</div>' if concl_text else ""
    )
    timings_block = f"<h2>Step timings</h2>{timings}" if timings else ""

    # Caveat lines explaining empty sections (missing GPU telemetry / Ragas
    # metrics) instead of leaving unexplained em-dashes.
    answer_quality_note = _answer_quality_note(rep)
    gpu_note = _gpu_unavailable_note(gpu)
    glossary_block = _glossary_html()

    import json as _json

    cfg_block = (
        "<h2>Configuration</h2><table><tbody>"
        + "".join(
            f'<tr><td>{_e(k)}</td><td class="num">{_e(_json.dumps(v) if isinstance(v, dict) else v)}</td></tr>'
            for k, v in cfg.items()
        )
        + "</tbody></table>"
    )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Benchmark {_e(did)}</title><style>{_CSS}</style></head><body><div class='page'>"
        "<header><span class='logo'><span class='t'>Innovation</span> Ways</span>"
        "<span class='sub'>· Benchmark Report</span><span class='spacer'></span>"
        f"<span class='sub'>{_e(rep.get('created_at') or '')}</span></header>"
        f"<h1>Benchmark Report — {_e(did)}</h1>"
        f"<div class='meta'>dataset: {_e(_dataset_label(rep))} · "
        f"queries: {_e(rep.get('n_queries', '?'))} · judged: {_e(rep.get('n_judged', '—'))} · "
        f"label: {_e(_clean_label(rep.get('report_label') or '—'))}</div>"
        f"<div class='intro'>{_intro(rep)}</div>"
        f"{concl_block}"
        f"<h2>Headline</h2><div class='stats'>{stats}</div>"
        f"{tiers}"
        f"{answer_quality_note}"
        f"{timings_block}"
        f"<h2>Speed &amp; latency</h2>{speed}"
        f"<h2>GPU &amp; memory</h2>{gpu_note}{gpu_html}"
        f"{cfg_block}"
        f"{glossary_block}"
        "<footer>Generated by the IW-RAG Benchmark Studio · fully local, single-GPU, open-weight · scores judged by local models (no hosted APIs).</footer>"
        "</div></body></html>"
    )


__all__ = ["build_report_html"]
