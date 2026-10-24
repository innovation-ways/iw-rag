# ruff: noqa: E501
"""The "New to benchmarks?" primer shared by the Benchmark, Results, Compare and Scoreboard pages.

A collapsed panel at the top of each page, written for someone who has never read a RAG
benchmark: what a run is, the four things it measures, and how to read a number.  The full
explanation of every metric lives in the Learn section; this links to it.
"""

from __future__ import annotations

PRIMER_CSS = """
  details.bench-primer { background: rgba(45,212,191,0.06); border: 1px solid var(--teal-deep); border-radius: var(--radius); padding: 0 18px; margin-bottom: 22px; }
  details.bench-primer > summary { cursor: pointer; padding: 13px 0; font-weight: 600; color: var(--teal); list-style: none; display: flex; align-items: center; gap: 10px; }
  details.bench-primer > summary::-webkit-details-marker { display: none; }
  details.bench-primer > summary::before { content: "\\25B8"; transition: transform .15s; }
  details.bench-primer[open] > summary::before { transform: rotate(90deg); }
  details.bench-primer .bp-body { padding: 0 0 16px; font-size: 14px; line-height: 1.6; color: var(--text); }
  details.bench-primer .bp-body p { margin: 0 0 10px; max-width: 78ch; }
  .bp-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 12px 0; }
  .bp-card { border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; background: var(--ink-2); font-size: 13px; line-height: 1.5; }
  .bp-card b { display: block; color: var(--teal); margin-bottom: 3px; font-size: 13.5px; }
  .bp-card small { color: var(--muted); display: block; margin-top: 4px; }
  details.bench-primer ul { margin: 6px 0 12px; padding-left: 20px; } details.bench-primer li { margin-bottom: 5px; max-width: 78ch; }
  details.bench-primer a { color: var(--teal); }
"""

PRIMER_HTML = """<details class="bench-primer" id="benchPrimer">
      <summary>New to benchmarks? Read this first</summary>
      <div class="bp-body">
        <p>A benchmark <strong>run</strong> puts a fixed list of questions to IW-RAG, questions whose correct sources are already known, and scores what comes back. Run the same questions through two setups and you can tell which is better, instead of guessing.</p>
        <div class="bp-grid">
          <div class="bp-card"><b>Did it find the right passages?</b>Recall, nDCG and MRR check whether the correct passage was in the results, and how high.<small>Higher is better.</small></div>
          <div class="bp-card"><b>Is the answer honest?</b>Faithfulness is the share of claims in the answer that the documents really support.<small>Higher is better.</small></div>
          <div class="bp-card"><b>How fast is it?</b>Seconds to answer, and how many tokens (word pieces) it writes per second. "p99" means the slow cases.<small>Lower time is better.</small></div>
          <div class="bp-card"><b>How much GPU memory?</b>The models must fit in one 32 GB card together.<small>Lower leaves more room.</small></div>
        </div>
        <ul>
          <li><strong>Scores run from 0 to 1</strong>, and higher is better unless the label says lower. Times are in seconds or milliseconds, and lower is better.</li>
          <li><strong>Hover the <em>i</em> beside any metric</strong> for what it means in plain words, with an example of how to read it.</li>
          <li><strong>Only compare runs on the same dataset.</strong> A different question list is a different test.</li>
          <li><strong>A small gap can be luck.</strong> With about 37 questions only fairly large differences are reliable, so look for "significant" before believing a gap.</li>
        </ul>
        <p>The full guide, with every metric explained: <a href="/console/learn/benchmarks">Reading the benchmarks</a>.</p>
      </div>
    </details>"""

_PLACEHOLDER = "__BENCH_PRIMER__"


def apply_benchmark_primer(html: str) -> str:
    """Fill a benchmark page's primer placeholder and add the primer's CSS.

    Args:
        html: A complete console document containing exactly one primer placeholder and
            at least one ``</style>``.

    Returns:
        The document with the primer in place.
    """
    return html.replace(_PLACEHOLDER, PRIMER_HTML, 1).replace(
        "</style>", PRIMER_CSS + "</style>", 1
    )


__all__ = ["PRIMER_CSS", "PRIMER_HTML", "apply_benchmark_primer"]
