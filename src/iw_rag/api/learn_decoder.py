# ruff: noqa: E501
"""The benchmark metric decoder shown in the "Reading the benchmarks" lesson.

Generated from :mod:`iw_rag.learn.metrics`, the same source the benchmark pages' tooltips
use, so a number is explained in exactly the same words on both.  Every term has an anchor
(``#term-<key>``) so a page can link straight to it.
"""

from __future__ import annotations

from html import escape

from iw_rag.learn.metrics import GROUPS, TERMS

DECODER_CSS = """
  .decoder { margin: 20px 0 6px; }
  .decoder h2.dec-title { font-size: 16px; margin: 0 0 4px; color: var(--teal); }
  .decoder .dec-sub { color: var(--muted); font-size: 14px; margin: 0 0 12px; max-width: 76ch; line-height: 1.55; }
  .dec-jump { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
  .dec-jump a { font-size: 12.5px; padding: 4px 11px; border: 1px solid var(--line); border-radius: 999px; color: var(--text); text-decoration: none; }
  .dec-jump a:hover { border-color: var(--teal); color: var(--teal); }
  .decoder h3.dec-group { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 22px 0 8px; scroll-margin-top: 90px; }
  .dec-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }
  .term { border: 1px solid var(--line); border-radius: 10px; padding: 11px 13px; background: rgba(24,29,41,0.5); scroll-margin-top: 90px; font-size: 13.5px; line-height: 1.5; }
  .term:target { border-color: var(--teal); box-shadow: 0 0 0 1px var(--teal) inset; }
  .term b.nm { display: block; font-size: 14px; margin-bottom: 4px; }
  .term .key { color: var(--muted); font-family: ui-monospace, monospace; font-size: 11px; font-weight: 400; margin-left: 6px; }
  .term .ex, .term .gd, .term .wt { color: var(--muted); margin-top: 5px; }
  .term .ex em, .term .gd em, .term .wt em { color: var(--text); font-style: normal; font-weight: 600; }
"""


def _slug(group: str) -> str:
    return "grp-" + "".join(c if c.isalnum() else "-" for c in group.lower()).strip("-")


def decoder_html() -> str:
    """Return the metric decoder: every benchmark number and setting, grouped and explained."""
    jump = "".join(f'<a href="#{_slug(g)}">{escape(g)}</a>' for g in GROUPS)
    sections = []
    for group in GROUPS:
        cards = "".join(
            f'<div class="term" id="term-{escape(t.key)}"><b class="nm">{escape(t.name)}<span class="key">{escape(t.key)}</span></b>'
            f"{escape(t.plain)}"
            f'<div class="ex"><em>Example.</em> {escape(t.example)}</div>'
            f'<div class="gd"><em>Good or bad?</em> {escape(t.good)}</div>'
            + (f'<div class="wt"><em>Watch out.</em> {escape(t.watch)}</div>' if t.watch else "")
            + "</div>"
            for t in TERMS
            if t.group == group
        )
        sections.append(
            f'<h3 class="dec-group" id="{_slug(group)}">{escape(group)}</h3><div class="dec-grid">{cards}</div>'
        )
    return (
        '<div class="decoder" id="metricDecoder">'
        '<h2 class="dec-title">The metric decoder</h2>'
        '<p class="dec-sub">Every number and setting on the Benchmark, Results, Compare and Scoreboard pages, in plain words. '
        "The same explanations appear when you hover the <b>i</b> beside a metric on those pages.</p>"
        f'<div class="dec-jump">{jump}</div>{"".join(sections)}</div>'
    )


__all__ = ["DECODER_CSS", "decoder_html"]
