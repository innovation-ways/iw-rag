# ruff: noqa: E501
"""Learn section: how IW-RAG itself works, step by step.

Three server-rendered views share the console shell: the hub (``/console/learn``,
the answer path and every step), one page per step (``/console/learn/{slug}``) and a
glossary.  Content comes from :mod:`iw_rag.learn`; this module only renders it, so
all text is HTML-escaped here.  No external assets (Invariant 8).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from html import escape

from iw_rag.api import (
    learn_chunking_lab_page,
    learn_guards_lab_page,
    learn_hybrid_lab_page,
    learn_rerank_lab_page,
)
from iw_rag.api.console_shell import render_console_page
from iw_rag.api.learn_decoder import DECODER_CSS, decoder_html
from iw_rag.api.learn_lab_base import BASE_CSS
from iw_rag.api.learn_visuals import VIZ_CSS, figure_html, hub_diagram, progress_html
from iw_rag.learn.glossary import GLOSSARY
from iw_rag.learn.lessons import LESSONS, Lesson, Phase

_CSS = """
  .learn-hero h1 { font-size: 28px; margin-bottom: 8px; }
  .learn-hero p { max-width: 70ch; line-height: 1.6; color: var(--muted); margin: 0 0 10px; }
  .phase-title { display: flex; align-items: baseline; gap: 10px; margin: 26px 0 4px; }
  .phase-title h2 { margin: 0; font-size: 18px; }
  .phase-sub { color: var(--muted); font-size: 14px; margin: 0 0 14px; }
  .step-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 14px; }
  a.step-card { display: block; text-decoration: none; color: inherit; background: rgba(34,38,49,0.72); border: 1px solid var(--line); border-radius: var(--radius); padding: 16px; transition: .15s ease; }
  a.step-card:hover { border-color: var(--teal-deep); transform: translateY(-2px); }
  .step-num { display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px; border-radius: 50%; background: var(--teal-deep); color: #06231f; font-weight: 700; font-size: 13px; margin-bottom: 10px; }
  a.step-card h3 { margin: 0 0 6px; font-size: 16px; font-family: "Space Grotesk", system-ui, sans-serif; }
  a.step-card p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.5; }
  .crumbs { font-size: 13px; color: var(--muted); margin-bottom: 14px; }
  .crumbs a { color: var(--teal); text-decoration: none; }
  .lesson h1 { font-size: 26px; }
  .lesson .ask { font-size: 18px; color: var(--teal); margin: 0 0 16px; }
  .lesson p { line-height: 1.65; max-width: 72ch; }
  .analogy { border-left: 3px solid var(--teal-deep); padding: 4px 0 4px 14px; color: var(--muted); font-style: italic; margin: 16px 0; max-width: 72ch; }
  .tradeoff { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 18px 0; }
  .tradeoff > div { border: 1px solid var(--line); border-radius: var(--radius); padding: 14px 16px; background: rgba(24,29,41,0.5); }
  .tradeoff h3 { margin: 0 0 8px; font-size: 14px; text-transform: uppercase; letter-spacing: .06em; }
  .tradeoff .gain h3 { color: var(--ok); } .tradeoff .loss h3 { color: var(--warn); }
  .tradeoff ul { margin: 0; padding-left: 18px; line-height: 1.55; font-size: 14px; }
  .tradeoff li { margin-bottom: 6px; }
  .in-iw { background: rgba(45,212,191,0.07); border: 1px solid var(--teal-deep); border-radius: var(--radius); padding: 14px 16px; max-width: 72ch; line-height: 1.6; font-size: 14px; }
  .in-iw strong { color: var(--teal); }
  .pager { display: flex; justify-content: space-between; margin-top: 24px; gap: 12px; }
  dl.glossary dt { font-weight: 600; margin-top: 14px; color: var(--teal); }
  dl.glossary dd { margin: 4px 0 0; color: var(--muted); line-height: 1.55; max-width: 72ch; }
  .flow-note { color: var(--muted); font-size: 13px; line-height: 1.6; max-width: 80ch; margin: 6px 0 0; }
  .flow-note b { color: var(--teal); font-weight: 600; }
  .lesson h2.sec { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 22px 0 8px; }
  .lesson ol.how { margin: 0; padding-left: 20px; line-height: 1.6; max-width: 76ch; font-size: 15px; }
  .lesson ol.how li { margin-bottom: 8px; }
  .evidence { border: 1px solid var(--line); border-left: 3px solid var(--warn); border-radius: 8px; padding: 12px 16px; line-height: 1.6; font-size: 14px; max-width: 76ch; background: rgba(245,158,11,0.06); }
  .evidence strong { color: var(--warn); }
  .try-links { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0; }
  .code-refs { color: var(--muted); font-size: 13px; margin: 10px 0 0; }
  .code-refs code { background: var(--ink-2); border: 1px solid var(--line); border-radius: 6px; padding: 2px 6px; margin-right: 6px; }
  @media (max-width: 640px) { .tradeoff { grid-template-columns: 1fr; } }
"""

_PHASES: tuple[tuple[Phase, str, str], ...] = (
    (
        "prepare",
        "Before a question: how a document gets in",
        "Offline, once per document. Nothing here runs while you wait for an answer.",
    ),
    (
        "answer",
        "When a question arrives: the verification loop",
        "Every question walks this path. Each step can stop, retry or refuse.",
    ),
    (
        "prove",
        "Beyond the loop: modes, proof and hardware",
        "What else the system can do, how it is measured, and what it runs on.",
    ),
)


@dataclass(frozen=True, slots=True)
class _Lab:
    """An interactive lab embedded in one lesson page."""

    html: Callable[[], str]
    css: str
    js: str


_LABS: dict[str, _Lab] = {
    "chunking": _Lab(
        learn_chunking_lab_page.lab_html,
        learn_chunking_lab_page.LAB_CSS,
        learn_chunking_lab_page.LAB_JS,
    ),
    "fail-closed": _Lab(
        learn_guards_lab_page.lab_html,
        learn_guards_lab_page.LAB_CSS,
        learn_guards_lab_page.LAB_JS,
    ),
    "rerank": _Lab(
        learn_rerank_lab_page.lab_html,
        learn_rerank_lab_page.LAB_CSS,
        learn_rerank_lab_page.LAB_JS,
    ),
    "search": _Lab(
        learn_hybrid_lab_page.lab_html,
        learn_hybrid_lab_page.LAB_CSS,
        learn_hybrid_lab_page.LAB_JS,
    ),
}


def _page(title: str, body: str, *, extra_css: str = "", view_js: str = "") -> str:
    """Wrap ``body`` in the shared console shell (the sign-in gate is inherited)."""
    return render_console_page(
        title=title, extra_css=_CSS + VIZ_CSS + DECODER_CSS + extra_css, body=body, view_js=view_js
    )


def _crumbs(*trail: tuple[str, str | None]) -> str:
    parts = [
        f'<a href="{href}">{escape(label)}</a>' if href else escape(label) for label, href in trail
    ]
    return '<div class="crumbs">' + " / ".join(parts) + "</div>"


def _flow() -> str:
    """The answer path: an animated, clickable diagram of the production graph."""
    return (
        hub_diagram()
        + '<p class="flow-note"><b>Self-check</b> can send the question back to <b>Search</b> '
        "(at most twice). When the budget is spent, or a guard objects, the system "
        "<b>fails closed</b>: an honest refusal instead of a guess. Every outcome is audited.</p>"
    )


def learn_hub_html() -> str:
    """Render the Learn hub: what IW-RAG is, the answer path, and every step by phase."""
    numbered = {lesson.slug: i for i, lesson in enumerate(LESSONS, start=1)}
    sections = []
    for phase, heading, sub in _PHASES:
        cards = "".join(
            f'<a class="step-card" href="/console/learn/{escape(lsn.slug)}">'
            f'<span class="step-num">{numbered[lsn.slug]}</span>'
            f"<h3>{escape(lsn.title)}</h3><p>{escape(lsn.question)}</p>"
            + ('<span class="lab-badge">Interactive lab</span>' if lsn.slug in _LABS else "")
            + "</a>"
            for lsn in LESSONS
            if lsn.phase == phase
        )
        sections.append(
            f'<div class="phase-title"><h2>{escape(heading)}</h2></div>'
            f'<p class="phase-sub">{escape(sub)}</p><div class="step-grid">{cards}</div>'
        )
    body = (
        '<div class="card learn-hero">'
        "<h1>How IW-RAG works</h1>"
        "<p>IW-RAG answers questions from <strong>your own documents</strong>, with citations and "
        "per-document access control, entirely on one local GPU. Its defining feature is not the "
        "answer model but the checking around it: the system grades its own evidence and its own "
        "draft, and <strong>refuses rather than guess</strong>.</p>"
        "<p>These pages follow one question through the real pipeline. For every step you get "
        "what it does, how, <em>why it was built that way</em>, what it costs, and the measured "
        "evidence where the project has some.</p>"
        "<h2 style='margin-top:18px'>The answer path</h2>"
        + _flow()
        + '<p style="margin-top:16px"><a class="btn" href="/console/learn/glossary">Open the glossary</a></p>'
        "</div>" + "".join(sections)
    )
    return _page("Learn", body)


def _bullets(items: tuple[str, ...]) -> str:
    return "<ul>" + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"


def _pager(lesson: Lesson) -> str:
    idx = LESSONS.index(lesson)
    prev = LESSONS[idx - 1] if idx > 0 else None
    nxt = LESSONS[idx + 1] if idx < len(LESSONS) - 1 else None
    left = (
        f'<a class="btn" href="/console/learn/{escape(prev.slug)}">&larr; {escape(prev.title)}</a>'
        if prev
        else "<span></span>"
    )
    right = (
        f'<a class="btn btn-primary" href="/console/learn/{escape(nxt.slug)}">{escape(nxt.title)} &rarr;</a>'
        if nxt
        else '<a class="btn btn-primary" href="/console/learn">Back to the map</a>'
    )
    return f'<div class="pager">{left}{right}</div>'


def _section(title: str, inner: str) -> str:
    return f'<h2 class="sec">{escape(title)}</h2>{inner}'


def learn_lesson_html(lesson: Lesson) -> str:
    """Render one lesson: what it does, how, why, the trade-off, evidence and where to look."""
    number = LESSONS.index(lesson) + 1
    how = "<ol class='how'>" + "".join(f"<li>{escape(h)}</li>" for h in lesson.how) + "</ol>"
    parts = [
        progress_html(number - 1, len(LESSONS)),
        f"<h1>{escape(lesson.title)}</h1>",
        f'<p class="ask">{escape(lesson.question)}</p>',
        f"<p>{escape(lesson.what)}</p>",
        figure_html(lesson.slug),
        decoder_html() if lesson.slug == "benchmarks" else "",
        _LABS[lesson.slug].html() if lesson.slug in _LABS else "",
        _section("How it works here", how),
        _section("Why it is built this way", f"<p>{escape(lesson.why)}</p>"),
        '<div class="tradeoff">'
        f'<div class="gain"><h3>What you gain</h3>{_bullets(lesson.gains)}</div>'
        f'<div class="loss"><h3>What it costs</h3>{_bullets(lesson.losses)}</div>'
        "</div>",
    ]
    if lesson.evidence:
        parts.append(
            f'<div class="evidence"><strong>Measured:</strong> {escape(lesson.evidence)}</div>'
        )
    if lesson.try_it:
        links = "".join(
            f'<a class="btn btn-sm" href="{escape(href)}">{escape(label)}</a>'
            for label, href in lesson.try_it
        )
        parts.append(_section("See it working", f'<div class="try-links">{links}</div>'))
    if lesson.code:
        refs = "".join(f"<code>{escape(c)}</code>" for c in lesson.code)
        parts.append(f'<p class="code-refs">Read the code: {refs}</p>')
    parts.append(_pager(lesson))
    body = (
        _crumbs(("Learn", "/console/learn"), (f"Step {number}: {lesson.title}", None))
        + '<div class="card lesson">'
        + "".join(parts)
        + "</div>"
    )
    lab = _LABS.get(lesson.slug)
    if lab is not None:
        return _page(lesson.title, body, extra_css=BASE_CSS + lab.css, view_js=lab.js)
    return _page(lesson.title, body)


def learn_glossary_html() -> str:
    """Render the glossary of plain-language definitions."""
    terms = "".join(f"<dt>{escape(t)}</dt><dd>{escape(d)}</dd>" for t, d in GLOSSARY)
    body = (
        _crumbs(("Learn", "/console/learn"), ("Glossary", None))
        + '<div class="card"><h1>Glossary</h1>'
        '<p class="sub">The words you will meet, in plain language.</p>'
        f'<dl class="glossary">{terms}</dl></div>'
    )
    return _page("Glossary", body)


__all__ = ["learn_glossary_html", "learn_hub_html", "learn_lesson_html"]
