# ruff: noqa: E501
"""Figures for the Learn section: the animated answer-path diagram and one drawing per lesson.

Everything is inline SVG styled by the console's CSS variables, so it needs no assets and
follows the theme.  Whenever a figure shows a number or an output, it is real: the
Prompt Guard 2 scores come from the recorded guards trace, the datamarking and the
citation parsing are the production functions, and the benchmark example is computed here
(and checked in the tests).  Figures that are only a schematic say so in their caption.
Motion is CSS only and is switched off under ``prefers-reduced-motion``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from html import escape

from iw_rag.agent.budgets import DEFAULT_MAX_RETRIES
from iw_rag.generation.answer import parse_citation_markers
from iw_rag.guardrails.sanitize import datamark
from iw_rag.learn.guards_lab import load_trace as load_guards_trace
from iw_rag.learn.guards_scenarios import Scenario, get_scenario
from iw_rag.learn.lessons import LESSONS

VIZ_CSS = """
  .viz { margin: 18px 0 8px; max-width: 820px; }
  .viz svg { width: 100%; height: auto; display: block; overflow: visible; }
  .viz figcaption { color: var(--muted); font-size: 12.5px; line-height: 1.5; margin-top: 6px; }
  .viz .scroll { overflow-x: auto; padding-bottom: 4px; }
  .viz .scroll svg { min-width: 640px; }
  .v-text { fill: var(--text); font: 13px Inter, system-ui, sans-serif; }
  .v-sm { fill: var(--muted); font: 11.5px Inter, system-ui, sans-serif; }
  .v-b { font-weight: 700; } .v-teal { fill: var(--teal); } .v-warn-t { fill: var(--warn); } .v-bad-t { fill: var(--danger); } .v-mono { font-family: ui-monospace, monospace; }
  .v-box { fill: var(--ink-2); stroke: var(--line); stroke-width: 1.2; }
  .v-box.ok { stroke: var(--teal-deep); } .v-box.bad { stroke: var(--danger); } .v-box.warn { stroke: var(--warn); } .v-box.dim { stroke-dasharray: 5 4; opacity: .75; }
  .v-fill-ok { fill: rgba(45,212,191,0.16); stroke: var(--teal-deep); } .v-fill-bad { fill: rgba(244,63,94,0.14); stroke: var(--danger); } .v-fill-warn { fill: rgba(245,158,11,0.14); stroke: var(--warn); }
  .v-line { stroke: #5b6382; stroke-width: 1.6; fill: none; } .v-line.bad { stroke: var(--danger); } .v-line.warn { stroke: var(--warn); } .v-line.ok { stroke: var(--teal); }
  .v-flow { stroke-dasharray: 7 6; animation: v-dash 1.1s linear infinite; }
  .v-bar { fill: var(--teal-deep); } .v-bar.bad { fill: var(--danger); } .v-bar.warn { fill: var(--warn); } .v-track { fill: var(--ink-3); }
  a.v-node { cursor: pointer; } a.v-node:hover .v-box, a.v-node:focus .v-box { stroke: var(--teal); }
  a.v-node:hover .v-text, a.v-node:focus .v-text { fill: var(--teal); }
  .v-pulse { animation: v-pulse 2.4s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
  .v-token { fill: var(--teal); offset-path: path("M220 73 H940 V173 H220"); offset-rotate: 0deg; animation: v-travel 10s linear infinite; }
  @keyframes v-dash { to { stroke-dashoffset: -26; } }
  @keyframes v-pulse { 50% { transform: scale(1.25); opacity: .6; } }
  @keyframes v-travel { from { offset-distance: 0%; } to { offset-distance: 100%; } }
  @media (prefers-reduced-motion: reduce) { .v-flow, .v-pulse, .v-token { animation: none; } .v-token { display: none; } }
  .lesson-progress { display: flex; gap: 4px; margin: 2px 0 14px; } .lesson-progress i { flex: 1; height: 4px; border-radius: 2px; background: var(--ink-3); }
  .lesson-progress i.done { background: var(--teal-deep); } .lesson-progress i.cur { background: var(--teal); }
  .lab-badge { display: inline-block; margin-top: 8px; font-size: 11px; color: var(--teal); border: 1px solid var(--teal-deep); border-radius: 999px; padding: 1px 8px; }
"""


def _scenario(scenario_id: str) -> Scenario:
    found = get_scenario(scenario_id)
    if found is None:
        raise LookupError(scenario_id)
    return found


def _t(x: float, y: float, s: str, cls: str = "v-text", anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>'


def _r(x: float, y: float, w: float, h: float, cls: str = "v-box", rx: int = 9) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'


def _l(
    x1: float, y1: float, x2: float, y2: float, cls: str = "v-line", head: str | None = None
) -> str:
    end = f' marker-end="url(#{head})"' if head else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"{end}/>'


def _defs(prefix: str) -> str:
    return (
        f'<defs><marker id="ah-{prefix}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0 0 L10 5 L0 10 z" fill="#8b93ad"/></marker></defs>'
    )


def _figure(
    prefix: str,
    width: int,
    height: int,
    label: str,
    body: str,
    caption: str,
    *,
    scroll: bool = False,
    fig_id: str | None = None,
) -> str:
    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}">'
        f"<title>{escape(label)}</title>{_defs(prefix)}{body}</svg>"
    )
    inner = f'<div class="scroll">{svg}</div>' if scroll else svg
    return f'<figure class="viz" id="{fig_id or "viz-" + prefix}">{inner}<figcaption>{escape(caption)}</figcaption></figure>'


# --------------------------------------------------------------------------- hub


def hub_diagram() -> str:
    """The real answer path as an animated, clickable diagram (hub page)."""
    answer = [lsn.slug for lsn in LESSONS if lsn.phase == "answer"]
    slug_for = dict(
        zip(
            (
                "input",
                "router",
                "search",
                "rerank",
                "sanitize",
                "grade",
                "generate",
                "self",
                "output",
            ),
            answer,
            strict=True,
        )
    )

    def node(key: str, x: int, y: int, label: str, w: int = 140, sub: str = "") -> str:
        text = _t(x + w / 2, y + (20 if sub else 26), label, "v-text v-b", "middle")
        if sub:
            text += _t(x + w / 2, y + 36, sub, "v-sm", "middle")
        body = f"{_r(x, y, w, 46, 'v-box ok')}{text}"
        return f'<a class="v-node" data-step="{slug_for[key]}" href="/console/learn/{slug_for[key]}" aria-label="{escape(label)}">{body}</a>'

    row1 = [
        ("input", 150, "Input guard", "Prompt Guard 2"),
        ("router", 330, "Router", "retrieve or skip"),
        ("search", 510, "Search", "dense + BM25"),
        ("rerank", 690, "Rerank", "top 8"),
        ("sanitize", 870, "Sanitise", "scan · scrub · mark"),
    ]
    row2 = [
        ("grade", 870, "Grade evidence", "relevant?"),
        ("generate", 690, "Generate", "cited draft"),
        ("self", 510, "Self-check", "supported?"),
        ("output", 330, "Output guard", "citations valid?"),
    ]
    parts = []
    # happy-path connectors
    for a, b in zip(row1, row1[1:], strict=False):
        parts.append(_l(a[1] + 140, 73, b[1], 73, "v-line v-flow", "ah-hub"))
    parts.append(_l(940, 96, 940, 150, "v-line v-flow", "ah-hub"))
    for a, b in zip(row2, row2[1:], strict=False):
        parts.append(_l(a[1], 173, b[1] + 140, 173, "v-line v-flow", "ah-hub"))
    parts.append(_l(330, 173, 290, 173, "v-line v-flow ok", "ah-hub"))
    # retry loop self-check -> search
    parts.append('<path d="M560 150 V96" class="v-line warn v-flow" marker-end="url(#ah-hub)"/>')
    parts.append(_t(572, 128, f"retry, at most {DEFAULT_MAX_RETRIES}×", "v-sm v-warn-t"))
    # terminals
    parts.append(
        _r(150, 150, 140, 46, "v-box ok")
        + _t(220, 177, "Cited answer", "v-text v-b v-teal", "middle")
    )
    parts.append(
        _r(10, 50, 110, 46, "v-box bad")
        + _t(65, 70, "Reject", "v-text v-b", "middle")
        + _t(65, 86, "never searched", "v-sm", "middle")
    )
    parts.append(_l(150, 73, 120, 73, "v-line bad", "ah-hub"))
    parts.append(
        f'<a class="v-node" href="/console/learn/{slug_for["output"]}" aria-label="Fail closed">{_r(10, 150, 110, 46, "v-box bad")}{_t(65, 170, "Fail closed", "v-text v-b", "middle")}{_t(65, 186, "honest refusal", "v-sm", "middle")}</a>'
    )
    parts.append('<path d="M400 196 V228 H65 V196" class="v-line bad" marker-end="url(#ah-hub)"/>')
    parts.append('<path d="M580 196 V228 H400" class="v-line bad"/>')
    parts.append(_t(232, 222, "blocked, or retries spent", "v-sm v-bad-t", "middle"))
    parts.append(
        _r(10, 246, 1000, 34, "v-box dim")
        + _t(
            510,
            268,
            "Every outcome (answered, refused or rejected) is written to the append-only audit log",
            "v-sm",
            "middle",
        )
    )
    parts += [node(k, x, 50, label, 140, sub) for k, x, label, sub in row1]
    parts += [node(k, x, 150, label, 140, sub) for k, x, label, sub in row2]
    parts.append('<circle class="v-token" r="6"/>')
    caption = "The path a question takes, as the production graph runs it. Click a step to read how it works. The dot is a question travelling the happy path; red is where the system stops instead of guessing."
    return _figure(
        "hub",
        1030,
        292,
        "The IW-RAG answer path: input guard, router, search, rerank, sanitise, grade evidence, generate, self-check and output guard, with a retry loop and fail-closed exits",
        "".join(parts),
        caption,
        scroll=True,
        fig_id="answerPath",
    )


# ----------------------------------------------------------------- lesson figures


def _parsing() -> str:
    blocks = [
        ("heading", "Annual leave"),
        ("text", "Every full-time employee receives 25 days…"),
        ("table", "| Situation | Days carried over |"),
        ("figure_caption", "Figure 2: leave accrual by month"),
    ]
    body = _r(20, 20, 120, 150, "v-box") + "".join(
        _l(34, 46 + i * 22, 126 - (i % 3) * 18, 46 + i * 22, "v-line") for i in range(6)
    )
    body += _t(80, 188, "leave_policy.pdf", "v-sm", "middle")
    body += _l(150, 95, 230, 95, "v-line v-flow", "ah-parse")
    for i, (kind, text) in enumerate(blocks):
        y = 20 + i * 40
        body += _r(240, y, 440, 34, "v-box ok" if i == 0 else "v-box")
        body += _t(252, y + 21, kind, "v-sm v-mono v-teal") + _t(380, y + 21, text)
    body += _t(240, 196, "Each block keeps the section path it sits in,", "v-sm")
    body += _t(240, 212, "for example: Northwind leave policy > Annual leave.", "v-sm")
    return _figure(
        "parse",
        700,
        224,
        "A PDF page turned into typed blocks: heading, text, table and figure caption",
        body,
        "Parsing turns a page into ordered, typed blocks that remember their section. "
        "The block types are the real ones the ingestion pipeline uses.",
    )


def _embedding() -> str:
    def cluster(cx: int, cy: int, label: str, pts: list[tuple[int, int]], cls: str) -> str:
        dots = "".join(
            f'<circle cx="{cx + dx}" cy="{cy + dy}" r="5" class="{cls}"/>' for dx, dy in pts
        )
        return dots + _t(cx, cy + 46, label, "v-sm", "middle")

    body = '<rect x="10" y="10" width="680" height="190" rx="10" class="v-box"/>'
    body += cluster(
        150, 90, "expenses", [(-22, -14), (8, -26), (24, 6), (-10, 14), (30, -12)], "v-bar"
    )
    body += cluster(360, 120, "leave", [(-24, -10), (6, -20), (20, 10), (-8, 16)], "v-bar warn")
    body += cluster(
        560, 80, "invoices", [(-20, -12), (4, -22), (22, 8), (-6, 14), (26, -10)], "v-bar bad"
    )
    qx, qy = 175, 70
    for dx, dy in ((-22, -14), (8, -26), (-10, 14)):
        body += _l(qx, qy, 150 + dx, 90 + dy, "v-line ok v-flow")
    body += f'<path class="v-pulse" d="M{qx} {qy - 10} l3 7 8 1 -6 5 2 8 -7 -4 -7 4 2 -8 -6 -5 8 -1z" fill="var(--teal)"/>'
    body += _t(qx + 16, qy - 14, "question", "v-sm v-teal")
    return _figure(
        "embed",
        700,
        212,
        "A two-dimensional map: similar chunks sit together and the question lands next to the expenses chunks",
        body,
        "A schematic, not real data: IW-RAG's vectors have 1,024 dimensions, which cannot be drawn. Chunks about the same thing land close together, and a question is embedded the same way, so its nearest neighbours are the candidates.",
    )


def _storage() -> str:
    layers = [
        ("pgvector", "dense vectors (HNSW)", "v-fill-ok"),
        ("BM25 index", "ParadeDB, exact words", "v-fill-ok"),
        ("Access tags", "who may read each chunk", "v-fill-warn"),
        ("Graph", "Apache AGE entities", "v-fill-ok"),
        ("Audit log", "append-only outcomes", "v-fill-bad"),
    ]
    top, first_row = 52, 72
    bottom = first_row + len(layers) * 34
    body = _t(20, 28, "chunk + vector + access tags", "v-sm")
    body += _l(60, 36, 60, 70, "v-line v-flow", "ah-store") + _r(20, 76, 180, 40, "v-box")
    body += _t(110, 101, "ingestion", "v-text v-b", "middle") + _l(
        200, 96, 262, 96, "v-line v-flow", "ah-store"
    )
    for i, (name, sub, cls) in enumerate(layers):
        y = first_row + i * 34
        body += f'<rect x="280" y="{y}" width="400" height="34" class="{cls}"/>'
        body += _t(298, y + 22, name, "v-text v-b") + _t(430, y + 22, sub, "v-sm")
    body += f'<ellipse cx="480" cy="{top}" rx="200" ry="20" class="v-box ok"/>'
    body += f'<path d="M280 {bottom} A200 20 0 0 0 680 {bottom}" class="v-line ok"/>'
    body += _t(480, top + 4, "one transactional Postgres", "v-text v-b v-teal", "middle")
    return _figure(
        "store",
        700,
        bottom + 40,
        "One Postgres database holding vectors, a BM25 index, access tags, a graph and the audit log",
        body,
        "Everything lives in one transactional database, so a retrieval and its access check are "
        "a single decision with a single audit trail.",
    )


def _input_guard() -> str:
    trace = load_guards_trace()
    scores, threshold = trace["pg2_scores"], trace["pg2_threshold"]
    ok_q, bad_q = _scenario("answered"), _scenario("jailbreak")
    rows = [
        (ok_q.question, scores[ok_q.question], "allowed"),
        (bad_q.question, scores[bad_q.question], "blocked"),
    ]
    bar_w = 500
    body = _t(
        20,
        22,
        "Prompt Guard 2 score: the probability the question is a jailbreak or injection",
        "v-sm",
    )
    for i, (q, score, verdict) in enumerate(rows):
        y = 52 + i * 62
        blocked = verdict == "blocked"
        body += _t(20, y, q if len(q) < 64 else q[:61] + "…", "v-text")
        body += f'<rect x="20" y="{y + 10}" width="{bar_w}" height="14" rx="7" class="v-track"/>'
        body += f'<rect x="20" y="{y + 10}" width="{max(3, round(bar_w * score))}" height="14" rx="7" class="v-bar {"bad" if blocked else ""}"/>'
        lx = 20 + bar_w * threshold
        body += _l(lx, y + 4, lx, y + 30, "v-line warn")
        body += _t(536, y + 22, f"{score:.3f}" if score >= 0.001 else "under 0.001", "v-text v-b")
        body += _t(680, y + 22, verdict, "v-sm " + ("v-bad-t" if blocked else "v-teal"), "end")
    body += _t(20 + bar_w * threshold + 6, 176, f"limit {threshold}", "v-sm v-warn-t")
    return _figure(
        "guard",
        700,
        190,
        "Two questions scored by Prompt Guard 2 against the 0.5 limit: one allowed, one blocked",
        body,
        "Real scores recorded from the live classifier. Granite Guardian also screens the question "
        "in the live system; its container was off when this was recorded.",
    )


def _router() -> str:
    routes = [
        ("skip", "no documents needed", "v-box ok", "answer directly"),
        ("retrieve", "the default", "v-box ok", "hybrid search"),
        ("graph", "entity questions", "v-box dim", "GraphRAG"),
        ("long_context", "whole documents", "v-box dim", "32K window"),
        ("visual", "page images", "v-box dim", "ColVision"),
    ]
    body = _r(20, 92, 150, 50, "v-box ok") + _t(95, 114, "Question", "v-text v-b", "middle")
    body += _t(95, 131, "classified", "v-sm", "middle")
    for i, (name, why, cls, what) in enumerate(routes):
        y = 14 + i * 46
        body += _l(170, 117, 300, y + 22, "v-line v-flow", "ah-route") + _r(300, y, 360, 40, cls)
        body += _t(316, y + 17, name, "v-text v-b v-mono") + _t(
            316, y + 32, f"{why} · {what}", "v-sm"
        )
    body += _t(
        20, 256, "Dashed routes collapse to retrieve unless their mode is switched on.", "v-sm"
    )
    body += _t(20, 272, "An unknown label also means retrieve.", "v-sm")
    return _figure(
        "route",
        700,
        284,
        "The router picks one of five routes; the three optional ones are dashed",
        body,
        "The router returns a query type and a route. Optional routes are recorded as intended but "
        "fall back to the normal search when their mode is off.",
    )


def _sanitize() -> str:
    marked = datamark("Travel expenses are reimbursed within 30 days of submission.", marker="◆")
    trace = load_guards_trace()
    scores, threshold = trace["pg2_scores"], trace["pg2_threshold"]
    by_id = {c.chunk_id: c for c in _scenario("poisoned").retrievals[0]}
    chunks = [(cid, by_id[cid].text, scores[by_id[cid].text]) for cid in ("f07", "x01", "f08")]
    body = _t(20, 22, f"1 · Scan each chunk with Prompt Guard 2 (limit {threshold})", "v-text v-b")
    for i, (cid, text, score) in enumerate(chunks):
        y = 34 + i * 30
        dropped = score >= threshold
        body += _r(20, y, 660, 24, "v-box " + ("bad" if dropped else "ok")) + _t(
            32, y + 16, cid, "v-sm v-mono"
        )
        body += _t(76, y + 16, (text[:56] + "…") if len(text) > 56 else text, "v-sm")
        body += _t(500, y + 16, f"{score:.3f}" if score >= 0.001 else "under 0.001", "v-sm v-b")
        body += _t(
            668,
            y + 16,
            "DROPPED" if dropped else "kept",
            "v-sm " + ("v-bad-t" if dropped else "v-teal"),
            "end",
        )
    body += _t(
        20, 148, "2 · Scrub personal data (Presidio, scoped to the user's clearance)", "v-text v-b"
    )
    body += _t(
        20,
        178,
        "3 · Datamark: a marker after every word, so the model can tell data from instructions",
        "v-text v-b",
    )
    body += _r(20, 190, 660, 32, "v-box ok") + _t(32, 211, marked, "v-text v-mono")
    return _figure(
        "sanit",
        700,
        236,
        "Three chunks scanned (one kept, two dropped), then a kept chunk datamarked with a marker after every word",
        body,
        "Real: the scores come from the live classifier and the marked text is the output of the "
        "production datamarking function (the real marker is an invisible character, shown here as "
        "◆). The innocent chunk about meals is dropped too: the guard fails closed.",
    )


def _grade_evidence() -> str:
    cases = [
        ("correct", "every chunk is relevant", "generate", "v-fill-ok", "v-teal"),
        (
            "ambiguous",
            "some chunks are relevant",
            "generate, from the relevant ones",
            "v-fill-warn",
            "v-warn-t",
        ),
        (
            "incorrect",
            "no chunk is relevant",
            f"rewrite the question and search again (while a retry is left, at most {DEFAULT_MAX_RETRIES})",
            "v-fill-bad",
            "v-bad-t",
        ),
    ]
    body = ""
    for i, (name, when, then, cls, tcls) in enumerate(cases):
        y = 14 + i * 58
        body += f'<rect x="20" y="{y}" width="150" height="46" rx="9" class="{cls}"/>' + _t(
            95, y + 29, name, f"v-text v-b {tcls}", "middle"
        )
        body += _t(190, y + 20, when, "v-text") + _t(190, y + 38, "→ " + then, "v-sm")
    return _figure(
        "grade",
        700,
        190,
        "Three evidence verdicts: correct and ambiguous go on to generate, incorrect triggers a retry",
        body,
        "The grader judges each retrieved chunk once, in one batched call. These three verdicts and where they lead are the production rule.",
    )


def _generate() -> str:
    draft = "Reimbursed within 30 days [1]. Late claims need written approval [7]."
    markers = parse_citation_markers(draft)
    n_sources = 2
    body = _r(20, 16, 330, 100, "v-box") + _t(34, 38, "Draft from the generator", "v-sm")
    body += _t(34, 62, "Reimbursed within 30 days [1].", "v-text")
    body += _t(34, 86, "Late claims need written approval [7].", "v-text")
    body += _t(
        34, 106, "markers found: " + ", ".join(f"[{m}]" for m in markers), "v-sm v-mono v-teal"
    )
    for i in range(n_sources):
        y = 16 + i * 52
        body += _r(480, y, 200, 44, "v-box ok") + _t(494, y + 19, f"Source [{i + 1}]", "v-text v-b")
        body += _t(494, y + 35, "a graded chunk", "v-sm")
    body += _l(350, 50, 480, 38, "v-line ok v-flow", "ah-gen") + _t(
        415, 36, "[1] resolves", "v-sm v-teal", "middle"
    )
    body += _l(350, 100, 480, 140, "v-line bad", "ah-gen")
    body += _r(480, 122, 200, 36, "v-box bad dim") + _t(
        494, 145, "[7]: no such source, dropped", "v-sm v-bad-t"
    )
    return _figure(
        "gen",
        700,
        172,
        "A draft with markers [1] and [7] over two sources: [1] resolves to a real chunk and [7] is dropped",
        body,
        "The markers are found with the production citation parser. A marker with no matching "
        "source is dropped instead of becoming a citation, so a made-up source cannot survive.",
    )


def _self_check() -> str:
    body = _r(20, 60, 150, 50, "v-box ok") + _t(95, 90, "Draft", "v-text v-b", "middle")
    body += _l(170, 85, 220, 85, "v-line v-flow", "ah-self")
    body += '<polygon points="220,85 290,48 360,85 290,122" class="v-box warn"/>'
    body += _t(290, 82, "supported?", "v-sm", "middle") + _t(290, 97, "answers?", "v-sm", "middle")
    body += _l(360, 85, 430, 85, "v-line ok v-flow", "ah-self") + _t(
        395, 76, "yes", "v-sm v-teal", "middle"
    )
    body += _r(430, 60, 150, 50, "v-box ok") + _t(505, 90, "Output guard", "v-text v-b", "middle")
    body += '<path d="M290 48 V22 H95 V60" class="v-line warn v-flow" marker-end="url(#ah-self)"/>'
    body += _t(
        190, 16, "no, and a retry is left: search again, write again", "v-sm v-warn-t", "middle"
    )
    for i in range(DEFAULT_MAX_RETRIES):
        body += (
            f'<rect x="{20 + i * 34}" y="142" width="28" height="12" rx="3" class="v-bar warn"/>'
        )
    body += _t(
        20 + DEFAULT_MAX_RETRIES * 34 + 6,
        153,
        f"{DEFAULT_MAX_RETRIES} retries in total, one counter shared by weak evidence and weak drafts",
        "v-sm",
    )
    body += _l(290, 122, 290, 176, "v-line bad", "ah-self") + _r(215, 178, 150, 36, "v-box bad")
    body += _t(290, 201, "Fail closed", "v-text v-b", "middle") + _t(
        380, 201, "when the retries are spent", "v-sm v-bad-t"
    )
    return _figure(
        "self",
        700,
        228,
        "The self-check: a grounded draft goes to the output guard, an unsupported one retries up to twice, then the system fails closed",
        body,
        "The loop's limits are plain predicates, not model judgement: one retry counter, plus "
        "LangGraph's recursion limit as a second brake.",
    )


def _modes() -> str:
    cards = [
        (
            "GraphRAG",
            "knowledge graph (AGE)",
            "nDCG@5 0.395 vs 0.292 on entity questions; 0 on policy questions",
            "v-box ok",
        ),
        (
            "Visual",
            "page images (ColVision)",
            "tied with captions at 1.000; text search finds about 0",
            "v-box warn",
        ),
        (
            "Long context",
            "whole document at 32K",
            "found a buried needle RAG missed; costs ~21K tokens a query",
            "v-box ok",
        ),
    ]
    body = ""
    for i, (name, what, result, cls) in enumerate(cards):
        x = 14 + i * 230
        body += (
            _r(x, 16, 216, 150, cls)
            + _t(x + 14, 42, name, "v-text v-b")
            + _t(x + 14, 60, what, "v-sm")
        )
        words, line, y = result.split(), "", 86
        for w in words:
            if len(line) + len(w) > 24:
                body += _t(x + 14, y, line, "v-text")
                line, y = w, y + 18
            else:
                line = (line + " " + w).strip()
        body += _t(x + 14, y, line, "v-text")
    return _figure(
        "modes",
        700,
        182,
        "Three optional modes with their measured results",
        body,
        "Each mode was measured before being trusted, and the negative and tied results are published. The numbers are from the project's own experiments.",
    )


def _benchmarks() -> str:
    ranks, correct = 5, 3
    recall = 1.0 if correct <= ranks else 0.0
    mrr = 1 / correct
    ndcg = (1 / math.log2(correct + 1)) / 1.0
    body = _t(
        20,
        22,
        "One question. The search returned five passages; the correct one is third.",
        "v-text",
    )
    for i in range(ranks):
        x = 20 + i * 70
        hit = i + 1 == correct
        body += _r(x, 34, 62, 44, "v-box ok" if hit else "v-box") + _t(
            x + 31, 61, f"#{i + 1}", "v-text v-b" + (" v-teal" if hit else ""), "middle"
        )
    metrics = [
        ("recall@5", recall, "was a correct passage in the top 5?"),
        ("MRR", mrr, "1 ÷ position of the first correct one"),
        ("nDCG@5", ndcg, "rewards correct results near the top"),
    ]
    for i, (name, value, what) in enumerate(metrics):
        y = 104 + i * 36
        body += (
            _t(20, y + 13, name, "v-text v-b")
            + f'<rect x="120" y="{y}" width="360" height="16" rx="8" class="v-track"/><rect x="120" y="{y}" width="{max(3, round(360 * value))}" height="16" rx="8" class="v-bar"/>'
        )
        body += _t(494, y + 13, f"{value:.3f}", "v-text v-b") + _t(550, y + 13, what, "v-sm")
    return _figure(
        "bench",
        800,
        220,
        "A worked example: the correct passage is third, giving recall at 5 of 1.000, MRR 0.333 and nDCG at 5 of 0.500",
        body,
        "Worked by hand for one question with one correct passage. Benchmark Studio averages these over a whole golden set and adds confidence intervals.",
        scroll=True,
    )


def _one_gpu() -> str:
    used, total = 23.9, 32.0
    body = (
        _t(20, 22, "One RTX 5090: 32 GB of video memory", "v-text v-b")
        + '<rect x="20" y="34" width="620" height="30" rx="8" class="v-track"/>'
    )
    body += (
        f'<rect x="20" y="34" width="{round(620 * used / total)}" height="30" rx="8" class="v-bar"/>'
        + _t(
            30,
            54,
            f"{used} GB resident: generator + embedder + reranker (measured peak)",
            "v-text v-b",
        )
    )
    body += _t(640, 54, "≈ 8 GB free", "v-sm", "end")
    body += (
        _t(20, 100, "Serving mode", "v-text v-b")
        + _r(20, 108, 300, 42, "v-box ok")
        + _t(34, 134, "generator + guards stay loaded; chat is on", "v-sm")
    )
    body += (
        _t(340, 100, "Benchmark mode", "v-text v-b")
        + _r(340, 108, 300, 42, "v-box warn")
        + _t(354, 134, "one model at a time is swapped in; chat is off", "v-sm")
    )
    body += _t(
        20,
        178,
        "The card does one job at a time. Heavy work (parsing, embedding, judging, benchmarks) is moved off the answer path.",
        "v-sm",
    )
    return _figure(
        "gpu",
        660,
        190,
        "A 32 GB GPU bar with 23.9 GB used by the resident models, and the two mutually exclusive modes",
        body,
        "The 23.9 GB peak is the measured co-residence of the three resident models. The two modes are mutually exclusive on the single card.",
    )


FIGURES: dict[str, Callable[[], str]] = {
    "parsing": _parsing,
    "embedding": _embedding,
    "storage": _storage,
    "input-guard": _input_guard,
    "router": _router,
    "sanitize": _sanitize,
    "grade-evidence": _grade_evidence,
    "generate": _generate,
    "self-check": _self_check,
    "modes": _modes,
    "benchmarks": _benchmarks,
    "one-gpu": _one_gpu,
}


def figure_html(slug: str) -> str:
    """Return the lesson's figure, or an empty string when the lesson has a lab instead."""
    build = FIGURES.get(slug)
    return build() if build else ""


def progress_html(index: int, total: int) -> str:
    """A thin progress strip: finished steps, the current one, and what is left."""
    cells = "".join(
        f'<i class="{"done" if i < index else ("cur" if i == index else "")}"></i>'
        for i in range(total)
    )
    return f'<div class="lesson-progress" role="img" aria-label="Step {index + 1} of {total}">{cells}</div>'


__all__ = ["FIGURES", "VIZ_CSS", "figure_html", "hub_diagram", "progress_html"]
