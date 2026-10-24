# ruff: noqa: E501
"""The single source of truth for the Operator Console navigation.

Every console page used to carry its own copy of the nav.  They now hold one
``__CONSOLE_NAV__`` placeholder, filled by :func:`apply_console_nav`.  A fixed left sidebar keeps every destination visible, grouped by intent:
Understand (Learn), Use (Chat, Upload), Measure (Benchmark family) and Operate (Config,
System). Pure CSS, no script.  Link ids
and hrefs are stable, so tests and browser checks can still target them.
"""

from __future__ import annotations

_GROUPS: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
    ("Understand", (("navLearn", "/console/learn", "How IW-RAG works"),)),
    (
        "Use",
        (("navChat", "/console/chat", "Chat"), ("navUpload", "/console/upload", "Upload")),
    ),
    (
        "Measure",
        (
            ("navBenchmark", "/console/benchmark", "Benchmark"),
            ("navResults", "/console/results", "Results"),
            ("navCompare", "/console/compare", "Compare"),
            ("navScoreboard", "/console/scoreboard", "Scoreboard"),
        ),
    ),
    (
        "Operate",
        (("navConfig", "/console/config", "Config"), ("navSystem", "/console/system", "System")),
    ),
)


def _build_nav() -> str:
    groups = "".join(
        f'\n  <div class="nav-group"><div class="nav-group-title">{title}</div>'
        + "".join(
            f'<a class="nav-item" id="{ident}" href="{href}">{label}</a>'
            for ident, href, label in items
        )
        + "</div>"
        for title, items in _GROUPS
    )
    return (
        '<nav class="console-nav hidden" id="consoleNav" aria-label="Console navigation">'
        '\n  <div class="nav-brand"><span class="i">I</span>nnovation <span class="w">W</span>ays'
        "<small>IW-RAG</small></div>" + groups + "\n</nav>"
    )


NAV_HTML = _build_nav()

NAV_CSS = """
  nav.console-nav { position: fixed; z-index: 6; top: 0; left: 0; bottom: 0; width: 232px; display: flex; flex-direction: column; flex-wrap: nowrap; align-items: stretch; gap: 18px; padding: 18px 12px; overflow-y: auto; background: rgba(24,29,41,0.92); border-right: 1px solid var(--line); border-bottom: none; }
  body:has(nav.console-nav:not(.hidden)) { padding-left: 232px; }
  body:has(nav.console-nav:not(.hidden)) header .logo, body:has(nav.console-nav:not(.hidden)) header .logo + .muted { display: none; }
  nav.console-nav .nav-brand { font-family: "Space Grotesk", system-ui, sans-serif; font-weight: 700; font-size: 18px; padding: 2px 10px 6px; }
  nav.console-nav .nav-brand .i, nav.console-nav .nav-brand .w { color: var(--teal); }
  nav.console-nav .nav-brand small { display: block; font-size: 12px; font-weight: 500; color: var(--muted); letter-spacing: .08em; }
  nav.console-nav .nav-group { display: flex; flex-direction: column; gap: 2px; }
  nav.console-nav .nav-group-title { font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); padding: 0 10px 4px; }
  nav.console-nav a.nav-item { display: block; text-decoration: none; padding: 8px 10px; font-size: 14px; }
  @media (max-width: 860px) {
    nav.console-nav { position: static; width: auto; flex-direction: row; flex-wrap: wrap; gap: 8px 18px; border-right: none; border-bottom: 1px solid var(--line); }
    body:has(nav.console-nav:not(.hidden)) { padding-left: 0; }
    body:has(nav.console-nav:not(.hidden)) header .logo, body:has(nav.console-nav:not(.hidden)) header .logo + .muted { display: inline; }
    nav.console-nav .nav-brand { display: none; }
    nav.console-nav .nav-group { flex-direction: row; flex-wrap: wrap; align-items: center; gap: 4px; }
    nav.console-nav .nav-group-title { padding: 0 6px 0 0; }
  }
"""

_PLACEHOLDER = "__CONSOLE_NAV__"


def apply_console_nav(html: str) -> str:
    """Fill a console page's nav placeholder and add the menu's CSS.

    Args:
        html: A complete console document containing exactly one nav placeholder,
            and one ``</style>``.

    Returns:
        The document with the shared navigation in place.
    """
    return html.replace(_PLACEHOLDER, NAV_HTML, 1).replace("</style>", NAV_CSS + "</style>", 1)


__all__ = ["NAV_CSS", "NAV_HTML", "apply_console_nav"]
