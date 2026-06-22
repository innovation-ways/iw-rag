"""Regression tests for the Innovation Ways favicon on every console document.

Before this fix the console served no favicon at all: every page's ``<head>``
omitted a ``<link rel="icon">``, so the browser fell back to requesting
``/favicon.ico`` (a 404) and the tab showed a blank icon.  The fix homes a
single IW mark in :data:`iw_rag.api.console_shell.FAVICON_LINK` and injects it —
via :func:`iw_rag.api.console_shell.add_favicon` for the standalone pages and the
``__FAVICON__`` placeholder for the shared shell — into *every* console document.

These tests pin that contract: each rendered page carries the favicon exactly
once, it is an inlined ``data:`` URI (no external CDN/host fetch — Invariant 8),
the base64 payload decodes to a real SVG, and it sits inside the document head
right after ``</title>``.  The single source of truth (one ``FAVICON_LINK``)
guards against the marks drifting apart across the nine console views.
"""

from __future__ import annotations

import base64
import re

import pytest

from iw_rag.api.benchmark_page import benchmark_page_html
from iw_rag.api.benchmark_runner_page import benchmark_runner_page_html
from iw_rag.api.config_page import config_page_html
from iw_rag.api.console_chat_page import console_chat_page_html
from iw_rag.api.console_page import console_page_html
from iw_rag.api.console_shell import FAVICON_LINK, add_favicon
from iw_rag.api.console_upload_page import console_upload_page_html
from iw_rag.api.documents_page import documents_page_html
from iw_rag.api.results_page import results_page_html
from iw_rag.api.scoreboard_page import scoreboard_page_html

# Every console document, by the operator-visible name of its view.  The shell
# pages (chat/upload) go through render_console_page; the rest assemble their own
# <head> and call add_favicon — both routes must end up with the same mark.
_PAGES = {
    "console": console_page_html,
    "config": config_page_html,
    "benchmark": benchmark_page_html,
    "runner": benchmark_runner_page_html,
    "results": results_page_html,
    "scoreboard": scoreboard_page_html,
    "documents": documents_page_html,
    "chat": console_chat_page_html,
    "upload": console_upload_page_html,
}


@pytest.mark.parametrize("name", sorted(_PAGES))
def test_console_page_has_exactly_one_favicon(name: str) -> None:
    """Each console document carries the IW favicon link exactly once.

    Exactly-once guards against both the original bug (no favicon) and a future
    refactor silently inserting a duplicate (e.g. a page templating its own mark
    *and* calling ``add_favicon``).
    """
    page = _PAGES[name]()
    assert page.count('rel="icon"') == 1, f"{name} must carry exactly one favicon link"
    # The unexpanded placeholder must never reach a rendered document.
    assert "__FAVICON__" not in page, f"{name} leaked the __FAVICON__ placeholder"


@pytest.mark.parametrize("name", sorted(_PAGES))
def test_console_favicon_is_inlined_not_external(name: str) -> None:
    """The favicon is an inlined ``data:`` URI — never an external fetch (Invariant 8).

    The console is local-only; even the tab icon must not trigger a request to a
    CDN or any ``http(s)`` host.
    """
    page = _PAGES[name]()
    link = re.search(r'<link rel="icon"[^>]*>', page)
    assert link is not None, f"{name} favicon link not found"
    href = re.search(r'href="([^"]+)"', link.group(0))
    assert href is not None
    assert href.group(1).startswith("data:image/svg+xml;base64,"), (
        f"{name} favicon must be an inlined SVG data URI (no external host)"
    )


@pytest.mark.parametrize("name", sorted(_PAGES))
def test_console_favicon_sits_in_head_after_title(name: str) -> None:
    """The favicon link lives in the document head, right after ``</title>``.

    ``add_favicon`` / the shell both insert the mark immediately after the
    closing title tag, so the link must precede ``</head>`` and follow
    ``</title>``.
    """
    page = _PAGES[name]()
    icon_at = page.index('rel="icon"')
    assert page.index("</title>") < icon_at < page.index("</head>"), (
        f"{name} favicon must sit in <head>, after </title>"
    )


def test_favicon_link_decodes_to_valid_svg() -> None:
    """The shared ``FAVICON_LINK`` payload is a well-formed SVG (the IW mark)."""
    payload = re.search(r"base64,([^\"]+)", FAVICON_LINK)
    assert payload is not None, "FAVICON_LINK is not a base64 data URI"
    svg = base64.b64decode(payload.group(1)).decode("utf-8")
    assert svg.startswith("<svg"), "decoded favicon is not an SVG element"
    assert svg.rstrip().endswith("</svg>"), "decoded favicon SVG is truncated"
    # The IW connected-nodes mark uses the brand blue + accent orange.
    assert "#2E86AB" in svg and "#F18F01" in svg, "decoded favicon lost the IW brand colours"


def test_add_favicon_injects_after_title_once() -> None:
    """``add_favicon`` inserts the link after the first ``</title>`` and only once."""
    doc = "<head><title>X</title></head>"
    out = add_favicon(doc)
    assert out.count('rel="icon"') == 1
    assert out.index("</title>") < out.index('rel="icon"') < out.index("</head>")
