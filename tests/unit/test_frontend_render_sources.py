"""Unit tests for the frontend citation renderers.

Targets CR-00098 S06: ``_render_sources`` and ``_render_provenance`` in
``frontend/app.py`` must emit a clickable ``<a href download>`` anchor
when a citation carries a ``download_url``, and fall back to the existing
plain ``<code>`` text when ``download_url`` is empty (no token minted —
the API's fail-soft path). Tests import ``frontend/app.py`` directly, the
same way the Chainlit runtime does, and do not need a real API or
Chainlit server.

TDD RED: anchors are expected but the current implementation emits no
``<a `` markup. This file pins the contract.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# The S06 frontend renderer tests import ``frontend/app.py`` directly the same
# way the Chainlit runtime does. ``app.py`` hard-imports ``chainlit`` at module
# load, so the entire test module is meaningless in an environment that does
# not have the chainlit extras installed (e.g. the S13 unit-gate venv, which
# intentionally keeps the dev dependency group lean). The S06 step verified
# these tests pass with ``uv run --with chainlit --with httpx --with fastapi``;
# the S07 step intentionally excluded this file from the plain ``pytest
# tests/unit`` run for the same reason (see CR-00098 S07 report §"Pre-existing
# fixture breakage"). Skip the whole module gracefully when chainlit is
# unavailable so the S13 gate does not fail on an environmental gap rather
# than a behaviour regression.
pytest.importorskip("chainlit")

# Add the frontend directory to sys.path so we can import app.py directly
# (it's a standalone Chainlit app module, not an installed package).
_FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"
if str(_FRONTEND) not in sys.path:
    sys.path.insert(0, str(_FRONTEND))


@pytest.fixture(scope="module")
def app_module() -> object:
    """Import frontend/app.py once per module (its import is heavy on chainlit).

    The public API base is read at render-time (inside the renderer), so
    tests can ``monkeypatch.setenv`` the env var before each call without
    needing to reload the module.
    """
    return importlib.import_module("app")


# Public base used for browser-reachable absolute URL construction.
# Mirrors the env var the implementation reads (IW_RAG_PUBLIC_API_BASE).
_TEST_PUBLIC_BASE = "https://iw.example.test"


class TestRenderSourcesAnchor:
    """_render_sources must emit a clickable anchor when download_url is present."""

    def test_emits_anchor_with_href_for_each_citation(
        self, app_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each citation becomes an <a href download> linking the public download URL."""
        monkeypatch.setenv("IW_RAG_PUBLIC_API_BASE", _TEST_PUBLIC_BASE)
        render_sources = app_module._render_sources
        citations = [
            {
                "marker": 1,
                "source_uri": "upload://analyst/abc/policy.md",
                "quote": "leave policy",
                "document_id": "11111111-1111-1111-1111-111111111111",
                "download_url": "/documents/11111111-1111-1111-1111-111111111111/raw?token=t1",
            },
            {
                "marker": 2,
                "source_uri": "upload://analyst/abc/handbook.md",
                "quote": "section 3",
                "document_id": "22222222-2222-2222-2222-222222222222",
                "download_url": "/documents/22222222-2222-2222-2222-222222222222/raw?token=t2",
            },
        ]
        html = render_sources(citations)
        assert html.count("<a ") == 2, html
        assert "href=" in html
        assert "download" in html
        assert (
            "https://iw.example.test/documents/11111111-1111-1111-1111-111111111111/raw?token=t1"
            in html
        )
        assert (
            "https://iw.example.test/documents/22222222-2222-2222-2222-222222222222/raw?token=t2"
            in html
        )

    def test_anchor_href_uses_document_id_path_not_token(
        self, app_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The href is the absolute download URL — base + download_url verbatim."""
        monkeypatch.setenv("IW_RAG_PUBLIC_API_BASE", "https://api.example.test")
        render_sources = app_module._render_sources
        citations = [
            {
                "marker": 1,
                "source_uri": "uri",
                "quote": "",
                "document_id": "abc",
                "download_url": "/documents/abc/raw?token=xyz",
            }
        ]
        html = render_sources(citations)
        assert html.count("<a ") == 1, f"Expected exactly one anchor; got:\n{html}"
        assert 'href="https://api.example.test/documents/abc/raw?token=xyz"' in html

    def test_falls_back_to_code_text_when_download_url_empty(
        self, app_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty download_url → current plain `source_uri` code text, no anchor."""
        monkeypatch.setenv("IW_RAG_PUBLIC_API_BASE", _TEST_PUBLIC_BASE)
        render_sources = app_module._render_sources
        citations = [
            {
                "marker": 1,
                "source_uri": "upload://analyst/abc/policy.md",
                "quote": "leave policy",
                "document_id": "abc",
                "download_url": "",  # fail-soft: no token minted
            }
        ]
        html = render_sources(citations)
        assert "<a " not in html
        # The source_uri must still surface — the existing Markdown code-span
        # formatting (backticks) is what Chainlit turns into a <code> chip
        # downstream.
        assert "upload://analyst/abc/policy.md" in html
        assert "`upload://analyst/abc/policy.md`" in html

    def test_preserves_marker_and_quote_snippet(
        self, app_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The [n] marker and the optional `_quote_` snippet remain visible."""
        monkeypatch.setenv("IW_RAG_PUBLIC_API_BASE", _TEST_PUBLIC_BASE)
        render_sources = app_module._render_sources
        citations = [
            {
                "marker": 1,
                "source_uri": "uri",
                "quote": "leave policy text",
                "document_id": "abc",
                "download_url": "/documents/abc/raw?token=t",
            }
        ]
        html = render_sources(citations)
        assert html.count("[1]") == 1, f"Marker [1] must appear exactly once; got:\n{html}"
        assert "[1]" in html
        assert "leave policy text" in html
        # And the iw-sources-title structure is preserved.
        assert "iw-sources-title" in html

    def test_empty_citations_returns_empty_string(self, app_module: object) -> None:
        """No citations → empty string (no rendering at all)."""
        render_sources = app_module._render_sources
        assert render_sources([]) == ""


class TestRenderProvenanceAnchor:
    """_render_provenance must emit an anchor for each 'Sources used' citation."""

    def _provenance(self, app_module: object, citations: list[dict]) -> str:
        render_provenance = app_module._render_provenance
        return render_provenance(
            {
                "grounded": True,
                "retriever": "spine",
                "citations": citations,
                "diagnostics": {},
            },
            "http://localhost:6006",
        )

    def test_emits_anchor_per_source_used(
        self, app_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each 'Sources used' line links the citation's absolute download URL."""
        monkeypatch.setenv("IW_RAG_PUBLIC_API_BASE", _TEST_PUBLIC_BASE)
        html = self._provenance(
            app_module,
            [
                {
                    "marker": 1,
                    "source_uri": "upload://x/a.md",
                    "quote": "q",
                    "document_id": "doc-1",
                    "download_url": "/documents/doc-1/raw?token=t",
                },
                {
                    "marker": 2,
                    "source_uri": "upload://x/b.md",
                    "quote": "q2",
                    "document_id": "doc-2",
                    "download_url": "/documents/doc-2/raw?token=t",
                },
            ],
        )
        assert html.count("<a ") == 2
        assert "https://iw.example.test/documents/doc-1/raw?token=t" in html
        assert "https://iw.example.test/documents/doc-2/raw?token=t" in html

    def test_falls_back_to_code_text_when_download_url_empty(
        self, app_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty download_url → keep the current `<code>` source_uri text."""
        monkeypatch.setenv("IW_RAG_PUBLIC_API_BASE", _TEST_PUBLIC_BASE)
        html = self._provenance(
            app_module,
            [
                {
                    "marker": 1,
                    "source_uri": "upload://x/a.md",
                    "quote": "",
                    "document_id": "doc-1",
                    "download_url": "",
                }
            ],
        )
        # No anchor for this citation; the source_uri is in <code> instead.
        assert "<a " not in html
        assert "<code>upload://x/a.md</code>" in html

    def test_anchor_href_is_html_escaped(
        self, app_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dynamic values in the anchor are HTML-escaped (defense in depth)."""
        monkeypatch.setenv("IW_RAG_PUBLIC_API_BASE", _TEST_PUBLIC_BASE)
        html = self._provenance(
            app_module,
            [
                {
                    "marker": 1,
                    "source_uri": "upload://x/<script>alert(1)</script>",
                    "quote": "",
                    "document_id": "doc-1",
                    "download_url": "/documents/doc-1/raw?token=t",
                }
            ],
        )
        # The raw payload must never leak into the markup.
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_no_citations_emits_no_anchor(
        self, app_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No citations at all → 'no anchor' anywhere in the disclosure."""
        monkeypatch.setenv("IW_RAG_PUBLIC_API_BASE", _TEST_PUBLIC_BASE)
        html = self._provenance(app_module, [])
        assert "<a " not in html
        # But the disclosure still renders its 'none' message.
        assert "iw-prov" in html


class TestPublicApiBaseResolution:
    """The implementation must read IW_RAG_PUBLIC_API_BASE and fall back gracefully."""

    def test_anchor_uses_public_base_when_set(
        self, app_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The public base from the env var is used to prefix the relative download URL."""
        monkeypatch.setenv("IW_RAG_PUBLIC_API_BASE", "https://public.example.test")
        render_sources = app_module._render_sources
        html = render_sources(
            [
                {
                    "marker": 1,
                    "source_uri": "uri",
                    "quote": "",
                    "document_id": "doc",
                    "download_url": "/documents/doc/raw?token=t",
                }
            ]
        )
        assert html.count("<a ") == 1, f"Expected exactly one anchor; got:\n{html}"
        assert "https://public.example.test/documents/doc/raw?token=t" in html

    def test_anchor_falls_back_to_api_base_when_unset(
        self, app_module: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset env var → fall back to the configured API base (same origin path)."""
        monkeypatch.delenv("IW_RAG_PUBLIC_API_BASE", raising=False)
        monkeypatch.setenv("IW_RAG_API_BASE", "http://api.internal.test:8000")
        render_sources = app_module._render_sources
        html = render_sources(
            [
                {
                    "marker": 1,
                    "source_uri": "uri",
                    "quote": "",
                    "document_id": "doc",
                    "download_url": "/documents/doc/raw?token=t",
                }
            ]
        )
        # The fallback base must NOT be hardcoded localhost — it must come from
        # the configured API base so docker / dev / prod can all be served.
        assert "localhost" not in html
        assert "http://api.internal.test:8000/documents/doc/raw?token=t" in html
