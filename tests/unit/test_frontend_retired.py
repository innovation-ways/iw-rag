"""Regression guard — the retired Chainlit frontend must stay retired.

The Chainlit frontend was retired in commit ``ecf60bf`` ("retire
Chainlit; single frontend"): the production ``docker/compose.yaml``
dropped the ``frontend`` service and Caddy now routes ``/`` →
``/console`` (the server-rendered htmx Operator Console inside the
FastAPI app is the only deployed frontend).  This file is the
structural guard that fails the moment a ``frontend/`` dir reappears,
a ``frontend`` service is re-added to ``compose.e2e.yaml``, or
``rag.sh`` re-references the ``iw-rag-frontend`` image — preventing a
partial revert or a future copy-paste from silently resurrecting the
dead surface (I-00132).

Pure filesystem/YAML introspection — no DB, no FastAPI, no network —
so ``tests/unit/`` is the correct home.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frontend_directory_is_removed() -> None:
    """The retired Chainlit app dir must not exist (fails pre-fix, passes post-fix)."""
    assert not (_REPO_ROOT / "frontend").exists(), (
        "frontend/ is the retired Chainlit app — it must be deleted (I-00132)"
    )


def test_e2e_compose_defines_no_frontend_service() -> None:
    """compose.e2e.yaml must not build the retired Chainlit frontend service.

    Asserts on the *parsed* services map (via ``yaml.safe_load``) rather
    than a raw substring of the file, so legitimate references in
    header comments or in the prod Caddyfile path do not cause a false
    fail / false pass.
    """
    compose = yaml.safe_load((_REPO_ROOT / "docker/compose.e2e.yaml").read_text())
    services: dict[str, Any] = compose.get("services", {})
    assert "frontend" not in services, (
        "docker/compose.e2e.yaml still defines a 'frontend' service (I-00132)"
    )
    # The edge must not declare a dependency on the removed service.
    caddy_deps: Any = services.get("caddy", {}).get("depends_on", {})
    deps = list(caddy_deps.keys()) if isinstance(caddy_deps, dict) else list(caddy_deps or [])
    assert "frontend" not in deps, "caddy still depends_on the removed 'frontend' (I-00132)"


def test_launcher_has_no_frontend_service_reference() -> None:
    """rag.sh must not list a 'frontend' service that compose.yaml no longer defines.

    Asserts the specific retired image tag (``iw-rag-frontend``) rather
    than a bare ``frontend`` substring, so that future legitimate uses
    of the word (e.g. a comment about the retired code) do not cause a
    false fail.
    """
    rag = (_REPO_ROOT / "rag.sh").read_text()
    assert "iw-rag-frontend" not in rag, (
        "rag.sh still references the iw-rag-frontend image/container (I-00132)"
    )
