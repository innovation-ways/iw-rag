"""Apply per-item E2E fixtures for a single work item (in-container).

Invoked by the daemon's ``browser_env._apply_per_item_fixtures`` after the
E2E stack is healthy:

    docker compose -p <project> exec e2e-dashboard \
        uv run python scripts/e2e_apply_item_fixtures.py <item_id>

It discovers and runs every ``seed(db)`` in
``ai-dev/active/<item_id>/e2e_fixtures/*.py`` (lexical order) against the
per-stack E2E Postgres. This is the same per-item fixture loading the api
container's entrypoint already performs at startup via ``e2e_seed.py``; the
daemon re-runs it here so a fix-cycle re-provision lands a deterministic DB
state without rebuilding the api image. Fixtures are idempotent, so the
double-apply (startup + here) is safe.

Exit codes:
    0 — success (including no-op when the fixtures directory is absent/empty)
    1 — a fixture raised, or IW_RAG__DATABASE__DSN is unset
    2 — wrong number of arguments

Environment:
    IW_RAG__DATABASE__DSN — DSN of the per-stack E2E Postgres (NOT the orch DB).
    CWD must be the repo root (WORKDIR /app) so ``ai-dev/active/...`` resolves.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(item_id: str) -> None:
    """Apply ``item_id``'s e2e_fixtures against the E2E DB (no-op when absent)."""
    # e2e_seed.py is the sibling script in scripts/; when this file is run as
    # ``python scripts/e2e_apply_item_fixtures.py`` that dir is sys.path[0], so
    # it imports as a top-level module. Its iw_rag imports are function-local,
    # so importing it here has no DB/heavy side effects.
    import e2e_seed  # noqa: PLC0415

    dsn = os.environ.get("IW_RAG__DATABASE__DSN", "").strip()
    if not dsn:
        sys.stderr.write("e2e_apply_item_fixtures: IW_RAG__DATABASE__DSN is not set\n")
        sys.stderr.flush()
        raise SystemExit(1)

    e2e_seed._load_item_fixtures(dsn, item_id)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write(
            f"usage: {Path(__file__).name} <item_id>\n"
            "  item_id: work item identifier, e.g. F-00096\n"
        )
        sys.stderr.flush()
        sys.exit(2)

    try:
        main(sys.argv[1])
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — surface any fixture failure as exit 1
        sys.stderr.write(f"e2e_apply_item_fixtures: FAILED for {sys.argv[1]}: {exc}\n")
        sys.stderr.flush()
        sys.exit(1)
