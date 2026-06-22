#!/usr/bin/env bash
# Entrypoint for the iw-rag E2E api image (docker/api.e2e.Dockerfile).
#
# Default (no args): migrate the isolated DB, optionally seed it, then launch
# the FastAPI app. With args (compose `command:`): exec them directly — used by
# the `e2e-models` stub service, which must NOT migrate or seed.
set -euo pipefail

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

echo "[e2e-entry] alembic upgrade head"
uv run --no-dev alembic upgrade head

if [ "${IW_E2E_SEED:-0}" = "1" ]; then
    echo "[e2e-entry] seeding (IW_ITEM_ID=${IW_ITEM_ID:-<unset>})"
    uv run --no-dev python scripts/e2e_seed.py
fi

echo "[e2e-entry] launching uvicorn on :8000"
exec uv run --no-dev uvicorn iw_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000
