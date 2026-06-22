#!/usr/bin/env bash
# Bring up the isolated iw-rag E2E stack for a browser_verification step.
#
# Called by the daemon via .iw-orch.json.browser_verification.env_up_command.
# CWD is the worktree root. Env set by orch.daemon.browser_env.allocate_browser_env:
#   COMPOSE_PROJECT_NAME, E2E_FRONTEND_PORT, E2E_API_PORT, E2E_DB_PORT,
#   IW_BROWSER_BASE_URL (= http://localhost:${E2E_FRONTEND_PORT}, the Caddy edge),
#   IW_ITEM_ID.
#
# NOTE: no --project-directory — Compose resolves the relative build contexts
# (context: ..) and bind mounts against the compose file's dir (docker/), so
# `..` is the worktree root. Passing --project-directory would break them.
set -euo pipefail

: "${COMPOSE_PROJECT_NAME:?COMPOSE_PROJECT_NAME must be set}"
: "${E2E_FRONTEND_PORT:?E2E_FRONTEND_PORT must be set}"
: "${E2E_API_PORT:?E2E_API_PORT must be set}"
: "${E2E_DB_PORT:?E2E_DB_PORT must be set}"
: "${IW_BROWSER_BASE_URL:?IW_BROWSER_BASE_URL must be set}"

COMPOSE=(docker compose -f docker/compose.e2e.yaml -p "${COMPOSE_PROJECT_NAME}")

echo "[e2e_up] project=${COMPOSE_PROJECT_NAME} edge=${E2E_FRONTEND_PORT} api=${E2E_API_PORT} db=${E2E_DB_PORT} item=${IW_ITEM_ID:-<unset>}"

# Clean any stale containers/volumes from a prior failed run of THIS project.
"${COMPOSE[@]}" down --remove-orphans --volumes >/dev/null 2>&1 || true

# Build from the worktree source + wait for healthchecks (db, models, api, frontend).
"${COMPOSE[@]}" up -d --build --wait

# Belt-and-braces host probe: the api through the Caddy edge (/api/* → api).
echo "[e2e_up] probing ${IW_BROWSER_BASE_URL}/api/health ..."
for _ in $(seq 1 30); do
    if curl --fail --silent --show-error --max-time 2 "${IW_BROWSER_BASE_URL}/api/health" >/dev/null; then
        echo "[e2e_up] stack healthy at ${IW_BROWSER_BASE_URL}"
        exit 0
    fi
    sleep 1
done

echo "[e2e_up] api never became healthy at ${IW_BROWSER_BASE_URL}/api/health"
"${COMPOSE[@]}" ps || true
"${COMPOSE[@]}" logs --tail 120 || true
exit 1
