#!/usr/bin/env bash
# Tear down the isolated iw-rag E2E stack. Idempotent, best-effort — never
# fail-fast (the env is "down" if the project namespace is already gone).
#
# `--rmi local` removes the per-project images e2e_up.sh built so they don't
# accumulate (~GBs) across work items; base images (paradedb, caddy) are kept.
set -uo pipefail

: "${COMPOSE_PROJECT_NAME:?COMPOSE_PROJECT_NAME must be set}"

echo "[e2e_down] project=${COMPOSE_PROJECT_NAME}"

docker compose -f docker/compose.e2e.yaml -p "${COMPOSE_PROJECT_NAME}" \
    down --remove-orphans --volumes --rmi local --timeout 20 || true

# Sweep untagged remnants orphaned by repeated --build cycles (each build
# loses the previous layer's tag but keeps the compose-project label).
dangling=$(docker images -a \
    --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" -q \
    2>/dev/null || true)
if [[ -n "${dangling}" ]]; then
    echo "[e2e_down] pruning $(echo "${dangling}" | wc -l) leftover image(s)"
    echo "${dangling}" | xargs -r docker rmi -f >/dev/null 2>&1 || true
fi

exit 0
