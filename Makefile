# iw-rag task runner. The hot-path services (vllm, tei) require the GPU overlay.
# Heavy/offline targets (ingest, eval) land in later phases.
# --project-directory . so the repo root is the project dir: root .env is read
# and ./docker/... bind-mount paths resolve correctly.
COMPOSE := docker compose --project-directory . -f docker/compose.yaml -f docker/compose.gpu.yaml

# Recreating the api WITHOUT the guards overlay silently drops the guard
# endpoint env: GUARDRAILS__ENABLED stays baked-true but GUARD_*__ENDPOINT revert to
# localhost defaults → input guard fails closed → every query rejected. So the deploy
# targets must include docker/compose.guards.yaml. It is untracked/local-only, so add
# it conditionally (wildcard) — absent on a fresh checkout, present on the live box.
GUARDS_OVERLAY := $(wildcard docker/compose.guards.yaml)
DEPLOY_COMPOSE := $(COMPOSE) $(if $(GUARDS_OVERLAY),-f docker/compose.guards.yaml)

.PHONY: up down spike lint type test cov check-assertions config ingest eval eval-ci db-migrate migration-pending api-deploy api-dev benchmark-mode serving-mode system-control visual-up visual-down visual-validate

# Dev hot-reload overlay (docker/compose.dev.yaml) on top of the deploy stack:
# bind-mounts host ./src over the baked /app/src and runs uvicorn --reload, so
# api edits are picked up live. Carries the guards overlay (DEPLOY_COMPOSE) so the
# guard endpoints don't revert to localhost. `make api-deploy` recreates from the
# baked image without this overlay, restoring production parity.
DEV_COMPOSE := $(DEPLOY_COMPOSE) -f docker/compose.dev.yaml

# VLM overlay (visual mode) — bring the Qwen2.5-VL caption server up/down in a
# dedicated GPU window (two-clocks rule; not part of `make up`).
VLM_COMPOSE := docker compose --project-directory . -f docker/compose.yaml -f docker/compose.gpu.yaml -f docker/compose.vlm.yaml

up:                ## start postgres + vllm + tei (GPU)
	$(COMPOSE) up -d

# MUST use $(COMPOSE) (--project-directory .) like `up`. A bare
# `docker compose -f docker/compose.yaml down` makes the project dir `docker/`, so
# the `./docker/postgres/init` bind path resolves to `docker/docker/postgres/init`
# and the daemon auto-creates that empty root-owned stray tree (it broke I-00131's
# pre-merge `git stash`). Keeping the project dir at the repo root resolves it to
# the real `docker/postgres/init`.
down:              ## stop services
	$(COMPOSE) down

# The api is also a baked image (COPY src/). Rebuild + recreate just the api
# container after backend changes (document management, ingestion, etc.).
api-deploy:        ## rebuild + recreate ONLY the FastAPI api container
	docker build -t iw-rag-api -f docker/api.Dockerfile .
	$(DEPLOY_COMPOSE) up -d --no-build --no-deps --force-recreate api

# DEV: recreate the api with host ./src bind-mounted + uvicorn --reload, so
# backend edits (incl. the server-rendered console *_page.py HTML) are picked up
# live with no rebuild. Run once; thereafter just save a file. Revert to the
# baked, production-parity container with `make api-deploy`.
api-dev:           ## DEV: run api with hot-reload (no rebuild on edits)
	$(DEV_COMPOSE) up -d --no-build --no-deps --force-recreate api

# Benchmark Studio = "benchmark mode": the single 32 GB GPU is freed of the
# pinned serving generator + guards so llama-swap can cold-swap the model
# catalog through it. This is mutually exclusive with normal chat/query serving
# (see guide/09-operations.md "Serving mode vs benchmark mode"). These two
# targets flip the box between modes; the read-only mode badge in the Operator
# Console shows which one is active. Both assume the iw-rag-api image exists
# (run `make api-deploy` once on a fresh box).
BENCH_COMPOSE := $(COMPOSE) -f docker/compose.benchmark.yaml

benchmark-mode:    ## switch the box to benchmark mode (Studio swap sidecars up, guards off)
	docker build -t iw-rag-llama-swap -f docker/llama-swap.Dockerfile .
	-docker stop iw-rag-vllm-1 iw-rag-vllm-guard injguard
	$(BENCH_COMPOSE) up -d --no-build
	-docker stop iw-rag-vllm-1 iw-rag-vllm-guard injguard
	@echo "+ benchmark mode active - open /console/benchmark (run 'make serving-mode' to restore chat)."

serving-mode:      ## switch the box back to normal serving (chat/query, guards on)
	-$(BENCH_COMPOSE) stop llama-swap swap-controller
	docker start iw-rag-vllm-1
	@echo "  waiting for the vllm generator to report healthy (generator BEFORE guards - avoids a KV-cache OOM race)..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' iw-rag-vllm-1 2>/dev/null)" = "healthy" ]; do sleep 5; done
	-docker start iw-rag-vllm-guard injguard
	$(DEPLOY_COMPOSE) up -d --no-build --no-deps --force-recreate api
	$(DEPLOY_COMPOSE) up -d --no-build --no-deps system-control
	@echo "+ serving mode active - chat/query restored (guards on)."

# Power switch sidecar (System view): the privileged, internal-only
# system-control container that owns the docker socket + NVML.  Recreate it on
# its own after an image rebuild (`make api-deploy` rebuilds iw-rag-api, then
# this picks up the new power_app code).  Always-on in both modes.
system-control:    ## (re)deploy ONLY the VRAM power-switch sidecar
	$(DEPLOY_COMPOSE) up -d --no-build --no-deps --force-recreate system-control

visual-up:         ## (GPU window) start the Qwen2.5-VL caption server (visual mode)
	$(VLM_COMPOSE) up -d vlm-caption

visual-down:       ## stop the Qwen2.5-VL caption server
	$(VLM_COMPOSE) stop vlm-caption

visual-validate:   ## caption a real image via the live VLM (IMG=path/to/image.png)
	uv run python scripts/validate_visual.py $(IMG)

config:            ## validate compose without starting anything
	$(COMPOSE) config >/dev/null && echo "compose OK"

spike:             ## GATING — VRAM co-residence spike (GUARDS_ON=1; EMBED_BACKEND=infinity on Blackwell)
	uv run --with nvidia-ml-py python scripts/spike_vram.py   # HTTP-based: needs only pynvml + httpx, not the full vllm wheel

lint:              ## ruff check + format check
	uv run ruff check . && uv run ruff format --check .

type:              ## mypy --strict on the package
	uv run --with playwright mypy --strict src  # playwright is in the `pdf` group; benchmark_routes.py lazy-imports it for PDF export

test:              ## unit + integration suites
	uv run pytest -q

cov:               ## full suite WITH coverage; fails under the [tool.coverage] floor
	uv run pytest -q --cov=iw_rag --cov-report=term-missing --cov-report=xml

check-assertions:  ## assertion-strength scanner — fails on NEW tautology/no-assert/broad-raises
	uv run python scripts/check_test_assertions.py --baseline tests/assertion_free_baseline.txt

ingest:            ## ingest the demo corpus (CORPUS=path/to/manifest.yaml)
	uv run --group ingest python scripts/ingest_corpus.py $(CORPUS)

eval:              ## (Phase 1) run the eval harness — generates baseline numbers
	uv run python scripts/run_eval.py

eval-ci:           ## (Phase 1) CI regression gate — fails on Recall@10 < 0.8 or faithfulness regression
	uv run --group eval-judges deepeval test run tests/eval/ -q

db-migrate:        ## (wired in 0.1) apply migrations
	uv run alembic upgrade head

# MUST emit down_revision="PENDING" — the actual mechanism (script.py.mako
# template or post-gen rewrite) is wired in task-0.1. NEVER hand-chain
# down_revision; NEVER call `alembic revision --autogenerate` directly.
migration-pending: ## (wired in 0.1) author a PENDING-rebased migration
	uv run alembic revision --autogenerate -m "$(MSG)"
