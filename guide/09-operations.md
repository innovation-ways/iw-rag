# Operations — deploy, configure, run

This chapter is the operator's manual for **iw-rag**: how to bring the stack up, run the gating
VRAM spike, configure it, ingest a corpus, query it (API + Chainlit), manage migrations, run the
quality gates, and read traces. Every command and port below is verified against the actual
`Makefile`, `docker/`, `config/`, and `src/iw_rag/` in this repo. Where something is demo-only or
deferred to a later phase, it says so.

> Canonical config rule, applied throughout: configuration flows through one typed `Settings`
> object, env vars map as `IW_RAG__SECTION__KEY` (double-underscore nested delimiter), and the
> caller's identity (`AccessContext.principal_groups`) is established **server-side from verified
> auth — never** from the request body, the query text, or the LLM.

---

## 1. Prerequisites

| Requirement | Detail |
|---|---|
| **GPU** | One **NVIDIA RTX 5090, 32 GB (Blackwell, sm_120)**. Dev = demo: the single card hosts the generator + embedder/reranker (and, guards-on, Granite Guardian) co-resident. |
| **NVIDIA driver** | `570.172.08` or compatible → CUDA 12.8. This pins `vllm/vllm-openai:v0.11.0` (the last cu128 tag that runs on Blackwell). FlashAttention-3 is broken on Blackwell, so the compose sets `VLLM_ATTENTION_BACKEND=TRITON_ATTN`. |
| **Docker** | Docker Engine + Compose v2 (`docker compose`), plus the **NVIDIA Container Toolkit** (the GPU overlay reserves the device via `deploy.resources.reservations.devices`). |
| **uv** | The Python toolchain runner. All `make` targets shell out to `uv run`. |
| **Python** | 3.12 (managed by uv; you do not need a system interpreter). |
| **Hugging Face** | Open-weight downloads. `HF_TOKEN` is optional (rate limits / gated repos); the demo models are ungated. |

A note on the embedder: the canonical embedding/rerank server is **TEI**, but TEI has **no
Blackwell (sm_120) build** — it crashes on the RTX 5090. iw-rag therefore uses the design's
sanctioned fallback, **`infinity`** (`michaelf34/infinity`, PyTorch cu128), which serves *both*
BGE-M3 and `bge-reranker-v2-m3` from one container. On sm_90-and-earlier hosts you would switch the
engine back to `tei` — that is a `config/models.yaml` + compose change, not a code change.

---

## 2. Bringing the stack up

The task runner is the `Makefile`. The `up` target composes the **base** file with the **GPU**
overlay:

```bash
COMPOSE := docker compose --project-directory . -f docker/compose.yaml -f docker/compose.gpu.yaml
```

First, copy and fill the env file (the password is mandatory — compose refuses to start without it):

```bash
cp .env.example .env
# edit .env: set POSTGRES_PASSWORD at minimum
```

Then bring everything up:

```bash
make up        # docker compose ... up -d  (postgres + vllm + infinity + phoenix + api + frontend + caddy)
make down      # stop services
make config    # validate the merged compose without starting anything → prints "compose OK"
```

### Services and ports

| Service | Image | Host port | Role |
|---|---|---|---|
| **caddy** | `caddy:2` | **`:8088` → :80** | Edge / single entrypoint. `/` → frontend, `/api/*` → api (prefix stripped). |
| **postgres** | `paradedb/paradedb:latest` | `:5432` | One auditable store: pgvector + ParadeDB `pg_search` + metadata + ACL. |
| **vllm** | `vllm/vllm-openai:v0.11.0` | `:8001` | Generator — Qwen3-14B-AWQ, FP8 KV, `--gpu-memory-utilization 0.55`, `--max-model-len 16384`. |
| **infinity** | `michaelf34/infinity:latest` | `:7997` | Embedder (BGE-M3) **and** reranker (bge-reranker-v2-m3), one container. |
| **phoenix** | `arizephoenix/phoenix:latest` | **`:6006`** (UI) | OTel/OpenInference trace backend. OTLP gRPC `:4317` is internal-only (`http://phoenix:4317`), not host-published, to avoid clashing with a local collector. |
| **api** | built from `docker/api.Dockerfile` | *(internal `8000`, not published)* | FastAPI delivery — reached only via Caddy at `/api/*`. |
| **frontend** | built from `frontend/Dockerfile` | *(internal `8501`, not published)* | Chainlit — reached only via Caddy at `/`. |

So in practice you talk to **`http://localhost:8088`** for the app and API, and
**`http://localhost:6006`** for traces.

### Overlays (profiles)

- **GPU (default for `make up`)** — `docker/compose.gpu.yaml`. Reserves the single NVIDIA device for
  `vllm` and `infinity`. Co-residence on the one card is bounded by vLLM's
  `--gpu-memory-utilization` cap and validated by the spike (§3).
- **Dev** — `docker/compose.dev.yaml`. Adds **Ollama** (`:11434`, profile `dev`) as a lightweight
  generator alternative. The hot-path embedder/reranker stay on `infinity` in both profiles. This
  overlay is intentionally minimal in Phase 0 (hot-reload mounts arrive in tasks 0.1/0.5). Bring it
  in explicitly, e.g.:
  ```bash
  docker compose --project-directory . \
    -f docker/compose.yaml -f docker/compose.gpu.yaml -f docker/compose.dev.yaml \
    --profile dev up -d
  ```
- **Benchmark / model-swap** — `docker/compose.benchmark.yaml`. Adds the **`llama-swap`** proxy
  (`:8080`, internal) and the **`swap-controller`** sidecar (`:8090`, internal) that power
  single-GPU generator cold-swaps for Benchmark Studio (see §6a). Bring it in only when running
  model-swap benchmarks; the base stack runs a single static generator without it.
- **Guards** — there is no separate compose file; guards are toggled by `IW_RAG__GUARDRAILS__ENABLED`
  (default `true`) and the guard model endpoints in `config/models.yaml`. The guard models
  (Prompt Guard 2, Granite Guardian) use `${VAR:-default}` endpoints so the API/ingest start without
  a live guard server, but **live guard serving is required when `guardrails.enabled=true`**. For a
  demo that exercises the core grounded-answer path without serving guard models, set
  `IW_RAG__GUARDRAILS__ENABLED=false` in `.env`.

### Serving mode vs benchmark mode

The single 32 GB GPU does two **mutually exclusive** jobs, so the stack runs in one of two
operating modes at a time. The difference is physical, not cosmetic: **serving** keeps one
generator pinned and hot (plus the guard models); **benchmarking** keeps the card mostly empty
so it can cold-**swap** a whole catalog of candidates through it — including a 32B (~21 GB) that
cannot fit alongside a pinned 14B + guards.

| Aspect | **Serving mode** (normal chat/query) | **Benchmark mode** (Benchmark Studio) |
|---|---|---|
| Optimizes for | one **stable, pinned** generator, always hot | **swapping many** generators in/out to measure each |
| api generator endpoint | `vllm:8001` (direct) | `llama-swap:8080` (swap proxy — swap-agnostic URL) |
| Resident generator | 14B pinned (~19 GB) | none until a run cold-swaps one in |
| Guardrails | **on** (input + output guards) | **off** (frees the ~7 GB guard model so a 32B candidate fits) |
| `vllm`, `vllm-guard`, `injguard` | running | stopped (free VRAM for swaps) |
| `llama-swap`, `swap-controller` | stopped | running |
| GPU at rest | ~31 GB (14B + guard + embedder) | ~4 GB (embedder only); spikes per run |
| Audience | end users | `eval-operator` (the Studio run gate) |

**Why they cannot coexist:** serving wants the 14B pinned + guards resident (~30 GB), which leaves
no headroom to cold-swap in a 32 GB-class benchmark candidate. To benchmark the larger models you
must evict the serving generator and guards first. Hence the empty Benchmark Studio picker
("No candidates — the catalog returned no eligible models") is the **expected** state in serving
mode: the swap sidecars are down, so `GET /models` has nothing to enumerate.

**Switching modes** is a one-liner each way (the targets handle VRAM, the swap sidecars, and the
generator-before-guards restart order):

```bash
make benchmark-mode    # free the card, bring up the swap sidecars, api on the proxy + guards off
make serving-mode      # sidecars down, generator (then guards), api back to base config
```

The Operator Console shows a **read-only mode badge** in the Benchmark Runner header (`Benchmark
mode` / `Serving mode`) plus a panel that, in serving mode, explains the empty picker and prints the
exact `make` command to switch. The actual flip stays a host-side `make` target by design — the api
cannot recreate its own container, so there is no privileged in-dashboard switch (`GET /benchmark/mode`
is read-only). The targets assume the `iw-rag-api` image exists (`make api-deploy` once on a fresh box).

> A faithfulness (judged) benchmark additionally needs the offline judge (`vllm-judge`, Lynx-8B
> bf16 ~15 GB) co-resident with the generator; on 32 GB that only fits alongside the 8B candidate
> (judge `--gpu-memory-utilization 0.53` + generator `0.25`). 14B/32B + judge do not co-reside —
> the integrated runner holds the generator resident through scoring, so the host-script two-phase
> judge swap is not available to it.

---

## 3. The gating step: the VRAM co-residence spike

This is **Task 0.0** — the first executable step, and it **gates the entire 32 GB design**. Nothing
downstream is trustworthy until co-residence is proven on the real card.

```bash
make spike                 # spine-only: spine + generator
GUARDS_ON=1 make spike      # guards-on: spine + generator + resident Granite Guardian (~5-6 GB)
```

The target runs `scripts/spike_vram.py` over HTTP (it needs only `pynvml` + `httpx`, not the full
vLLM wheel):

```makefile
spike:
	uv run --with nvidia-ml-py python scripts/spike_vram.py
```

**What it validates.** It drives one realistic agentic step — embed (BGE-M3) + rerank
(bge-reranker-v2-m3) + a short streamed generation — *simultaneously*, polls **true device memory**
via `pynvml`/NVML at ≥2 Hz (NOT framework self-report), keeps the peak, and **fails (exit 1) if the
peak co-resident total exceeds the 32 GB card budget**. It also reports TTFT and per-stage latency,
and **warns** if it fits but leaves under the 10% headroom target.

You must run it **twice** — spine-only and `GUARDS_ON=1` — and **both must fit 32 GB**. On Blackwell
the embed/rerank backend is `infinity` (set `EMBED_BACKEND=infinity`); the `INFINITY_URL` and
`VLLM_URL` knobs in `.env` point it at the live services. Sample output ends with
`RESULT: PASS — co-residence confirmed` or `FAIL — exceeds 32 GB card budget`.

Knobs (`.env`): `SPIKE_BUDGET_GB=32`, `GUARDS_ON`, `VLLM_URL`, `INFINITY_URL`, `EMBED_BACKEND`.

---

## 4. Configuration

### One typed `Settings`, three layers

All configuration is one authoritative `Settings` object (`src/iw_rag/settings.py`), loaded once at
startup via `get_settings()`. Precedence, lowest to highest:

```
config/default.yaml   <   .env   <   process env
```

Nothing else hardcodes endpoints, ports, or model names. A missing required value (e.g. an unset
model endpoint) raises `ConfigError` **immediately at startup**, before any traffic — misconfig
fails fast.

### The `IW_RAG__SECTION__KEY` convention

Env vars map onto nested settings with prefix `IW_RAG__` and `__` as the section delimiter
(`ENV_PREFIX` / `ENV_NESTED_DELIMITER` in `settings.py`). Examples (from `.env.example`):

```bash
IW_RAG__DATABASE__DSN=postgresql+psycopg://iw_rag:change-me-locally@localhost:5432/iw_rag
IW_RAG__MODELS__GENERATOR__ENDPOINT=http://localhost:8001/v1
IW_RAG__MODELS__EMBEDDER__ENDPOINT=http://localhost:7997
IW_RAG__MODELS__RERANKER__ENDPOINT=http://localhost:7997
IW_RAG__API__AUTH_MODE=static_token
IW_RAG__GUARDRAILS__ENABLED=true
```

Secrets (the DB DSN, model endpoints) come from **env only** — they are deliberately *not* in
`config/default.yaml`. Inside the compose network the api gets `postgres:5432`, `vllm:8001`,
`infinity:7997` injected automatically (see `docker/compose.yaml`).

### `config/default.yaml` — non-secret defaults

The baseline knobs for ingestion, retrieval (`candidate_k=40`, `fused_k=80`, `top_k=8`, RRF
`k=60`), generation (`temperature=0.1`, `cite_required=true`), the agent loop (`max_retries=2`,
`recursion_limit=25`), guardrails, eval thresholds, and the optional modes (all `enabled: false`
until their experiment proves the win). Override any of these via the matching `IW_RAG__…` env var.

### `config/models.yaml` — the model registry (model-agnostic)

Modules reference a **logical role** (`generator`, `embedder`, `reranker`, the offline `judge_*`,
the Phase-3 `guard_*`, the Phase-4 batch models), and the registry maps each role to a
`served_name` + `engine` + `endpoint`:

```yaml
generator:
  served_name: "Qwen/Qwen3-14B-AWQ"
  engine: vllm
  endpoint: "${IW_RAG__MODELS__GENERATOR__ENDPOINT}"   # http://vllm:8001/v1
embedder:
  served_name: "BAAI/bge-m3"
  engine: infinity                                      # Blackwell fallback for TEI
  endpoint: "${IW_RAG__MODELS__EMBEDDER__ENDPOINT}"     # http://infinity:7997
```

**Swapping a model is a config change, never a code change** — edit the `served_name`/`engine`/
`endpoint` here (and the env). Endpoint `${ENV_VAR}` placeholders are resolved at load time; an
unset *required* endpoint → `ConfigError`. Guard/judge/batch endpoints use `${VAR:-default}` so the
hot path starts without them (the "two clocks" rule — offline work never blocks the hot path).

### Demo auth (POC only)

Phase 0 uses **`static_token`** auth: a configured token → groups map. **It is demo-only.** The
production-shaped swap is OIDC (`auth_mode` also allows `oidc`/`mtls`/`trusted_header`).

- API side: `IW_RAG__API__STATIC_TOKENS` (JSON `{token: [groups]}`) and
  `IW_RAG__API__OPERATOR_TOKENS` (JSON list). `StaticTokenAuthenticator` resolves a bearer token to
  an `AccessContext` with those `principal_groups`.
- Frontend side (`frontend/auth.py`): `IW_RAG_DEMO_USERS` maps `username → {password, token}`.
  Defaults:

  | Username | Password | Token | Groups (set these in `IW_RAG__API__STATIC_TOKENS`) |
  |---|---|---|---|
  | `analyst` | `analyst` | `demo-finance` | `finance` |
  | `hr` | `hr` | `demo-hr` | `hr` |

  Login carries **only a token** — never groups. The same tokens must appear on the API side so the
  server can resolve their groups. A token's groups are the *only* source of `principal_groups`.

Example API-side wiring (in `.env`) granting each demo token its group plus `public`:

```bash
IW_RAG__API__STATIC_TOKENS={"demo-finance":["finance","public"],"demo-hr":["hr","public"]}
IW_RAG__API__OPERATOR_TOKENS=["demo-finance"]
```

---

## 5. Ingesting a corpus

Ingestion is **offline/batch** (the slow clock): parse → chunk → embed → store. It is driven by a
**manifest** whose `acl_groups` are the ACL authority — entitlements come from source-system
metadata, **never inferred from content**.

```bash
make ingest CORPUS=tests/fixtures/corpus/manifest.yaml
# → uv run --group ingest python scripts/ingest_corpus.py tests/fixtures/corpus/manifest.yaml
```

The demo corpus under `tests/fixtures/corpus/` is public + synthetic (no confidential data) and maps
each doc to a group:

```yaml
documents:
  - uri: "corpus://public/intro.md"        # acl_groups: [public]
  - uri: "corpus://hr/leave_policy.md"      # acl_groups: [hr]
  - uri: "corpus://finance/invoice_q3.md"   # acl_groups: [finance]
```

The `POST /api/ingest` route exists but is an **operator-only ack of the seam** — bulk corpus
ingestion runs offline via `make ingest` (the route returns `202` with a note to that effect).

**Online (per-user) ingestion.** Separately, end users ingest their **own** documents through the
document manager (`POST /api/documents`, §6a): the online light-parser path (txt / md / PDF + OCR) →
HybridChunker (bounded parent expansion) → BGE-M3 embed → persist, with `owner_id` + `visibility`
ACL set from the authenticated session. Same embedding space and same ACL pre-filter as the offline
corpus — just the fast clock, one document at a time.

---

## 6. Querying the API

The core endpoints (`src/iw_rag/api/routes.py`) are `/api/query`, `/api/query/stream`,
`/api/health`, `/api/ingest`, plus `/api/auth/login`. Two further routers add the document-management
and model-swap surfaces (`documents_routes.py`, `models_routes.py`) — see §6a below. Reach them all
through Caddy at `http://localhost:8088/api/*`. Every query
requires a `Bearer` token; the request body carries **no** acl/groups field by design. The demo
tokens map to ACL groups server-side: `demo-finance` → `[finance, public]`, `demo-hr` → `[hr, public]`
(configured in `.env` / `IW_RAG__API__STATIC_TOKENS`). The examples below pass the token via a shell
variable:

```bash
TOKEN=demo-finance        # or: TOKEN=demo-hr
```

### `POST /api/query` — JSON, cited answer

```bash
curl -s http://localhost:8088/api/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Q3 invoice total?"}'
```

The response (`QueryResponse`) carries:

```jsonc
{
  "answer": "The Q3 invoice total is 100 [1].",
  "grounded": true,
  "citations": [
    {"marker": 1, "chunk_id": "…", "source_uri": "corpus://finance/invoice_q3.md", "quote": "the total due is 100"}
  ],
  "used_chunk_ids": ["…"],
  "retriever": "spine",
  "trace_id": "…",
  "diagnostics": { "...": "operator/debug only — timings, arm sizes, trace_ref" }
}
```

`diagnostics` (timings, retrieval arm sizes, and the Phoenix `trace_ref`) is operator-only and gated
by `environment`. It carries the trace reference that resolves in Phoenix (§10).

**Fail-closed refusal** — when the loop cannot ground an answer (or the caller lacks access to any
relevant document), it returns **HTTP 200** with `grounded: false`. This is deliberate: a `404`
would *leak the existence* of restricted documents.

```bash
# An HR-only token asking a finance question → honest refusal, not a leak
TOKEN=demo-hr
curl -s http://localhost:8088/api/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Q3 invoice total?"}'
```

```jsonc
{ "answer": "I couldn't find a grounded answer in the documents you can access.",
  "grounded": false, "citations": [], "used_chunk_ids": [], "trace_id": "…" }
```

ACL is enforced as a **pre-filter** before retrieval — the HR token's `principal_groups` simply do
not include `finance`, so the finance chunks are never candidates. A missing/unknown token → `401`;
calling `/api/ingest` without an operator token → `403`.

`/api/query` also accepts `?loop=off` to bypass the agent graph and call the Phase-0
spine+assembler directly (enables the loop-on-vs-off experiment).

### `POST /api/query/stream` — SSE

```bash
curl -N http://localhost:8088/api/query/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise the leave policy."}'
```

Returns `text/event-stream`. Events follow the canonical SSE taxonomy — a `: ping` heartbeat first,
then `step` · `source` · `verdict` · `final` (grounded answer + citations + `trace_ref`) or
`refusal` (`reason_code: "no_grounded_answer"` + `trace_ref`):

```
: ping

event: final
data: {"answer":"…[1]","citations":[…],"trace_ref":"…","acl_applied":true,"attempts":1}
```

### Health

```bash
curl -s http://localhost:8088/api/health            # liveness
curl -s "http://localhost:8088/api/health?deep=true" # + database + vllm readiness → status "ok"|"degraded"
```

---

## 6a. Document management & model-swap endpoints

Two additional routers landed after the phase 0–4 milestone. Both enforce identity server-side from
the bearer token, never from the body.

**Documents** (`documents_routes.py`) — per-user, ACL-scoped. Uploaded content is parsed
(txt / md / PDF + OCR), chunked (online HybridChunker with bounded parent expansion), embedded into
the **same BGE-M3 space** as the corpus, and persisted — so a user's own documents are retrieved and
**ACL pre-filtered** exactly like the corpus.

| Route | Method | Purpose |
|---|---|---|
| `/api/documents` | GET | List the caller's own + `common` documents (newest first), ACL-scoped. |
| `/api/documents` | POST | Upload (multipart); `visibility=specific` → `acl_tags:[user:<owner>]`, `common` → `[public]`. |
| `/api/documents/{id}` | DELETE | Owner-only; purges chunks + embeddings + visual_pages + graph back-refs. |
| `/api/documents/{id}/raw` | GET | Stream the original bytes (visibility-gated). |
| `/api/attachments` | POST | **Ephemeral** per-chat attachment; server issues a `session:<id>` ACL tag, TTL-purged. |
| `/api/attachments/{session_id}` | DELETE | Purge all attachments for a chat session (on chat end). |
| `/api/documents/ui` | GET | Standalone document-manager page (self-contained login). |

Ephemeral attachments are validated against `EphemeralSessionRepository` (existence + ownership + TTL)
on **every** query — a chat's files are visible only inside that chat, to its owner, until it ends.

**Models / swap** (`models_routes.py`) — operator-only (the `eval-operator` group), the control plane
for the single-GPU model-swap (see ch. on serving). Backed by the `swap-controller` sidecar.

| Route | Method | Purpose |
|---|---|---|
| `/api/models` | GET | Catalog with per-model state (`available` / `loading` / `active` / `unreachable`) + `active_model_id`. |
| `/api/models/activate` | POST | Activate (cold-swap) a generator; proxies to the sidecar, blocks until ready/failed/timeout. |

The swap plumbing runs in a dedicated overlay (`docker/compose.benchmark.yaml`): a ~15 MB Go
**`llama-swap`** proxy (`:8080`, internal) fronts the vLLM generator containers as an *exclusive*
group (activating one stops the others — the single-resident-model guarantee), and a
**`swap-controller`** sidecar (`:8090`, internal) is the swap-state authority the API talks to. Both
ports are internal to the Compose network. Tune via `IW_RAG__SERVING__*` (`llama_swap_base_url`,
`swap_controller_base_url`, `health_timeout_s` — which **must** match `config/llama-swap.yaml`'s
`healthCheckTimeout`).

---

## 7. The Chainlit frontend

Open **`http://localhost:8088/`**. The frontend is a pure API client — it holds no retrieval logic
and streams `/api/query/stream` over SSE.

1. **Login** with a demo user (`analyst`/`analyst` or `hr`/`hr`). Login exchanges credentials for a
   bearer **token** (e.g. `demo-finance`); it never carries groups.
2. It surfaces, per answer: the **grounded answer with inline citation markers**, the **sources**
   (resolved citations → `source_uri`), a **"🔎 How grounded?" provenance panel** (groundedness
   verdict, retrieval route, guard checks, sources), a **fail-closed banner** when `grounded=false`,
   and a **Phoenix trace link** (built from `IW_RAG_PHOENIX_BASE_URL`, default
   `http://localhost:6006`) that jumps to the request's trace.
3. The **paperclip** attaches files to the current chat only — ephemeral, session-scoped, auto-purged
   when the chat ends (wired to `/api/attachments`).
4. **⚙️ Settings → "general knowledge" (assist mode)** sends `allow_ungrounded=true`, permitting a
   clearly-marked ungrounded answer; all guards stay enforced.
5. The **Documents** link opens the standalone **document manager** (`/api/documents/ui`, an SSO
   bridge from the chat session) to upload / list / delete your own documents.

This is demo-only auth. Because each demo user maps to a different group, you can demonstrate the ACL
pre-filter live: the same finance question answers for `analyst` and fails closed for `hr`.

---

## 8. Migrations

Schema lives in Alembic (`alembic/versions/` — currently `0001`–`0008`: initial schema, ACL RLS,
audit log, visual pages, GraphRAG/AGE, **`0006` document ownership** (`documents.owner_id` +
`visibility`), **`0007` ephemeral sessions** (chat-scoped attachment ACL + TTL), and **`0008`
benchmark reports** (Benchmark Studio T1/T2/T3 + latency/resource + CI, append-only)). The rule is
strict:

- **Authoring (devs / agents):** use the Makefile target only —
  ```bash
  make migration-pending MSG="add foo column"
  ```
  This writes `down_revision = "PENDING"`; the orchestration daemon rebases it onto the real head at
  merge. **NEVER** run `alembic revision --autogenerate` directly, and **NEVER** hand-chain
  `down_revision`. When a migration merges, update any `_HEAD_REVISION` / migration-count test
  constants.
- **Applying:**
  ```bash
  make db-migrate      # uv run alembic upgrade head
  ```
  **NEVER apply an uncommitted migration to a shared/long-lived DB** — commit the revision file
  first, or it breaks other worktrees. Integration tests apply migrations to a disposable
  testcontainers Postgres, never a shared/live DB.

---

## 9. Quality and tests

```bash
make lint    # ruff check . && ruff format --check .
make type    # mypy --strict src
make test    # uv run pytest -q  (unit + integration)
```

All three must be green — including **`mypy --strict`** — before anything is "done". Integration
tests stand up Postgres/ParadeDB via **testcontainers** (needs Docker); they never touch a shared or
live DB.

The eval harness has its own targets (Phase 1): `make eval` (baseline numbers) and `make eval-ci`
(CI regression gate — fails on Recall@10 < 0.8 or a faithfulness regression). See chapter 8.

**IW dashboard.** The quality gates and test suites can also be launched per-project from the IW
dashboard, configured by `.iw-orch.json` (`quality_config` / `test_config`). It runs the same
commands (`uv run ruff …`, `uv run mypy --strict src`, `uv run pytest …`) and collects **Allure
reports** (`--alluredir=allure-results`), with bundled "Run All Quality Gates" and "Run All Tests"
categories.

---

## 10. Observability

Phoenix is the trace backend, UI at **`http://localhost:6006`**. The api exports OTLP over the
internal compose network to `http://phoenix:4317` (the gRPC port is intentionally not host-published).

Each request produces **one `iw_rag.query` trace** with nested spans for the retrieval **spine** and
the agent **loop** (route → retrieve → grade → rewrite/re-retrieve → generate → grade → fail-closed).
The `trace_ref` carried in a query response's `diagnostics` (and in the SSE `final`/`refusal` events)
resolves to that trace in Phoenix — and the frontend's per-answer trace link points straight at it.

Sampling and log level are config: `IW_RAG__OBSERVABILITY__TRACE_SAMPLE_RATE` (default `1.0` — trace
everything in the demo) and `IW_RAG__OBSERVABILITY__LOG_LEVEL` (default `INFO`).

> Deferred / optional: an `otel-collector` service (profile `obs`, off by default) sits between the
> api and Phoenix to run PII redaction on spans and act as the swap point for an alternative OTLP
> backend. It is optional for a minimal demo and recommended once the corpus carries sensitive
> (HR / financial) data. Without it, the api exports straight to Phoenix.

---

## Related

- [Overview](./01-overview.md)
- [Architecture](./02-architecture.md)
- [Trust and evaluation](./08-trust-and-eval.md)
