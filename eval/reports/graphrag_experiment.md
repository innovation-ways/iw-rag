# Experiment #6 — GraphRAG route vs hybrid spine (LIVE, 2026-06-08)

**Headline: measured NEGATIVE result. The graph route does not win → stays OFF by
config.** This is a valid, publishable outcome per the program's "measure, don't
assert" rule (ADR-009 / task-4a: a mode that doesn't pass its experiment stays off).

## What it took to run it live (the infra was never validated before)

GraphRAG needs Apache AGE **co-located** in the store (the code shares one DSN for
chunks + graph; the hybrid baseline needs paradedb's pg_search). The production
paradedb is **Postgres 18.4**; Apache AGE only has a **release-candidate** PG18
build (`PG18/v1.7.0-rc0`). So:

1. Stood up an **isolated** paradedb copy (`iw-rag-graph-pg`, port 5434) — the live
   demo store untouched.
2. Built **AGE PG18-RC** into it from source (`make … PG_CONFIG=…/18/bin/pg_config`).
   Smoke-tested: `create_graph`, Cypher `CREATE`/`MATCH`, traversal all work on PG18.
3. Restored the 142-chunk corpus (pgvector + pg_search + AGE in one DB).
4. Indexed with `scripts/index_graph.py` (extraction reused the running Qwen3-14B).

The store/indexer had **three never-live-tested bugs** (only faked in unit tests),
all fixed here (ruff + mypy clean):

| # | Bug | Fix |
|---|-----|-----|
| 1 | `cypher(CAST(:gname AS text), …)` → AGE: *"a name constant is expected"* | inline the validated graph name as a literal (`store.py`) |
| 2 | `create_graph` on an existing graph aborts the txn; `contextlib.suppress` hides the Python error but not the poisoned Postgres txn → `InFailedSqlTransaction` cascade | check `ag_catalog.ag_graph` before creating (`store.py`) |
| 3 | relation **types** with spaces (`request through`) are invalid Cypher identifiers → every edge silently dropped | normalize to `UPPER_SNAKE` (`_cypher_rel_type`, `store.py`) |

Plus `/no_think` on the Qwen3 extraction prompt (faster, cleaner JSON). Result of a
clean index: **147 vertices, 289 edges** (29/142 chunks yielded entities; the
structured invoice tables mostly yielded none — a corpus/extraction note).

## The numbers

`IW_RAG__DATABASE__DSN=…@localhost:5434/iw_rag IW_RAG__MODES__GRAPHRAG__ENABLED=true uv run python scripts/run_graphrag_eval.py`

| subset | arm | recall@5 | recall@10 | nDCG@5 |
|--------|-----|--:|--:|--:|
| multi_hop  | **spine** | 0.375 | **0.625** | **0.347** |
| multi_hop  | graph | 0.000 | 0.000 | 0.000 |
| procedural | **spine** | 1.000 | **1.000** | — |
| procedural | graph | 0.000 | 0.000 | — |

**multihop_win = False · procedural_no_loss = False · GATE = NOT PASSED.**

## Why the graph arm returns empty (precise root causes)

1. **Hot-path seed extraction is keyword-based, by design.** `GraphRetriever`
   tokenizes the query (`"How long are audit logs retained"` → `long, audit, logs,
   retained`) and does an **exact** `MATCH (e:Entity {name: <seed>})`. The graph's
   entities are proper-case multi-word names (`Ada Lovelace`, `Principal`), so the
   seeds match nothing → empty subgraph. The two-clocks rule deliberately forbids
   the NER/extraction model on the hot path, so model-based entity linking (the
   thing that would fix this) is **out of scope for the hot path** — this is the
   documented WI-029 deferral, not a regression.
2. **A variable-length-path incompatibility on AGE PG18-RC**: `MATCH p=(e:Entity
   {name:…}) -[*0..2]->() RETURN e` raises *"No function matches the given name and
   argument types"* (a VLE quirk on the RC build), compounding (1). Even if fixed,
   (1) keeps the arm near-empty.

## Update — WI-031 hot-path retriever (the graph route now *works* and wins on entities)

The first run scored the graph arm **0.000 everywhere**. Two causes were addressed:

- **Crude seed extraction** → **fuzzy seed→entity resolution** (`AgeGraphStore._resolve_seeds`):
  query keywords are resolved to actual entity names via a **substring lookup against the
  known entity names** in `graph_chunk_backrefs` — pure SQL, **no model on the hot path**
  (two-clocks preserved). `acme`→`Acme Search`, `umbrella`→`Umbrella Inc`, etc. (each
  returned 0 chunks before; now 4 / 2 / 3).
- **Two further read-path bugs the first run had hidden** behind the empty results:
  (a) `_ensure_age` cached the check *and* skipped the connection-scoped `LOAD 'age'` /
  `SET search_path` on later sessions → `cypher(unknown,unknown) does not exist`;
  (b) AGE serialises a vertex as `{…}::vertex`, so `json.loads` failed with "Extra data".
  Both fixed; 74 graph unit tests still pass.

On an **entity-centric** golden (`eval/golden/graphrag_entity_golden.yaml` — the entity
is *named* in the query), the graph route now **wins**:

| subset | graph recall@5 | graph nDCG@5 | spine nDCG@5 |
|---|--:|--:|--:|
| entity-named (multi_hop slot) | **0.500** | **0.395** | 0.292 |
| procedural / policy | 0.000 | 0.000 | 1.000 |

`multihop_win = True` (graph 0.395 > spine 0.292) — **but** the pure-graph arm scores 0
on procedural/policy queries (entities don't help there), so `procedural_no_loss = False`
and the full gate still doesn't pass. That is exactly the documented reason for LightRAG
**"mix" mode** (blend graph + vector so the graph never regresses the procedural floor) —
the next increment.

## Conclusion

GraphRAG went from *never run against real AGE* → **running, and winning on its target
query shape**. Five latent store bugs were fixed (literal graph name; create-graph txn
abort; relation-type identifiers; per-session AGE load; agtype `::vertex` parsing) plus
the WI-031 fuzzy seed resolution. The remaining gap is the **multi-hop-with-unnamed-bridge**
case (the query names only the answer — inferring the bridge needs the extraction model,
which two-clocks keeps offline) and the **procedural regression** (→ LightRAG mix mode).
The route stays off by config until mix mode closes the procedural gap.

> Isolated build artifacts: container `iw-rag-graph-pg` (port 5434) holds the
> AGE-enabled copy + the indexed graph. Tear down with `docker rm -f iw-rag-graph-pg`
> when no longer needed (the live store is unaffected).
