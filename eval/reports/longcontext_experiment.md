# Experiment #8 — long-context (native 32K) vs RAG (LIVE, 2026-06-08)

**Headline: native-32K long-context recovers a needle that top-k RAG misses — even
when RAG retrieves the right chunk.** A measured long-context WIN on the needle
probe (the "why not just stuff the window?" question, answered with evidence).

## Setup

- Generator restarted at **native 32K** (`--max-model-len 32768`, FP8 KV, prefix
  caching). The 32 GB card needed `--enforce-eager` + `--max-num-batched-tokens 2048`
  (chunked prefill) to fit the long-prompt activation spike alongside infinity; guards
  dropped for the KV VRAM, then restored (generator-before-guards to avoid the KV race).
- Probe doc: `data/longdoc_corpus` — a **~21K-token** synthetic "operations handbook"
  with one **NoLiMa-style needle** buried at ~60% depth, worded *obliquely* relative to
  the question (the needle says "custodial recovery passphrase for the cold-storage
  enclosure"; the question asks for "the recovery passphrase for the archive vault").
  141 chunks ingested (augment). Runner: `scripts/run_longcontext_eval.py`.

## Result (deterministic needle-recovery; no judge)

| arm | context fed | needle in context | **needle recovered** | latency |
|-----|--:|:--:|:--:|--:|
| **long_context** (whole 21K-tok doc, native 32K) | ~21,250 tok | — | **YES** ✓ | 461 ms |
| rag (top-8 chunks) | ~1,300 tok | **YES** | **NO** ✗ | 680 ms |

- LC answer: *"The recovery passphrase for the archive vault is **mauve-otter-49-banister**."*
- RAG answer: *"The document does not contain any information regarding the recovery
  passphrase for the archive vault."*

## The interesting part

**RAG retrieved the needle chunk (`needle_in_chunks = True`) but still failed to
answer.** The oblique wording means that, presented in isolation among 7 other
top-k chunks, the model didn't connect "custodial recovery passphrase / cold-storage
enclosure" to the question — whereas the full-document pass, with the surrounding
context, recovered it. This is the precise case the long-context route exists for:
global/oblique facts where pinpoint retrieval surfaces the right text but the model
needs the whole document to use it.

## Needle-depth sweep (NoLiMa/RULER-style, `scripts/run_longcontext_sweep.py`)

Same ~21K-token document, needle moved across the depth axis:

| needle depth | 5% | 25% | 50% | 75% | 95% |
|---|:--:|:--:|:--:|:--:|:--:|
| recovered | ✓ | ✓ | ✓ | ✓ | ✓ |
| latency | 7705 ms | 4885 ms | 4199 ms | 3143 ms | 1914 ms |

**No "lost in the middle" effect at 21K tokens** — recovery is **5/5** across the
document, so the native-32K window is usable end to end (not just at the edges). The
latency trend (cold 7.7 s → warm 1.9 s) is the **prefix-cache amortisation** the route
relies on: successive queries reuse the shared document prefix from the KV cache.
`eval/reports/longcontext_depth_sweep.json`.

## Context-length sweep (`run_longcontext_sweep.py --axis length`)

Needle fixed at 50% depth, document grown toward the 32K ceiling:

| context | ~9K tok (60 paras) | ~18K (120) | ~27K (175) | ~31K (205) |
|---|:--:|:--:|:--:|:--:|
| recovered | ✓ | ✓ | ✓ | ✓ |

**No effective-length cliff within the native 32K window** — recovery is **4/4** up to
~31K tokens (right at the ceiling). Together with the depth sweep (5/5), this
characterises both NoLiMa/RULER axes: Qwen3-14B's 32K window is usable end to end for
needle recovery, so **accuracy is not the limiter — cost is** (prefill + concurrency).
`eval/reports/longcontext_length_sweep.json`.

## Honest caveats

- **Demo-scale; single needle.** Both axes recover the needle, but this is one
  obliquely-worded fact in one synthetic corpus — indicative, not a multi-needle
  RULER benchmark. The route stays gated on cost, not accuracy.
- **Cost side not free.** The LC arm prefills ~21K tokens every query (≈6× concurrency
  ceiling at 32K on this card per vLLM's KV report); the spine's bounded window is far
  cheaper. The route's value is real but **gated** — it earns its place only on
  whole-doc/global queries, exactly as ADR-012 intends (off by default for ACL-scoped
  and pinpoint traffic).
- The canonical `run_experiment_8_long_context` harness scores *retrieval* recall/nDCG
  (a weaker proxy for the generation question this probe measures directly); persisting
  that arm is a follow-up.

Persisted: `eval/reports/longcontext_needle.json`.
