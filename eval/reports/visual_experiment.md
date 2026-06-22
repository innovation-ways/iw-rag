# Experiment #7 — visual: ColVision vs parse-then-caption (LIVE, 2026-06-08)

**Both visual arms perfectly recover image-only facts that text retrieval cannot;
they tie at 1.000 on this corpus.** The full ColVision late-interaction arm — which
the codebase left as an injected/faked client (WI-031) — was **built and run live**.

## Corpus

`data/visual_corpus` — 10 PIL-rendered report pages (bar charts + tables). Each of 3
"needle" pages carries a value visible **only in the rendered image** (e.g. a bar
labelled `Northwind 4,820`, a cell `RX-19 → Bin 43`); the other 7 are distractors.
The values appear in **no text document**, so the text spine scores ~0 on these by
construction. Golden: `eval/golden/visual_golden.yaml` (4 queries). Runner:
`scripts/run_visual_eval.py` (three phases).

## Two arms (both built live)

- **parse-then-caption** — Qwen2.5-VL-7B (served via vLLM) transcribes each page to a
  dense caption ("Northwind: 4,820 units …"); captions + queries embedded with BGE-M3
  (infinity); pages ranked by cosine. *The VLM captioned every figure exactly.*
- **ColVision late-interaction** — `vidore/colqwen2.5-v0.2` encodes each page **image**
  and the query into multi-vectors; pages ranked by **MaxSim**. No OCR, no caption.

## Result

| arm | recall@5 | recall@10 | nDCG@5 | nDCG@10 |
|-----|--:|--:|--:|--:|
| colvision  | 1.000 | 1.000 | 1.000 | 1.000 |
| caption    | 1.000 | 1.000 | 1.000 | 1.000 |
| *(text spine — by construction)* | *~0* | *~0* | *~0* | *~0* |

`colvision_wins (nDCG@5) = False` — a **tie**, not a loss: both visual methods rank
the correct page #1 for all 4 queries. The headline is the **visual vs text** gap
(both ≈1.0 vs text ≈0 on image-only facts). Separating ColVision from caption needs a
**denser/blurrier** visual corpus where VLM transcription is lossy (small fonts,
overlapping series, scanned pages) — this clean synthetic set doesn't stress that.

## Building the ColVision arm — the version puzzle (documented so it's reproducible)

The ColVision encoder is an *injected client* in the codebase (faked in tests), so it
had to be built. The blocker was a three-way version conflict:

- `colqwen2.5-v0.2`'s LoRA adapter keys only match **colpali-engine 0.3.10 +
  transformers 4.51.3**. Newer colpali (0.3.17, transformers 5.10 — what the project
  venv has) renames `model.layers → language_model.layers`, so the adapter loads as
  **random weights** (rankings become noise). Older colpali (≤0.3.7) lacks the
  `ColQwen2_5` class entirely.
- That pinned trio then hit the **Blackwell sm_120** wall: a default `torch` has no
  RTX 5090 kernels (`CUDA error: no kernel image is available`). Fixed with
  `torch==2.11.0+cu130`.

Net: the ColVision phase runs in an **isolated venv** (`colpali-engine==0.3.10`,
`transformers==4.51.3`, `torch==2.11.0+cu130`) and writes its rankings to JSON; the
score phase (project venv) reads both arms' rankings and runs `run_experiment_7_visual`.
This keeps the project venv (transformers 5.10, needed by the spine) untouched.

## Denser corpus follow-up (caption-lossy pages)

To try to *separate* the arms, three **dense** pages were added (30-row tables of
`PX-1xx → bin`, `node-xx → port`, `SKU-2xx → stock`) with a query targeting a specific
deep cell. The VLM caption is **value-lossy** here — at the 300-token cap it truncates
*before* the queried row (e.g. d2's caption stops at `node-04`, so `node-21 → 8021` is
absent from the caption) — verified directly.

**Yet both arms still tie at 1.000**, including on the dense queries. The reason is
instructive: **page-level retrieval is topical, not value-level.** The query "what port
does node-21 listen on?" matches the *only* server/port page on topic — so the caption
arm ranks it #1 even without the specific value. ColVision's patch-level advantage
therefore doesn't show at the *retrieval* level; the caption's value-lossiness would
bite at **answer** time (the caption can't supply `8021`; a ColVision→generator read of
the page patch could). This **supports the project's own guidance** that the simpler
parse-then-caption fallback is the right default — the heavy ColVision late-interaction
earns its place only in narrower regimes (topically-confusable pages, or value-level
answer extraction), not for page retrieval on this corpus.

## Honest caveats

- Demo-scale (10 pages, 4 queries). Indicative, not a benchmark.
- The arms tie because the corpus is clean; the ColVision-vs-caption separation (and
  the pooled-vector index-size / quality-cliff sweep) needs a harder visual corpus —
  the documented heavy deferral.
- Persisted: `eval/reports/` + `eval_runs` (`visual_colvision` / `visual_caption`).
