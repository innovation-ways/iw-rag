# Operations Runbook

On-call procedures for the retrieval service.

- To reindex after an embedder change, run the ingestion pipeline with
  `--reembed` during the maintenance window.
- If search latency exceeds 800 ms p95, check HNSW `ef_search` (see INC-2026-001).
- Resident VRAM must stay under 32 GB; the health probe fails closed otherwise.
