# Configuration Reference

Settings are environment-driven (`IW_RAG__SECTION__KEY`).

- `IW_RAG__RETRIEVAL__RRF_K`: reciprocal-rank-fusion constant, default 60.
- `IW_RAG__RETRIEVAL__TOP_K`: candidates before rerank, default 40.
- `IW_RAG__GENERATION__MAX_TOKENS`: answer cap, default 1024.
- `IW_RAG__AGENT__MAX_RETRIES`: loop budget, default 2.
- `IW_RAG__GUARDRAILS__ENABLED`: input/output guards, default true.
