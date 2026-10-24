"""iw-rag — fully-local, single-GPU, open-weight agentic RAG.

This package holds the hot-path and offline layers (domain, storage, ingestion,
retrieval, generation, agent, guardrails, modes, eval, serving, observability,
api). The only deployed frontend is the server-rendered htmx Operator Console
inside ``api/`` (``console_*.py`` / ``*_page.py``); the top-level ``frontend/``
Chainlit app is retired. See ``docs/build/00-architecture/repo-structure.md`` for
the layering contract.
"""

__version__ = "0.1.0"
