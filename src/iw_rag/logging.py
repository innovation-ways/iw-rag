"""Structured logging setup (scaffold stub).

The real configuration wires structlog with trace correlation (the Phoenix
``trace_ref``) in a later task. This stub exposes the canonical logger-name
prefix so log/span names stay aligned (``iw_rag.<layer>.<operation>``).
"""

from __future__ import annotations

LOGGER_NAMESPACE = "iw_rag"
