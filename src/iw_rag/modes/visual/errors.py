"""Visual mode error types (WI-031 / Task 4b — late-interaction path).

These exceptions encode the degradation semantics required by the spec:
encoder OOM / GPU busy / timeout → VisualEncodeUnavailable / VisualRouteTimeout
→ the router collapses to the spine (degrade, never block).

References:
- task-4b-visual-rag-route §11 (error handling = degrade, never block)
- lld-multimodal §5
"""

from __future__ import annotations


class VisualModeError(Exception):
    """Base class for visual mode errors."""


class VisualEncodeUnavailable(VisualModeError):
    """Raised when the ColVision encoder is unavailable (OOM, GPU busy, load failure).

    The router/node catches this and falls back to the spine retriever.
    Never propagated as a 500 to the caller.
    """


class VisualRouteTimeout(VisualModeError):
    """Raised when the visual retriever exceeds its latency budget.

    The router/node catches this and falls back to the spine retriever.
    Never propagated as a 500 to the caller.
    """


class VisualStoreUnavailable(VisualModeError):
    """Raised at startup when backend=qdrant but Qdrant is unreachable.

    Unlike encode errors (which degrade gracefully), a missing Qdrant backend
    fails fast at startup so the misconfiguration is surfaced immediately.
    """
