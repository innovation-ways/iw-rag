"""iw_rag.modes.long_context — the gated 4th router branch (Phase 4, WI-032).

Long-context route: bypasses chunk retrieval and stuffs whole document(s) into
the native 32K window.  Still emits a :class:`RetrievalResult`-shaped object
(the stuffed docs as document-level ``RetrievedChunk``s) so the generator/eval
stay mode-agnostic.  Falls back to RAG on any entry/exit failure.

**YaRN invariant:** this route NEVER enables the 131K YaRN window.  Native 32K
only; ``LongContextConfig.enable_yarn=False`` is validated at startup.

**Two structural exclusions (T4):**
- ACL-scoped queries → forced to RAG (``acl_required``).
- Per-claim-attribution queries → forced to RAG (``attribution_strict``).

``acl_applied=False`` on the emitted :class:`RetrievalResult` is the SINGLE
sanctioned carve-out to the spine's ``acl_applied=True`` invariant
(CONVENTIONS.md §6) — correct because the entry predicate ensures nothing to
pre-filter.

Public API::

    from iw_rag.modes.long_context import (
        LongContextRoute,
        LongContextConfig,
        LongContextOverBudget,
        EntryDecision,
        WholeDocAssembler,
    )

References:
- lld-long-context.md (the design this module implements)
- task-4c-long-context-route (build guide)
- CONVENTIONS.md §6 (acl_applied carve-out)
- R-00109 (routing economics, YaRN tax, entry/exit predicates)
"""

from iw_rag.modes.long_context.assembler import AssembledContext, WholeDocAssembler
from iw_rag.modes.long_context.route import LongContextOverBudget, LongContextRoute
from iw_rag.modes.long_context.types import EntryDecision, LongContextConfig

__all__ = [
    "LongContextRoute",
    "LongContextConfig",
    "LongContextOverBudget",
    "EntryDecision",
    "WholeDocAssembler",
    "AssembledContext",
]
