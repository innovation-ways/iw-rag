"""Typed exception hierarchy for iw-rag.

Every error raised by the system subclasses :class:`IwRagError`, so callers can
catch the whole family or a specific failure. These are referenced throughout
the later LLDs (storage, ingestion, retrieval, generation, api); the hierarchy
is defined once here so layers map their failures onto a stable taxonomy.
"""

from __future__ import annotations


class IwRagError(Exception):
    """Base class for all iw-rag domain errors."""


class ConfigError(IwRagError):
    """Invalid, missing, or contradictory configuration."""


class ParseError(IwRagError):
    """A source document could not be parsed during ingestion."""


class EmbedError(IwRagError):
    """An embedding request failed or returned an unusable result."""


class IngestionError(IwRagError):
    """The ingestion pipeline failed for a document or batch."""


class RetrievalError(IwRagError):
    """Retrieval (dense/sparse/fusion/rerank/ACL) failed."""


class RerankError(RetrievalError):
    """The reranker stage failed; a subclass of :class:`RetrievalError`."""


class GenerationError(IwRagError):
    """Answer generation failed."""


class AuthenticationError(IwRagError):
    """The caller's identity could not be established."""


class AuthorizationError(IwRagError):
    """The caller is authenticated but not permitted to perform the action."""


class NotFoundError(IwRagError):
    """A requested resource does not exist or is not visible to the caller.

    Returned as HTTP 404 — deliberately indistinguishable from "exists but not
    visible" so the API never leaks the existence of another user's documents.
    """


class ValidationError(IwRagError):
    """A request payload was structurally valid but semantically rejected.

    Used for upload failures the framework cannot catch (unsupported file type,
    file too large, unparseable content) — returned as HTTP 400.
    """


class JudgeUnavailable(IwRagError):
    """The offline judge endpoint is unreachable or returned an unusable response.

    Evaluation runs MUST fail fast on this error — never silently substitute the
    hot-path generator as judge (generator ≠ judge invariant,
    lld-eval-observability §9).
    """
