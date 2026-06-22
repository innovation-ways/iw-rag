"""Token-budgeting for prompts assembled from retrieved chunks (I-CTXOVERFLOW).

Every LLM call that stuffs retrieved chunk content into its prompt MUST fit
inside the generator's ``max_model_len`` window *minus* the tokens reserved for
the model's own output.  Before this module the only guard was a per-candidate
**character** clamp (``RetrievalSettings.parent_expansion_max_chars``) which is
blind to two things:

1. **Aggregate size** — N candidates each within the per-candidate char budget
   still sum past the window (the clamp bounds one chunk, never the total).
2. **Datamark token inflation** — with guardrails on, ``guardrails.sanitize``
   intersperses a ``U+2063``+hex spotlighting marker after *every word*.  That
   inflates the *token* count of a chunk ~12x (measured: raw text tokenizes at
   ~6.8 chars/token; the same text datamarked tokenizes at ~1.5 chars/token).
   A char budget of "4000 chars ≈ 1000 tokens" (4 chars/token) is therefore off
   by ~3x for the content the grader/generator actually sees, so eight clamped
   chunks blew past the 16,384 window and vLLM returned ``400 Bad Request`` —
   surfaced to the user as a misleading "over its context limit" error.

This module replaces the char clamp with a real **token budget**: a fast,
marker-aware token *estimator* (deliberately conservative — it over-estimates so
we stay under the window) and a greedy *fitter* that keeps the highest-ranked
whole sources, drops the lowest-ranked tail when the budget is exhausted, and
only truncates a single source when that source alone exceeds the budget.

The estimator is intentionally local (no tokenizer dependency, no per-request
call to vLLM ``/tokenize``) so budgeting adds no latency and no new package to
the thin API image.  It is calibrated against the served Qwen3 tokenizer; the
constants are conservative on the safe side.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: The datamark separator (``U+2063`` INVISIBLE SEPARATOR) prefixing each
#: session-randomized spotlighting marker emitted by
#: ``guardrails.sanitize.generate_session_marker`` (``"⁣" + token_hex(6)``).
#: Its presence in content means the text is datamarked and tokenizes far worse
#: than plain prose; we count occurrences to price that inflation precisely.
_MARKER_SENTINEL = "⁣"

#: Average characters consumed by one *marked word* in datamarked text: the
#: 13-char marker (``U+2063`` + 12 hex) plus an assumed ~6-char word.
_AVG_MARKED_WORD_CHARS = 19

#: Conservative token cost of one *marked word* (the marker itself tokenizes to
#: ~12 tokens; the word it trails adds a few more).  Measured per-marked-word
#: cost ranged 10.6 (prose) to 15.7 (dense code) under the Qwen3 tokenizer; 16
#: covers the dense end so the estimate never under-counts datamarked content —
#: which is the only content the production (guards-on) pipeline ever sends.
_PER_MARKED_WORD_TOKENS = 16

#: Conservative chars-per-token for raw (non-datamarked) text.  Measured values
#: span ~6.8 for English prose down to ~2.3 for dense code / JSON / markdown;
#: 1.5 sizes below the dense end so the estimate never under-counts.  This path
#: is only hit in guards-off mode (no datamarking) — production is datamarked.
_RAW_CHARS_PER_TOKEN = 1.5


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` (marker-aware, conservative-by-design.)

    The estimate is biased HIGH on purpose: over-filling the ``max_model_len``
    window is a hard ``400`` failure, while under-filling only costs a little
    retrieved context that ``headroom_tokens`` already pads against.

    Two regimes:

    * **Datamarked** (guards-on — the production path): ``guardrails.sanitize``
      appends a ``U+2063``+hex marker after every word, so the marker count ≈
      the word count.  We price each *marked word* as a unit
      (:data:`_PER_MARKED_WORD_TOKENS`), which dominates and is counted exactly,
      then add any residual non-word characters at the dense raw rate.
    * **Raw** (no markers — guards-off / tests): a straight
      ``len(text) / 1.5`` over-estimate.

    Args:
        text: The text to size (may be datamarked).

    Returns:
        A conservative (high-biased) token-count estimate, ``>= 0``.
    """
    if not text:
        return 0
    n_markers = text.count(_MARKER_SENTINEL)
    if n_markers == 0:
        return int(-(-len(text) // _RAW_CHARS_PER_TOKEN))  # ceil(len / 1.5)
    # Datamarked: ~one marker per word. Price marker+word together; charge any
    # characters beyond the marked words at the dense raw rate (handles a long
    # non-datamarked tail, e.g. when marking stopped early).
    residual_chars = max(0, len(text) - n_markers * _AVG_MARKED_WORD_CHARS)
    residual_tokens = int(-(-residual_chars // _RAW_CHARS_PER_TOKEN))
    return n_markers * _PER_MARKED_WORD_TOKENS + residual_tokens


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` so its :func:`estimate_tokens` is ``<= max_tokens``.

    Used only when a *single* source exceeds the whole budget (e.g. a
    whole-document chunk from the long-context corpus).  The cut is a
    deterministic prefix slice — datamarked content is opaque data, so a clean
    word boundary is unnecessary; we only need to guarantee the slice fits.

    Args:
        text: The text to truncate.
        max_tokens: The token ceiling for the result.

    Returns:
        A prefix of ``text`` whose estimated tokens are ``<= max_tokens`` (empty
        when ``max_tokens <= 0``).
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    # First cut by the average chars/token ratio of THIS text (captures the
    # datamark density), with a 0.9 safety factor, then trim down deterministically
    # until the estimate fits.
    est = estimate_tokens(text)
    ratio = len(text) / est if est else _RAW_CHARS_PER_TOKEN
    cut = max(1, int(max_tokens * ratio * 0.9))
    candidate = text[:cut]
    # Shrink in modest steps if still over (rare; only when density is uneven).
    while candidate and estimate_tokens(candidate) > max_tokens:
        candidate = candidate[: int(len(candidate) * 0.9)]
    return candidate


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """The token window available for retrieved content in one LLM call.

    Attributes:
        max_model_len: The served model's context window (tokens).
        reserved_output_tokens: Tokens the model needs for its own completion
            (the call's ``max_tokens``); never available for input.
        headroom_tokens: Tokens reserved for everything in the prompt that is
            NOT the budgeted chunk content — the system prompt, the question,
            the chat-template scaffolding, plus a safety margin.
    """

    max_model_len: int
    reserved_output_tokens: int
    headroom_tokens: int

    def available_for_sources(self, *, extra_fixed_tokens: int = 0) -> int:
        """Tokens left for chunk content after output + fixed overhead.

        Args:
            extra_fixed_tokens: Additional already-known fixed prompt tokens to
                subtract (e.g. the draft answer the grader must also see).  Pass
                an estimate via :func:`estimate_tokens`.

        Returns:
            The non-negative token budget available for the chunk content.
        """
        return max(
            0,
            self.max_model_len
            - self.reserved_output_tokens
            - self.headroom_tokens
            - extra_fixed_tokens,
        )


def fit_items_to_budget[T](
    items: Sequence[T],
    content_of: Callable[[T], str],
    *,
    available_tokens: int,
    separator_tokens: int = 4,
) -> list[tuple[T, str]]:
    """Greedily keep items whose content fits ``available_tokens``.

    Items are assumed pre-ranked (highest relevance first).  Whole items are
    kept while they fit; once the budget is exhausted the remaining (lowest-
    ranked) items are dropped — *never* truncated mid-source — EXCEPT the
    degenerate case where the very first item alone exceeds the budget, which is
    truncated so at least some evidence survives (a whole-document chunk must
    not vanish entirely).

    Args:
        items: The ranked items (e.g. ``RetrievedChunk`` objects).
        content_of: Extracts the budget-relevant text from an item.
        available_tokens: The token budget for the joined content.
        separator_tokens: Tokens charged for the join separator between kept
            items (``"\\n\\n"`` plus any per-source header).

    Returns:
        A list of ``(item, fitted_content)`` for the kept items, in rank order.
        ``fitted_content`` equals the item's content except for a single
        first-item truncation.  Empty when ``available_tokens <= 0``.
    """
    if available_tokens <= 0:
        return []
    kept: list[tuple[T, str]] = []
    used = 0
    for item in items:
        content = content_of(item)
        sep = separator_tokens if kept else 0
        cost = estimate_tokens(content) + sep
        if used + cost <= available_tokens:
            kept.append((item, content))
            used += cost
            continue
        # Does not fit whole. If nothing has been kept yet, this single source
        # is larger than the entire budget — truncate it so some evidence
        # survives. Otherwise stop: the tail (lowest-ranked) is dropped.
        if not kept:
            truncated = truncate_to_tokens(content, available_tokens)
            if truncated:
                kept.append((item, truncated))
        break
    return kept
