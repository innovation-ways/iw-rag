"""Input guard — scan the user turn before routing (lld-guardrails §3.2).

Two local-model clients compose the decision:

1. **Llama Prompt Guard 2** (``prompt-guard-2-86m``) — a mDeBERTa binary
   jailbreak/benign classifier served via TEI/transformers.  Returns
   ``P(jailbreak)`` as a float in ``[0, 1]``.  Hot-path cheap (BERT-class,
   single-digit-ms target).

2. **IBM Granite Guardian 3.3-8B** (``granite-guardian-3.3-8b``) — harm/jailbreak
   risk served on the same vLLM instance used by the groundedness judge in
   Task 3.3.  One workhorse, two jobs, minimising resident VRAM (R-00108 F6).

Both endpoints are resolved from ``config/models.yaml`` via ``get_settings()``
(logical roles ``guard_injection`` and ``guard_safety``); **no hardcoded
URLs** — swapping a model is a config change, never a code change.

Fail-closed invariant (lld-guardrails §5):
- If either client raises or times out after the bounded retry, the function
  returns a ``block`` verdict with ``reason="guard_unavailable"``.
- A down injection scanner must NOT become an open door.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import GuardVerdict
from iw_rag.settings import get_settings

logger = logging.getLogger(__name__)

# Bounded retry: attempt the guard call this many times before fail-closed.
_MAX_RETRIES: int = 2
# Per-attempt HTTP timeout in seconds (BERT-class: single-digit-ms target).
_TIMEOUT_S: float = 5.0


async def prompt_guard_score(question: str) -> float:
    """Call Llama Prompt Guard 2 and return P(jailbreak) in [0, 1].

    The client calls the TEI/transformers HTTP classify endpoint for the
    ``guard_injection`` logical model role.  The response is a softmax
    probability for the ``JAILBREAK`` class.

    On any transient error the caller (``scan_input``) handles retry and
    fail-closed logic — this function raises on every error after one attempt.

    Args:
        question: The user question text to classify.

    Returns:
        A float in ``[0, 1]`` — the probability that the question is a
        jailbreak/injection attempt.

    Raises:
        httpx.HTTPError: On any HTTP-level failure.
        RuntimeError: On unexpected response shape from the serving endpoint.
    """
    cfg = get_settings()
    entry = cfg.models.get("guard_injection")
    url = f"{entry.endpoint.rstrip('/')}/classify"

    payload: dict[str, object] = {"inputs": question}
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    # TEI classify returns a list of label/score dicts (one per token in
    # multi-label mode).  Find the JAILBREAK label score.
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("label", "").upper() == "JAILBREAK":
                return float(item.get("score", 0.0))
        # If no JAILBREAK label, return 0 (benign by default).
        return 0.0
    raise RuntimeError(f"Unexpected response shape from guard_injection endpoint: {data!r}")


async def granite_harm_score(question: str) -> float:
    """Call Granite Guardian 3.3-8B and return a harm/jailbreak risk score in [0, 1].

    The client posts a harm-risk classification prompt to the vLLM completions
    endpoint for the ``guard_safety`` logical model role.  Granite Guardian
    returns a risk assessment; this function parses the ``"Yes"``/``"No"``
    token and converts it to a probability.

    Args:
        question: The user question text to assess for harm.

    Returns:
        A float in ``[0, 1]`` — the harm risk probability.

    Raises:
        httpx.HTTPError: On any HTTP-level failure.
        RuntimeError: On unexpected response shape from the serving endpoint.
    """
    cfg = get_settings()
    entry = cfg.models.get("guard_safety")
    url = f"{entry.endpoint.rstrip('/')}/chat/completions"

    # Granite Guardian applies its OWN risk-detection chat template; we pass the
    # turn to assess as the user message and read its verdict.  (A custom system
    # prompt is ignored by the template, so we keep the request minimal.)
    messages = [{"role": "user", "content": question}]
    payload: dict[str, object] = {
        "model": entry.served_name,
        "messages": messages,
        "max_tokens": 16,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        text = str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected response shape from guard_safety endpoint: {data!r}"
        ) from exc
    return _parse_harm_verdict(text)


def _parse_harm_verdict(text: str) -> float:
    """Parse a guard model's harm verdict into a risk score in [0, 1].

    Thin wrapper over :func:`iw_rag.guardrails._verdict.parse_risk_verdict`
    (shared with the output-guard groundedness check): handles Granite
    Guardian's ``<score>yes/no</score>``, a ``{"risk": <float>}`` JSON fallback,
    and a bare ``yes``/``no`` token.

    Args:
        text: The raw guard model output (a ``<think>`` block is ignored).

    Returns:
        ``1.0`` for a harmful/yes verdict, ``0.0`` for safe/no.

    Raises:
        RuntimeError: If no verdict can be parsed (caller fails closed).
    """
    from iw_rag.guardrails._verdict import parse_risk_verdict

    return parse_risk_verdict(text, json_keys=("risk",))


async def scan_input(question: str, access_context: AccessContext) -> GuardVerdict:
    """Scan the user turn for prompt-injection/jailbreak and harm.

    Combines Llama Prompt Guard 2 (injection classifier, BERT-class hot path)
    and Granite Guardian 3.3-8B (harm/jailbreak risk, same vLLM instance as
    Task 3.3 groundedness) into a single aggregated verdict.

    Aggregation rule (lld-guardrails §3.2):
    - ``score = max(prompt_guard_score, granite_harm_score)``
    - ``score >= threshold`` → ``block``;  tag = ``"injection"`` if pg >= gg
      else ``"harm"``.
    - ``score < threshold``  → ``allow``.

    The ``access_context`` is threaded through for audit attribution only —
    it is NOT used for the guard decision.  ``principal_groups`` are never
    derived from the question text or model output (CONVENTIONS §2).

    Fail-closed (lld-guardrails §5): if either guard client raises after
    :data:`_MAX_RETRIES` attempts, the function returns
    ``GuardVerdict(layer="input", decision="block", reason="guard_unavailable")``.
    A down injection scanner must NOT become an open door.

    Args:
        question: The user question text to scan.
        access_context: The authenticated access context (for audit attribution
            only; not used in the guard decision).

    Returns:
        A :class:`~iw_rag.domain.models.GuardVerdict` with:
        - ``layer="input"``
        - ``decision`` in ``{"allow", "block"}``
        - ``score`` = ``max(pg, gg)``
        - ``reason`` = ``"clean"`` or ``"<tag>:<score:.2f>"`` or
          ``"guard_unavailable"``
        - ``detail`` = ``{"prompt_guard": pg, "granite": gg}`` (or ``{}``
          on unavailability)
    """
    cfg = get_settings()
    threshold: float = cfg.guardrails.input.threshold

    # ------------------------------------------------------------------
    # Call both guard clients concurrently with bounded retry.
    # Either raising → fail-closed block (lld-guardrails §5).
    # ------------------------------------------------------------------
    try:
        pg, gg = await asyncio.gather(
            _with_retry(prompt_guard_score, question, retries=_MAX_RETRIES),
            _with_retry(granite_harm_score, question, retries=_MAX_RETRIES),
        )
    except Exception:
        logger.warning(
            "input_guard: guard client unavailable for question=%r — failing closed",
            question[:80],
        )
        return GuardVerdict(
            layer="input",
            decision="block",
            score=1.0,
            reason="guard_unavailable",
            detail={},
        )

    score: float = max(pg, gg)

    if score >= threshold:
        # Determine the dominant signal for the tag.
        tag = "injection" if pg >= gg else "harm"
        reason = f"{tag}:{score:.2f}"
        logger.info(
            "input_guard: BLOCKED question=%r score=%.3f tag=%s (pg=%.3f gg=%.3f)",
            question[:80],
            score,
            tag,
            pg,
            gg,
        )
        return GuardVerdict(
            layer="input",
            decision="block",
            score=score,
            reason=reason,
            detail={"prompt_guard": pg, "granite": gg},
        )

    logger.debug(
        "input_guard: allowed question=%r score=%.3f (pg=%.3f gg=%.3f)",
        question[:80],
        score,
        pg,
        gg,
    )
    return GuardVerdict(
        layer="input",
        decision="allow",
        score=score,
        reason="clean",
        detail={"prompt_guard": pg, "granite": gg},
    )


async def _with_retry(
    fn: object,
    question: str,
    *,
    retries: int,
) -> float:
    """Call an async guard function with bounded exponential-backoff retry.

    Retries on any exception up to ``retries`` times, then re-raises.  The
    caller (``scan_input``) converts the final exception into a fail-closed
    block verdict.

    Args:
        fn: The async callable to call (``prompt_guard_score`` or
            ``granite_harm_score``).
        question: The user question text to pass to ``fn``.
        retries: Maximum number of attempts (including the first).

    Returns:
        The float score returned by the successful call.

    Raises:
        Exception: Re-raises the last exception if all attempts fail.
    """
    import asyncio as _asyncio

    last_exc: BaseException | None = None
    for attempt in range(max(retries, 1)):
        try:
            # fn is typed as object to satisfy mypy; it is always callable.
            result = await fn(question)  # type: ignore[operator]
            return float(result)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                backoff = 0.05 * (2**attempt)
                logger.debug(
                    "_with_retry: attempt %d/%d failed (%s); backing off %.2fs",
                    attempt + 1,
                    retries,
                    exc,
                    backoff,
                )
                await _asyncio.sleep(backoff)
            else:
                logger.warning(
                    "_with_retry: all %d attempt(s) failed for %r: %s",
                    retries,
                    getattr(fn, "__name__", repr(fn)),
                    exc,
                )
    # mypy: last_exc is always set when retries >= 1.
    raise last_exc  # type: ignore[misc]
