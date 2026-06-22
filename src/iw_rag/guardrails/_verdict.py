"""Shared parsing for guard-model risk verdicts (Granite Guardian et al.).

Live validation showed Granite Guardian 3.3-8B returns
``<think>…</think><score> yes </score>`` regardless of a custom JSON-asking
prompt — so the input-guard harm check and the output-guard groundedness check
both need to read that format (plus a JSON fallback for model-agnosticism)
rather than assuming ``json.loads`` succeeds. This module centralises that
parsing so both call sites stay consistent.
"""

from __future__ import annotations

import json as _json
import re as _re

_THINK = _re.compile(r"<think>.*?</think>", _re.DOTALL | _re.IGNORECASE)
_SCORE = _re.compile(r"<score>\s*(yes|no)\s*</score>", _re.IGNORECASE)


def parse_risk_verdict(text: str, *, json_keys: tuple[str, ...] = ("risk",)) -> float:
    """Parse a guard model's risk verdict into a score in ``[0, 1]``.

    Handles, in order: Granite Guardian's ``<score>yes/no</score>`` (yes → 1.0,
    no → 0.0), a JSON object carrying any of ``json_keys`` (model-agnostic
    fallback), and a bare ``yes``/``no`` token. A leading ``<think>`` reasoning
    block is ignored.

    Args:
        text: Raw guard-model output.
        json_keys: JSON keys to try (first match wins) for the model-agnostic
            JSON fallback, e.g. ``("risk",)`` or ``("groundedness_risk", "risk")``.

    Returns:
        ``1.0`` for a yes/risk verdict, ``0.0`` for no/safe, or the parsed JSON value.

    Raises:
        RuntimeError: If no verdict can be parsed (caller should fail closed).
    """
    m = _SCORE.search(text)
    if m:
        return 1.0 if m.group(1).lower() == "yes" else 0.0

    try:
        parsed = _json.loads(text)
        if isinstance(parsed, dict):
            for key in json_keys:
                if key in parsed:
                    return float(parsed[key])
    except (ValueError, TypeError, _json.JSONDecodeError):
        pass

    tail = _THINK.sub("", text).strip().lower()
    if tail.startswith("yes"):
        return 1.0
    if tail.startswith("no"):
        return 0.0
    raise RuntimeError(f"Could not parse a risk verdict from guard output: {text!r}")
