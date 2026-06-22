"""Tests for the shared guard-model risk-verdict parser (guardrails/_verdict.py).

Covers the formats live validation proved Granite Guardian actually emits, plus
the model-agnostic JSON fallbacks used by the input-guard harm check and the
output-guard groundedness check.
"""

from __future__ import annotations

import pytest

from iw_rag.guardrails._verdict import parse_risk_verdict


@pytest.mark.parametrize(
    ("text", "json_keys", "expected"),
    [
        ("<score> yes </score>", ("risk",), 1.0),  # Granite: risky
        ("<score>no</score>", ("risk",), 0.0),  # Granite: safe
        ("<think>\n reasoning \n</think>\n<score> yes </score>", ("risk",), 1.0),
        ('{"risk": 0.8}', ("risk",), 0.8),  # JSON fallback (harm)
        ('{"groundedness_risk": 0.25}', ("groundedness_risk", "risk"), 0.25),  # groundedness key
        ('{"risk": 0.4}', ("groundedness_risk", "risk"), 0.4),  # secondary key match
        ("yes", ("risk",), 1.0),  # bare token
        ("no", ("risk",), 0.0),
    ],
)
def test_parse_risk_verdict(text: str, json_keys: tuple[str, ...], expected: float) -> None:
    """The shared parser handles <score>, JSON keys, and bare yes/no."""
    assert parse_risk_verdict(text, json_keys=json_keys) == pytest.approx(expected)


def test_parse_risk_verdict_unparseable_raises() -> None:
    """Unparseable guard output raises RuntimeError naming the bad input."""
    with pytest.raises(RuntimeError, match="Could not parse a risk verdict"):
        parse_risk_verdict("the sky is blue", json_keys=("risk",))


def test_groundedness_inversion_semantics() -> None:
    """A 'yes' groundedness-risk verdict inverts to a low groundedness probability."""
    # yes → risk 1.0 → grounded_prob = 1 - 1 = 0.0 (ungrounded)
    risk_yes = parse_risk_verdict("<score>yes</score>", json_keys=("groundedness_risk", "risk"))
    assert 1.0 - risk_yes == pytest.approx(0.0)
    # no → risk 0.0 → grounded_prob = 1.0 (grounded)
    risk_no = parse_risk_verdict("<score>no</score>", json_keys=("groundedness_risk", "risk"))
    assert 1.0 - risk_no == pytest.approx(1.0)
