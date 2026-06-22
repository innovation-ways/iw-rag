"""Unit tests for prompt token-budgeting (I-CTXOVERFLOW).

These pin the contract that stops the grader/generator prompts from overflowing
the generator's ``max_model_len`` window:

* :func:`estimate_tokens` is *conservative* (biased high) and marker-aware —
  datamarked content (guards-on, ~12x token inflation) must estimate far higher
  than the same raw text, so the fitter keeps a real safety margin.
* :func:`fit_items_to_budget` keeps the highest-ranked whole items that fit,
  drops the lowest-ranked overflow tail, and truncates only a lone oversized
  first item.
* :func:`truncate_to_tokens` never returns text whose estimate exceeds the cap.

Pure logic — no network, no tokenizer, no live models.
"""

from __future__ import annotations

from dataclasses import dataclass

from iw_rag.generation.context_budget import (
    ContextBudget,
    estimate_tokens,
    fit_items_to_budget,
    truncate_to_tokens,
)

#: One datamark marker (U+2063 + 12 hex) as emitted by guardrails.sanitize.
_MARKER = "⁣abc123def456"


def _datamark(text: str) -> str:
    """Approximate ``guardrails.sanitize.datamark``: marker after every word."""
    return "".join(w + _MARKER for w in text.split(" "))


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty_is_zero() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_raw_is_ceil_len_over_1_5() -> None:
    # 30 chars, no markers → ceil(30 / 1.5) == 20.
    assert estimate_tokens("a" * 30) == 20


def test_estimate_tokens_is_monotonic_in_length() -> None:
    assert estimate_tokens("word " * 10) < estimate_tokens("word " * 100)


def test_estimate_tokens_datamarked_far_exceeds_raw() -> None:
    """The whole point: datamarked content must price MUCH higher than raw.

    If the estimator were length-only it would treat the two as comparable and
    the fitter would happily overflow the window — the live 400 bug.
    """
    raw = "the operations handbook describes the escalation procedure " * 20
    marked = _datamark(raw)
    assert estimate_tokens(marked) > 3 * estimate_tokens(raw)


def test_estimate_tokens_scales_with_marker_count() -> None:
    one = "alpha" + _MARKER
    many = ("alpha" + _MARKER) * 50
    assert estimate_tokens(many) >= 50 * estimate_tokens(one) * 0.9


# ---------------------------------------------------------------------------
# truncate_to_tokens
# ---------------------------------------------------------------------------


def test_truncate_zero_budget_is_empty() -> None:
    assert truncate_to_tokens("anything", 0) == ""
    assert truncate_to_tokens("anything", -5) == ""


def test_truncate_keeps_text_already_within_budget() -> None:
    text = "short text"
    assert truncate_to_tokens(text, 10_000) == text


def test_truncate_result_never_exceeds_budget() -> None:
    big = _datamark("alpha beta gamma delta epsilon " * 400)
    for cap in (50, 250, 1000):
        out = truncate_to_tokens(big, cap)
        assert estimate_tokens(out) <= cap, f"truncation overshoots budget {cap}"


# ---------------------------------------------------------------------------
# ContextBudget
# ---------------------------------------------------------------------------


def test_context_budget_available_subtracts_output_headroom_and_fixed() -> None:
    b = ContextBudget(max_model_len=16384, reserved_output_tokens=1024, headroom_tokens=2048)
    assert b.available_for_sources() == 16384 - 1024 - 2048
    assert b.available_for_sources(extra_fixed_tokens=300) == 16384 - 1024 - 2048 - 300


def test_context_budget_available_never_negative() -> None:
    b = ContextBudget(max_model_len=1000, reserved_output_tokens=900, headroom_tokens=900)
    assert b.available_for_sources() == 0


# ---------------------------------------------------------------------------
# fit_items_to_budget
# ---------------------------------------------------------------------------


@dataclass
class _Item:
    content: str


def _content(it: _Item) -> str:
    return it.content


def test_fit_keeps_ranked_prefix_and_drops_overflow_tail() -> None:
    # Each item ~ ceil(60/1.5)=40 tokens; budget 130 fits 3 (40*3=120 + seps),
    # the 4th+ are dropped.
    items = [_Item("x" * 60) for _ in range(8)]
    kept = fit_items_to_budget(items, _content, available_tokens=130, separator_tokens=0)
    assert 0 < len(kept) < len(items)
    # Kept items are the ranked prefix (same objects, in order).
    assert [it for it, _ in kept] == items[: len(kept)]
    # Total estimated tokens stay within budget.
    total = sum(estimate_tokens(text) for _, text in kept)
    assert total <= 130


def test_fit_truncates_lone_oversized_first_item() -> None:
    # A single item larger than the entire budget must be TRUNCATED (not dropped)
    # so some evidence survives — a whole-document chunk must not vanish.
    items = [_Item("y" * 100_000)]
    kept = fit_items_to_budget(items, _content, available_tokens=500, separator_tokens=0)
    assert len(kept) == 1
    _, fitted = kept[0]
    assert len(fitted) < 100_000
    assert estimate_tokens(fitted) <= 500


def test_fit_empty_budget_returns_nothing() -> None:
    items = [_Item("a"), _Item("b")]
    assert fit_items_to_budget(items, _content, available_tokens=0) == []


def test_fit_all_fit_returns_all_with_original_content() -> None:
    items = [_Item("tiny one"), _Item("tiny two")]
    kept = fit_items_to_budget(items, _content, available_tokens=10_000)
    assert [text for _, text in kept] == ["tiny one", "tiny two"]


def test_fit_separator_tokens_are_charged() -> None:
    # With a large separator charge, fewer items fit than with none.
    items = [_Item("z" * 15) for _ in range(6)]  # ~10 tokens each
    no_sep = fit_items_to_budget(items, _content, available_tokens=45, separator_tokens=0)
    big_sep = fit_items_to_budget(items, _content, available_tokens=45, separator_tokens=20)
    assert len(big_sep) < len(no_sep)
