"""The plain-language benchmark glossary: complete, consistent with the code, and correct."""

from __future__ import annotations

import dataclasses
import math
import re

import pytest

from iw_rag.api.benchmark_render import REPORT_JS
from iw_rag.eval.benchmark import scoreboard as scoreboard_module
from iw_rag.learn.metrics import GROUPS, TERMS, get_term, plain_map

KEYS = [t.key for t in TERMS]
INTERNAL = re.compile(r"\bS\d{2}\b|Invariant|\bF-\d{5}\b|\bPOST /|\bGET /|\bSSE\b|AC\d")


def test_keys_are_unique_and_every_group_is_used_and_declared() -> None:
    assert len(KEYS) == len(set(KEYS))
    assert {t.group for t in TERMS} == set(GROUPS)


@pytest.mark.parametrize("term", TERMS, ids=lambda t: t.key)
def test_every_term_explains_itself_with_an_example_and_a_direction(term) -> None:  # type: ignore[no-untyped-def]
    assert len(term.plain) >= 25 and term.plain.endswith((".", "?"))
    assert len(term.example) >= 20 and len(term.good) >= 5
    assert not INTERNAL.search(
        " ".join((term.name, term.plain, term.example, term.good, term.watch))
    )


def test_every_metric_the_report_renderer_knows_has_a_plain_explanation() -> None:
    glossary_keys = re.findall(r"^    (\w+):\s*\{ label:", REPORT_JS, re.M)
    assert len(glossary_keys) >= 25
    assert sorted(set(glossary_keys) - set(KEYS)) == []


def test_every_scoreboard_board_has_a_plain_explanation() -> None:
    boards = [f.name for f in dataclasses.fields(scoreboard_module.Scoreboard)]
    assert boards == [
        "faithfulness",
        "ndcg_at_5",
        "tok_per_s",
        "latency",
        "vram_efficiency",
        "composite",
    ]
    assert sorted(set(boards) - set(KEYS)) == []


def test_every_statistic_column_on_the_ab_pages_has_a_plain_explanation() -> None:
    needed = {
        "delta",
        "p_value",
        "p_corrected",
        "ci_95",
        "significant",
        "underpowered",
        "winner",
        "baseline",
    }
    assert sorted(needed - set(KEYS)) == []


def test_the_composite_weights_in_the_text_match_the_scoreboard_service() -> None:
    weights = scoreboard_module._COMPOSITE_WEIGHTS
    stated = sorted(int(n) for n in re.findall(r"(\d+)%", get_term("composite").plain))  # type: ignore[union-attr]
    assert stated == sorted(round(w * 100) for w in weights.values())
    assert math.isclose(sum(weights.values()), 1.0)


def test_the_worked_examples_in_the_text_are_arithmetically_right() -> None:
    ndcg = get_term("ndcg_at_5").example  # type: ignore[union-attr]
    assert (1 / math.log2(1 + 1), round(1 / math.log2(3 + 1), 2)) == (1.0, 0.5)
    assert "1.0" in ndcg and "about 0.5" in ndcg
    assert (
        round(1 / math.log2(10 + 1), 2) == 0.29 and "about 0.29" in get_term("ndcg_at_10").example
    )  # type: ignore[union-attr]
    assert (round(1 / 3, 2), 1 / 2) == (0.33, 0.5) and "0.33" in get_term("mrr").example  # type: ignore[union-attr]
    assert round(1 / 0.02) == 50 and "50 tokens a second" in get_term("tpot_p99_s").example  # type: ignore[union-attr]
    assert 50 * 0.75 == 37.5 and "37 words" in get_term("tok_per_s").example  # type: ignore[union-attr]


def test_direction_is_stated_correctly_for_the_lower_is_better_metrics() -> None:
    for key in ("noise_sensitivity", "latency"):
        assert "LOWER is better" in get_term(key).good  # type: ignore[union-attr]
    for key in ("recall_at_5", "ndcg_at_5", "mrr", "faithfulness"):
        assert get_term(key).good.startswith("Higher is better")  # type: ignore[union-attr]
    assert "B minus A" in get_term("delta").name  # type: ignore[union-attr]


def test_the_faithfulness_caveat_names_the_refusal_effect() -> None:
    # Refusals are trivially faithful, so a high score can mean 'refused a lot'.
    watch = get_term("faithfulness").watch  # type: ignore[union-attr]
    assert watch.startswith("A refusal such as 'I do not have enough information' is trivially")
    assert watch.count("faithful") == 1


def test_plain_map_carries_every_term_with_all_five_fields() -> None:
    mapping = plain_map()
    assert list(mapping) == KEYS
    assert all(set(v) == {"name", "plain", "example", "good", "watch"} for v in mapping.values())
