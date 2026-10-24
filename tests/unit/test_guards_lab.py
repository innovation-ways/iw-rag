"""Guards lab: a recorded run of the production graph, replayed as explained steps."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from iw_rag.learn.guards_lab import (
    SOURCE_LABELS,
    load_trace,
    run_guards_lab,
    summaries,
)
from iw_rag.learn.guards_scenarios import SCENARIOS, all_texts, get_scenario

TRACE = load_trace()
MAX_RETRIES = TRACE["max_retries"]
KNOWN_NODES = {
    "input_guard",
    "reject",
    "router",
    "retrieve",
    "sanitize",
    "grade_documents",
    "consume_retry",
    "rewrite_query",
    "generate",
    "grade_generation",
    "output_guard",
    "done",
    "fail_closed",
    "audit",
}


def _run(scenario_id: str):  # type: ignore[no-untyped-def]
    scenario = get_scenario(scenario_id)
    assert scenario is not None
    return run_guards_lab(scenario)


def _nodes(scenario_id: str) -> list[str]:
    return [s.node for s in _run(scenario_id).steps]


def test_every_scenario_was_recorded_and_every_pg2_text_has_a_real_score() -> None:
    assert {s["id"] for s in TRACE["scenarios"]} == {s.scenario_id for s in SCENARIOS}
    assert set(TRACE["pg2_scores"]) == set(all_texts())
    assert all(0.0 <= v <= 1.0 for v in TRACE["pg2_scores"].values())


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.scenario_id)
def test_graph_invariants_hold_in_every_recorded_run(scenario) -> None:  # type: ignore[no-untyped-def]
    result = run_guards_lab(scenario)
    nodes = [s.node for s in result.steps]
    assert set(nodes) <= KNOWN_NODES
    assert nodes[0] == "input_guard" and nodes[-1] == "audit"  # every run ends in the audit
    assert max(s.state["retries"] for s in result.steps) <= MAX_RETRIES
    disposition = result.final["disposition"]
    assert disposition in {"answered", "refused"}
    if disposition == "answered":
        # Only 'done' may promote a draft to an answer, and only after the output guard.
        assert nodes.index("output_guard") < nodes.index("done")
        assert result.final["answer"] and result.steps[-2].state["gen_verdict"] == "grounded"
        assert "fail_closed" not in nodes
    else:
        assert not result.final.get("answer") and result.final["refusal"]
        assert "done" not in nodes


def test_a_jailbreak_dies_at_the_input_guard_before_any_search() -> None:
    nodes = _nodes("jailbreak")
    assert nodes == ["input_guard", "reject", "audit"]
    step = _run("jailbreak").steps[0]
    assert step.outcome == "stop" and "blocked" in step.headline
    assert TRACE["pg2_scores"][get_scenario("jailbreak").question] >= TRACE["pg2_threshold"]  # type: ignore[union-attr]


def test_the_poisoned_document_is_dropped_and_never_reaches_the_answer() -> None:
    result = _run("poisoned")
    sanitize = next(s for s in result.steps if s.node == "sanitize")
    assert "x01" in sanitize.state["dropped"] and "x01" not in sanitize.state["kept"]
    assert any("x01" in i and "DROPPED" in i for i in sanitize.items)
    assert all("x01" not in json.dumps(asdict(s)["state"]["kept"]) for s in result.steps)
    generate = next(s for s in result.steps if s.node == "generate")
    assert all("x01" not in i for i in generate.items)
    assert sanitize.marked_sample is not None and "◆" in sanitize.marked_sample


def test_an_innocent_chunk_is_dropped_too_and_the_lab_says_so() -> None:
    # A real classifier score: the meal-allowance note is harmless but scores over the limit.
    sanitize = next(s for s in _run("poisoned").steps if s.node == "sanitize")
    assert "f08" in sanitize.state["dropped"]
    assert any("f08" in i and "DROPPED" in i for i in sanitize.items)
    assert "false positive" in sanitize.detail


def test_a_bad_first_search_is_recovered_by_a_retry() -> None:
    nodes = _nodes("retry")
    assert nodes.count("retrieve") == 2 and "consume_retry" in nodes
    assert _run("retry").final["disposition"] == "answered"
    assert _run("retry").steps[-1].state["retries"] == 1


@pytest.mark.parametrize("scenario_id", ["unsupported", "uncited", "no-evidence"])
def test_exhausting_the_retry_budget_ends_in_a_refusal(scenario_id: str) -> None:
    result = _run(scenario_id)
    nodes = [s.node for s in result.steps]
    assert nodes.count("consume_retry") == MAX_RETRIES
    assert nodes[-2] == "fail_closed"
    assert result.steps[-1].state["retries"] == MAX_RETRIES


def test_an_uncited_draft_can_never_be_grounded() -> None:
    result = _run("uncited")
    drafts = [s for s in result.steps if s.node == "generate"]
    assert drafts and all(not s.items and s.outcome == "stop" for s in drafts)
    assert "citation is required" in drafts[0].detail.lower()
    assert {s.state["gen_verdict"] for s in result.steps if s.node == "grade_generation"} == {
        "unsupported"
    }


def test_the_retry_counter_is_shown_rising_step_by_step() -> None:
    counts = [s.state["retries"] for s in _run("no-evidence").steps]
    assert counts == sorted(counts) and counts[0] == 0 and counts[-1] == MAX_RETRIES


def test_summaries_report_the_outcome_and_the_deciding_checkpoint() -> None:
    cards = {c.scenario_id: c for c in summaries()}
    assert cards["answered"].disposition == "answered"
    assert cards["jailbreak"].decided_by == "Input guard"
    assert cards["unsupported"].decided_by == "Fail closed"
    assert {c.disposition for c in cards.values()} == {"answered", "refused"}


def test_provenance_is_stated_and_every_step_names_its_source() -> None:
    assert set(TRACE["provenance"]) == {"real", "scripted", "not_run"}
    assert any("Granite" in x for x in TRACE["provenance"]["not_run"])
    for scenario in SCENARIOS:
        assert {s.source for s in run_guards_lab(scenario).steps} <= set(SOURCE_LABELS)
    # The two steps whose signal is a real classifier score are labelled as recorded.
    steps = _run("answered").steps
    assert {s.node for s in steps if s.source == "recorded"} == {"input_guard", "sanitize"}
