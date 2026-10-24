"""Reranker lab: a recorded real spine run, replayed under the spine's own knobs."""

from __future__ import annotations

import pytest

from iw_rag.learn.rerank_corpus import QUERIES
from iw_rag.learn.rerank_lab import (
    MAX_BUDGET,
    MAX_TOP_K,
    MIN_BUDGET,
    _unique_chars,
    get_query,
    load_trace,
    run_rerank_lab,
)

TRACE = load_trace()
CONFIGS = {
    "parent_up": {},
    "leaf_up": {"expansion": "leaf"},
    "parent_down": {"reranker_on": False},
    "sections_off": {"sections": False},
}


def _lab(query_id: str, **kw):  # type: ignore[no-untyped-def]
    query = get_query(query_id)
    assert query is not None
    return run_rerank_lab(query, **kw)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.query_id)
def test_replay_reproduces_what_the_real_spine_returned(query, config) -> None:  # type: ignore[no-untyped-def]
    recorded = next(q for q in TRACE["queries"] if q["id"] == query.query_id)["spine"][config]
    replay = run_rerank_lab(query, **CONFIGS[config])
    real = [(c["id"], c["chars"], c["expanded"], tuple(c["merged"])) for c in recorded["chunks"]]
    assert [(s.chunk_id, s.sent_chars, s.expanded, s.merged_ids) for s in replay.selected] == real
    assert recorded["degraded"] is (config == "parent_down")
    # The spine's own diagnostics agree with the replay's facts.
    assert recorded["diagnostics"].get("repeats_merged", 0) == replay.facts["repeats_merged"]
    assert (
        recorded["diagnostics"].get("sections_set_aside", {}) == replay.facts["sections_set_aside"]
    )


def test_trace_structure_is_consistent_with_the_documents() -> None:
    parents = {p["id"]: p for p in TRACE["parents"]}
    leaves = {x["id"]: x for x in TRACE["leaves"]}
    assert len({c["uuid"] for c in TRACE["parents"] + TRACE["leaves"]}) == len(parents) + len(
        leaves
    )
    for leaf in leaves.values():
        assert leaf["parent"] in parents and leaf["text"] in parents[leaf["parent"]]["text"]
    for q in TRACE["queries"]:
        assert {c["id"] for c in q["candidates"]} <= set(parents) | set(leaves)
    all_text = " ".join(x["text"] for x in leaves.values())
    assert all(query.answer in all_text for query in QUERIES)


def test_kept_chunks_are_a_prefix_that_grows_with_top_k() -> None:
    for k in range(1, MAX_TOP_K):
        smaller, larger = _lab("alcohol", top_k=k), _lab("alcohol", top_k=k + 1)
        assert len(smaller.kept_ids) == k
        assert larger.kept_ids[:k] == smaller.kept_ids


def test_reranker_off_keeps_the_fused_order() -> None:
    off = _lab("late-claim", reranker_on=False)
    fused_ids = [c.chunk_id for c in off.fused]
    positions = [fused_ids.index(i) for i in off.kept_ids if i in fused_ids]
    assert off.kept_ids[0] == fused_ids[0] and positions == sorted(positions)
    assert list(_lab("late-claim").kept_ids) != list(off.kept_ids)


def test_the_reranker_pulls_a_buried_answer_leaf_up_and_a_section_can_still_carry_it() -> None:
    on = _lab("late-claim")
    gold = on.facts["gold_ranks"]
    assert gold["fused"] == 11 and gold["reranked"] == 2
    assert any("moved the answer leaf from #11 to #2" in n for n in on.notes)
    assert on.facts["carrier_fused_rank"] < gold["fused"]  # its section ranks far higher
    assert any("a section that contains it ranks" in n for n in on.notes)


def test_without_the_reranker_a_buried_leaf_is_cut_but_its_section_still_carries_the_answer() -> (
    None
):
    off = _lab("late-claim", reranker_on=False, top_k=8)
    assert any("answer leaf (#11) is cut" in n for n in off.notes)
    assert off.facts["answer_in_context"]  # via the section that was kept
    assert not any(s.kind == "leaf" and s.is_answer for s in off.selected)
    assert any(s.kind == "section" and s.is_answer for s in off.selected)


def test_a_small_top_k_cuts_a_lower_ranked_answer_leaf() -> None:
    result = _lab("badge", top_k=3)
    assert result.facts["gold_ranks"]["reranked"] == 5
    assert any("answer leaf (#5) is cut" in n for n in result.notes)


def test_the_answer_is_lost_for_real_when_neither_leaf_nor_section_is_kept() -> None:
    result = _lab("badge", top_k=1)
    assert not result.facts["answer_in_context"]
    assert any("never reaches" in n or "not in anything" in n for n in result.notes)


def test_leaf_only_sends_the_matched_text_and_never_expands() -> None:
    for s in _lab("invoice", expansion="leaf").selected:
        assert not s.expanded and s.sent_text == s.matched_text
        assert "Expansion is off" in s.note or s.kind == "section"


def test_the_budget_decides_whether_a_leaf_is_expanded() -> None:
    def invoice_leaf(budget: int):  # type: ignore[no-untyped-def]
        return next(s for s in _lab("invoice", budget_chars=budget).selected if s.chunk_id == "L17")

    tight, roomy = invoice_leaf(MIN_BUDGET), invoice_leaf(1000)
    assert not tight.expanded and tight.sent_text == tight.matched_text
    assert "over the 500-character budget" in tight.note
    assert roomy.expanded and roomy.sent_chars > tight.sent_chars
    assert tight.matched_text in roomy.sent_text  # the matched leaf sits inside its section


def test_an_oversized_section_is_set_aside_before_the_reranker_so_it_is_never_sent_whole() -> None:
    result = _lab("badge", top_k=10)
    assert result.facts["sections_set_aside"] == {"oversize": 1}
    assert result.facts["eligible"] == result.candidates_total - 1
    assert "S8" not in {c.chunk_id for c in result.fused} | {c.chunk_id for c in result.reranked}
    assert all(s.sent_chars <= 4000 for s in result.selected)
    assert any("set aside before reranking" in n for n in result.notes)


def test_the_flat_document_leaf_is_not_expanded_at_the_real_budget_but_is_at_a_large_one() -> None:
    at_default = _lab("badge", top_k=10)
    leaf = next(s for s in at_default.selected if s.chunk_id == "L23")
    assert not leaf.expanded and "over the 4,000-character budget" in leaf.note
    roomy = _lab("badge", top_k=10, budget_chars=MAX_BUDGET)
    assert roomy.facts["sections_set_aside"] == {}  # the 4,126-character section now fits
    whole = roomy.selected[0]
    assert whole.sent_chars == 4126 and "L23" in whole.merged_ids  # one chunk stands for them all


def test_repeats_are_merged_so_the_generator_never_reads_the_same_text_twice() -> None:
    result = _lab("late-claim")
    assert result.facts["repeats_merged"] == 8 and result.facts["chars_duplicated"] == 0
    assert any("repeated copies merged" in n for n in result.notes)
    s3 = next(s for s in result.selected if s.chunk_id == "S3")
    assert set(s3.merged_ids) == {"L05", "L06", "L07"}  # its own leaves, folded into it
    assert s3.matched_text == next(x for x in TRACE["leaves"] if x["id"] == "L06")["text"]


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.query_id)
def test_no_two_chunks_handed_over_carry_the_same_text_and_none_is_also_a_merged_id(
    query, config
) -> None:  # type: ignore[no-untyped-def]
    result = run_rerank_lab(query, **CONFIGS[config])
    texts = [s.sent_text for s in result.selected]
    assert len(texts) == len(set(texts))
    ids = {s.chunk_id for s in result.selected}
    merged = {m for s in result.selected for m in s.merged_ids}
    assert ids & merged == set()


def test_leaf_mode_and_the_section_switch_remove_section_rows_as_candidates() -> None:
    leaf = _lab("alcohol", expansion="leaf")
    off = _lab("alcohol", sections=False)
    for result, reason in ((leaf, "leaf_only"), (off, "section_hits_off")):
        assert result.facts["sections_set_aside"] == {reason: 8}
        assert all(c.kind == "leaf" for c in result.fused + result.reranked)
        assert all(s.kind == "leaf" for s in result.selected)
    assert any("only leaves can match" in n for n in off.notes)


def test_merging_before_the_cut_gives_a_full_set_of_distinct_chunks() -> None:
    result = _lab("late-claim", top_k=8)
    assert len(result.selected) == 8 and result.facts["chunks_sent"] == 8


def test_unique_chars_counts_contained_and_identical_texts_once() -> None:
    assert _unique_chars(["abcdef", "bcd"]) == 6
    assert _unique_chars(["same", "same"]) == 4
    assert _unique_chars(["one", "two"]) == 6
    assert _unique_chars([]) == 0


def test_settings_are_clamped_to_the_lab_range() -> None:
    result = _lab("alcohol", top_k=999, budget_chars=1)
    assert result.settings["top_k"] == MAX_TOP_K and result.settings["budget_chars"] == MIN_BUDGET
    assert _lab("alcohol", expansion="nonsense").settings["expansion"] == "parent"
