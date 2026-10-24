"""Chunking lab: the real chunker over the bundled samples, and what the lab reports."""

from __future__ import annotations

import pytest

from iw_rag.learn.chunking_lab import (
    MAX_LEAF_CHARS,
    MIN_LEAF_CHARS,
    PARENT_EXPANSION_MAX_CHARS,
    run_chunking_lab,
)
from iw_rag.learn.samples import SAMPLES, get_sample

WITH_HEADINGS = SAMPLES[0]
NO_HEADINGS = SAMPLES[1]


def test_probe_sentences_exist_verbatim_in_every_sample() -> None:
    for sample in SAMPLES:
        for probe in sample.probes:
            assert probe.sentence in sample.text


def test_get_sample_unknown_is_none() -> None:
    assert get_sample("nope") is None
    assert get_sample("with-headings") is WITH_HEADINGS


@pytest.mark.parametrize("leaf", [50, 150, 500, 1600, 2500])
def test_leaves_never_exceed_the_requested_size_and_keep_all_text(leaf: int) -> None:
    result = run_chunking_lab(WITH_HEADINGS, leaf)
    assert all(x.chars <= leaf for x in result.leaves)
    # Leaves of a block are contiguous slices, so joining them restores every sentence.
    body = "".join(x.text for x in result.leaves)
    assert all(probe.sentence in body for probe in WITH_HEADINGS.probes)


def test_default_size_cuts_the_sick_leave_answer_in_two() -> None:
    probe = run_chunking_lab(WITH_HEADINGS, 500).probes[0]
    assert probe.status == "split"
    assert len(probe.leaf_indexes) == 2
    assert probe.verdict.startswith("Answer cut across leaves")


def test_larger_leaves_keep_the_answer_whole_and_expand_to_the_section() -> None:
    result = run_chunking_lab(WITH_HEADINGS, 800)
    probe = result.probes[0]
    assert probe.status == "intact"
    sick = next(p for p in result.parents if p.section.endswith("Sick leave"))
    assert sick.expanded
    assert probe.model_reads_chars == sick.chars


def test_no_headings_makes_one_oversized_parent_so_the_model_reads_only_the_leaf() -> None:
    result = run_chunking_lab(NO_HEADINGS, 800)
    assert len(result.parents) == 1
    assert result.parents[0].chars > PARENT_EXPANSION_MAX_CHARS
    assert result.oversized_parents == 1
    probe = result.probes[0]
    assert probe.status == "bare"
    leaf = result.leaves[probe.leaf_indexes[0]]
    assert probe.model_reads_chars == leaf.chars
    assert "no surrounding context" in probe.detail


def test_headings_variant_has_no_oversized_parent() -> None:
    assert run_chunking_lab(WITH_HEADINGS, 800).oversized_parents == 0


def test_smaller_leaves_mean_more_of_them_and_more_mid_sentence_cuts() -> None:
    small = run_chunking_lab(WITH_HEADINGS, 150)
    large = run_chunking_lab(WITH_HEADINGS, 1600)
    assert len(small.leaves) > len(large.leaves)
    assert small.mid_sentence_cuts > large.mid_sentence_cuts


def test_leaf_size_is_clamped_to_the_lab_range() -> None:
    assert run_chunking_lab(WITH_HEADINGS, 1).leaf_chars == MIN_LEAF_CHARS
    assert run_chunking_lab(WITH_HEADINGS, 10**6).leaf_chars == MAX_LEAF_CHARS


def test_a_cut_sentence_is_found_even_when_the_cut_is_near_its_end() -> None:
    # At 550 the sick-leave sentence is cut 11 characters before it ends; it must be
    # reported as split across two leaves, never as missing.
    result = run_chunking_lab(WITH_HEADINGS, 550)
    assert all(p.status in {"intact", "bare", "split"} for p in result.probes)
    assert result.probes[0].status == "split"
