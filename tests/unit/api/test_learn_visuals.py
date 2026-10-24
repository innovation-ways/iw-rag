"""Learn section figures: coverage, accessibility, motion safety, and the numbers they show."""

from __future__ import annotations

import math
import re

import pytest

from iw_rag.api.learn_page import learn_hub_html, learn_lesson_html
from iw_rag.api.learn_visuals import FIGURES, VIZ_CSS, figure_html, hub_diagram, progress_html
from iw_rag.domain.models import RetrievedChunk
from iw_rag.generation.answer import _resolve, parse_citation_markers
from iw_rag.guardrails.sanitize import datamark
from iw_rag.learn.guards_lab import load_trace
from iw_rag.learn.lessons import LESSONS

LAB_SLUGS = {"chunking", "search", "rerank", "fail-closed"}


def test_every_lesson_has_exactly_one_visual_a_figure_or_a_lab() -> None:
    slugs = {lesson.slug for lesson in LESSONS}
    assert set(FIGURES) | LAB_SLUGS == slugs
    assert not set(FIGURES) & LAB_SLUGS  # a lesson with a lab does not also get a figure
    assert figure_html("chunking") == ""


@pytest.mark.parametrize("slug", sorted(FIGURES))
def test_each_figure_is_an_accessible_self_contained_svg(slug: str) -> None:
    html = figure_html(slug)
    assert html.count("<svg") == 1 and html.count("</svg>") == 1
    assert 'role="img"' in html and re.search(r'aria-label="[^"]{20,}"', html)
    assert "<title>" in html and "<figcaption>" in html
    assert "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html
    assert "<script" not in html and "<image" not in html and "<foreignObject" not in html
    # every arrow marker a figure references is defined in that same figure
    for ref in set(re.findall(r"url\(#([\w-]+)\)", html)):
        assert f'id="{ref}"' in html, (slug, ref)


def test_lesson_pages_carry_their_figure_and_a_progress_strip() -> None:
    for lesson in LESSONS:
        page = learn_lesson_html(lesson)
        assert page.count('class="lesson-progress"') == 1
        assert f'aria-label="Step {LESSONS.index(lesson) + 1} of {len(LESSONS)}"' in page
        assert ('<figure class="viz"' in page) == (lesson.slug in FIGURES)


def test_motion_is_css_only_and_switched_off_for_reduced_motion() -> None:
    assert "@media (prefers-reduced-motion: reduce)" in VIZ_CSS
    reduced = VIZ_CSS.split("prefers-reduced-motion: reduce")[1]
    assert ".v-flow" in reduced and ".v-token" in reduced and "animation: none" in reduced
    assert "<animate" not in hub_diagram() and "<script" not in hub_diagram()


def test_hub_diagram_links_the_answer_steps_in_loop_order_and_shows_the_exits() -> None:
    diagram = hub_diagram()
    steps = re.findall(r'data-step="([a-z-]+)"', diagram)
    assert steps == [lesson.slug for lesson in LESSONS if lesson.phase == "answer"]
    for text in (
        "Reject",
        "Fail closed",
        "Cited answer",
        "append-only audit log",
        "retry, at most 2",
    ):
        assert text in diagram
    assert 'id="answerPath"' in diagram and learn_hub_html().count('id="answerPath"') == 1


def test_the_retry_budget_drawn_matches_the_production_default() -> None:
    from iw_rag.agent.budgets import DEFAULT_MAX_RETRIES

    assert f"retry, at most {DEFAULT_MAX_RETRIES}×" in hub_diagram()
    assert figure_html("self-check").count("v-bar warn") == DEFAULT_MAX_RETRIES


def test_input_guard_figure_shows_the_recorded_classifier_scores() -> None:
    trace = load_trace()
    html = figure_html("input-guard")
    assert f"limit {trace['pg2_threshold']}" in html
    scores = sorted(trace["pg2_scores"].values())
    assert "1.000" in html and scores[-1] >= 0.999  # the jailbreak question scores 1.000
    assert "under 0.001" in html  # the benign question


def test_sanitize_figure_shows_real_scores_the_false_positive_and_real_datamarking() -> None:
    html = figure_html("sanitize")
    assert "0.892" in html and html.count("DROPPED") == 2 and html.count("kept") >= 1
    marked = datamark("Travel expenses are reimbursed within 30 days of submission.", marker="◆")
    assert marked in html


def test_generate_figure_matches_the_production_citation_rules() -> None:
    draft = "Reimbursed within 30 days [1]. Late claims need written approval [7]."
    assert parse_citation_markers(draft) == [1, 7]
    assert "markers found: [1], [7]" in figure_html("generate")
    # With two sources, [1] resolves to a real chunk and [7] is dropped, as the figure shows.
    sources = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            document_id="d",
            content=f"text {i}",
            score=1.0,
            source_uri="learn://x",
        )
        for i in (1, 2)
    ]
    citations, dropped = _resolve([1, 7], sources)
    assert [c.chunk_id for c in citations] == ["c1"] and dropped == [7]


def test_benchmark_example_numbers_are_correct() -> None:
    html = figure_html("benchmarks")
    ndcg = 1 / math.log2(3 + 1)  # one relevant passage at rank 3, ideal DCG = 1
    assert (round(1 / 3, 3), round(ndcg, 3)) == (0.333, 0.5)
    for value in ("1.000", "0.333", "0.500"):
        assert value in html


def test_vram_figure_draws_the_measured_peak_to_scale_and_states_the_headroom() -> None:
    html = figure_html("one-gpu")
    used_width = round(620 * 23.9 / 32.0)  # the bar is drawn on a 620-unit track for 32 GB
    assert html.count(f'width="{used_width}"') == 1
    assert html.count("23.9 GB resident") == 1 and html.count("≈ 8 GB free") == 1


def test_progress_strip_marks_done_current_and_remaining() -> None:
    strip = progress_html(2, 5)
    assert (
        strip.count('class="done"') == 2
        and strip.count('class="cur"') == 1
        and strip.count("<i ") == 5
    )


def test_hub_cards_flag_exactly_the_lessons_with_a_lab() -> None:
    page = learn_hub_html()
    assert page.count('class="lab-badge"') == len(LAB_SLUGS)
    for slug in LAB_SLUGS:
        card = re.search(rf'<a class="step-card" href="/console/learn/{slug}">.*?</a>', page, re.S)
        assert card and "Interactive lab" in card.group(0)
