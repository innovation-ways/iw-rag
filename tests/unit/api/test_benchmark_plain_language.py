"""The benchmark pages in plain language: primer, tooltips, labels, and no internal jargon."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap

import pytest

from iw_rag.api.benchmark_primer import PRIMER_HTML, apply_benchmark_primer
from iw_rag.api.benchmark_render import REPORT_JS, plain_terms_json
from iw_rag.api.benchmark_runner_page import benchmark_runner_page_html
from iw_rag.api.compare_page import compare_page_html
from iw_rag.api.learn_page import learn_lesson_html
from iw_rag.api.results_page import results_page_html
from iw_rag.api.scoreboard_page import scoreboard_page_html
from iw_rag.learn.lessons import LESSONS
from iw_rag.learn.metrics import TERMS, plain_map

PAGES = {
    "scoreboard": scoreboard_page_html,
    "results": results_page_html,
    "compare": compare_page_html,
    "benchmark": benchmark_runner_page_html,
}
INTERNAL = ("Invariant", "(S05)", "F-00094", "SSE", "POST /", "GET /", "config_snapshot", "(AC6)")


def _visible_text(page: str) -> str:
    page = re.sub(r"<(script|style)\b.*?</\1>", " ", page, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))


@pytest.mark.parametrize("name", sorted(PAGES))
def test_every_benchmark_page_opens_with_the_primer_and_links_to_the_guide(name: str) -> None:
    page = PAGES[name]()
    assert page.count('id="benchPrimer"') == 1 and "__BENCH_PRIMER__" not in page
    assert 'href="/console/learn/benchmarks"' in page
    assert "details.bench-primer" in page  # its styles were added
    assert page.index('id="benchPrimer"') < page.index('class="card"', page.index('id="appView"'))


@pytest.mark.parametrize("name", sorted(PAGES))
def test_the_visible_copy_has_no_internal_jargon(name: str) -> None:
    text = _visible_text(PAGES[name]())
    leaked = [token for token in INTERNAL if token in text]
    assert leaked == [], leaked


def test_the_primer_covers_what_a_run_is_the_four_questions_and_how_to_read_a_number() -> None:
    text = _visible_text(PRIMER_HTML)
    for fragment in (
        "fixed list of questions",
        "Did it find the right passages?",
        "Is the answer honest?",
        "How fast is it?",
        "How much GPU memory?",
        "0 to 1",
        "same dataset",
        "significant",
    ):
        assert text.count(fragment) >= 1, fragment
    assert apply_benchmark_primer("<style></style>__BENCH_PRIMER__").count("benchPrimer") == 1


def _embedded_plain(js: str) -> dict[str, dict[str, str]]:
    match = re.search(r"var IW_PLAIN = (\{.*?\});\n", js, re.S)
    assert match is not None
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_the_plain_language_data_reaches_the_report_renderer_and_the_scoreboard_intact() -> None:
    assert _embedded_plain(REPORT_JS) == plain_map()
    assert _embedded_plain(scoreboard_page_html()) == plain_map()
    assert "__IW_PLAIN__" not in REPORT_JS and "</script" not in plain_terms_json().lower()
    for page in (results_page_html(), compare_page_html()):
        assert _embedded_plain(page) == plain_map()


def test_the_ab_statistics_columns_carry_hover_explanations_on_both_compare_views() -> None:
    for key in ("delta", "p_value", "p_corrected", "ci_95", "significant"):
        assert f'IW_tip("{key}")' in REPORT_JS, key
    page = compare_page_html()
    for key in ("delta", "p_corrected", "ci_95", "significant"):
        assert f'IW_tip("{key}")' in page, key


def test_the_tooltip_shows_meaning_then_example_then_whether_high_or_low_is_good() -> None:
    assert REPORT_JS.index("pl.plain") < REPORT_JS.index("pl.example") < REPORT_JS.index("pl.good")
    assert "Good or bad?" in REPORT_JS and "Example:" in REPORT_JS


def test_the_colour_legend_states_the_same_thresholds_the_report_actually_uses() -> None:
    code = re.search(
        r"var good = lower \? \(v <= ([\d.]+)\) : \(v >= ([\d.]+)\);\s*"
        r"var mid = lower \? \(v <= ([\d.]+)\) : \(v >= ([\d.]+)\);",
        REPORT_JS,
    )
    legend = re.search(
        r"green</span> is ([\d.]+) or more, .*?amber</span> is ([\d.]+) to ([\d.]+), "
        r".*?below ([\d.]+)\.",
        REPORT_JS,
        re.S,
    )
    assert code is not None and legend is not None
    good_hi, mid_hi = float(code.group(2)), float(code.group(4))  # higher-is-better cut-offs
    stated = tuple(float(g) for g in legend.groups())
    assert stated == (good_hi, mid_hi, good_hi, mid_hi)
    assert REPORT_JS.count("html += IW_legend();") == 1


def test_scoreboard_tabs_use_friendly_names_and_keep_the_technical_labels() -> None:
    page = scoreboard_page_html()
    tabs = re.findall(r'<span>([^<]+) <small class="tech">([^<]+)</small></span>', page)
    assert tabs == [
        ("Honest answers", "faithfulness"),
        ("Search quality", "ndcg_at_5"),
        ("Writing speed", "tok/s"),
        ("Worst-case wait", "latency"),
        ("Quality per GB", "VRAM-efficiency"),
        ("All-round", "composite"),
    ]


def test_runner_settings_each_get_a_one_line_explanation() -> None:
    page = benchmark_runner_page_html()
    assert page.count('class="field-help"') == 6
    for label in ("Passages kept (top_k)", "Retrieval mode", "GPU memory budget"):
        assert label in page
    for element_id in ("pickGenerator", "pickEmbedder", "pickReranker", "pickTopK", "pickMode"):
        assert page.count(f'id="{element_id}"') == 1  # behaviour hooks are untouched


def test_the_lesson_decoder_lists_every_term_once_with_an_anchor() -> None:
    page = learn_lesson_html(next(x for x in LESSONS if x.slug == "benchmarks"))
    assert page.count('id="metricDecoder"') == 1
    for term in TERMS:
        assert page.count(f'id="term-{term.key}"') == 1, term.key
    assert page.count('class="dec-group"') == 7
    for other in LESSONS:
        if other.slug != "benchmarks":
            assert "metricDecoder" not in learn_lesson_html(other)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_scoreboard_date_and_configuration_formatting_behave_correctly_in_a_real_js_engine() -> (
    None
):
    page = scoreboard_page_html()

    def source(name: str) -> str:
        match = re.search(rf"  function {name}\(.*?\n  \}}\n", page, re.S)
        assert match is not None, name
        return match.group(0)

    script = textwrap.dedent(
        """
        const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
        %s
        %s
        const out = {
          when: fmtWhen("2026-06-17T11:43:43.726573+00:00"),
          single: fmtWhen("2026-01-05T09:07:00+00:00"),
          bad: fmtWhen("not a date"),
          none: fmtWhen(null),
          fields: configFields({
            generator: "qwen3-8b-awq", embedder: "bge-m3",
            reranker: "bge-reranker-v2-m3", grader: "judge_faithfulness"
          }, "benchmark-golden"),
          empty: configFields(null, null),
          escaped: configFields({generator: "<b>x</b>"}, null),
        };
        console.log(JSON.stringify(out));
        """
    ) % (source("fmtWhen"), source("configFields"))
    run = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    out = json.loads(run.stdout)
    assert out["when"] == "17 Jun 2026, 11:43 UTC" and out["single"] == "5 Jan 2026, 09:07 UTC"
    assert out["bad"] == "not a date" and out["none"] == "—" and out["empty"] == ""
    labels = re.findall(r"<i>([^<]+)</i>", out["fields"])
    assert labels == ["Generator", "Embedder", "Reranker", "Judge", "Dataset"]
    assert "qwen3-8b-awq" in out["fields"] and "benchmark-golden" in out["fields"]
    assert "<b>x</b>" not in out["escaped"] and "&lt;b>" in out["escaped"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_ab_result_gets_a_plain_one_sentence_summary_in_a_real_js_engine() -> None:
    match = re.search(r"  function IW_plainVerdict\(.*?\n  \}\n", REPORT_JS, re.S)
    assert match is not None
    script = (
        'const IW_esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");\n'
        + match.group(0)
        + textwrap.dedent(
            """
            const row = (metric, delta, significant) => ({metric, delta, significant});
            const out = {
              split: IW_plainVerdict({
                winner: null,
                rows: [row("ndcg_at_5", 0.0, false), row("faithfulness", -0.081, false)]
              }),
              win: IW_plainVerdict({
                winner: "B",
                rows: [row("ndcg_at_5", 0.06, true), row("faithfulness", 0.05, true)]
              }),
              missing: IW_plainVerdict({winner: null, rows: [row("faithfulness", 0.2, true)]}),
              empty: IW_plainVerdict({}),
            };
            console.log(JSON.stringify(out));
            """
        )
    )
    out = json.loads(
        subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    )
    assert out["split"] == (
        '<div class="iw-intro">In short. Search quality (nDCG@5): B is about the same as A. '
        "Honesty (faithfulness): B is 0.081 lower than A (not significant, so it may be luck). "
        "There is no clear winner: neither run is better on both.</div>"
    )
    assert "B is 0.060 higher than A (a real difference)" in out["win"]
    assert out["win"].endswith("Run B wins: it is better on both.</div>")
    assert "Search quality (nDCG@5): not measured in both runs" in out["missing"]
    assert (
        "not measured in both runs. Honesty (faithfulness): not measured in both runs"
        in out["empty"]
    )


def test_the_underpowered_warning_carries_its_own_explanation() -> None:
    assert REPORT_JS.count('IW_tip("underpowered")') == 1
    # The tip sits in the warning itself, before its "read this with care" sentence.
    assert REPORT_JS.index('IW_tip("underpowered")') < REPORT_JS.index("Read this with care")
    assert plain_map()["underpowered"]["name"] == "Underpowered"
