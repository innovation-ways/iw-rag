"""Learn section: content integrity, rendering, routes and the shared console nav."""

from __future__ import annotations

import html
import re

import pytest
from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator
from iw_rag.api.compare_page import compare_page_html
from iw_rag.api.config_page import config_page_html
from iw_rag.api.console_chat_page import console_chat_page_html
from iw_rag.api.console_upload_page import console_upload_page_html
from iw_rag.api.learn_page import learn_glossary_html, learn_hub_html, learn_lesson_html
from iw_rag.api.results_page import results_page_html
from iw_rag.api.scoreboard_page import scoreboard_page_html
from iw_rag.api.system_page import system_page_html
from iw_rag.learn.glossary import GLOSSARY
from iw_rag.learn.lessons import LESSONS, get_lesson

_PAGES = {
    "learn": learn_hub_html,
    "chat": console_chat_page_html,
    "upload": console_upload_page_html,
    "config": config_page_html,
    "results": results_page_html,
    "compare": compare_page_html,
    "scoreboard": scoreboard_page_html,
    "system": system_page_html,
}


def _client() -> TestClient:
    auth = StaticTokenAuthenticator.from_config({"u": ["u"]}, operator_tokens=[])
    return TestClient(create_app(authenticator=auth), follow_redirects=False)


def test_lessons_are_complete_and_slugs_unique() -> None:
    slugs = [lesson.slug for lesson in LESSONS]
    assert len(slugs) == len(set(slugs)) == 16
    for lesson in LESSONS:
        assert re.fullmatch(r"[a-z-]+", lesson.slug)
        assert lesson.gains and lesson.losses, f"{lesson.slug} must show both sides"
        assert lesson.what and lesson.how and lesson.why


def test_get_lesson_unknown_is_none() -> None:
    assert get_lesson("nope") is None
    assert get_lesson("chunking") is not None


def test_hub_links_every_lesson_in_pipeline_order() -> None:
    page = learn_hub_html()
    cards = [h for h in re.findall(r'class="step-card" href="/console/learn/([a-z-]+)"', page)]
    assert cards == [lesson.slug for lesson in LESSONS]
    # The answer-path flow links every answer-phase step, in loop order.
    flow = re.findall(r'data-step="([a-z-]+)"', page)
    assert flow == [lesson.slug for lesson in LESSONS if lesson.phase == "answer"]
    assert 'href="/console/learn/glossary"' in page


def test_lesson_page_shows_gain_and_cost_and_links_neighbours() -> None:
    lesson = LESSONS[1]
    page = learn_lesson_html(lesson)
    assert "What you gain" in page and "What it costs" in page
    assert f'href="/console/learn/{LESSONS[0].slug}"' in page
    assert f'href="/console/learn/{LESSONS[2].slug}"' in page


def test_content_is_html_escaped() -> None:
    page = learn_glossary_html()
    assert all(html.escape(term) in page for term, _ in GLOSSARY)
    assert "<script>alert" not in page


@pytest.mark.parametrize("name", sorted(_PAGES))
def test_every_console_page_has_the_shared_nav(name: str) -> None:
    page = _PAGES[name]()
    assert "__CONSOLE_NAV__" not in page
    assert page.count('id="consoleNav"') == 1
    for anchor in (
        "navLearn",
        "navChat",
        "navUpload",
        "navConfig",
        "navBenchmark",
        "navResults",
        "navCompare",
        "navScoreboard",
        "navSystem",
    ):
        assert f'id="{anchor}"' in page, anchor
    # Learn comes first; Use, Measure and Operate follow, all visible.
    order = [page.index(f'id="{a}"') for a in ("navLearn", "navChat", "navBenchmark", "navConfig")]
    assert order == sorted(order)


def test_learn_pages_load_no_external_assets() -> None:
    for page in (learn_hub_html(), learn_lesson_html(LESSONS[0]), learn_glossary_html()):
        assert "<script src" not in page and "https://" not in page.replace("http://www.w3.org", "")


def test_routes() -> None:
    client = _client()
    root = client.get("/console")
    assert root.status_code == 307 and root.headers["location"] == "/console/learn"
    assert client.get("/console/learn").status_code == 200
    assert client.get("/console/learn/glossary").status_code == 200
    assert client.get("/console/learn/chunking").status_code == 200
    assert client.get("/console/learn/not-a-step").status_code == 404


def test_lesson_code_refs_point_at_tracked_paths() -> None:
    # docs/ is gitignored, so a public reader could not follow a reference into it.
    for lesson in LESSONS:
        assert not any(ref.startswith("docs/") for ref in lesson.code), lesson.slug


def test_chunking_lesson_embeds_the_lab_and_other_lessons_do_not() -> None:
    chunking = learn_lesson_html(next(x for x in LESSONS if x.slug == "chunking"))
    assert 'id="chunkLab"' in chunking and "/learn/lab/chunking" in chunking
    assert 'id="labLeaf"' in chunking
    parsing = learn_lesson_html(next(x for x in LESSONS if x.slug == "parsing"))
    assert "chunkLab" not in parsing


def test_lab_endpoint_requires_sign_in_and_runs_the_chunker() -> None:
    client = _client()
    assert client.get("/learn/lab/chunking?sample=with-headings&leaf=500").status_code == 401
    ok = client.get(
        "/learn/lab/chunking?sample=with-headings&leaf=500",
        headers={"Authorization": "Bearer u"},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["leaf_chars"] == 500 and body["leaves"] and body["probes"][0]["status"] == "split"


def test_lab_endpoint_validates_its_input() -> None:
    client = _client()
    auth = {"Authorization": "Bearer u"}
    assert client.get("/learn/lab/chunking?sample=nope&leaf=500", headers=auth).status_code == 404
    assert (
        client.get("/learn/lab/chunking?sample=with-headings&leaf=5", headers=auth).status_code
        == 422
    )
    assert (
        client.get("/learn/lab/chunking?sample=with-headings&leaf=99999", headers=auth).status_code
        == 422
    )


def test_search_lesson_embeds_the_hybrid_lab_and_not_the_chunking_lab() -> None:
    page = learn_lesson_html(next(x for x in LESSONS if x.slug == "search"))
    assert 'id="hybridLab"' in page and "/learn/lab/hybrid" in page
    assert "chunkLab" not in page
    assert "hybridLab" not in learn_lesson_html(next(x for x in LESSONS if x.slug == "chunking"))


def test_hybrid_lab_endpoint_requires_sign_in_and_validates_input() -> None:
    client = _client()
    url = "/learn/lab/hybrid?query=paraphrase&principal=analyst"
    assert client.get(url).status_code == 401
    auth = {"Authorization": "Bearer u"}
    ok = client.get(url + "&w_dense=1&w_sparse=0.5&k=30", headers=auth)
    assert ok.status_code == 200
    body = ok.json()
    assert body["rrf_k"] == 30 and body["gold"]["chunk_id"] == "f07" and body["fused"]
    assert (
        client.get("/learn/lab/hybrid?query=nope&principal=analyst", headers=auth).status_code
        == 404
    )
    assert (
        client.get("/learn/lab/hybrid?query=paraphrase&principal=zed", headers=auth).status_code
        == 404
    )
    assert client.get(url + "&w_dense=9", headers=auth).status_code == 422
    assert client.get(url + "&k=0", headers=auth).status_code == 422


def test_hybrid_lab_script_distinguishes_an_arm_that_is_off_from_one_that_found_nothing() -> None:
    page = learn_lesson_html(next(x for x in LESSONS if x.slug == "search"))
    assert page.count('"<b>off</b>') + page.count("<b>off</b>") == 1
    assert "d.weights.sparse === 0)" in page and "not found" in page


def test_fail_closed_lesson_embeds_the_guards_lab() -> None:
    page = learn_lesson_html(next(x for x in LESSONS if x.slug == "fail-closed"))
    assert 'id="guardsLab"' in page and "/learn/lab/guards" in page
    assert page.count('class="gd-card"') == 7
    assert "chunkLab" not in page and "hybridLab" not in page


def test_guards_lab_endpoint_requires_sign_in_and_validates_input() -> None:
    client = _client()
    auth = {"Authorization": "Bearer u"}
    assert client.get("/learn/lab/guards?scenario=answered").status_code == 401
    ok = client.get("/learn/lab/guards?scenario=jailbreak", headers=auth)
    assert ok.status_code == 200
    body = ok.json()
    assert [s["node"] for s in body["steps"]] == ["input_guard", "reject", "audit"]
    assert body["final"]["disposition"] == "refused"
    assert client.get("/learn/lab/guards?scenario=nope", headers=auth).status_code == 404


def test_guard_lessons_point_at_the_guards_lab() -> None:
    for slug in ("input-guard", "sanitize"):
        lesson = next(x for x in LESSONS if x.slug == slug)
        assert any(href == "/console/learn/fail-closed" for _, href in lesson.try_it), slug


def test_rerank_lesson_embeds_the_rerank_lab_only() -> None:
    page = learn_lesson_html(next(x for x in LESSONS if x.slug == "rerank"))
    assert 'id="rerankLab"' in page and "/learn/lab/rerank" in page
    assert "hybridLab" not in page and "chunkLab" not in page and "guardsLab" not in page


def test_rerank_lab_endpoint_requires_sign_in_and_validates_input() -> None:
    client = _client()
    auth = {"Authorization": "Bearer u"}
    url = "/learn/lab/rerank?query=late-claim"
    assert client.get(url).status_code == 401
    ok = client.get(url + "&top_k=3&reranker=off&expansion=leaf&budget=1500", headers=auth)
    assert ok.status_code == 200
    body = ok.json()
    assert body["settings"] == {
        "top_k": 3,
        "reranker_on": False,
        "expansion": "leaf",
        "budget_chars": 1500,
        "sections": True,
    }
    assert len(body["selected"]) == 3
    assert client.get("/learn/lab/rerank?query=nope", headers=auth).status_code == 404
    off = client.get(url + "&sections=off", headers=auth).json()
    assert off["settings"]["sections"] is False and off["facts"]["sections_set_aside"]
    for bad in (
        "&top_k=0",
        "&top_k=11",
        "&reranker=maybe",
        "&expansion=window",
        "&budget=100",
        "&sections=maybe",
    ):
        assert client.get(url + bad, headers=auth).status_code == 422, bad
