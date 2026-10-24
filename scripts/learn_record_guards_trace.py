"""Record the Learn section's guards lab by running the PRODUCTION LangGraph graph.

For each scenario in ``learn/guards_scenarios.py`` this builds the real compiled graph
with the guards switched on and streams every node it visits.  What is real and what is
scripted is recorded in the output:

* real: the graph, every routing function, the retry budget, the sanitising pipeline,
  the answer assembler (citation parsing), the output guard and the audit-free terminals;
  and the Prompt Guard 2 scores, asked of the live classifier (``injguard``).
* scripted: the language-model steps (router, relevance grader, generator, self-check,
  query rewrite), so each scenario is reproducible and a failure can be forced.
* not run: Granite Guardian (its container is off) and Presidio PII redaction.

Writes ``src/iw_rag/learn/traces/guards.json``.  Needs the ``iw-rag-api-1`` container
(to reach the classifier on the Docker network)::

    uv run python scripts/learn_record_guards_trace.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any

os.environ["IW_RAG__GUARDRAILS__ENABLED"] = "true"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import iw_rag.agent.nodes as nodes_mod  # noqa: E402
import iw_rag.guardrails.input_guard as input_guard_mod  # noqa: E402
import iw_rag.guardrails.output_guard as output_guard_mod  # noqa: E402
import iw_rag.guardrails.sanitize as sanitize_mod  # noqa: E402
import iw_rag.settings as settings_mod  # noqa: E402
from iw_rag.agent.graph import RECURSION_LIMIT, build_graph  # noqa: E402
from iw_rag.domain.acl import AccessContext  # noqa: E402
from iw_rag.domain.models import RetrievalResult, RetrievedChunk  # noqa: E402
from iw_rag.generation.answer import AnswerAssembler  # noqa: E402
from iw_rag.generation.llm import GenerationParams, LlmResponse  # noqa: E402
from iw_rag.generation.prompts import CitationPrompt  # noqa: E402
from iw_rag.learn.guards_scenarios import SCENARIOS, Scenario, all_texts  # noqa: E402
from iw_rag.settings import AgentSettings  # noqa: E402

OUT = ROOT / "src/iw_rag/learn/traces/guards.json"
CTX = AccessContext(principal_id="lab", principal_groups=("finance", "public"))
_DOC = re.compile(r"\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\Z)", re.S)
_MARKER = re.compile("⁣[0-9a-f]{12}")

_PG2_SCRIPT = """
import json, sys, httpx
out = {}
for t in json.load(sys.stdin):
    r = httpx.post("http://injguard:8003/classify", json={"inputs": t}, timeout=15)
    r.raise_for_status()
    out[t] = next(i["score"] for i in r.json() if i["label"].upper() == "JAILBREAK")
print(json.dumps(out))
"""


def _real_pg2_scores(texts: list[str]) -> dict[str, float]:
    """Ask the live Prompt Guard 2 service (reachable inside the api container)."""
    proc = subprocess.run(  # noqa: S603 - fixed argv
        ["docker", "exec", "-i", "iw-rag-api-1", "python", "-c", _PG2_SCRIPT],  # noqa: S607
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        check=True,
    )
    scores: dict[str, float] = json.loads(proc.stdout)
    return scores


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage: dict[str, int] = {}
        self.tool_calls: list[Any] = []
        self.finish_reason = "stop"


class ScriptedGraderLlm:
    """Router, relevance grader, self-check and rewrite: all answered from the script."""

    def __init__(self, scenario: Scenario) -> None:
        self._s = scenario
        self._gen_checks = 0
        self._ans_checks = 0
        self._rewrites = 0

    async def complete(self, messages: Any, params: Any) -> Any:  # noqa: ARG002
        text = self._s.rewrites[min(self._rewrites, len(self._s.rewrites) - 1)]
        self._rewrites += 1
        return _Resp(text)

    async def complete_structured(self, messages: Any, schema: Any, params: Any) -> Any:  # noqa: ARG002
        required = schema.get("required", []) if isinstance(schema, dict) else []
        if "route" in required:
            return _Resp('{"route": "retrieve", "query_type": "factual"}')
        if "grades" in required:
            docs = [m.group(2) for m in _DOC.finditer(messages[-1]["content"])]
            marker = self._s.relevant_marker
            grades = [{"relevant": bool(marker) and marker in d} for d in docs]
            return _Resp(json.dumps({"grades": grades}))
        if "supported" in required:
            seq = self._s.supported
            value = seq[min(self._gen_checks, len(seq) - 1)]
            self._gen_checks += 1
            return _Resp(json.dumps({"supported": value}))
        if "answers" in required:
            seq = self._s.answers
            value = seq[min(self._ans_checks, len(seq) - 1)]
            self._ans_checks += 1
            return _Resp(json.dumps({"answers": value}))
        return _Resp("{}")


class ScriptedGeneratorLlm:
    """The answer generator: returns the scenario's scripted draft."""

    def __init__(self, scenario: Scenario) -> None:
        self._s = scenario
        self._calls = 0

    async def complete(self, messages: Any, params: Any) -> LlmResponse:  # noqa: ARG002
        drafts = self._s.drafts or ("",)
        text = drafts[min(self._calls, len(drafts) - 1)]
        self._calls += 1
        return LlmResponse(text=text, usage={}, tool_calls=[], finish_reason="stop")


class ScriptedSpine:
    """Returns the scenario's chunks for each retrieval pass."""

    def __init__(self, scenario: Scenario) -> None:
        self._s = scenario
        self._calls = 0

    async def retrieve(self, query: str, ctx: Any) -> RetrievalResult:  # noqa: ARG002
        passes = self._s.retrievals
        chosen = passes[min(self._calls, len(passes) - 1)] if passes else ()
        self._calls += 1
        chunks = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=f"doc-{c.document}",
                content=c.text,
                score=1.0 / (i + 1),
                source_uri=f"learn://{c.document}",
            )
            for i, c in enumerate(chosen)
        ]
        return RetrievalResult(
            chunks=chunks,
            retriever="spine",
            acl_applied=True,
            query=query,
            candidate_count=len(chunks),
        )


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _summarise(update: dict[str, Any]) -> dict[str, Any]:
    """Keep what a reader needs from one node's state update."""
    out: dict[str, Any] = {}
    for key in (
        "input_decision",
        "route",
        "query_type",
        "doc_verdict",
        "gen_verdict",
        "retries",
        "output_guard_decision",
        "disposition",
        "answer",
        "refusal",
        "draft",
        "question",
    ):
        if key in update and update[key] is not None:
            out[key] = update[key]
    for key in ("chunks", "graded"):
        if key in update:
            out[key] = [c.chunk_id for c in update[key]]
    if update.get("chunks"):
        out["first_chunk_marked"] = _MARKER.sub("◆", update["chunks"][0].content)
    if "citations" in update:
        out["citations"] = [
            {"marker": c.marker, "chunk_id": c.chunk_id, "quote": _MARKER.sub("", c.quote)}
            for c in update["citations"]
        ]
    if "verdicts" in update:
        out["verdicts"] = [_plain(v) for v in update["verdicts"]]
    diag = update.get("diagnostics") or {}
    if "sanitize_dropped_ids" in diag:
        out["dropped_ids"] = list(diag["sanitize_dropped_ids"])
    return out


async def _run(scenario: Scenario, pg2: dict[str, float]) -> dict[str, Any]:
    async def pg_score(text: str) -> float:
        return pg2[text]

    async def granite(text: str) -> float:  # noqa: ARG001
        return 0.0  # Granite Guardian is not running: recorded as "not run", scores nothing

    async def no_pii(text: str, ctx: Any, *, allow_text: str | None = None) -> str:  # noqa: ARG001
        return text

    patches = [
        (input_guard_mod, "prompt_guard_score", pg_score),
        (input_guard_mod, "granite_harm_score", granite),
        (sanitize_mod, "prompt_guard_score", pg_score),
        (sanitize_mod, "_presidio_redact", no_pii),
        (output_guard_mod, "_presidio_redact", no_pii),
        (nodes_mod, "_presidio_redact", no_pii),
    ]
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    for mod, name, fn in patches:
        setattr(mod, name, fn)
    settings_mod.get_settings.cache_clear()
    try:
        assembler = AnswerAssembler(
            ScriptedGeneratorLlm(scenario), CitationPrompt(), GenerationParams(model="scripted")
        )
        graph = build_graph(
            ScriptedSpine(scenario),
            assembler,
            ScriptedGraderLlm(scenario),
            settings=AgentSettings(),
        )
        init = {
            "question": scenario.question,
            "original_question": scenario.question,
            "access_context": CTX,
            "retries": 0,
            "max_retries": AgentSettings().max_retries,
            "allow_ungrounded": False,
            "history_chunks": [],
            "verdicts": [],
        }
        steps: list[dict[str, Any]] = []
        final: dict[str, Any] = {}
        config = {
            "configurable": {"thread_id": scenario.scenario_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        async for event in graph.astream(init, config=config, stream_mode="updates"):
            for node, update in event.items():
                summary = _summarise(update or {})
                steps.append({"node": node, "update": summary})
                final.update(
                    {k: v for k, v in summary.items() if k in ("disposition", "answer", "refusal")}
                )
        return {"steps": steps, "final": final}
    finally:
        for mod, name, fn in saved:
            setattr(mod, name, fn)
        settings_mod.get_settings.cache_clear()


async def _main() -> None:
    texts = all_texts()
    pg2 = await asyncio.to_thread(_real_pg2_scores, texts)
    scenarios = []
    for scenario in SCENARIOS:
        result = await _run(scenario, pg2)
        scenarios.append({"id": scenario.scenario_id, **result})
        print(
            f"{scenario.scenario_id:12} -> {result['final'].get('disposition')}  "
            f"({' > '.join(s['node'] for s in result['steps'])})"
        )
    sha = (
        await asyncio.to_thread(
            subprocess.run,
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
    ).stdout.strip()
    trace = {
        "recorded": date.today().isoformat(),
        "git_commit": sha,
        "pg2_threshold": settings_mod.get_settings().guardrails.input.threshold,
        "max_retries": AgentSettings().max_retries,
        "pg2_scores": pg2,
        "provenance": {
            "real": [
                "the production LangGraph graph and every routing function",
                "the retry budget",
                "the sanitising pipeline and datamarking",
                "the answer assembler (citation parsing)",
                "the output guard",
                "Prompt Guard 2 scores, from the live classifier",
            ],
            "scripted": [
                "the router, relevance grader, generator, self-check and query rewrite "
                "(language-model steps)"
            ],
            "not_run": ["Granite Guardian (its container is off)", "Presidio PII redaction"],
        },
        "scenarios": scenarios,
    }
    OUT.write_text(json.dumps(trace, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(_main())
