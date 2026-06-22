"""Production composition root for the F-00095 Benchmark Studio runner.

``create_app`` leaves ``benchmark_runner`` injectable; until it is wired the
``POST /benchmark/run`` route refuses with a typed error. This module builds
the live :class:`~iw_rag.eval.benchmark.runner.BenchmarkRunner` from the
already-constructed app collaborators (the live retrieval+generation pipeline,
the shared database, the F-00094 swap-controller sidecar) so the run route is
reachable end-to-end on the single-GPU deployment.

The runner's collaborator Protocols do not all line up 1:1 with the concrete
clients, so this module supplies thin adapters:

* :class:`_SwapControllerAdapter` — the runner expects
  ``activate(model_config: dict)`` + ``wait_for_ready(*, timeout_s)``; the real
  :class:`~iw_rag.serving.swap_controller_client.SwapControllerClient` exposes
  ``activate(model_id: str)`` (which already blocks until ready) + a
  ``get_status()`` poll. The adapter maps ``model_config["generator"]`` to the
  catalog id and translates a non-ready terminal state to
  :class:`~iw_rag.eval.benchmark.runner.SwapTimeout`.
* :class:`_DatabaseGoldenSetLoader` — resolves a dataset uuid to the golden
  query list via :class:`~iw_rag.eval.repository.EvalRepository`.
* :class:`_SpineAssemblerQueryRunner` — runs one query through the live spine +
  assembler and returns the ``(RetrievalResult, GeneratedAnswer, contexts)``
  triple the runner records.
* :class:`_SessionBoundReportRepository` — opens a session per
  ``create_report`` call (the F-00093 repository does not own a session; the
  :meth:`Database.session` context manager commits on clean exit).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import NotFoundError

logger = logging.getLogger(__name__)

# Cold-swap of the 32B AWQ generator costs ~70-110 s (R-00123 §5); the HTTP
# call to the sidecar's blocking POST /catalog/activate must not time out
# before that. 600 s leaves generous headroom over the 300 s health gate.
_SWAP_CLIENT_TIMEOUT_S = 600.0


class _SwapControllerAdapter:
    """Adapt :class:`SwapControllerClient` to the runner's swap Protocol."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def activate(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Activate the generator named by ``model_config["generator"]``.

        The sidecar's ``POST /catalog/activate`` blocks until the model is
        ready / failed / timed out, so this returns the post-activation status.
        """
        model_id = str(model_config.get("generator") or "")
        if not model_id:
            raise NotFoundError("model_config.generator is required to activate a model")
        result = await self._client.activate(model_id)
        return dict(result) if isinstance(result, dict) else {}

    async def wait_for_ready(self, *, timeout_s: float) -> dict[str, Any]:
        """Poll the sidecar status until ``ready`` (raise ``SwapTimeout``).

        ``activate`` already blocks until readiness, so the first poll is
        normally already ``ready``; the loop is a defensive backstop.
        """
        from iw_rag.eval.benchmark.runner import SwapTimeout

        loop = asyncio.get_event_loop()
        deadline = loop.time() + float(timeout_s)
        last: dict[str, Any] = {}
        while True:
            status = await self._client.get_status()
            last = dict(status) if isinstance(status, dict) else {}
            state = str(last.get("state") or "").lower()
            if state == "ready":
                return last
            if state == "failed":
                raise SwapTimeout(f"swap-controller reported failed: {last.get('failure_reason')}")
            if loop.time() >= deadline:
                raise SwapTimeout(
                    f"swap-controller never reported ready within {timeout_s}s "
                    f"(last state={state!r})"
                )
            await asyncio.sleep(1.0)


class _DatabaseGoldenSetLoader:
    """Load golden queries for a dataset uuid from ``eval_queries``."""

    def __init__(self, database: Any) -> None:
        self._database = database

    async def load(self, dataset_id: str) -> list[Any]:
        from iw_rag.eval.repository import EvalRepository

        async with self._database.session() as session:
            return await EvalRepository(session).get_queries(dataset_id)


class _SpineAssemblerQueryRunner:
    """Run one query through the live spine + a per-run assembler.

    The live assembler is configured with the static ``settings.generation``
    model name, but under the benchmark overlay the generator endpoint is the
    llama-swap proxy, which routes by the **active catalog id** (the swapped
    model changes per run). So the generation request must name the active
    model — otherwise llama-swap cannot route it (404). We rebuild the
    assembler for the active model (read from the swap-controller status) at
    the start of each run, reusing the live llm / prompt / cite settings.
    """

    def __init__(self, spine: Any, base_assembler: Any, swap_client: Any) -> None:
        self._spine = spine
        self._base = base_assembler
        self._swap_client = swap_client
        self._assembler: Any = None

    async def _assembler_for_active_model(self) -> Any:
        from dataclasses import replace

        from iw_rag.generation.answer import AnswerAssembler

        active = None
        try:
            status = await self._swap_client.get_status()
            if isinstance(status, dict):
                active = status.get("active_model_id")
        except Exception:
            active = None
        if not active:
            return self._base
        params = replace(self._base._params, model=str(active))
        return AnswerAssembler(
            llm=self._base._llm,
            prompt=self._base._prompt,
            params=params,
            cite_required=self._base._cite_required,
        )

    async def run_query(
        self, question: str, ctx: AccessContext, i: int, total: int
    ) -> tuple[Any, Any, list[str]]:
        # Refresh the active-model assembler at the start of each run (i == 0)
        # or if it has not been built yet.
        if self._assembler is None or i == 0:
            self._assembler = await self._assembler_for_active_model()
        result = await self._spine.retrieve(question, ctx)
        answer = await self._assembler.generate(question, result)
        contexts = [chunk.content for chunk in result.chunks]
        return result, answer, contexts


class _SessionBoundReportRepository:
    """Open a session per ``create_report`` (the F-00093 repo holds none).

    ``benchmark_reports.run_id`` is a FK to ``eval_runs.id``, but the runner
    mints a synthetic ``run_id`` (its docstring notes "the S02 layer will
    supply the real eval_runs id"). This composition root IS that S02 layer:
    it inserts the backing ``eval_runs`` row in the same transaction so the FK
    is satisfied, then writes the report.
    """

    def __init__(self, database: Any) -> None:
        self._database = database

    async def create_report(self, **kwargs: Any) -> str:
        import uuid

        from sqlalchemy import insert

        from iw_rag.eval.benchmark.repository import BenchmarkReportRepository
        from iw_rag.storage import schema

        run_id = kwargs["run_id"]
        dataset_id = kwargs.get("dataset_id")
        model_config = kwargs.get("model_config") or {}
        async with self._database.session() as session:
            await session.execute(
                insert(schema.eval_runs).values(
                    id=uuid.UUID(str(run_id)),
                    dataset_id=uuid.UUID(str(dataset_id)) if dataset_id else None,
                    experiment="benchmark",
                    git_sha=kwargs.get("git_sha"),
                    config=dict(model_config),
                )
            )
            report_id = await BenchmarkReportRepository(session).create_report(**kwargs)
            # Database.session() does NOT auto-commit (it rolls back uncommitted
            # work on close); persist explicitly like the other write paths.
            await session.commit()
            return report_id


def _build_vllm_fetcher(swap_client: Any) -> Any:
    """Return an async fetcher of the ACTIVE generator's vLLM ``/metrics``.

    llama-swap does NOT expose the upstream's Prometheus metrics (``/metrics``
    on the proxy is 404), so we hit the active generator container directly.
    Its name follows the ``config/llama-swap.yaml`` convention
    ``vllm-<catalog-id>`` and the active catalog id comes from the
    swap-controller status. Degrades to an empty result on any failure so an
    unreachable endpoint never fails the run.
    """

    async def _fetch() -> str:
        try:
            status = await swap_client.get_status()
            active = status.get("active_model_id") if isinstance(status, dict) else None
            if not active:
                return ""
            url = f"http://vllm-{active}:8001/metrics"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.text
        except Exception:
            return ""

    return _fetch


def _build_ragas_runner(settings: Any, generator: Any, judge_entry: Any) -> Any:
    """Construct a RagasRunner wired to the local judge LLM + BGE-M3 embedder.

    Ragas computes the reference-required T2/T3 metrics (context_precision,
    context_recall, factual_correctness, noise_sensitivity) via an LLM judge +
    embeddings.  Both are LOCAL, OpenAI-compatible endpoints:

    * the judge LLM is the general judge (Selene) — pointed at whatever
      ``judge_entry.endpoint`` resolves to (the llama-swap proxy in benchmark
      mode), a DIFFERENT family from the Qwen generator (ADR-013); and
    * the embedder is BGE-M3 on the infinity server.

    Returns ``None`` (graceful) when the eval-judges deps (ragas + langchain)
    are not installed in this image, or when construction fails — the run then
    still produces T1 + faithfulness, just without the Ragas metrics.
    """
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from pydantic import SecretStr

        from iw_rag.eval.ragas_runner import RagasRunner

        # Local OpenAI-compatible endpoints need no real key; a placeholder
        # SecretStr satisfies the client (and the langchain type signature).
        local_key = SecretStr("local")
        embedder_entry = settings.models.get("embedder")
        judge_llm = ChatOpenAI(
            base_url=judge_entry.endpoint,
            model=judge_entry.served_name,
            api_key=local_key,
            temperature=0.0,
            timeout=300,
        )
        judge_embed = OpenAIEmbeddings(
            base_url=embedder_entry.endpoint,
            model=embedder_entry.served_name,
            api_key=local_key,
            check_embedding_ctx_length=False,  # infinity is not tiktoken-based
        )
        return RagasRunner(
            judge_llm=judge_llm,
            embed=judge_embed,
            generator_model=generator.served_name,
            judge_model=judge_entry.served_name,
        )
    except Exception:  # noqa: BLE001 — ragas/langchain optional (eval-judges)
        logger.warning(
            "Ragas not wired (eval-judges deps missing or construction failed) — "
            "context_precision/recall/factual_correctness/noise_sensitivity will be empty.",
            exc_info=True,
        )
        return None


def build_benchmark_runner(settings: Any, database: Any, query_service: Any) -> Any:
    """Compose the live :class:`BenchmarkRunner` (F-00095 S02 composition root).

    Args:
        settings: The validated ``Settings``.
        database: The shared ``Database`` handle (from ``_build_query_service``).
        query_service: The live ``AgentQueryService`` (exposes ``_spine`` and
            ``_assembler``, the Phase-0 retrieval+generation pair).

    Returns:
        A fully-wired ``BenchmarkRunner``.
    """
    from iw_rag.eval.benchmark.metrics_engine import MetricsEngine
    from iw_rag.eval.benchmark.perf import GpuSampler, VllmMetricsScraper
    from iw_rag.eval.benchmark.runner import BenchmarkRunner
    from iw_rag.eval.judges import FaithfulnessJudge, GradingJudge
    from iw_rag.generation.llm import VllmClient
    from iw_rag.serving.swap_controller_client import SwapControllerClient

    generator = settings.models.get(settings.generation.model)
    faithfulness_entry = settings.models.get(settings.eval.judges.faithfulness_model)
    grading_entry = settings.models.get(settings.eval.judges.general_model)

    # Judges run against the offline judge endpoint (a DIFFERENT family from the
    # generator — self-preference de-bias, ADR-013). enable_thinking=False keeps
    # the judge output a bare verdict. When the endpoint is unreachable at
    # compute time the MetricsEngine records the T2/T3 metrics as None
    # (graceful) — the run still persists T1 retrieval + perf + VRAM.
    faithfulness_judge = FaithfulnessJudge(
        judge_client=VllmClient(faithfulness_entry.endpoint, enable_thinking=False),
        generator_model=generator.served_name,
        judge_model=faithfulness_entry.served_name,
    )
    grading_judge = GradingJudge(
        judge_client=VllmClient(grading_entry.endpoint, enable_thinking=False),
        generator_model=generator.served_name,
        judge_model=grading_entry.served_name,
    )
    # Ragas (context_precision/recall + factual_correctness + noise_sensitivity)
    # lives in the optional eval-judges group.  Wire it against the LOCAL
    # general judge (Selene — a DIFFERENT family from the Qwen generator,
    # self-preference de-bias) + the BGE-M3 embedder.  Degrades to None when
    # the deps are absent so a lean api image still runs (faithfulness via the
    # FaithfulnessJudge only in that case).
    ragas_runner = _build_ragas_runner(settings, generator, grading_entry)
    metrics_engine = MetricsEngine(
        faithfulness_judge=faithfulness_judge,
        grading_judge=grading_judge,
        ragas_runner=ragas_runner,
    )

    swap_client = SwapControllerClient(
        base_url=settings.serving.swap_controller_base_url,
        timeout_s=_SWAP_CLIENT_TIMEOUT_S,
    )

    return BenchmarkRunner(
        swap_controller=_SwapControllerAdapter(swap_client),
        metrics_engine=metrics_engine,
        repository=_SessionBoundReportRepository(database),
        gpu_sampler=GpuSampler(),
        vllm_scraper=VllmMetricsScraper(fetcher=_build_vllm_fetcher(swap_client)),
        golden_loader=_DatabaseGoldenSetLoader(database),
        query_runner=_SpineAssemblerQueryRunner(
            query_service._spine, query_service._assembler, swap_client
        ),
        conclusion_fn=build_conclusion_fn(settings, database),
    )


def build_conclusion_fn(settings: Any, database: Any) -> Any:
    """Build the run-time conclusion writer closure (or ``None``).

    Returns an ``async (this_row, dataset_id) -> str | None`` callable that
    reads the same-dataset comparable reports, then has the LOCAL general judge
    (Selene — already resident after the grading phase, so no extra swap) write
    a grounded conclusion.  ``None`` when the deps are unavailable; the run
    then persists without a conclusion.
    """
    try:
        from iw_rag.eval.benchmark.conclusion import ConclusionWriter, summarize_report
        from iw_rag.generation.llm import VllmClient

        judge_entry = settings.models.get(settings.eval.judges.general_model)
        writer = ConclusionWriter(
            chat_client=VllmClient(judge_entry.endpoint, enable_thinking=False),
            model=judge_entry.served_name,
        )
    except Exception:  # noqa: BLE001 — optional feature; degrade to None
        logger.warning("benchmark conclusion writer not wired", exc_info=True)
        return None

    async def _conclusion_fn(this_row: dict[str, Any], dataset_id: str) -> str | None:
        rows = await _read_same_dataset_reports(database, dataset_id)
        comparables = [summarize_report(r) for r in rows]
        return await writer.write(summarize_report(this_row), comparables)

    return _conclusion_fn


def build_compare_conclusion_fn(settings: Any) -> Any:
    """Build the N-way comparison conclusion writer closure (or ``None``).

    Returns an ``async (payload) -> str | None`` callable that has the
    LOCAL general judge (Selene) write a grounded conclusion + reasoning
    from an already-computed multi-compare payload (per-run means +
    baseline-relative deltas + significance).  Unlike
    :func:`build_conclusion_fn` it needs no database — every figure the
    LLM narrates is already in the payload.  ``None`` when the deps are
    unavailable; the Compare view + PDF then render without a conclusion.
    """
    try:
        from iw_rag.eval.benchmark.conclusion import CompareConclusionWriter
        from iw_rag.generation.llm import VllmClient

        judge_entry = settings.models.get(settings.eval.judges.general_model)
        writer = CompareConclusionWriter(
            chat_client=VllmClient(judge_entry.endpoint, enable_thinking=False),
            model=judge_entry.served_name,
        )
    except Exception:  # noqa: BLE001 — optional feature; degrade to None
        logger.warning("benchmark compare conclusion writer not wired", exc_info=True)
        return None

    async def _compare_conclusion_fn(payload: dict[str, Any]) -> str | None:
        return await writer.write(payload)

    return _compare_conclusion_fn


async def _read_same_dataset_reports(database: Any, dataset_id: str) -> list[dict[str, Any]]:
    """Read recent persisted reports for a dataset (the conclusion's comparables)."""
    import uuid as _uuid

    from sqlalchemy import select

    from iw_rag.storage import schema

    try:
        stmt = (
            select(schema.benchmark_reports)
            .where(schema.benchmark_reports.c.dataset_id == _uuid.UUID(str(dataset_id)))
            .order_by(schema.benchmark_reports.c.created_at.desc())
            .limit(12)
        )
        async with database.session() as session:
            return [dict(r) for r in (await session.execute(stmt)).mappings()]
    except Exception:  # noqa: BLE001 — comparables are best-effort
        logger.warning("benchmark comparables read failed", exc_info=True)
        return []


async def resolve_dataset_uuid(database: Any, dataset_id: str) -> str:
    """Resolve a dataset id-or-name to a real ``eval_datasets`` uuid.

    The console + runner default to the logical id ``"benchmark_golden"`` but
    the runner threads ``dataset_id`` straight into ``assemble_report`` →
    ``uuid.UUID(dataset_id)``, so it must be a real uuid. Accept either:

    * a uuid string (returned verbatim), or
    * a name — resolved to the matching ``eval_datasets`` row, tolerant of the
      ``_``/``-`` seed inconsistency (``benchmark_golden`` ↔ ``benchmark-golden``)
      and preferring the most-populated dataset on a name collision.

    Falls back to the single most-populated dataset when no name matches, so a
    run is never launched against an empty dataset.

    Raises:
        NotFoundError: When no dataset with any query exists at all.
    """
    import uuid

    from sqlalchemy import func, select

    from iw_rag.storage import schema

    try:
        uuid.UUID(dataset_id)
        return dataset_id
    except (ValueError, AttributeError, TypeError):
        pass

    name_variants = {
        dataset_id,
        dataset_id.replace("_", "-"),
        dataset_id.replace("-", "_"),
    }
    counts = (
        select(
            schema.eval_datasets.c.id,
            schema.eval_datasets.c.name,
            func.count(schema.eval_queries.c.id).label("n"),
        )
        .select_from(
            schema.eval_datasets.outerjoin(
                schema.eval_queries,
                schema.eval_queries.c.dataset_id == schema.eval_datasets.c.id,
            )
        )
        .group_by(schema.eval_datasets.c.id, schema.eval_datasets.c.name)
        .order_by(func.count(schema.eval_queries.c.id).desc())
    )
    async with database.session() as session:
        rows = (await session.execute(counts)).all()

    named = [r for r in rows if r.name in name_variants and r.n > 0]
    if named:
        return str(named[0].id)
    populated = [r for r in rows if r.n > 0]
    if populated:
        return str(populated[0].id)
    raise NotFoundError(f"no populated eval dataset found for {dataset_id!r}")
