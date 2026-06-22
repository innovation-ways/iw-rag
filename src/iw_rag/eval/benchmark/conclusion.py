"""Run-time benchmark conclusion — a grounded, local-LLM verdict per run.

The Operator Console shows, under each report, a short narrative conclusion:
*how did this configuration do, and how does it compare to prior runs on the
same dataset* — e.g. "for ~12% more VRAM you get ~8% higher faithfulness at
~5% slower end-to-end; a good pick when grounding is the priority."

The numbers in that sentence must be **real**, never hallucinated.  So this
module computes the comparison deltas deterministically (:func:`build_basis`)
and the local LLM only writes the prose around them (:class:`ConclusionWriter`
feeds the judge a tightly-scoped prompt carrying the measured figures).  It is
generated **during the run** (after scoring, before the row is inserted) and
stored in the immutable report's ``metrics`` blob under ``"conclusion"``.

Local-only (Invariant 1): the writer is the general judge model (Selene),
already resident on the GPU after the grading phase — so no extra swap.  The
module is **offline-only** (two-clocks rule) and degrades to ``None`` on any
failure so a benchmark run never fails because the conclusion could not be
written.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic summary + comparison (the LLM never invents these numbers)
# ---------------------------------------------------------------------------


def _f(v: Any) -> float | None:
    """Coerce to ``float`` or ``None``."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (f != f) else f  # drop NaN


def summarize_report(row: dict[str, Any]) -> dict[str, Any]:
    """Extract the headline numbers used by the conclusion + comparison.

    Accepts a ``benchmark_reports`` row dict (promoted columns + ``latency`` /
    ``resource`` JSONB).  Returns a flat dict of the figures the conclusion
    talks about — quality, speed, and GPU footprint — plus a config label.
    """
    cfg = row.get("model_config") or row.get("config_snapshot") or {}
    latency = row.get("latency") or {}
    resource = row.get("resource") or {}
    gpu = resource.get("gpu") or {}
    per_stage = latency.get("per_stage_ms") or {}
    return {
        "label": row.get("report_label") or cfg.get("generator") or "run",
        "generator": cfg.get("generator"),
        "faithfulness": _f(row.get("faithfulness_mean")),
        "ndcg_at_5": _f(row.get("ndcg_at_5")),
        "context_recall": _f(row.get("context_recall")),
        "factual_correctness": _f(row.get("factual_correctness")),
        # Speed: prefer the vLLM e2e p99; fall back to the per-query mean.
        "e2e_p99_s": _f(latency.get("e2e_p99_s")),
        "retrieve_generate_ms": _f(per_stage.get("retrieve_generate_ms")),
        # GPU footprint.
        "vram_peak_mb": _f(gpu.get("vram_peak_mb")),
        "vram_incremental_mb": _f(gpu.get("vram_incremental_mb")),
    }


def _pct(new: float | None, base: float | None) -> float | None:
    """Percent change from ``base`` to ``new`` (``None`` when not computable)."""
    if new is None or base is None or base == 0:
        return None
    return (new - base) / abs(base) * 100.0


def build_basis(this: dict[str, Any], comparables: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the deterministic comparison basis for the conclusion prompt.

    Picks the most relevant comparable (the run with the highest faithfulness
    among same-dataset runs with a *different* generator, else the closest by
    VRAM) and computes the signed percent deltas in quality, speed, and VRAM.
    These exact figures are what the LLM is told to narrate.

    Args:
        this: :func:`summarize_report` of the current run.
        comparables: :func:`summarize_report` of the other same-dataset runs.

    Returns:
        ``{"this": ..., "baseline": ... | None, "deltas": {...}}`` — ``deltas``
        is empty when there is no comparable to anchor against.
    """
    others = [c for c in comparables if c.get("generator") != this.get("generator")]
    pool = others or comparables
    # Anchor on the best-grounded comparable (highest faithfulness), else first.
    ranked = sorted(
        pool,
        key=lambda c: (c.get("faithfulness") is None, -(c.get("faithfulness") or 0.0)),
    )
    baseline = ranked[0] if ranked else None
    deltas: dict[str, float | None] = {}
    if baseline is not None:
        deltas = {
            "faithfulness_pct": _pct(this.get("faithfulness"), baseline.get("faithfulness")),
            "ndcg_at_5_pct": _pct(this.get("ndcg_at_5"), baseline.get("ndcg_at_5")),
            "e2e_pct": _pct(this.get("e2e_p99_s"), baseline.get("e2e_p99_s")),
            "speed_pct": _pct(
                this.get("retrieve_generate_ms"), baseline.get("retrieve_generate_ms")
            ),
            "vram_pct": _pct(this.get("vram_peak_mb"), baseline.get("vram_peak_mb")),
        }
    return {"this": this, "baseline": baseline, "deltas": deltas}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a precise ML benchmarking analyst. Write a SHORT conclusion (3-5 "
    "sentences, no bullet points, no headings) about a single RAG benchmark run, "
    "comparing it to prior runs on the same dataset. Use ONLY the numbers given — "
    "never invent figures. State the trade-off plainly (e.g. 'for ~X% more VRAM you "
    "get ~Y% higher faithfulness at ~Z% slower end-to-end') and finish with a one-line "
    "recommendation of when this configuration is the right pick (what priority it "
    "favours: grounding, speed, or memory). Be factual and concise."
)


def _fmt_num(v: float | None, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.1f}%"


def build_user_prompt(basis: dict[str, Any]) -> str:
    """Render the deterministic basis into the user prompt (real numbers only)."""
    this = basis["this"]
    base = basis.get("baseline")
    d = basis.get("deltas") or {}
    lines = [
        "THIS RUN:",
        f"  config: {this.get('label')}",
        f"  faithfulness: {_fmt_num(this.get('faithfulness'))}",
        f"  nDCG@5: {_fmt_num(this.get('ndcg_at_5'))}",
        f"  context_recall: {_fmt_num(this.get('context_recall'))}",
        f"  factual_correctness: {_fmt_num(this.get('factual_correctness'))}",
        f"  e2e p99: {_fmt_num(this.get('e2e_p99_s'), 2)} s",
        f"  per-query mean: {_fmt_num(this.get('retrieve_generate_ms'), 0)} ms",
        f"  VRAM peak: {_fmt_num(this.get('vram_peak_mb'), 0)} MB",
    ]
    if base is not None:
        lines += [
            "",
            f"BASELINE (best comparable on the same dataset): {base.get('label')}",
            f"  faithfulness: {_fmt_num(base.get('faithfulness'))}",
            f"  nDCG@5: {_fmt_num(base.get('ndcg_at_5'))}",
            f"  e2e p99: {_fmt_num(base.get('e2e_p99_s'), 2)} s",
            f"  VRAM peak: {_fmt_num(base.get('vram_peak_mb'), 0)} MB",
            "",
            "DELTAS (this run vs baseline, signed):",
            f"  faithfulness: {_fmt_pct(d.get('faithfulness_pct'))}",
            f"  nDCG@5: {_fmt_pct(d.get('ndcg_at_5_pct'))}",
            f"  end-to-end latency: {_fmt_pct(d.get('e2e_pct'))} (positive = slower)",
            f"  per-query latency: {_fmt_pct(d.get('speed_pct'))} (positive = slower)",
            f"  VRAM: {_fmt_pct(d.get('vram_pct'))} (positive = more memory)",
        ]
    else:
        lines += [
            "",
            "No comparable run on this dataset yet — describe this run on its own merits.",
        ]
    lines.append("")
    lines.append("Write the conclusion now.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class _ChatClientLike(Protocol):
    """The minimal LLM surface the writer needs (mirrors VllmClient.complete)."""

    async def complete(self, messages: list[dict[str, str]], params: Any) -> Any: ...


class ConclusionWriter:
    """Writes the grounded conclusion via a LOCAL judge LLM.

    The deltas are computed deterministically (:func:`build_basis`); the LLM
    only writes the prose.  Degrades to ``None`` on any failure (the run still
    persists without a conclusion).
    """

    def __init__(self, *, chat_client: Any, model: str, max_tokens: int = 320) -> None:
        """Bind the writer to a local chat client + model.

        Args:
            chat_client: An object exposing ``complete(messages, params)`` (the
                project's ``VllmClient``), pointed at a LOCAL endpoint.
            model: The served model name (the general judge, e.g. Selene —
                resident after the grading phase, so no extra swap).
            max_tokens: Cap on the generated conclusion length.
        """
        self._client = chat_client
        self._model = model
        self._max_tokens = max_tokens

    async def write(
        self, this_summary: dict[str, Any], comparables: list[dict[str, Any]]
    ) -> str | None:
        """Generate the conclusion text, or ``None`` on failure."""
        try:
            basis = build_basis(this_summary, comparables)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": build_user_prompt(basis)},
            ]
            from iw_rag.generation.llm import GenerationParams

            params = GenerationParams(
                model=self._model, temperature=0.2, max_tokens=self._max_tokens
            )
            resp = await self._client.complete(messages, params)
            text = getattr(resp, "text", None) or (resp if isinstance(resp, str) else None)
            if not text:
                return None
            return str(text).strip() or None
        except Exception:  # noqa: BLE001 — never fail the run on the conclusion
            logger.warning("benchmark conclusion generation failed", exc_info=True)
            return None


# ---------------------------------------------------------------------------
# Multi-run comparison conclusion (the N-way Compare view + its PDF)
# ---------------------------------------------------------------------------

_COMPARE_SYSTEM = (
    "You are a precise ML benchmarking analyst. You are given the results of a "
    "comparison of 2-5 RAG configurations evaluated on the SAME dataset, each "
    "compared against a designated baseline run. Write a SHORT analysis (4-6 "
    "sentences, no bullet points, no headings): FIRST state the overall conclusion "
    "(which configuration is best and for what priority — grounding/faithfulness, "
    "retrieval quality, or a balance), THEN give the reasoning grounded ONLY in the "
    "numbers provided — cite the key metric deltas vs the baseline and whether they "
    "are statistically significant. If an underpowered caveat is present, respect it "
    "(do not overclaim small, non-significant differences). Never invent figures."
)


def _fmt_signed(v: float | None, digits: int = 3) -> str:
    """Format a signed delta (``+0.050`` / ``-0.020`` / ``n/a``)."""
    return "n/a" if v is None else f"{v:+.{digits}f}"


def _compare_mods(run: dict[str, Any]) -> str:
    """A short ``generator / embedder / reranker / grader`` module summary."""
    cfg = run.get("config_snapshot") or {}
    parts = [str(cfg[k]) for k in ("generator", "embedder", "reranker", "grader") if cfg.get(k)]
    return " / ".join(parts) or "—"


def build_compare_user_prompt(payload: dict[str, Any]) -> str:
    """Render the multi-compare payload into the grounded user prompt.

    The payload is the ``BenchmarkMultiCompareResponse`` shape (a plain
    dict): per-run ``means`` + baseline-relative ``deltas`` with
    ``p_corrected`` + ``significant``.  Only these real numbers are
    handed to the LLM — it never sees raw vectors and is told not to
    invent figures.

    Args:
        payload: The multi-compare response dict.

    Returns:
        The rendered user-prompt string.
    """
    metrics: list[str] = list(payload.get("metrics") or [])
    runs: list[dict[str, Any]] = list(payload.get("runs") or [])
    lines: list[str] = [f"DATASET QUERIES: n={payload.get('n_queries', '?')}"]
    if payload.get("underpowered"):
        lines.append(f"CAVEAT: {payload.get('caveat') or 'underpowered (small sample)'}")
    lines.append("")
    for run in runs:
        tag = " (BASELINE)" if run.get("is_baseline") else ""
        label = run.get("display_id") or run.get("label") or run.get("id")
        lines.append(f"RUN {label}{tag} — modules: {_compare_mods(run)}")
        means = run.get("means") or {}
        for m in metrics:
            lines.append(f"  {m}: {_fmt_num(_f(means.get(m)))}")
        deltas = run.get("deltas") or []
        if deltas:
            lines.append("  vs baseline (delta = this run − baseline; positive = better):")
            for d in deltas:
                sig = "significant" if d.get("significant") else "not significant"
                lines.append(
                    f"    {d.get('metric')}: {_fmt_signed(_f(d.get('delta')))} "
                    f"(p_corrected={_fmt_num(_f(d.get('p_corrected')), 4)}, {sig})"
                )
        lines.append("")
    lines.append("Write the comparison analysis now.")
    return "\n".join(lines)


class CompareConclusionWriter:
    """Writes a grounded conclusion+reasoning for an N-way comparison.

    Mirrors :class:`ConclusionWriter` but consumes the already-computed
    multi-compare payload (per-run means + baseline-relative deltas +
    significance) — no DB read is needed because every figure the LLM
    narrates is already in the payload.  Local-only (Invariant 1): the
    writer is the general judge model (Selene).  Degrades to ``None`` on
    any failure.
    """

    def __init__(self, *, chat_client: Any, model: str, max_tokens: int = 420) -> None:
        """Bind the writer to a local chat client + model.

        Args:
            chat_client: An object exposing ``complete(messages, params)``
                (the project's ``VllmClient``), pointed at a LOCAL endpoint.
            model: The served model name (the general judge, e.g. Selene).
            max_tokens: Cap on the generated conclusion length.
        """
        self._client = chat_client
        self._model = model
        self._max_tokens = max_tokens

    async def write(self, payload: dict[str, Any]) -> str | None:
        """Generate the comparison conclusion text, or ``None`` on failure."""
        try:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": _COMPARE_SYSTEM},
                {"role": "user", "content": build_compare_user_prompt(payload)},
            ]
            from iw_rag.generation.llm import GenerationParams

            params = GenerationParams(
                model=self._model, temperature=0.2, max_tokens=self._max_tokens
            )
            resp = await self._client.complete(messages, params)
            text = getattr(resp, "text", None) or (resp if isinstance(resp, str) else None)
            if not text:
                return None
            return str(text).strip() or None
        except Exception:  # noqa: BLE001 — never fail the request on the conclusion
            logger.warning("benchmark compare conclusion generation failed", exc_info=True)
            return None


__all__ = [
    "CompareConclusionWriter",
    "ConclusionWriter",
    "build_basis",
    "build_compare_user_prompt",
    "build_user_prompt",
    "summarize_report",
]
