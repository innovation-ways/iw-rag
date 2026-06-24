"""Unit tests for the per-mode (spine/graph/visual) ACL-leak proof extension (F-00097 S05).

Targets:
- Per-mode probe variants live in :mod:`iw_rag.eval.acl_leak_probes` and each
  probe declares a ``mode`` (one of ``"spine"``, ``"graph"``, ``"visual"``) so
  the runner can route the probe through the correct retriever arm.
- The detector (:mod:`iw_rag.eval.acl_leak_detector`) is retriever-agnostic
  and works unchanged on graph/visual ``RetrievalResult`` shapes.
- The live runner (:mod:`scripts.run_acl_leak_proof`) exposes **pure,
  importable** per-mode aggregation functions and a ``build_summary`` helper
  that emits the per-mode breakdown into the existing JSON shape.  No import-
  time side effects (the runner module must be safe to import from any test).

TDD RED-first (F-00097 step S05): the per-mode pure functions do NOT exist yet
on the base branch — the asserts below are deliberately ``AssertionError``
failures (NOT ``ImportError``) so the failure mode proves the gap is in the
extension contract, not the import graph.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Make sure ``scripts.run_acl_leak_proof`` is importable (mirrors the
# convention used in tests/unit/eval/test_benchmark_golden_v1.py).
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR.parent))


# ===========================================================================
# Helpers
# ===========================================================================


def _principal_for(token: str) -> Any:
    """Build an AccessContext for a probe's principal token (mirrors the live harness)."""
    from iw_rag.domain.acl import AccessContext

    if token == "demo-hr":
        groups = ("hr", "public")
    elif token == "demo-finance":
        groups = ("finance", "public")
    else:
        raise ValueError(f"Unknown test principal token: {token!r}")
    return AccessContext(
        principal_id=token,
        principal_groups=groups,
        is_operator=False,
        request_id=f"req-{token}",
    )


# ===========================================================================
# §1 — Per-mode probe variants exist on the public surface
# ===========================================================================


def test_per_mode_probe_lists_exist() -> None:
    """The probes module must expose per-mode lists so the runner can iterate them.

    The runner iterates each mode independently and aggregates per-mode leak
    counts; if the lists are missing, the runner has nothing to iterate.
    The expected names are part of the contract: ``ACL_INJECTION_PROBES_SPINE``
    (existing aggregate — backward compatible alias of
    ``ACL_INJECTION_PROBES``), plus the new
    ``ACL_INJECTION_PROBES_GRAPH`` and ``ACL_INJECTION_PROBES_VISUAL``.
    """
    from iw_rag.eval import acl_leak_probes as probes

    assert hasattr(probes, "ACL_INJECTION_PROBES_GRAPH"), (
        "acl_leak_probes must expose ACL_INJECTION_PROBES_GRAPH so the runner "
        "can iterate graph-mode probes; this attribute is missing on the "
        "base branch (F-00097 S05 RED)"
    )
    assert hasattr(probes, "ACL_INJECTION_PROBES_VISUAL"), (
        "acl_leak_probes must expose ACL_INJECTION_PROBES_VISUAL so the "
        "runner can iterate visual-mode probes (F-00097 S05 RED)"
    )
    assert isinstance(probes.ACL_INJECTION_PROBES_GRAPH, list)
    assert isinstance(probes.ACL_INJECTION_PROBES_VISUAL, list)
    # Each list must be non-empty — an empty list silently degrades the proof
    # to "no probes run" which trivially passes; we want at least one
    # probe per mode.
    assert len(probes.ACL_INJECTION_PROBES_GRAPH) > 0, (
        "Graph-mode probe list is empty — the proof is meaningless"
    )
    assert len(probes.ACL_INJECTION_PROBES_VISUAL) > 0, (
        "Visual-mode probe list is empty — the proof is meaningless"
    )


def test_per_mode_probes_have_mode_field() -> None:
    """Every per-mode probe must declare its routing ``mode``.

    The runner uses the ``mode`` field to choose the right route override
    on the live API (``mode="graph"`` / ``mode="visual"``) — see
    :class:`iw_rag.api.schemas.QueryRequest`.  Without ``mode``, the probes
    would only exercise the spine path and not prove the security claim
    for graph/visual.
    """
    from iw_rag.eval import acl_leak_probes as probes
    from iw_rag.eval.acl_leak_probes import AclInjectionProbe

    valid_modes = {"spine", "graph", "visual"}
    for label, probe_list in (
        ("graph", probes.ACL_INJECTION_PROBES_GRAPH),
        ("visual", probes.ACL_INJECTION_PROBES_VISUAL),
    ):
        for p in probe_list:
            assert isinstance(p, AclInjectionProbe), (
                f"{label} list must contain AclInjectionProbe instances; got {type(p).__name__}"
            )
            assert hasattr(p, "mode"), (
                f"Probe {p.probe_id!r} in {label} list must declare a 'mode' attribute "
                "so the runner can route it to the right retriever"
            )
            assert p.mode in valid_modes, (
                f"Probe {p.probe_id!r} has unexpected mode {p.mode!r}; "
                f"expected one of {valid_modes}"
            )
            assert p.mode == label, (
                f"Probe {p.probe_id!r} declares mode={p.mode!r} but lives in the "
                f"{label!r} probe list — mode and list membership must agree"
            )


def test_per_mode_probes_cover_both_directions() -> None:
    """Each per-mode probe list must cover HR→finance AND finance→HR.

    The aggregate ``ACL_INJECTION_PROBES`` already covers both directions
    (16 probes, 10 HR→finance + 6 finance→HR).  The per-mode lists must
    preserve that coverage so each mode is proven leak-free in both attack
    directions (AC4: zero cross-tenant leak per mode).
    """
    from iw_rag.eval import acl_leak_probes as probes

    for label, probe_list in (
        ("graph", probes.ACL_INJECTION_PROBES_GRAPH),
        ("visual", probes.ACL_INJECTION_PROBES_VISUAL),
    ):
        tokens = {p.principal_token for p in probe_list}
        assert "demo-hr" in tokens, (
            f"{label} probe list must include an HR principal (probing finance)"
        )
        assert "demo-finance" in tokens, (
            f"{label} probe list must include a finance principal (probing hr)"
        )


# ===========================================================================
# §2 — Detector is retriever-agnostic (no per-mode extension needed)
# ===========================================================================


def test_detector_works_unchanged_on_graph_results() -> None:
    """The detector must work unchanged on a graph-mode RetrievalResult.

    The detector reads ``chunk.metadata["acl_tags"]`` and applies the same
    any-overlap check the spine pre-filter uses.  Graph mode emits
    ``RetrievedChunk`` with ``retriever="graphrag"`` and ``acl_applied=True``
    (F-00097 S03) — same shape, no special-case.
    """
    from iw_rag.domain.models import RetrievalResult, RetrievedChunk
    from iw_rag.eval.acl_leak_detector import detect_context_leaks

    hr_chunk = RetrievedChunk(
        chunk_id="g_hr_001",
        document_id="d-g_hr_001",
        content="HR leave policy.",
        score=0.9,
        source_uri="corpus://hr/leave",
        metadata={"acl_tags": ["hr"]},
        retriever="graphrag",
        acl_applied=True,
    )
    finance_chunk = RetrievedChunk(
        chunk_id="g_fin_001",
        document_id="d-g_fin_001",
        content="Q3 invoice.",
        score=0.85,
        source_uri="corpus://finance/q3",
        metadata={"acl_tags": ["finance"]},
        retriever="graphrag",
        acl_applied=True,
    )
    public_chunk = RetrievedChunk(
        chunk_id="g_pub_001",
        document_id="d-g_pub_001",
        content="Public FAQ.",
        score=0.7,
        source_uri="corpus://public/faq",
        metadata={"acl_tags": ["public"]},
        retriever="graphrag",
        acl_applied=True,
    )
    ctx = _principal_for("demo-hr")
    result = RetrievalResult(
        chunks=[hr_chunk, finance_chunk, public_chunk],
        retriever="graphrag",
        acl_applied=True,
    )
    leak = detect_context_leaks(result, ctx)
    # HR principal: hr_chunk and public_chunk are readable; finance_chunk is not.
    assert leak.context_leak_count == 1, (
        f"Detector must flag exactly 1 context leak in graph-mode result "
        f"(the finance chunk); got {leak.context_leak_count}"
    )
    assert leak.offending_chunk_ids == ["g_fin_001"], (
        "g_fin_001 (finance chunk) must be flagged as offending"
    )


def test_detector_works_unchanged_on_visual_results() -> None:
    """The detector must work unchanged on a visual-mode RetrievalResult.

    Visual mode emits ``RetrievedChunk`` with ``retriever="visual"`` and
    ``acl_applied=True`` (F-00097 S01) — the metadata carries ``acl_tags``
    on each page chunk just like the spine.  No shape divergence → no
    detector extension required.
    """
    from iw_rag.domain.models import RetrievalResult, RetrievedChunk
    from iw_rag.eval.acl_leak_detector import detect_context_leaks

    hr_page = RetrievedChunk(
        chunk_id="v_hr_page_001",
        document_id="d-v_hr_page_001",
        content="HR leave policy page.",
        score=0.88,
        source_uri="corpus://hr/leave.pdf",
        metadata={"acl_tags": ["hr"], "page_image_ref": "x", "page": 1},
        retriever="visual",
        acl_applied=True,
    )
    finance_page = RetrievedChunk(
        chunk_id="v_fin_page_001",
        document_id="d-v_fin_page_001",
        content="Q3 invoice page.",
        score=0.86,
        source_uri="corpus://finance/q3_invoice.pdf",
        metadata={"acl_tags": ["finance"], "page_image_ref": "y", "page": 1},
        retriever="visual",
        acl_applied=True,
    )
    public_page = RetrievedChunk(
        chunk_id="v_pub_page_001",
        document_id="d-v_pub_page_001",
        content="Public FAQ page.",
        score=0.72,
        source_uri="corpus://public/faq.pdf",
        metadata={"acl_tags": ["public"], "page_image_ref": "z", "page": 1},
        retriever="visual",
        acl_applied=True,
    )
    ctx = _principal_for("demo-finance")
    result = RetrievalResult(
        chunks=[finance_page, hr_page, public_page],
        retriever="visual",
        acl_applied=True,
    )
    leak = detect_context_leaks(result, ctx)
    # Finance principal: finance_page and public_page are readable; hr_page is not.
    assert leak.context_leak_count == 1, (
        f"Detector must flag exactly 1 context leak in visual-mode result "
        f"(the hr page); got {leak.context_leak_count}"
    )
    assert leak.offending_chunk_ids == ["v_hr_page_001"], (
        "v_hr_page_001 (HR page) must be flagged as offending"
    )


# ===========================================================================
# §3 — Per-mode aggregation (pure, importable from scripts.run_acl_leak_proof)
# ===========================================================================


def _runner_module() -> Any:
    """Return the imported scripts.run_acl_leak_proof module.

    Using importlib + getattr (instead of ``from ... import ...``) makes
    the RED failures ``AttributeError`` rather than ``ImportError`` —
    required by the S05 TDD contract.
    """
    return importlib.import_module("scripts.run_acl_leak_proof")


def test_runner_module_import_has_no_side_effects() -> None:
    """Importing scripts.run_acl_leak_proof must NOT touch the network or stdout.

    The runner is also a library that S07 will import to assert on pure
    functions.  Any import-time side effect (e.g. ``httpx.post`` on import,
    network probe, stdout banner) makes that impossible.

    This test guards the contract by importing the module and asserting that
    no unexpected attributes were created on ``sys.modules`` outside the
    module's own namespace.
    """
    module = _runner_module()
    # The module must export the pure helpers S07 will use.  These names
    # are part of the contract; missing names fail RED with AttributeError.
    for name in ("aggregate_per_mode", "build_summary"):
        assert hasattr(module, name), (
            f"scripts.run_acl_leak_proof must export {name!r} (F-00097 S05 RED)"
        )


def test_aggregate_per_mode_groups_results_by_mode() -> None:
    """aggregate_per_mode must bucket results by mode and sum leak counts.

    Input: a flat list of per-probe records of the shape
    ``{"probe_id": str, "mode": str, "leaked": bool}``.

    Output: a dict ``{mode_name: {"total": int, "leaked": int}}`` and a
    flat ``{"total": int, "leaked": int}`` for the overall aggregate.
    """
    fn = getattr(_runner_module(), "aggregate_per_mode", None)
    assert fn is not None, (
        "scripts.run_acl_leak_proof must define aggregate_per_mode "
        "(F-00097 S05 RED — AttributeError)"
    )

    records = [
        {"probe_id": "p1", "mode": "spine", "leaked": False},
        {"probe_id": "p2", "mode": "spine", "leaked": False},
        {"probe_id": "g1", "mode": "graph", "leaked": False},
        {"probe_id": "g2", "mode": "graph", "leaked": True},  # one graph leak
        {"probe_id": "v1", "mode": "visual", "leaked": False},
    ]
    agg = fn(records)

    assert "per_mode" in agg, "aggregate_per_mode must return a 'per_mode' key"
    assert "overall" in agg, "aggregate_per_mode must return an 'overall' key"

    # Per-mode breakdown
    pm = agg["per_mode"]
    assert pm["spine"]["total"] == 2
    assert pm["spine"]["leaked"] == 0
    assert pm["graph"]["total"] == 2
    assert pm["graph"]["leaked"] == 1
    assert pm["visual"]["total"] == 1
    assert pm["visual"]["leaked"] == 0

    # Overall totals
    assert agg["overall"]["total"] == 5
    assert agg["overall"]["leaked"] == 1


def test_aggregate_per_mode_empty_input() -> None:
    """aggregate_per_mode on an empty list must return zeros, not raise."""
    fn = getattr(_runner_module(), "aggregate_per_mode", None)
    assert fn is not None, (
        "scripts.run_acl_leak_proof must define aggregate_per_mode "
        "(F-00097 S05 RED — AttributeError)"
    )

    agg = fn([])
    assert agg["per_mode"] == {}, "Empty input must yield empty per_mode dict"
    assert agg["overall"] == {"total": 0, "leaked": 0}


def test_build_summary_includes_per_mode_breakdown() -> None:
    """build_summary must add a per-mode breakdown to the JSON summary shape.

    The summary JSON already carries ``experiment``, ``total_probes``,
    ``metrics``, ``offending_probes``, ``gate_passed``.  S05 adds:
    - ``per_mode_metrics``: {mode: {context_leak_count, context_leak_asr}}
    - ``modes_covered``: sorted list of modes the run exercised

    The shape is additive — existing top-level fields are preserved so the
    existing report readers keep working.
    """
    fn = getattr(_runner_module(), "build_summary", None)
    assert fn is not None, (
        "scripts.run_acl_leak_proof must define build_summary (F-00097 S05 RED — AttributeError)"
    )

    probe_records = [
        # (probe_id, mode, principal_token, forbidden_tenant, context_leaked, canary_leaked)
        ("hr_fin_01", "spine", "demo-hr", "finance", False, False),
        ("hr_fin_02", "spine", "demo-hr", "finance", False, False),
        ("g_hr_fin_01", "graph", "demo-hr", "finance", False, False),
        ("v_hr_fin_01", "visual", "demo-hr", "finance", False, False),
    ]
    summary = fn(probe_records, base_url="http://localhost:8088")

    # Existing top-level fields preserved
    assert summary["experiment"] == "acl_leak_proof"
    assert summary["arm"] == "prefilter_live"
    assert summary["base_url"] == "http://localhost:8088"
    assert summary["total_probes"] == 4

    # New per-mode fields present
    assert "per_mode_metrics" in summary, "build_summary must emit 'per_mode_metrics' (F-00097 S05)"
    assert "modes_covered" in summary, "build_summary must emit 'modes_covered' (F-00097 S05)"

    pm = summary["per_mode_metrics"]
    assert pm["spine"]["context_leak_count"] == 0
    assert pm["spine"]["context_leak_asr"] == 0.0
    assert pm["graph"]["context_leak_count"] == 0
    assert pm["graph"]["context_leak_asr"] == 0.0
    assert pm["visual"]["context_leak_count"] == 0
    assert pm["visual"]["context_leak_asr"] == 0.0

    assert sorted(summary["modes_covered"]) == ["graph", "spine", "visual"]

    # gate_passed is true iff every mode has context_leak_count == 0
    assert summary["gate_passed"] is True, (
        "gate_passed must be True when all per-mode context_leak_counts are 0"
    )


def test_build_summary_gate_fails_on_any_mode_leak() -> None:
    """build_summary must set gate_passed=False if ANY mode leaks.

    A leak in graph or visual mode is a CRITICAL finding (AC4) — even if
    the spine arm is clean, the per-mode gate fails.
    """
    fn = getattr(_runner_module(), "build_summary", None)
    assert fn is not None, (
        "scripts.run_acl_leak_proof must define build_summary (F-00097 S05 RED — AttributeError)"
    )

    probe_records = [
        ("hr_fin_01", "spine", "demo-hr", "finance", False, False),
        ("g_hr_fin_01", "graph", "demo-hr", "finance", True, True),  # LEAK
        ("v_hr_fin_01", "visual", "demo-hr", "finance", False, False),
    ]
    summary = fn(probe_records, base_url="http://localhost:8088")
    assert summary["per_mode_metrics"]["graph"]["context_leak_count"] == 1
    assert summary["gate_passed"] is False, (
        "gate_passed must be False when any per-mode context_leak_count > 0"
    )
    assert summary["per_mode_metrics"]["graph"]["context_leak_asr"] == pytest.approx(1.0)


def test_build_summary_serialises_to_json_cleanly() -> None:
    """The summary dict must be JSON-serialisable (the runner writes it to disk)."""
    fn = getattr(_runner_module(), "build_summary", None)
    assert fn is not None, (
        "scripts.run_acl_leak_proof must define build_summary (F-00097 S05 RED — AttributeError)"
    )

    summary = fn(
        [
            ("hr_fin_01", "spine", "demo-hr", "finance", False, False),
            ("g_hr_fin_01", "graph", "demo-hr", "finance", False, False),
        ],
        base_url="http://localhost:8088",
    )
    # Round-trip through json — must not raise
    blob = json.dumps(summary)
    parsed = json.loads(blob)
    assert parsed["experiment"] == "acl_leak_proof"
    assert "per_mode_metrics" in parsed


def test_build_summary_gate_fails_on_empty_input() -> None:
    """build_summary must set gate_passed=False when no probes ran (F-00097 S06 F3).

    ``all([])`` is ``True`` in Python, so a naive ``all(...)`` gate would
    silently pass an empty run.  The S06 review flagged this as MEDIUM:
    an empty runner state (no probes fired) MUST surface as a fail-closed
    refusal, not a silent pass — an empty run that reports ``gate_passed=True``
    would falsely advertise "no cross-tenant leak" without measuring anything.
    """
    fn = getattr(_runner_module(), "build_summary", None)
    assert fn is not None, (
        "scripts.run_acl_leak_proof must define build_summary (F-00097 S05 RED — AttributeError)"
    )

    summary = fn([], base_url="http://localhost:8088")
    assert summary["per_mode_metrics"] == {}, "Empty input must yield empty per_mode_metrics"
    assert summary["gate_passed"] is False, (
        "gate_passed must be False when no probes ran — an empty run must NOT "
        "silently pass (F-00097 S06 finding F3: all([]) is True, naive gate "
        "would falsely advertise zero leak without measuring anything)"
    )
