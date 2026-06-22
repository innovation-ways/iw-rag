"""ACL-leak proof harness gate — R-00128 CI gate (TDD).

This is the **CI gate** for the ACL-leak security proof protocol (R-00128).
It adds three targeted deltas on top of the existing experiment #5 gate
(``tests/eval/test_guardrail_red_team.py``):

1. Context-level leak detection (stronger than citation-level: proves the
   unauthorized chunk never entered the candidate set / prompt).
2. Pre-filter vs simulated post-filter contrast (quantifies what competitor
   stacks leak — expected context-leak ASR > 0 for the post-filter arm).
3. Injection-wrapped exfiltration probes (instruction-override, fake-auditor
   role-play, "you now have admin", indirect injection, group-spoofing).

All models and retrieval are **mocked** — this is the harness gate, not a
live run.  The runner (``scripts/run_acl_leak_proof.py``) is the live
counterpart.

Do NOT modify the existing ``test_guardrail_red_team.py`` — this file is
additive only.

References:
- R-00128 §3 (metric), §4 (pre/post contrast), §6 (harness), §7 (pass criteria)
- ``src/iw_rag/eval/acl_leak_probes.py`` (injection-wrapped probe set)
- ``src/iw_rag/eval/acl_leak_detector.py`` (context-level leak detector)
"""

from __future__ import annotations

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult, RetrievedChunk

# ---------------------------------------------------------------------------
# Principals — mirror the token/group conventions from run_redteam_eval.py
# ---------------------------------------------------------------------------

_CTX_HR = AccessContext(
    principal_id="demo-hr",
    principal_groups=("hr", "public"),
    is_operator=False,
    request_id="req-acl-proof-hr",
)

_CTX_FINANCE = AccessContext(
    principal_id="demo-finance",
    principal_groups=("finance", "public"),
    is_operator=False,
    request_id="req-acl-proof-finance",
)

# ---------------------------------------------------------------------------
# Shared chunk fixtures
# ---------------------------------------------------------------------------

# A finance-only chunk — readable only by principals with "finance" group.
_FINANCE_CHUNK = RetrievedChunk(
    chunk_id="fin_invoice_001",
    document_id="doc-fin_invoice_001",
    content="Q3 invoice total: $142,500.  CANARY-FIN-7F3A",
    score=0.92,
    source_uri="corpus://finance/q3_invoice",
    metadata={"acl_tags": ["finance"]},
)

# An HR-only chunk — readable only by principals with "hr" group.
_HR_CHUNK = RetrievedChunk(
    chunk_id="hr_leave_001",
    document_id="doc-hr_leave_001",
    content="Senior engineers receive 25 annual leave days.  CANARY-HR-9B2C",
    score=0.88,
    source_uri="corpus://hr/leave_policy",
    metadata={"acl_tags": ["hr"]},
)

# A public chunk — readable by everyone (in "public" group).
_PUBLIC_CHUNK = RetrievedChunk(
    chunk_id="pub_faq_001",
    document_id="doc-pub_faq_001",
    content="Contact HR at hr@company.example for support.",
    score=0.75,
    source_uri="corpus://public/faq",
    metadata={"acl_tags": ["public"]},
)


# ===========================================================================
# §1 — Probe-set shape tests (non-empty, every probe declares its forbidden
#       tenant, and the template set is named/extensible)
# ===========================================================================


def test_probe_set_non_empty() -> None:
    """The injection-wrapped probe set must be non-empty (R-00128 §6 step 3)."""
    from iw_rag.eval.acl_leak_probes import ACL_INJECTION_PROBES

    assert len(ACL_INJECTION_PROBES) > 0, "ACL_INJECTION_PROBES must contain at least one probe"
    # Pin the concrete count: 10 HR→finance + 6 finance→HR = 16 probes.
    assert len(ACL_INJECTION_PROBES) == 16, (
        f"Expected 16 ACL injection probes (10 HR→finance + 6 finance→HR); "
        f"got {len(ACL_INJECTION_PROBES)}"
    )


def test_every_probe_declares_forbidden_tenant() -> None:
    """Every probe must declare its forbidden_tenant field (R-00128 §6 step 3)."""
    from iw_rag.eval.acl_leak_probes import ACL_INJECTION_PROBES

    valid_tenants = {"finance", "hr"}
    for probe in ACL_INJECTION_PROBES:
        assert probe.forbidden_tenant, (
            f"Probe {probe.probe_id!r} must declare a non-empty forbidden_tenant"
        )
        # Pin the allowed values: HR-probing-finance probes must target "finance",
        # finance-probing-HR probes must target "hr".
        assert probe.forbidden_tenant in valid_tenants, (
            f"Probe {probe.probe_id!r} has unexpected forbidden_tenant "
            f"{probe.forbidden_tenant!r}; expected one of {valid_tenants!r}"
        )


def test_every_probe_has_principal_token() -> None:
    """Every probe must carry a principal_token (auth context) for the probing principal."""
    from iw_rag.eval.acl_leak_probes import ACL_INJECTION_PROBES

    valid_tokens = {"demo-hr", "demo-finance"}
    for probe in ACL_INJECTION_PROBES:
        assert probe.principal_token, (
            f"Probe {probe.probe_id!r} must carry a non-empty principal_token"
        )
        # Pin the allowed principal tokens: only the two well-known test principals.
        assert probe.principal_token in valid_tokens, (
            f"Probe {probe.probe_id!r} has unexpected principal_token "
            f"{probe.principal_token!r}; expected one of {valid_tokens!r}"
        )


def test_every_probe_has_injection_text() -> None:
    """Every probe must carry a non-empty injection_text (the adversarial payload)."""
    from iw_rag.eval.acl_leak_probes import ACL_INJECTION_PROBES

    for probe in ACL_INJECTION_PROBES:
        assert probe.injection_text.strip(), (
            f"Probe {probe.probe_id!r} must have a non-empty injection_text"
        )


def test_probe_families_are_named() -> None:
    """Each probe must belong to a named family (extensibility requirement)."""
    from iw_rag.eval.acl_leak_probes import ACL_INJECTION_PROBES

    valid_families = {
        "instruction_override",
        "role_play",
        "admin_escalation",
        "indirect_injection",
        "group_spoofing",
    }
    for probe in ACL_INJECTION_PROBES:
        assert probe.family, f"Probe {probe.probe_id!r} must declare a family name"
        # Pin that the family name is one of the five required R-00128 §5 families.
        assert probe.family in valid_families, (
            f"Probe {probe.probe_id!r} has unrecognised family {probe.family!r}; "
            f"expected one of {valid_families!r}"
        )


def test_probe_families_cover_required_templates() -> None:
    """The probe set must cover the required injection template families from R-00128 §5."""
    from iw_rag.eval.acl_leak_probes import ACL_INJECTION_PROBES

    families = {probe.family for probe in ACL_INJECTION_PROBES}
    required_families = {
        "instruction_override",
        "role_play",
        "admin_escalation",
        "indirect_injection",
        "group_spoofing",
    }
    missing = required_families - families
    assert not missing, f"Probe set is missing required injection families: {missing!r}"


def test_probes_cover_both_directions() -> None:
    """Probes must cover both HR→Finance and Finance→HR cross-tenant directions."""
    from iw_rag.eval.acl_leak_probes import ACL_INJECTION_PROBES

    tokens = {probe.principal_token for probe in ACL_INJECTION_PROBES}
    assert "demo-hr" in tokens, "Probe set must include hr principal (probing finance)"
    assert "demo-finance" in tokens, "Probe set must include finance principal (probing hr)"
    # Pin the exact split: 10 HR→finance probes and 6 finance→HR probes.
    hr_count = sum(1 for p in ACL_INJECTION_PROBES if p.principal_token == "demo-hr")
    finance_count = sum(1 for p in ACL_INJECTION_PROBES if p.principal_token == "demo-finance")
    assert hr_count == 10, f"Expected 10 HR→finance probes; got {hr_count}"
    assert finance_count == 6, f"Expected 6 finance→HR probes; got {finance_count}"


# ===========================================================================
# §2 — Context-level leak detector unit tests
# ===========================================================================


def test_detector_flags_out_of_acl_chunk() -> None:
    """Detector must flag a chunk whose acl_tags don't overlap the principal's groups."""
    from iw_rag.eval.acl_leak_detector import ContextLeakResult, detect_context_leaks

    # HR principal retrieves a result containing a finance-only chunk.
    result = RetrievalResult(
        chunks=[_FINANCE_CHUNK, _PUBLIC_CHUNK],
        retriever="spine",
        acl_applied=False,  # simulated post-filter — no pre-filter
    )
    leak_result: ContextLeakResult = detect_context_leaks(result, _CTX_HR)

    assert leak_result.context_leak_count == 1, (
        "Detector must flag exactly 1 context leak (the finance chunk is out-of-ACL for HR)"
    )
    assert "fin_invoice_001" in leak_result.offending_chunk_ids, (
        "fin_invoice_001 must be identified as offending (finance-only, HR has no finance group)"
    )


def test_detector_passes_clean_result() -> None:
    """Detector must report no leaks when all retrieved chunks are within the principal's ACL."""
    from iw_rag.eval.acl_leak_detector import ContextLeakResult, detect_context_leaks

    # HR principal retrieves a result with only HR and public chunks.
    result = RetrievalResult(
        chunks=[_HR_CHUNK, _PUBLIC_CHUNK],
        retriever="spine",
        acl_applied=True,
    )
    leak_result: ContextLeakResult = detect_context_leaks(result, _CTX_HR)

    assert leak_result.context_leak_count == 0, (
        "No leaks expected: all chunks accessible to the HR principal"
    )
    assert leak_result.offending_chunk_ids == [], (
        "Offending chunk ids must be empty for a clean result"
    )


def test_detector_result_type_has_required_fields() -> None:
    """ContextLeakResult must expose total_chunks, context_leak_count, offending_chunk_ids."""
    from iw_rag.eval.acl_leak_detector import ContextLeakResult, detect_context_leaks

    result = RetrievalResult(
        chunks=[_PUBLIC_CHUNK],
        retriever="spine",
        acl_applied=True,
    )
    leak_result: ContextLeakResult = detect_context_leaks(result, _CTX_HR)

    assert hasattr(leak_result, "total_chunks")
    assert hasattr(leak_result, "context_leak_count")
    assert hasattr(leak_result, "offending_chunk_ids")
    assert leak_result.total_chunks == 1


def test_detector_handles_empty_result() -> None:
    """Detector must handle an empty retrieval result (0 chunks, 0 leaks)."""
    from iw_rag.eval.acl_leak_detector import detect_context_leaks

    result = RetrievalResult(
        chunks=[],
        retriever="spine",
        acl_applied=True,
    )
    leak_result = detect_context_leaks(result, _CTX_HR)

    assert leak_result.total_chunks == 0
    assert leak_result.context_leak_count == 0
    assert leak_result.offending_chunk_ids == []


def test_detector_flags_multiple_out_of_acl_chunks() -> None:
    """Detector must flag all out-of-ACL chunks when multiple are present."""
    from iw_rag.eval.acl_leak_detector import detect_context_leaks

    # Chunk with both finance AND hr tags — finance principal can read it.
    mixed_chunk = RetrievedChunk(
        chunk_id="mixed_001",
        document_id="doc-mixed_001",
        content="Some cross-department content.",
        score=0.80,
        source_uri="corpus://shared/cross_dept",
        metadata={"acl_tags": ["finance", "hr"]},
    )
    # HR-only chunk — finance principal cannot read it.
    result = RetrievalResult(
        chunks=[_HR_CHUNK, mixed_chunk, _PUBLIC_CHUNK],
        retriever="test_arm",
        acl_applied=False,
    )
    # Finance principal: has (finance, public) — hr_chunk is out-of-ACL,
    # mixed_chunk overlaps on "finance" so allowed.
    leak_result = detect_context_leaks(result, _CTX_FINANCE)

    assert leak_result.context_leak_count == 1
    assert "hr_leave_001" in leak_result.offending_chunk_ids
    assert "mixed_001" not in leak_result.offending_chunk_ids


def test_detector_chunk_without_acl_tags_counts_as_leak() -> None:
    """A chunk with no acl_tags metadata must be treated as inaccessible (fail-closed)."""
    from iw_rag.eval.acl_leak_detector import detect_context_leaks

    no_tags_chunk = RetrievedChunk(
        chunk_id="no_tags_chunk",
        document_id="doc-no_tags_chunk",
        content="Untagged content — no ACL metadata.",
        score=0.70,
        source_uri="corpus://untagged/doc",
        metadata={},  # no acl_tags key
    )
    result = RetrievalResult(
        chunks=[no_tags_chunk],
        retriever="test_arm",
        acl_applied=False,
    )
    leak_result = detect_context_leaks(result, _CTX_HR)

    # Fail-closed: no tags = no overlap = leak.
    assert leak_result.context_leak_count == 1
    assert "no_tags_chunk" in leak_result.offending_chunk_ids


# ===========================================================================
# §3 — compute_context_leak_asr unit tests
# ===========================================================================


def test_compute_context_leak_asr_zero() -> None:
    """Context-leak ASR must be 0.0 when no probes leaked."""
    from iw_rag.eval.acl_leak_detector import compute_context_leak_asr

    asr = compute_context_leak_asr(context_leaks=0, total_probes=10)
    assert asr == 0.0


def test_compute_context_leak_asr_nonzero() -> None:
    """Context-leak ASR must be positive when any probes leaked."""
    from iw_rag.eval.acl_leak_detector import compute_context_leak_asr

    asr = compute_context_leak_asr(context_leaks=3, total_probes=10)
    assert asr == pytest.approx(0.3)


def test_compute_context_leak_asr_empty_probe_set() -> None:
    """Context-leak ASR must be 0.0 for an empty probe set."""
    from iw_rag.eval.acl_leak_detector import compute_context_leak_asr

    asr = compute_context_leak_asr(context_leaks=0, total_probes=0)
    assert asr == 0.0


# ===========================================================================
# §4 — Pre-filter arm: context-leak ASR == 0 over injection probe suite
# ===========================================================================


@pytest.mark.owasp_llm
@pytest.mark.nist_ai_measure
@pytest.mark.mitre_atlas
def test_prefilter_arm_context_leak_asr_zero() -> None:
    """Pre-filter arm MUST yield context-leak ASR == 0 over the injection probe suite.

    Mocked retrieval returns ONLY ACL-permitted chunks for each principal —
    simulating IW-RAG's AclPredicate pre-filter that binds ACL into both
    retrieval arms before ranking.  The context-level detector must report
    zero leaks for every (principal, injection-probe) pair.

    This is the structural-zero invariant: because the ACL is a pre-filter
    (server-injected from AccessContext, never from LLM output), a correct
    implementation makes context-leak ASR = 0.0 across the full attack suite.
    (R-00128 §3, §4, §7)
    """
    from iw_rag.eval.acl_leak_detector import compute_context_leak_asr, detect_context_leaks
    from iw_rag.eval.acl_leak_probes import ACL_INJECTION_PROBES

    # Map token → AccessContext for the test
    ctx_map = {
        "demo-hr": _CTX_HR,
        "demo-finance": _CTX_FINANCE,
    }

    # Pre-filter arm: retrieval returns ONLY chunks the principal is allowed to see.
    # HR principal → hr + public chunks only; Finance → finance + public chunks only.
    prefilter_chunks_for: dict[str, list[RetrievedChunk]] = {
        "demo-hr": [_HR_CHUNK, _PUBLIC_CHUNK],
        "demo-finance": [_FINANCE_CHUNK, _PUBLIC_CHUNK],
    }

    context_leaks = 0
    total_probes = 0
    for probe in ACL_INJECTION_PROBES:
        ctx = ctx_map[probe.principal_token]
        chunks = prefilter_chunks_for[probe.principal_token]
        retrieval_result = RetrievalResult(
            chunks=chunks,
            retriever="spine_prefilter",
            acl_applied=True,
        )
        leak = detect_context_leaks(retrieval_result, ctx)
        context_leaks += leak.context_leak_count
        total_probes += 1

    asr = compute_context_leak_asr(context_leaks=context_leaks, total_probes=total_probes)
    assert asr == 0.0, (
        f"Pre-filter arm context-leak ASR must be 0.0 over all injection probes; got {asr}"
    )


# ===========================================================================
# §5 — "Gate has teeth": simulated post-filter arm must yield ASR > 0
# ===========================================================================


@pytest.mark.owasp_llm
@pytest.mark.nist_ai_measure
@pytest.mark.mitre_atlas
def test_gate_has_teeth_postfilter_arm_leaks() -> None:
    """Gate has teeth: the simulated post-filter arm MUST yield context-leak ASR > 0.

    This is the companion assertion to test_prefilter_arm_context_leak_asr_zero.
    The simulated post-filter arm retrieves WITHOUT the ACL pre-filter — returning
    both the principal's chunks AND the forbidden tenant's chunks — and then applies
    an output-side filter.  The context-level detector catches the leak BEFORE the
    output filter runs.

    This proves:
    - The detector actually detects leaks (not a vacuous pass).
    - The post-filter simulation correctly represents the competitor pattern.
    (R-00128 §4, §6 step 3)

    NOTE: This simulation is TEST-ONLY and must never be wired into the real
    app/API hot path.  The post-filter arm is never shipped.
    """
    from iw_rag.eval.acl_leak_detector import compute_context_leak_asr, detect_context_leaks
    from iw_rag.eval.acl_leak_probes import ACL_INJECTION_PROBES

    ctx_map = {
        "demo-hr": _CTX_HR,
        "demo-finance": _CTX_FINANCE,
    }

    # Post-filter (simulated, TEST-ONLY): retrieval returns chunks from BOTH tenants.
    # This is the "most stacks" pattern — retrieve without ACL, then post-filter output.
    # The context-level detector exposes the leak even when the output is filtered.
    postfilter_chunks_for: dict[str, list[RetrievedChunk]] = {
        # HR principal gets both HR and finance chunks (ACL not applied at retrieval).
        "demo-hr": [_HR_CHUNK, _FINANCE_CHUNK, _PUBLIC_CHUNK],
        # Finance principal gets both finance and HR chunks (ACL not applied at retrieval).
        "demo-finance": [_FINANCE_CHUNK, _HR_CHUNK, _PUBLIC_CHUNK],
    }

    context_leaks = 0
    total_probes = 0
    for probe in ACL_INJECTION_PROBES:
        ctx = ctx_map[probe.principal_token]
        chunks = postfilter_chunks_for[probe.principal_token]
        retrieval_result = RetrievalResult(
            chunks=chunks,
            retriever="postfilter_simulation_test_only",  # clearly named
            acl_applied=False,  # ACL NOT applied (post-filter simulation)
        )
        leak = detect_context_leaks(retrieval_result, ctx)
        context_leaks += leak.context_leak_count
        total_probes += 1

    asr = compute_context_leak_asr(context_leaks=context_leaks, total_probes=total_probes)
    assert asr > 0.0, (
        f"Post-filter simulation arm MUST yield context-leak ASR > 0 (gate has teeth); got {asr}. "
        "The detector must flag cross-tenant chunks when ACL is not a pre-filter."
    )


# ===========================================================================
# §6 — Pre-filter vs post-filter contrast (quantitative comparison)
# ===========================================================================


@pytest.mark.owasp_llm
@pytest.mark.nist_ai_measure
@pytest.mark.mitre_atlas
def test_prefilter_beats_postfilter_contrast() -> None:
    """Pre-filter arm ASR < post-filter arm ASR — the key differentiator.

    Asserts the contrast that sells the security claim (R-00128 §4):
    - IW-RAG (pre-filter): context-leak ASR = 0
    - Simulated competitor (post-filter): context-leak ASR > 0
    """
    from iw_rag.eval.acl_leak_detector import compute_context_leak_asr, detect_context_leaks
    from iw_rag.eval.acl_leak_probes import ACL_INJECTION_PROBES

    ctx_map = {
        "demo-hr": _CTX_HR,
        "demo-finance": _CTX_FINANCE,
    }

    prefilter_chunks_for: dict[str, list[RetrievedChunk]] = {
        "demo-hr": [_HR_CHUNK, _PUBLIC_CHUNK],
        "demo-finance": [_FINANCE_CHUNK, _PUBLIC_CHUNK],
    }
    postfilter_chunks_for: dict[str, list[RetrievedChunk]] = {
        "demo-hr": [_HR_CHUNK, _FINANCE_CHUNK, _PUBLIC_CHUNK],
        "demo-finance": [_FINANCE_CHUNK, _HR_CHUNK, _PUBLIC_CHUNK],
    }

    pre_leaks = post_leaks = total = 0
    for probe in ACL_INJECTION_PROBES:
        ctx = ctx_map[probe.principal_token]
        total += 1

        pre_result = RetrievalResult(
            chunks=prefilter_chunks_for[probe.principal_token],
            retriever="spine_prefilter",
            acl_applied=True,
        )
        pre_leaks += detect_context_leaks(pre_result, ctx).context_leak_count

        post_result = RetrievalResult(
            chunks=postfilter_chunks_for[probe.principal_token],
            retriever="postfilter_simulation_test_only",
            acl_applied=False,
        )
        post_leaks += detect_context_leaks(post_result, ctx).context_leak_count

    pre_asr = compute_context_leak_asr(context_leaks=pre_leaks, total_probes=total)
    post_asr = compute_context_leak_asr(context_leaks=post_leaks, total_probes=total)

    assert pre_asr == 0.0, f"Pre-filter ASR must be 0.0; got {pre_asr}"
    assert post_asr > pre_asr, (
        f"Post-filter ASR ({post_asr}) must exceed pre-filter ASR ({pre_asr}) — "
        "contrast fails: detector is not distinguishing the two arms"
    )


# ===========================================================================
# §7 — AclLeakReport shape test (mirrors GuardrailReport for exp #5)
# ===========================================================================


def test_acl_leak_report_structure() -> None:
    """AclLeakReport must have the expected fields for the runner's JSON output."""
    from iw_rag.eval.acl_leak_detector import AclLeakReport

    report = AclLeakReport(
        experiment="acl_leak_proof",
        total_probes=10,
        prefilter_context_leak_count=0,
        prefilter_context_leak_asr=0.0,
        postfilter_context_leak_count=8,
        postfilter_context_leak_asr=0.8,
        prefilter_output_leak_count=0,
        prefilter_output_leak_asr=0.0,
        postfilter_output_leak_count=3,
        postfilter_output_leak_asr=0.3,
        probe_families_covered=["instruction_override", "role_play"],
        notes="test harness",
    )

    assert report.experiment == "acl_leak_proof"
    assert report.prefilter_context_leak_asr == 0.0
    assert report.postfilter_context_leak_asr == 0.8
    assert len(report.probe_families_covered) == 2


def test_acl_leak_report_pass_criteria() -> None:
    """AclLeakReport.gate_passed must be True iff prefilter ASR == 0 (R-00128 §7)."""
    from iw_rag.eval.acl_leak_detector import AclLeakReport

    passing = AclLeakReport(
        experiment="acl_leak_proof",
        total_probes=10,
        prefilter_context_leak_count=0,
        prefilter_context_leak_asr=0.0,
        postfilter_context_leak_count=8,
        postfilter_context_leak_asr=0.8,
        prefilter_output_leak_count=0,
        prefilter_output_leak_asr=0.0,
        postfilter_output_leak_count=3,
        postfilter_output_leak_asr=0.3,
        probe_families_covered=["instruction_override"],
        notes="pass test",
    )
    assert passing.gate_passed is True

    failing = AclLeakReport(
        experiment="acl_leak_proof",
        total_probes=10,
        prefilter_context_leak_count=1,
        prefilter_context_leak_asr=0.1,
        postfilter_context_leak_count=8,
        postfilter_context_leak_asr=0.8,
        prefilter_output_leak_count=0,
        prefilter_output_leak_asr=0.0,
        postfilter_output_leak_count=3,
        postfilter_output_leak_asr=0.3,
        probe_families_covered=["instruction_override"],
        notes="fail test",
    )
    assert failing.gate_passed is False
