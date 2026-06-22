"""Shared fixtures for the eval test suite (tests/eval/).

Provides the committed baseline loader fixture and helpers shared across
the retrieval gate, faithfulness gate, and any future eval gate tests.

The ``baseline`` fixture is always available (pure, no I/O beyond reading
the committed JSON file) so it can be used by both live-guarded and
always-on tests without any opt-in flag.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iw_rag.eval.ci_gate import Baseline, load_baseline

# Canonical path to the committed baseline file (relative to repo root).
_BASELINE_PATH = Path(__file__).resolve().parent / "baselines" / "phase0.json"


@pytest.fixture(scope="session")
def baseline() -> Baseline:
    """Load the committed Phase-0 regression baseline.

    Raises ``ConfigError`` loudly if the file is missing — a missing baseline
    is a misconfiguration, not a silent pass.

    Returns:
        The parsed :class:`~iw_rag.eval.ci_gate.Baseline` instance.
    """
    return load_baseline(_BASELINE_PATH)


def _judge_endpoint_up() -> bool:
    """Probe whether the offline judge endpoint is reachable.

    Reads ``VLLM_JUDGE_URL`` from the environment and issues a lightweight
    HTTP GET to its ``/health`` path (1-second timeout).  Returns ``False``
    on any error so the faithfulness gate can skip-with-warning when no
    judge is available in CI.

    Returns:
        ``True`` when the judge endpoint responds with HTTP 200; ``False``
        otherwise (network error, timeout, wrong status, env var absent).
    """
    import os

    import httpx

    judge_url = os.environ.get("VLLM_JUDGE_URL", "")
    if not judge_url:
        return False
    try:
        resp = httpx.get(f"{judge_url.rstrip('/')}/health", timeout=1.0)
        return resp.status_code == 200
    except Exception:
        return False
