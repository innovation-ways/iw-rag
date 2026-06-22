"""Coverage guard — the Phase-0 manual query set exercises every disposition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MANIFEST = Path(__file__).resolve().parents[1] / "fixtures" / "manual_queries.yaml"


def _expects() -> set[str]:
    data: dict[str, Any] = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return {q["expect"] for q in data["queries"]}


def test_manual_set_covers_grounded_and_refused() -> None:
    """Manual query set contains exactly the two expected disposition labels."""
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    queries = data["queries"]
    expects = {q["expect"] for q in queries}
    grounded_count = sum(1 for q in queries if q["expect"] == "grounded")
    refused_count = sum(1 for q in queries if q["expect"] == "refused")
    assert "grounded" in expects, "need at least one grounded case"
    assert "refused" in expects, "need at least one fail-closed refusal case"
    assert grounded_count >= 2, (
        f"Phase-0 set must have ≥2 grounded cases for meaningful coverage, got {grounded_count}"
    )
    assert refused_count >= 1, (
        f"Phase-0 set must have ≥1 refused case for fail-closed coverage, got {refused_count}"
    )


def test_manual_set_dispositions_are_known() -> None:
    assert _expects() <= {"grounded", "refused"}
