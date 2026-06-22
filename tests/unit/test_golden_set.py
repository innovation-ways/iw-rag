"""Unit tests for the contamination guard (generator must not be a judge)."""

from __future__ import annotations

import pytest

from iw_rag.domain.errors import ConfigError
from iw_rag.eval.golden_set import assert_generator_not_judge


def test_passes_when_judge_is_different_family() -> None:
    # Must return None (no exception) when generator is not among the judges.
    result = assert_generator_not_judge(
        "Qwen/Qwen3-14B-AWQ", ["PatronusAI/Lynx-8B", "AtlaAI/Selene-Mini-8B"]
    )
    assert result is None


def test_raises_when_generator_is_a_judge() -> None:
    with pytest.raises(ConfigError, match="Contamination"):
        assert_generator_not_judge("Qwen/Qwen3-14B-AWQ", ["Qwen/Qwen3-14B-AWQ"])


def test_empty_judges_passes() -> None:
    # An empty judges list cannot contain the generator — must return None silently.
    result = assert_generator_not_judge("gen", [])
    assert result is None
