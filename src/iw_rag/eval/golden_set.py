"""Golden set — the contamination guard + (deferred) Ragas generation.

The trust story's integrity rule: **the generator must never be a judge** (and
gold labels are human-curated, never model-asserted). This module owns the
startup contamination guard. The Ragas ``TestsetGenerator`` (heavy, offline,
local-LLM) that drafts candidate queries for human curation is deferred to the
``eval`` runtime path; the guard + the persistence (:mod:`iw_rag.eval.repository`)
are the load-bearing, dependency-light pieces.
"""

from __future__ import annotations

from collections.abc import Iterable

from iw_rag.domain.errors import ConfigError


def assert_generator_not_judge(generator_model: str, judge_models: Iterable[str]) -> None:
    """Fail closed if the generator model is also used as a judge.

    Args:
        generator_model: The served generator model name.
        judge_models: The configured judge model names.

    Raises:
        ConfigError: If the generator appears among the judges (contamination).
    """
    judges = set(judge_models)
    if generator_model in judges:
        raise ConfigError(
            f"Contamination: generator '{generator_model}' is also a judge "
            f"({sorted(judges)}). Judges MUST be a different model/family."
        )
