"""User-facing generator catalog (F-00094, Benchmark Studio).

The catalog is a YAML file (default ``config/model_catalog.yaml``) that lists
the generator models an ``eval-operator`` can activate on the single 32 GB
GPU.  This module is the *only* place that parses it; every other layer
consumes the resulting :class:`Catalog` object.

**Fail-fast contract** (mirrors ``_load_model_registry`` in
``src/iw_rag/settings.py``): a missing/empty/malformed catalog file raises
:class:`~iw_rag.domain.errors.ConfigError` at startup so misconfig surfaces
before any traffic arrives.  An unknown model id (resolved at request time)
raises :class:`~iw_rag.domain.errors.NotFoundError` so the S03 API can map it
to HTTP 404.

**vLLM sleep-mode fallback** (R-00123 Limitations 1): the catalog itself
holds NO vLLM sleep-mode fields (``--enable-sleep-mode``,
``VLLM_SERVER_DEV_MODE=1``).  Sleep mode is a *narrow* fallback for parking
the **same** model's VRAM (cannot cross-swap to a different base model);
``VLLM_SERVER_DEV_MODE`` must NEVER be exposed to the network.  This
fallback is **not** the swap path — the exclusive ``generators`` group in
``config/llama-swap.yaml`` is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from iw_rag.domain.errors import ConfigError, NotFoundError


class CatalogModel(BaseModel):
    """A single catalog entry — one generator the operator may activate.

    Attributes:
        id: Stable catalog id (also the ``POST /models/activate``
            ``model_id`` field and the llama-swap entry name).
        served_name: Hugging Face repo id passed to vLLM as ``--model``.
        role: Logical role of the model.  v1 only fills
            ``"generation"``; the schema is roomy enough to not preclude
            future ``"embedder"`` / ``"rerank"`` entries (which would
            extend the catalog beyond generators).
        quantization: Quantisation scheme — v1 only fills ``"awq"`` (FP8
            weights underperform on Blackwell sm_120; R-00118 / R-00123).
        expected_vram_gb: VRAM budget figure from R-00123 §5
            (7B ~4-5, 14B ~8-10, 32B ~19-22).  Informational; the real
            budget is enforced by vLLM's ``--gpu-memory-utilization``.
        max_context: vLLM ``--max-model-len``.
        vllm_flags: vLLM command-line flags (Blackwell sm_120 pins
            documented at the top of ``config/model_catalog.yaml``).
    """

    id: str
    served_name: str
    role: Literal["generation", "embedder", "rerank"] = "generation"
    quantization: str = "awq"
    expected_vram_gb: float = Field(ge=0.0)
    max_context: int = Field(gt=0)
    vllm_flags: list[str] = Field(default_factory=list)


class Catalog(BaseModel):
    """The validated user-facing generator catalog.

    Attributes:
        models: Map of catalog id → :class:`CatalogModel`.  The keys
            mirror the YAML structure (one entry per id).
    """

    models: dict[str, CatalogModel]

    def get(self, model_id: str) -> CatalogModel:
        """Resolve a catalog id to its entry.

        Args:
            model_id: The catalog id (e.g. ``"qwen3-14b-awq"``).

        Returns:
            The resolved :class:`CatalogModel`.

        Raises:
            NotFoundError: If the id is not present in the catalog.
        """
        if model_id not in self.models:
            raise NotFoundError(
                f"Unknown catalog model id '{model_id}'. Available: {sorted(self.models)}"
            )
        return self.models[model_id]

    def __contains__(self, model_id: str) -> bool:  # pragma: no cover - convenience
        return model_id in self.models

    def __iter__(self) -> Any:  # pragma: no cover - convenience
        return iter(self.models)


def load_catalog(path: str) -> Catalog:
    """Load and validate ``config/model_catalog.yaml`` into a :class:`Catalog`.

    Mirrors the fail-fast contract of ``_load_model_registry`` in
    :mod:`iw_rag.settings`: a missing file, an empty file, a file with no
    top-level ``models:`` key, or a file whose entries fail Pydantic
    validation all raise :class:`~iw_rag.domain.errors.ConfigError`.

    Args:
        path: Path to the catalog YAML (relative to cwd or absolute).

    Returns:
        A validated :class:`Catalog` instance.

    Raises:
        ConfigError: If the file is missing, unparseable, missing the
            ``models`` key, or has invalid entries.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"model_catalog.yaml not found at '{path}'. "
            "Check serving.catalog_path in default.yaml or cwd."
        )
    try:
        with p.open() as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"model_catalog.yaml is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("model_catalog.yaml must have a top-level 'models:' key (a mapping).")
    if "models" not in raw:
        raise ConfigError("model_catalog.yaml must have a top-level 'models:' key.")
    models_raw = raw["models"]
    if not isinstance(models_raw, dict) or not models_raw:
        raise ConfigError("model_catalog.yaml 'models:' must be a non-empty mapping.")
    try:
        return Catalog(models={k: CatalogModel(**v) for k, v in models_raw.items()})
    except (ValidationError, TypeError) as exc:
        raise ConfigError(f"model_catalog.yaml has invalid entries: {exc}") from exc
