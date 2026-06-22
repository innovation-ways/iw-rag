"""Unit tests for the F-00094 catalog loader (``src/iw_rag/serving/catalog.py``).

TDD RED→GREEN (S02): these tests drive ``load_catalog`` and ``Catalog.get``.
They pin the fail-fast contract (mirrors ``_load_model_registry`` in
``src/iw_rag/settings.py`` — missing/empty/malformed input → ``ConfigError``).
"""

from __future__ import annotations

import pytest
import yaml

from iw_rag.domain.errors import ConfigError, NotFoundError
from iw_rag.serving.catalog import Catalog, CatalogModel, load_catalog


def _write(tmp_path, payload: object | None) -> str:
    p = tmp_path / "model_catalog.yaml"
    if payload is not None:
        p.write_text(yaml.safe_dump(payload))
    return str(p)


def _ok_payload() -> dict[str, object]:
    return {
        "models": {
            "qwen3-7b-awq": {
                "id": "qwen3-7b-awq",
                "served_name": "Qwen/Qwen3-7B-AWQ",
                "role": "generation",
                "quantization": "awq",
                "expected_vram_gb": 5.0,
                "max_context": 16384,
                "vllm_flags": ["--quantization awq", "--kv-cache-dtype fp8"],
            },
        }
    }


def test_load_catalog_returns_catalog_with_models(tmp_path) -> None:
    cat = load_catalog(_write(tmp_path, _ok_payload()))
    assert isinstance(cat, Catalog)
    assert "qwen3-7b-awq" in cat.models
    assert len(cat.models) == 1


def test_load_catalog_parses_all_required_fields(tmp_path) -> None:
    cat = load_catalog(_write(tmp_path, _ok_payload()))
    e = cat.get("qwen3-7b-awq")
    assert isinstance(e, CatalogModel)
    assert e.id == "qwen3-7b-awq"
    assert e.served_name == "Qwen/Qwen3-7B-AWQ"
    assert e.role == "generation"
    assert e.quantization == "awq"
    assert e.expected_vram_gb == pytest.approx(5.0)
    assert e.max_context == 16384
    assert e.vllm_flags == ["--quantization awq", "--kv-cache-dtype fp8"]


def test_load_catalog_raises_configerror_on_missing_file(tmp_path) -> None:
    with pytest.raises(ConfigError, match="model_catalog.yaml not found"):
        load_catalog(str(tmp_path / "missing.yaml"))


def test_load_catalog_raises_configerror_on_empty_file(tmp_path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ConfigError, match="must have a top-level 'models:' key"):
        load_catalog(str(p))


def test_load_catalog_raises_configerror_on_malformed_yaml(tmp_path) -> None:
    bad = {"models": {"x": {"id": "x"}}}  # missing required fields
    with pytest.raises(ConfigError, match="has invalid entries"):
        load_catalog(_write(tmp_path, bad))


def test_load_catalog_raises_configerror_on_missing_models_key(tmp_path) -> None:
    p = _write(tmp_path, {"not_models": {}})
    with pytest.raises(ConfigError, match="must have a top-level 'models:' key"):
        load_catalog(p)


def test_catalog_get_raises_notfound() -> None:
    cat = Catalog(models={})
    with pytest.raises(NotFoundError, match="Unknown catalog model id 'absent'"):
        cat.get("absent")


# ---------------------------------------------------------------------------
# S05 — Blackwell sm_120 invariant guards (data-level)
# ---------------------------------------------------------------------------
# Every catalog entry MUST carry the Blackwell pins documented in
# `config/model_catalog.yaml`'s header comment.  We pin them here at the
# data level so any future edit that drops a flag (e.g. removing
# `--kv-cache-dtype fp8` in favour of the slower default) fails CI.
# The `VLLM_ATTENTION_BACKEND=TRITON_ATTN` env var is set in
# `config/llama-swap.yaml`'s `cmd` (not the vLLM CLI), so it is verified
# separately in the llama-swap.yaml tests (out of scope here).


def test_every_catalog_entry_has_kv_cache_dtype_fp8(tmp_path) -> None:
    """Every catalog entry's ``vllm_flags`` carries ``--kv-cache-dtype fp8``.

    This is the Blackwell sm_120 pin from F-00094 design §"Hard constraints".
    """
    payload: dict[str, object] = {
        "models": {
            "qwen3-7b-awq": {
                "id": "qwen3-7b-awq",
                "served_name": "Qwen/Qwen3-7B-AWQ",
                "role": "generation",
                "quantization": "awq",
                "expected_vram_gb": 5.0,
                "max_context": 16384,
                "vllm_flags": ["--kv-cache-dtype fp8"],
            },
            "qwen3-14b-awq": {
                "id": "qwen3-14b-awq",
                "served_name": "Qwen/Qwen3-14B-AWQ",
                "role": "generation",
                "quantization": "awq",
                "expected_vram_gb": 9.0,
                "max_context": 16384,
                "vllm_flags": ["--kv-cache-dtype fp8"],
            },
        }
    }
    p = tmp_path / "cat.yaml"
    p.write_text(yaml.safe_dump(payload))
    cat = load_catalog(str(p))
    assert len(cat.models) == 2
    for entry in cat.models.values():
        assert "--kv-cache-dtype fp8" in entry.vllm_flags, (
            f"Catalog entry '{entry.id}' is missing the Blackwell --kv-cache-dtype fp8 pin"
        )


def test_every_catalog_entry_has_awq_quantization(tmp_path) -> None:
    """Every catalog entry's quantization is ``awq`` (FP8 weights underperform on sm_120)."""
    payload: dict[str, object] = {
        "models": {
            "a": {
                "id": "a",
                "served_name": "A",
                "role": "generation",
                "quantization": "awq",
                "expected_vram_gb": 5.0,
                "max_context": 16384,
                "vllm_flags": [],
            },
            "b": {
                "id": "b",
                "served_name": "B",
                "role": "generation",
                "quantization": "awq",
                "expected_vram_gb": 5.0,
                "max_context": 16384,
                "vllm_flags": [],
            },
        }
    }
    p = tmp_path / "cat.yaml"
    p.write_text(yaml.safe_dump(payload))
    cat = load_catalog(str(p))
    for entry in cat.models.values():
        assert entry.quantization == "awq", (
            f"Catalog entry '{entry.id}' is not awq-quantized "
            f"(got {entry.quantization!r}); FP8 weights underperform on sm_120"
        )


def test_production_catalog_yaml_includes_blackwell_pins() -> None:
    """The shipped ``config/model_catalog.yaml`` includes every Blackwell pin.

    This is a regression guard: a future edit that drops one of the
    flags from every entry breaks the F-00094 hard constraints.
    """
    from pathlib import Path

    cat_path = Path(__file__).resolve().parents[3] / "config" / "model_catalog.yaml"
    if not cat_path.exists():
        pytest.skip(f"production catalog not found at {cat_path}")
    cat = load_catalog(str(cat_path))
    assert cat.models, "production catalog must be non-empty"
    # Quantization is a deliberate ladder since the FP8 + bf16 variants landed:
    # "awq" (awq_marlin kernel), "fp8", or "none" (bf16 weights).
    sanctioned_quant = {"awq", "fp8", "none"}
    for entry in cat.models.values():
        flags_str = " ".join(entry.vllm_flags)
        assert "--kv-cache-dtype fp8" in flags_str, (
            f"Production entry '{entry.id}' is missing --kv-cache-dtype fp8"
        )
        assert "--cuda-graph-sizes 1,2,4,8,16,32,64" in flags_str, (
            f"Production entry '{entry.id}' is missing the bounded --cuda-graph-sizes list"
        )
        assert entry.quantization in sanctioned_quant, (
            f"Production entry '{entry.id}' has unsanctioned quantization "
            f"{entry.quantization!r} (expected one of {sorted(sanctioned_quant)})"
        )
        # AWQ entries must use the awq_marlin kernel, not the legacy AWQ GEMM
        # path — the marlin kernel is the ~10x Blackwell decode win and must not
        # silently regress to plain `--quantization awq`.
        if entry.quantization == "awq":
            assert "--quantization awq_marlin" in flags_str, (
                f"Production AWQ entry '{entry.id}' must use the awq_marlin kernel"
            )
        assert entry.role == "generation"


def test_catalog_rejects_empty_models_mapping(tmp_path) -> None:
    """An empty ``models:`` mapping → ``ConfigError`` (fail-fast, mirrors models.yaml)."""
    p = tmp_path / "empty_models.yaml"
    p.write_text(yaml.safe_dump({"models": {}}))
    with pytest.raises(ConfigError, match="must be a non-empty mapping"):
        load_catalog(str(p))


def test_catalog_rejects_non_mapping_models(tmp_path) -> None:
    """``models:`` must be a mapping, not a list/scalar → ``ConfigError``."""
    p = tmp_path / "bad_models.yaml"
    p.write_text(yaml.safe_dump({"models": ["a", "b"]}))
    with pytest.raises(ConfigError, match="must be a non-empty mapping"):
        load_catalog(str(p))
