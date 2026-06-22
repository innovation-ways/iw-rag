"""Unit tests for the v1-pinned benchmark golden set + deterministic corpus.

The behavioural unit under test is **determinism / reproducibility** of
``scripts/gen_benchmark_corpus.py``.  Two successive generations of the v1
pinned golden + the corpus doc list must produce byte-identical output —
otherwise the Benchmark Studio cannot compare a run today with a run next
month without silent drift (the very risk the generator's own comments
warn about, "eval golden set drifted from a silent corpus re-gen").

The tests cover:

* **Determinism**: two successive generations of the v1 golden (and the
  corpus doc list) yield byte-identical results.
* **Version pin**: the emitted/pinned dataset block has ``version == "v1"``
  exactly.
* **Golden integrity**: every query with a ``source`` resolves to a
  present keyword — the ``(source_uri, keyword)`` golden model holds.

Exact-value assertions only; every ``pytest.raises`` uses ``match=``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

# The generator lives under ``scripts/`` (a top-level namespace).  The
# ``conftest.py`` at tests/ root already places the repo root on sys.path,
# so ``scripts.gen_benchmark_corpus`` is importable.
SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR.parent))

from scripts import gen_benchmark_corpus as gbc  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — the canonical pinned v1 golden path + dataset name.
# ---------------------------------------------------------------------------

PINNED_GOLDEN_PATH = Path("eval/golden/benchmark_golden_v1.yaml")
PINNED_DATASET_NAME = "benchmark-golden"
PINNED_DATASET_VERSION = "v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pinned_golden_dict(include_scale: bool = False) -> dict[str, Any]:
    """Build the in-memory pinned-v1 golden document the same way the
    generator does (mirrors the writer block in ``main()``).

    Args:
        include_scale: When True, append the scale-up golden queries so the
            pinned v1 set is the full set the generator supports.

    Returns:
        The full pinned-v1 golden document as a dict (no YAML serialisation).
    """
    golden = gbc._build_golden()  # type: ignore[attr-defined]
    if include_scale:
        golden = golden + gbc._build_scale_golden()  # type: ignore[attr-defined]
    groups = sorted({d[1] for d in gbc._build_docs()})  # type: ignore[attr-defined]
    return {
        "dataset": {"name": PINNED_DATASET_NAME, "version": PINNED_DATASET_VERSION},
        "principal_groups": groups,
        "queries": golden,
    }


def _sha256(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of *data*.

    Args:
        data: The raw bytes to hash.

    Returns:
        The 64-character hex digest.
    """
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# 1. Determinism — the central invariant.
# ---------------------------------------------------------------------------


def test_v1_pinned_golden_is_byte_identical_across_two_generations(
    tmp_path: Path,
) -> None:
    """Generating the v1 pinned golden twice yields byte-identical output.

    Two consecutive writes of the v1 golden dict into the same target file
    must be exactly equal — the determinism invariant that the Benchmark
    Studio depends on.  SHA-256 equality is asserted in addition to
    bytes-equality so a regression is obvious from the failure message.
    """
    target = tmp_path / "v1.yaml"
    first = _pinned_golden_dict(include_scale=False)
    second = _pinned_golden_dict(include_scale=False)

    target.write_text(
        gbc.GOLDEN_HEADER_V1 + _yaml_dump(first)  # type: ignore[attr-defined]
    )
    first_bytes = target.read_bytes()
    target.write_text(
        gbc.GOLDEN_HEADER_V1 + _yaml_dump(second)  # type: ignore[attr-defined]
    )
    second_bytes = target.read_bytes()

    assert first_bytes == second_bytes, (
        f"v1 golden drift: {_sha256(first_bytes)} vs {_sha256(second_bytes)}"
    )


def test_v1_pinned_corpus_doc_list_is_byte_identical_across_two_generations(
    tmp_path: Path,
) -> None:
    """Generating the corpus doc list twice yields byte-identical output.

    The doc list is the ``(relative_path, acl_group, body)`` triple list
    that drives the corpus manifest.  Two successive calls to
    ``_build_docs()`` must produce exactly the same tuples, in the same
    order, with the same body bytes.
    """
    first = list(gbc._build_docs())  # type: ignore[attr-defined]
    second = list(gbc._build_docs())  # type: ignore[attr-defined]

    assert first == second
    # Belt-and-braces: the concatenated body bytes must hash equal too.
    first_blob = "\n".join(p for _, _, p in first).encode("utf-8")
    second_blob = "\n".join(p for _, _, p in second).encode("utf-8")
    assert _sha256(first_blob) == _sha256(second_blob)


def test_v1_pinned_scale_golden_is_byte_identical_across_two_generations(
    tmp_path: Path,
) -> None:
    """The v1+scale golden (full set) is byte-identical across two generations."""
    target = tmp_path / "v1_scale.yaml"
    first = _pinned_golden_dict(include_scale=True)
    second = _pinned_golden_dict(include_scale=True)

    target.write_text(gbc.GOLDEN_HEADER_V1 + _yaml_dump(first))  # type: ignore[attr-defined]
    first_bytes = target.read_bytes()
    target.write_text(gbc.GOLDEN_HEADER_V1 + _yaml_dump(second))  # type: ignore[attr-defined]
    second_bytes = target.read_bytes()

    assert first_bytes == second_bytes


# ---------------------------------------------------------------------------
# 2. Version pin — the v1 stamp.
# ---------------------------------------------------------------------------


def test_v1_pinned_dataset_block_has_version_v1() -> None:
    """The pinned-v1 dataset block carries ``version == "v1"`` exactly."""
    pinned = _pinned_golden_dict(include_scale=False)
    assert pinned["dataset"]["name"] == PINNED_DATASET_NAME
    assert pinned["dataset"]["version"] == PINNED_DATASET_VERSION
    assert pinned["dataset"]["version"] == "v1"


def test_v1_pinned_dataset_block_is_pinned_via_generator_constant() -> None:
    """The script exposes ``DATASET_VERSION_V1 == "v1"`` so callers can pin."""
    assert gbc.DATASET_VERSION_V1 == "v1"  # type: ignore[attr-defined]


def test_v1_pinned_golden_file_uses_pinned_path() -> None:
    """The script exposes the canonical pinned v1 golden path constant."""
    assert Path(gbc.PINNED_GOLDEN_PATH) == PINNED_GOLDEN_PATH  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 3. Golden integrity — the (source_uri, keyword) model.
# ---------------------------------------------------------------------------


def test_v1_pinned_golden_every_source_resolves_to_a_present_keyword() -> None:
    """Every query with a ``source`` has a ``keyword`` that occurs in the
    target doc (case-insensitive).  The ``(source_uri, keyword)`` golden
    model must hold for the v1 pinned set — otherwise recall is inflated
    or falsified.
    """
    docs_by_uri = {
        gbc._uri(rel): body  # type: ignore[attr-defined]
        for rel, _group, body in gbc._build_docs()  # type: ignore[attr-defined]
    }
    pinned = _pinned_golden_dict(include_scale=False)
    answerable = [q for q in pinned["queries"] if q["source"] and q["keyword"]]

    assert len(answerable) > 0, "v1 golden must have at least one answerable query"

    for q in answerable:
        body = docs_by_uri[q["source"]]
        assert q["keyword"].lower() in body.lower(), (
            f"keyword {q['keyword']!r} not present in {q['source']} for query {q['question']!r}"
        )


def test_v1_pinned_golden_unanswerable_queries_have_null_source_and_keyword() -> None:
    """The ``unanswerable`` query type carries ``source: null`` and
    ``keyword: null`` exactly (zero-gold, must fail closed)."""
    pinned = _pinned_golden_dict(include_scale=False)
    unanswerable = [q for q in pinned["queries"] if q["query_type"] == "unanswerable"]
    assert len(unanswerable) >= 1, "v1 golden must include unanswerable queries"
    for q in unanswerable:
        assert q["source"] is None
        assert q["keyword"] is None


def test_v1_pinned_golden_query_count_matches_legacy_default() -> None:
    """The v1 pinned set has exactly the same query count as the legacy
    non-versioned default (``21`` queries).  This pins the v1 freeze to the
    pre-scale generator output so existing baselines remain comparable.
    """
    pinned = _pinned_golden_dict(include_scale=False)
    assert len(pinned["queries"]) == 21


def test_v1_pinned_golden_principal_groups_match_corpus_groups() -> None:
    """The ``principal_groups`` block in the v1 golden equals the sorted
    set of ACL groups across the v1 corpus (eng, finance, hr, public)."""
    pinned = _pinned_golden_dict(include_scale=False)
    assert pinned["principal_groups"] == ["eng", "finance", "hr", "public"]


# ---------------------------------------------------------------------------
# 4. CLI surface — the script supports the v1 pin via --version / --golden.
# ---------------------------------------------------------------------------


def test_v1_pinned_golden_cli_writes_to_default_v1_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoking the script with ``--version v1`` writes the pinned
    ``eval/golden/benchmark_golden_v1.yaml`` (relative to CWD) by default.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["gen_benchmark_corpus.py", "--version", "v1", "--out", "corpus_out"],
    )
    rc = gbc.main()
    assert rc == 0
    assert (tmp_path / "corpus_out" / "manifest.yaml").is_file()
    assert (tmp_path / "eval" / "golden" / "benchmark_golden_v1.yaml").is_file()


def test_v1_pinned_golden_default_run_still_writes_legacy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default run (no ``--version``) keeps the legacy
    ``eval/golden/benchmark_golden.yaml`` path so existing workflows
    are not broken.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["gen_benchmark_corpus.py", "--out", "corpus_out"],
    )
    rc = gbc.main()
    assert rc == 0
    assert (tmp_path / "corpus_out" / "manifest.yaml").is_file()
    assert (tmp_path / "eval" / "golden" / "benchmark_golden.yaml").is_file()


# ---------------------------------------------------------------------------
# Internal helper — wrap yaml.safe_dump with the pinned defaults the
# generator uses.  Imported lazily so the test can document the wire
# format the script emits.
# ---------------------------------------------------------------------------


def _yaml_dump(obj: dict[str, Any]) -> str:
    """Serialise *obj* the same way ``gen_benchmark_corpus.main()`` does.

    Args:
        obj: The golden document dict.

    Returns:
        A YAML string with ``sort_keys=False`` and ``width=100``.
    """
    import yaml

    return yaml.safe_dump(obj, sort_keys=False, width=100)
