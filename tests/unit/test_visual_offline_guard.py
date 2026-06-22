"""Offline-only import guard for the visual captioner and indexer (two-clocks rule).

Asserts that no hot-path module (api/, agent/, retrieval/, generation/) imports
``iw_rag.modes.visual.captioner``, ``iw_rag.modes.visual.indexer``, or
``scripts.index_visual``.  These are OFFLINE/batch only — they MUST NOT be
reachable from the ``/query`` hot path (VRAM rule: captioner + generator cannot
be co-resident; lld-multimodal §3, CLAUDE.md two-clocks rule).

The ColVisionEncoder is allowed on the hot path (``encode_query`` is called by
VisualRetriever), but the VisualIndexer is offline-only (``encode_pages`` +
``pool_to_n`` + ``upsert`` — all batch-time operations).

The check walks each .py file and inspects all ``import`` and ``from … import``
statements via a pure static AST walk — no code execution.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Directories considered hot-path (must not import offline visual modules)
_HOT_PATH_DIRS = ["api", "agent", "retrieval", "generation"]

# Modules that are offline-only for the visual path (WI-030 + WI-031).
# Note: iw_rag.modes.visual.encoder IS allowed on the hot path (encode_query
# is called by VisualRetriever); the INDEXER is offline-only.
_FORBIDDEN_VISUAL_MODULES = {
    "iw_rag.modes.visual.captioner",
    "iw_rag.modes.visual.indexer",  # WI-031: VisualIndexer is OFFLINE/batch only
    "iw_rag.modes.visual",  # top-level import pulls captioner + indexer
    "scripts.index_visual",
    "index_visual",
}

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "iw_rag"
_SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"


def _collect_imports(source: str) -> list[str]:
    """Parse a Python source string and return all imported module names.

    Args:
        source: Python source code as a string.

    Returns:
        Flat list of all module name strings referenced by any ``import`` or
        ``from … import`` statement anywhere in the file (including inside
        functions/classes — checked transitively for safety).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
            for alias in node.names:
                imported.append(f"{node.module}.{alias.name}")
    return imported


def _find_visual_violations() -> list[tuple[Path, str]]:
    """Walk hot-path directories and collect any forbidden visual imports.

    Returns:
        List of ``(file_path, forbidden_module)`` tuples for every violation found.
    """
    violations: list[tuple[Path, str]] = []
    for dir_name in _HOT_PATH_DIRS:
        hot_dir = _SRC_ROOT / dir_name
        if not hot_dir.exists():
            continue
        for py_file in hot_dir.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            imports = _collect_imports(source)
            for imp in imports:
                for forbidden in _FORBIDDEN_VISUAL_MODULES:
                    if imp == forbidden or imp.startswith(forbidden + "."):
                        violations.append((py_file, forbidden))
    return violations


def test_hot_path_does_not_import_visual_captioner() -> None:
    """No hot-path module (api/agent/retrieval/generation) imports modes.visual.captioner."""
    violations = _find_visual_violations()
    if violations:
        lines = [
            f"  {f.relative_to(_SRC_ROOT.parent.parent)}: imports '{m}'" for f, m in violations
        ]
        pytest.fail(
            "Visual-captioner offline-only guard violated — these hot-path files import "
            "the offline VlmCaptioner (two-clocks rule / VRAM rule):\n" + "\n".join(lines)
        )
    assert violations == [], "Expected zero violations; unreachable but explicit for the scanner"


def test_hot_path_does_not_import_visual_indexer() -> None:
    """No hot-path module imports modes.visual.indexer (VisualIndexer is OFFLINE-only).

    The ColVisionEncoder IS allowed on the hot path (encode_query for VisualRetriever)
    but the VisualIndexer is the OFFLINE batch pipeline and MUST NOT appear in any
    hot-path module (api/, agent/, retrieval/, generation/).
    """
    # Collect violations for the indexer specifically so the error message is targeted
    violations: list[tuple[Path, str]] = []
    indexer_module = "iw_rag.modes.visual.indexer"
    for dir_name in _HOT_PATH_DIRS:
        hot_dir = _SRC_ROOT / dir_name
        if not hot_dir.exists():
            continue
        for py_file in hot_dir.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            imports = _collect_imports(source)
            for imp in imports:
                if imp == indexer_module or imp.startswith(indexer_module + "."):
                    violations.append((py_file, indexer_module))
    if violations:
        lines = [
            f"  {f.relative_to(_SRC_ROOT.parent.parent)}: imports '{m}'" for f, m in violations
        ]
        pytest.fail(
            "VisualIndexer offline-only guard violated (WI-031 two-clocks rule) — "
            "these hot-path files import the offline VisualIndexer:\n" + "\n".join(lines)
        )


def test_index_visual_script_exists_as_offline_only_entrypoint() -> None:
    """scripts/index_visual.py exists as the offline captioning entrypoint."""
    script = _SCRIPTS_ROOT / "index_visual.py"
    assert script.exists(), (
        "scripts/index_visual.py is missing — the offline visual index entrypoint "
        "must exist and be reachable only from batch/offline scripts, never from /query."
    )


def test_index_visual_script_does_not_import_api_or_agent_routes() -> None:
    """scripts/index_visual.py does not import API routes or the agent loop.

    The script may import iw_rag.generation.llm (for VllmClient) because that
    module is a shared client used both online and offline — it is not itself a
    hot-path router.  The structural rule is that the script must not reach the
    API request-handling surface or the LangGraph agent nodes, which would couple
    the offline batch path to the query runtime.
    """
    script = _SCRIPTS_ROOT / "index_visual.py"
    if not script.exists():
        pytest.skip("index_visual.py not yet created")

    # These are the true hot-path entry surfaces (never batch-callable):
    # api/ = FastAPI request handlers; agent/ = LangGraph orchestration nodes;
    # retrieval/ = the spine that runs on /query.
    # generation/llm.py is a shared VllmClient — used offline too (judges, captioner).
    forbidden_hot_path = {
        "iw_rag.api",
        "iw_rag.agent",
        "iw_rag.retrieval",
    }
    source = script.read_text(encoding="utf-8")
    imports = _collect_imports(source)
    violations = [
        (imp, f)
        for imp in imports
        for f in forbidden_hot_path
        if imp == f or imp.startswith(f + ".")
    ]
    if violations:
        lines = [f"  imports '{imp}' (forbidden: '{f}')" for imp, f in violations]
        pytest.fail(
            "scripts/index_visual.py imports hot-path routing modules — "
            "it must be OFFLINE/batch only:\n" + "\n".join(lines)
        )
