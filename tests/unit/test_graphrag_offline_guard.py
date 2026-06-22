"""Offline-only import guard for GraphRAG indexer and index_graph script (two-clocks rule).

Asserts that no hot-path module (api/, agent/, retrieval/, generation/) imports
``iw_rag.modes.graphrag.indexer``, ``iw_rag.modes.graphrag.store``, or
``scripts.index_graph``.  These are OFFLINE/batch only — they MUST NOT be
reachable from the ``/query`` hot path.

GPU-contention is the existential risk (R-00106/F2, CLAUDE.md two-clocks rule):
the extraction model (Qwen3-4B) must never queue against the hot-path generator
(Qwen3-14B).  The indexer runs overnight/batch via ``scripts/index_graph.py``
only.

The check uses a pure static AST walk — no code execution.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Directories considered hot-path (must not import offline graphrag modules)
_HOT_PATH_DIRS = ["api", "agent", "retrieval", "generation"]

# Modules that are offline-only for the graphrag path.
# Note: iw_rag.modes.graphrag.types and iw_rag.modes.graphrag.store are offline-only
# (store issues Cypher/SQL that must not run on hot path). The retriever (WI-029)
# is the hot-path component — but it does not exist yet.
_FORBIDDEN_GRAPHRAG_MODULES = {
    "iw_rag.modes.graphrag.indexer",  # OFFLINE/batch only — extraction model
    "iw_rag.modes.graphrag.store",  # OFFLINE/batch only — AGE graph SQL owner
    "iw_rag.modes.graphrag",  # top-level import pulls indexer + store
    "scripts.index_graph",
    "index_graph",
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


def _find_graphrag_violations() -> list[tuple[Path, str]]:
    """Walk hot-path directories and collect any forbidden graphrag imports.

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
                for forbidden in _FORBIDDEN_GRAPHRAG_MODULES:
                    if imp == forbidden or imp.startswith(forbidden + "."):
                        violations.append((py_file, forbidden))
    return violations


def test_hot_path_does_not_import_graphrag_indexer() -> None:
    """No hot-path module (api/agent/retrieval/generation) imports modes.graphrag.indexer.

    GraphIndexer is OFFLINE/batch only — the two-clocks rule (CLAUDE.md) and the
    GPU-contention risk (R-00106/F2) require it to NEVER be called from the /query
    hot path.
    """
    violations = _find_graphrag_violations()
    if violations:
        lines = [
            f"  {f.relative_to(_SRC_ROOT.parent.parent)}: imports '{m}'" for f, m in violations
        ]
        pytest.fail(
            "GraphRAG offline-only guard violated — these hot-path files import "
            "the offline GraphIndexer or AgeGraphStore (two-clocks rule / GPU-contention "
            "rule, R-00106/F2):\n" + "\n".join(lines)
        )


def test_index_graph_script_exists_as_offline_only_entrypoint() -> None:
    """scripts/index_graph.py exists as the offline graph indexing entrypoint."""
    script = _SCRIPTS_ROOT / "index_graph.py"
    assert script.exists(), (
        "scripts/index_graph.py is missing — the offline graph index entrypoint "
        "must exist and be reachable only from batch/offline scripts, never from /query. "
        "(two-clocks rule, CLAUDE.md)"
    )


def test_index_graph_script_does_not_import_api_or_agent_routes() -> None:
    """scripts/index_graph.py does not import API routes or the agent loop.

    The script may import iw_rag.generation.llm (for VllmClient — it is a
    shared client used both online and offline) but must not reach the API
    request-handling surface or the LangGraph agent nodes (those are coupled
    to the query runtime).
    """
    script = _SCRIPTS_ROOT / "index_graph.py"
    if not script.exists():
        pytest.skip("index_graph.py not yet created")

    # True hot-path entry surfaces (never batch-callable)
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
            "scripts/index_graph.py imports hot-path routing modules — "
            "it must be OFFLINE/batch only:\n" + "\n".join(lines)
        )
