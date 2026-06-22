"""Offline-only import guard — static AST check (lld-eval-observability §11).

Asserts that no hot-path module (api/, agent/, retrieval/, generation/) imports
``iw_rag.eval.judges`` or ``iw_rag.eval.ragas_runner``.  This is a hard
architectural invariant: judges are eval/batch-time only (two-clocks rule,
CONVENTIONS §10) and must never contaminate the hot path.

The check walks each .py file and inspects all ``import`` and ``from … import``
statements.  It does NOT exec the code — purely static.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Directories that are "hot-path" (must not import the offline eval judges)
_HOT_PATH_DIRS = ["api", "agent", "retrieval", "generation"]

# Modules that are offline-only (must never appear in hot-path imports)
_FORBIDDEN_MODULES = {
    "iw_rag.eval.judges",
    "iw_rag.eval.ragas_runner",
}

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "iw_rag"


def _collect_imports(source: str) -> list[str]:
    """Parse a Python source string and return all top-level imported module names.

    Args:
        source: Python source code as a string.

    Returns:
        Flat list of all module name strings referenced by ``import`` or
        ``from … import`` statements anywhere in the file (including nested
        inside functions/classes — we check transitively for safety).
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
            # Also collect "from iw_rag.eval import judges" style
            for alias in node.names:
                imported.append(f"{node.module}.{alias.name}")
    return imported


# Allowlisted composition seams: files in a hot-path package that are NOT on
# the query hot path but are the deliberate composition roots that bridge the
# api into the offline benchmark feature (POST /benchmark/run).  They import the
# offline eval collaborators (judges / ragas_runner / conclusion) LAZILY inside
# factory functions, so nothing offline loads during a /query request — the
# two-clocks rule the guard protects is preserved.  Keep this list tiny.
_ALLOWED_SEAMS = {"benchmark_composition.py"}


def _find_violations() -> list[tuple[Path, str]]:
    """Walk hot-path directories and collect any forbidden imports.

    Returns:
        List of ``(file_path, forbidden_module)`` tuples for every violation.
    """
    violations: list[tuple[Path, str]] = []
    for dir_name in _HOT_PATH_DIRS:
        hot_dir = _SRC_ROOT / dir_name
        if not hot_dir.exists():
            continue
        for py_file in hot_dir.rglob("*.py"):
            if py_file.name in _ALLOWED_SEAMS:
                continue
            source = py_file.read_text(encoding="utf-8")
            imports = _collect_imports(source)
            for imp in imports:
                for forbidden in _FORBIDDEN_MODULES:
                    # Match exact module or submodule (startswith handles
                    # "iw_rag.eval.judges.SomeClass" style dotted references)
                    if imp == forbidden or imp.startswith(forbidden + "."):
                        violations.append((py_file, forbidden))
    return violations


def test_hot_path_does_not_import_offline_judges() -> None:
    """No hot-path module (api/agent/retrieval/generation) imports eval.judges or ragas_runner."""
    violations = _find_violations()
    if violations:
        lines = [
            f"  {f.relative_to(_SRC_ROOT.parent.parent)}: imports '{m}'" for f, m in violations
        ]
        pytest.fail(
            "Offline-only import guard violated — these hot-path files import "
            "offline eval modules (two-clocks rule):\n" + "\n".join(lines)
        )


def test_hot_path_dirs_exist_or_are_known_missing() -> None:
    """At least generation and retrieval hot-path dirs exist from Phase 0."""
    existing = {d for d in _HOT_PATH_DIRS if (_SRC_ROOT / d).exists()}
    # generation and retrieval are Phase-0 deliverables and must always be present
    for required in ("generation", "retrieval"):
        assert required in existing, (
            f"Hot-path dir '{required}' not found under {_SRC_ROOT}; "
            "the guard would pass vacuously without it. "
            f"Checked dirs: {_HOT_PATH_DIRS}"
        )
