"""Assertion-strength static scanner for the iw-rag test suite.

Walks every ``tests/**/test_*.py`` file and flags ``test_*`` functions whose
assertions are too weak to provide meaningful coverage signal.  Five violation
codes are reported:

``NOASSERT``
    The function has zero ``assert`` statements and no ``pytest.raises`` /
    ``pytest.warns`` context.

``TAUTOLOGY``
    Every assertion in the function is a weak-form check: ``assert x is not
    None``, ``assert isinstance(x, ...)``, ``assert len(x) > 0`` / ``>= 1`` /
    ``!= 0``, ``assert "literal" in x``, or bare truthiness ``assert x`` /
    ``assert x.attr``.  At least one *strong* assertion (``==``, ``<``, ``>``,
    ``pytest.approx``, ``.startswith``, ``match=``) is required to pass.

``MOCKONLY``
    Every assertion in the function is a mock-verification call:
    ``mock.assert_called*`` / ``assert_awaited*`` / ``.called`` /
    ``.call_count``.

``RAISES_BROAD``
    A ``pytest.raises(Exception)`` (bare ``Exception``) or a
    ``pytest.raises(SomeError)`` with **no** ``match=`` keyword.

``RAISES_MULTISTMT``
    More than one statement inside a ``with pytest.raises(...)`` block.

Exit codes
----------
0 — no new (non-baselined) violations.
1 — at least one new violation found.

Baseline support
----------------
``--baseline <file>``    Suppress baselined violations; exit 0 when only those
                         remain, exit 1 on any new violation.
``--write-baseline <file>``
                         Write the full current violation set (sorted, stable)
                         to the baseline file and exit 0.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Violation dataclass (plain tuple for stdlib-only)
# ---------------------------------------------------------------------------


class Violation:
    """A single scanner finding attached to a specific test function.

    Args:
        path: Absolute path to the source file.
        line: Line number of the ``def test_*`` statement.
        code: One of NOASSERT / TAUTOLOGY / MOCKONLY / RAISES_BROAD / RAISES_MULTISTMT.
        name: The bare test function name (e.g. ``test_foo``).
        reason: Human-readable explanation shown on the terminal.
    """

    __slots__ = ("path", "line", "code", "name", "reason")

    def __init__(
        self,
        path: str,
        line: int,
        code: str,
        name: str,
        reason: str,
    ) -> None:
        self.path = path
        self.line = line
        self.code = code
        self.name = name
        self.reason = reason

    def display(self) -> str:
        """Return the terminal-display line for this violation."""
        return f"{self.path}:{self.line}: {self.code} {self.name} — {self.reason}"

    def baseline_key(self) -> str:
        """Return the stable baseline identifier for this violation."""
        return f"{self.path}::{self.name}::{self.code}"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _is_pytest_raises_call(node: ast.expr) -> bool:
    """Return True if *node* is a ``pytest.raises(...)`` call expression.

    Args:
        node: An AST expression node to inspect.

    Returns:
        True when the node looks like ``pytest.raises(...)``.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    # pytest.raises(...)
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "raises"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def _is_pytest_warns_call(node: ast.expr) -> bool:
    """Return True if *node* is a ``pytest.warns(...)`` call expression.

    Args:
        node: An AST expression node to inspect.

    Returns:
        True when the node looks like ``pytest.warns(...)``.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "warns"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def _raises_with_nodes(body: list[ast.stmt]) -> list[ast.With]:
    """Collect all ``with pytest.raises(...)`` statement nodes in *body* (shallow).

    Args:
        body: A list of AST statement nodes (the function body).

    Returns:
        All ``ast.With`` nodes whose first context-manager is pytest.raises.
    """
    result: list[ast.With] = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if (
            isinstance(node, ast.With)
            and node.items
            and _is_pytest_raises_call(node.items[0].context_expr)
        ):
            result.append(node)
    return result


def _has_pytest_raises_or_warns(body: list[ast.stmt]) -> bool:
    """Return True if the body contains any ``with pytest.raises/warns`` context.

    Args:
        body: A list of AST statement nodes.

    Returns:
        True when at least one pytest.raises / pytest.warns context is present.
    """
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.With):
            for item in node.items:
                ce = item.context_expr
                if _is_pytest_raises_call(ce) or _is_pytest_warns_call(ce):
                    return True
    return False


def _has_skip_or_fail(body: list[ast.stmt]) -> bool:
    """Return True if the body calls ``pytest.skip``/``pytest.fail``/``pytest.xfail``.

    These are valid test terminals: ``pytest.skip(...)`` marks a deliberate,
    documented DEFERRED placeholder, and ``pytest.fail(...)`` IS the assertion
    mechanism for AST/manifest-guard tests (fail-on-violation). Neither is a
    "no-assert" violation.

    Args:
        body: A list of AST statement nodes (the function body).

    Returns:
        True when at least one pytest.skip / pytest.fail / pytest.xfail call exists.
    """
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("skip", "fail", "xfail")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest"
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Assertion-strength classification
# ---------------------------------------------------------------------------


def _is_weak_assert(node: ast.Assert) -> bool:
    """Return True when the assert statement is a recognised weak/tautology form.

    Weak forms (all return True):
    - ``assert x``                       bare truthiness
    - ``assert x.attr``                  attribute truthiness
    - ``assert x is not None``
    - ``assert isinstance(x, ...)``
    - ``assert len(x) > 0`` / ``>= 1`` / ``!= 0``
    - ``assert "literal" in x``

    Args:
        node: An ``ast.Assert`` node.

    Returns:
        True when the assertion matches a weak pattern.
    """
    test = node.test

    # bare truthiness: assert x  (Name or Attribute)
    if isinstance(test, (ast.Name, ast.Attribute)):
        return True

    # assert x is not None
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.IsNot):
        comp = test.comparators[0]
        if isinstance(comp, ast.Constant) and comp.value is None:
            return True

    # assert isinstance(x, ...)
    if (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Name)
        and test.func.id == "isinstance"
    ):
        return True

    # assert len(x) > 0 / >= 1 / != 0
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left = test.left
        if isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "len":
            op = test.ops[0]
            comp = test.comparators[0]
            if isinstance(comp, ast.Constant):
                if isinstance(op, ast.Gt) and comp.value == 0:
                    return True
                if isinstance(op, ast.GtE) and comp.value == 1:
                    return True
                if isinstance(op, ast.NotEq) and comp.value == 0:
                    return True

    # assert "literal" in x  — string constant is the left side
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.In)
        and isinstance(test.left, ast.Constant)
        and isinstance(test.left.value, str)
    ):
        return True

    # assert isinstance(x, T) and len(x) > 0  — compound of only weak forms
    # (BoolOp And whose operands are all weak)
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        sub_asserts = [ast.Assert(test=v, msg=None) for v in test.values]
        if all(_is_weak_assert(s) for s in sub_asserts):
            return True

    return False


def _is_mock_assert_call(node: ast.stmt) -> bool:
    """Return True when *node* is a mock assertion call statement.

    Recognised patterns:
    - ``mock.assert_called*()``
    - ``mock.assert_awaited*()``
    - ``assert mock.called``
    - ``assert mock.call_count ...``
    - ``_ = mock.called``  (attribute access as expression)

    Args:
        node: An AST statement node.

    Returns:
        True when the statement is a recognised mock-verification form.
    """
    # expr statement: mock.assert_called_once()
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Attribute):
            attr = call.func.attr
            if attr.startswith("assert_called") or attr.startswith("assert_awaited"):
                return True

    # assert statement: assert mock.called / assert mock.call_count > 0
    if isinstance(node, ast.Assert):
        test = node.test
        # assert mock.called  / assert mock.call_count
        if isinstance(test, ast.Attribute) and test.attr in ("called", "call_count"):
            return True
        # assert mock.call_count == N  or similar
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Attribute)
            and test.left.attr in ("called", "call_count")
        ):
            return True

    return False


# ---------------------------------------------------------------------------
# Per-function analysis
# ---------------------------------------------------------------------------


def _analyse_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    path: str,
) -> list[Violation]:
    """Analyse a single test function and return any violations.

    Args:
        func: The AST function definition node.
        path: The file path (used in violation output).

    Returns:
        A list of Violation objects (may be empty).
    """
    body = func.body
    line = func.lineno
    name = func.name

    # Collect all assert nodes and with-pytest.raises nodes inside the function.
    all_stmts: list[ast.stmt] = list(ast.walk(ast.Module(body=body, type_ignores=[])))

    assert_nodes = [n for n in all_stmts if isinstance(n, ast.Assert)]
    raises_withs = _raises_with_nodes(body)
    has_raises_or_warns = _has_pytest_raises_or_warns(body)
    has_skip_or_fail = _has_skip_or_fail(body)

    violations: list[Violation] = []

    # ------------------------------------------------------------------
    # NOASSERT: no assert AND no pytest.raises/warns context AND no
    # pytest.skip/fail/xfail terminal (a deliberate DEFERRED skip or a
    # fail-on-violation AST guard is not a missing-assertion).
    # ------------------------------------------------------------------
    if not assert_nodes and not has_raises_or_warns and not has_skip_or_fail:
        violations.append(
            Violation(
                path=path,
                line=line,
                code="NOASSERT",
                name=name,
                reason="no assert statements and no pytest.raises/warns context",
            )
        )
        return violations  # no point checking other codes

    # ------------------------------------------------------------------
    # RAISES_BROAD / RAISES_MULTISTMT — inspect each with pytest.raises block
    # ------------------------------------------------------------------
    for with_node in raises_withs:
        call = with_node.items[0].context_expr
        assert isinstance(call, ast.Call)  # guaranteed by _raises_with_nodes

        # RAISES_BROAD: bare Exception or no match= keyword
        exc_arg = call.args[0] if call.args else None
        has_match_kwarg = any(kw.arg == "match" for kw in call.keywords)

        is_bare_exception = (
            exc_arg is not None and isinstance(exc_arg, ast.Name) and exc_arg.id == "Exception"
        )

        if is_bare_exception:
            violations.append(
                Violation(
                    path=path,
                    line=with_node.lineno,
                    code="RAISES_BROAD",
                    name=name,
                    reason="pytest.raises(Exception) — use a specific exception type with match=",
                )
            )
        elif not has_match_kwarg:
            violations.append(
                Violation(
                    path=path,
                    line=with_node.lineno,
                    code="RAISES_BROAD",
                    name=name,
                    reason="pytest.raises(...) without match= — add match= to anchor the message",
                )
            )

        # RAISES_MULTISTMT: more than one statement in the with body
        # (comments and pass don't count; filter to real statements)
        real_stmts = [s for s in with_node.body if not isinstance(s, ast.Pass)]
        if len(real_stmts) > 1:
            violations.append(
                Violation(
                    path=path,
                    line=with_node.lineno,
                    code="RAISES_MULTISTMT",
                    name=name,
                    reason=(
                        f"pytest.raises block has {len(real_stmts)} statements "
                        "(only one statement should be inside the raises context)"
                    ),
                )
            )

    # If there are no assert statements at all (only pytest.raises contexts), stop here.
    if not assert_nodes:
        return violations

    # ------------------------------------------------------------------
    # Strength classification of assert statements
    # ------------------------------------------------------------------
    weak_asserts = [n for n in assert_nodes if _is_weak_assert(n)]
    mock_asserts = [n for n in assert_nodes if _is_mock_assert_call(n)]

    # A strong assert is one that is neither weak nor a mock check.
    strong_asserts = [
        n for n in assert_nodes if not _is_weak_assert(n) and not _is_mock_assert_call(n)
    ]

    # TAUTOLOGY: every assert is weak (no strong assert present)
    if assert_nodes and not strong_asserts and mock_asserts == [] and weak_asserts:
        violations.append(
            Violation(
                path=path,
                line=line,
                code="TAUTOLOGY",
                name=name,
                reason=(
                    "all assertions are weak-form checks "
                    "(is not None / isinstance / len > 0 / bare truthiness)"
                ),
            )
        )

    # MOCKONLY: every assert is a mock verification call
    if assert_nodes and not strong_asserts and not weak_asserts and mock_asserts:
        violations.append(
            Violation(
                path=path,
                line=line,
                code="MOCKONLY",
                name=name,
                reason=(
                    "all assertions are mock.assert_called* / .called / .call_count checks — "
                    "add a value assertion"
                ),
            )
        )

    return violations


# ---------------------------------------------------------------------------
# File-level walker
# ---------------------------------------------------------------------------


def _scan_file(path: Path) -> list[Violation]:
    """Parse *path* and return violations for every ``test_*`` function.

    Args:
        path: Absolute path to a Python test file.

    Returns:
        A (possibly empty) list of Violation objects found in the file.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        print(f"WARNING: skipping {path} — SyntaxError: {exc}", file=sys.stderr)
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            violations.extend(_analyse_function(node, str(path)))
    return violations


def _scan_tree(root: Path) -> list[Violation]:
    """Walk the tests tree and collect violations from all ``test_*.py`` files.

    Args:
        root: The ``tests/`` directory to scan.

    Returns:
        A flat list of all violations found, in file-path order.
    """
    violations: list[Violation] = []
    for test_file in sorted(root.rglob("test_*.py")):
        violations.extend(_scan_file(test_file))
    return violations


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def _read_baseline(path: Path) -> set[str]:
    """Parse a baseline file and return the set of suppression keys.

    Lines starting with ``#`` are comments and are ignored.  Blank lines are
    also ignored.

    Args:
        path: Path to the baseline file.

    Returns:
        A set of ``path::name::CODE`` strings to suppress.
    """
    keys: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        keys.add(line)
    return keys


def _write_baseline(path: Path, violations: list[Violation]) -> None:
    """Write all violations as a sorted stable baseline file.

    The file includes a header comment explaining it is a cleanup backlog,
    then one ``path::name::CODE`` key per violation.

    Args:
        path: Destination path for the baseline file.
        violations: All violations to grandfather.
    """
    keys = sorted({v.baseline_key() for v in violations})
    lines = [
        "# Assertion-strength baseline — cleanup backlog, NOT an accept-list.",
        "# Every entry here is a pre-existing weak-assertion violation that must",
        "# be fixed over time.  Do NOT add new entries without fixing an old one.",
        "# Generated by: uv run python scripts/check_test_assertions.py --write-baseline",
        "#",
        "# Format: path::test_name::CODE",
        "#",
    ]
    lines.extend(keys)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured ArgumentParser ready to parse sys.argv.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Assertion-strength static scanner.  "
            "Walks tests/**/test_*.py and reports weak test functions."
        )
    )
    parser.add_argument(
        "tests_root",
        nargs="?",
        default=None,
        help="Path to the tests directory (default: <repo-root>/tests).",
    )
    parser.add_argument(
        "--baseline",
        metavar="FILE",
        help=(
            "Baseline file of grandfathered violations (path::name::CODE per line). "
            "Suppresses listed violations; exits 1 only on new ones."
        ),
    )
    parser.add_argument(
        "--write-baseline",
        metavar="FILE",
        dest="write_baseline",
        help=(
            "Write the full current violation set to FILE as a baseline "
            "and exit 0 (used to grandfather existing violations)."
        ),
    )
    return parser


def main() -> None:
    """CLI entry point — parse arguments, run scanner, report findings."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Resolve tests root
    if args.tests_root:
        tests_root = Path(args.tests_root).resolve()
    else:
        # Default: <repo-root>/tests, where repo-root is two levels up from this script.
        tests_root = Path(__file__).resolve().parent.parent / "tests"

    if not tests_root.is_dir():
        print(f"ERROR: tests root not found: {tests_root}", file=sys.stderr)
        sys.exit(1)

    violations = _scan_tree(tests_root)

    # --write-baseline mode: write and exit 0
    if args.write_baseline:
        out_path = Path(args.write_baseline).resolve()
        _write_baseline(out_path, violations)
        print(
            f"Wrote {len({v.baseline_key() for v in violations})} entries to baseline: {out_path}"
        )
        sys.exit(0)

    # Load baseline (if provided)
    baseline_keys: set[str] = set()
    if args.baseline:
        baseline_path = Path(args.baseline).resolve()
        if not baseline_path.is_file():
            print(f"ERROR: baseline file not found: {baseline_path}", file=sys.stderr)
            sys.exit(1)
        baseline_keys = _read_baseline(baseline_path)

    # Partition violations into new vs suppressed
    new_violations = [v for v in violations if v.baseline_key() not in baseline_keys]
    suppressed = [v for v in violations if v.baseline_key() in baseline_keys]

    # Print new violations
    for v in new_violations:
        print(v.display())

    # Summary line
    total = len(violations)
    new_count = len(new_violations)
    sup_count = len(suppressed)
    print(f"\n{total} violation(s) found: {new_count} new, {sup_count} suppressed by baseline.")

    if new_violations:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
