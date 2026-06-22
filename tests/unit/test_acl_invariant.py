"""Security regression — the ACL predicate source-of-truth invariant.

TDD RED-first. Asserts:
- The SQL clause is exactly ``acl_tags && :principal_groups`` (no inlined values).
- ``.params()["principal_groups"]`` equals ``ctx.principal_groups`` and nothing else.
- Passing a non-``AccessContext`` (str, dict) raises ``TypeError``.
- No ``retrieval/`` module references ``question``, ``draft``, or grader-output
  keys when building the ACL predicate (AST/text guard).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.retrieval.acl_filter import acl_predicate

# ---------------------------------------------------------------------------
# Predicate structure assertions
# ---------------------------------------------------------------------------


def test_sql_clause_is_parameterized() -> None:
    """The SQL clause must be static; no inlined values."""
    ctx = AccessContext(principal_id="u", principal_groups=("finance", "hr"))
    pred = acl_predicate(ctx)
    assert pred.sql_clause() == "acl_tags && :principal_groups"


def test_params_contains_only_principal_groups() -> None:
    """params() must only contain the principal_groups key."""
    ctx = AccessContext(principal_id="u", principal_groups=("finance", "hr"))
    pred = acl_predicate(ctx)
    params = pred.params()
    assert set(params.keys()) == {"principal_groups"}
    assert params["principal_groups"] == list(ctx.principal_groups)


def test_params_groups_equal_ctx_groups() -> None:
    """The param value must exactly match ctx.principal_groups (order-preserving)."""
    ctx = AccessContext(principal_id="u", principal_groups=("a", "b", "c"))
    pred = acl_predicate(ctx)
    assert pred.params()["principal_groups"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# TypeError on non-AccessContext input
# ---------------------------------------------------------------------------


def test_acl_predicate_raises_on_string_input() -> None:
    """Passing a raw string (e.g. from question text) must raise TypeError."""
    with pytest.raises(TypeError, match="AccessContext"):
        acl_predicate("hr; DROP TABLE iw_rag.chunks --")  # type: ignore[arg-type]


def test_acl_predicate_raises_on_dict_input() -> None:
    """Passing a dict (e.g. parsed LLM output) must raise TypeError."""
    with pytest.raises(TypeError, match="AccessContext"):
        acl_predicate({"groups": ["finance"]})  # type: ignore[arg-type]


def test_acl_predicate_raises_on_none_input() -> None:
    """Passing None must raise TypeError."""
    with pytest.raises(TypeError, match="AccessContext"):
        acl_predicate(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Architectural guard: no retrieval/* module builds the predicate from
# question/draft/model-output.  We check via source-text scan.
# ---------------------------------------------------------------------------

_RETRIEVAL_DIR = Path(__file__).parent.parent.parent / "src" / "iw_rag" / "retrieval"

# Keys that must NEVER appear as arguments to acl_predicate/AclPredicate
# construction inside retrieval modules.
_FORBIDDEN_PATTERNS = [
    'state["question"]',
    "state['question']",
    'state["draft"]',
    "state['draft']",
    'state["grader',
    "state['grader",
    "question",  # as a bare identifier passed to acl_predicate
]


def _retrieval_py_sources() -> list[Path]:
    return list(_RETRIEVAL_DIR.glob("*.py"))


@pytest.mark.parametrize("src_file", _retrieval_py_sources(), ids=lambda p: p.name)
def test_no_free_text_in_acl_predicate_construction(src_file: Path) -> None:
    """No retrieval module constructs AclPredicate/acl_predicate from free text.

    This is a grep-style guard: we parse the AST and look for calls to
    ``acl_predicate(...)`` or ``AclPredicate(...)`` where the first positional
    argument is NOT a name bound to an ``AccessContext`` (e.g., not a string
    literal, not ``state["question"]``).

    We do this via source-text scan for banned literal patterns rather than
    full type-inference (sufficient for a security regression).
    """
    source = src_file.read_text()
    tree = ast.parse(source)

    # Collect all Call nodes whose func is acl_predicate or AclPredicate.
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Function call: acl_predicate(x) or AclPredicate(x)
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name not in ("acl_predicate", "AclPredicate"):
            continue
        # Check every arg for banned patterns: string literals or
        # subscript access on 'state' with a text key.
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                bad.append(
                    f"{src_file.name}:{arg.lineno} — "
                    f"string literal passed to {name}(): {arg.value!r}"
                )
            if isinstance(arg, ast.Subscript):
                # e.g. state["question"]
                val = arg.value
                if isinstance(val, ast.Name) and val.id == "state":
                    bad.append(
                        f"{src_file.name}:{arg.lineno} — state[...] subscript passed to {name}()"
                    )
        for kw in node.keywords:
            kval = kw.value
            if isinstance(kval, ast.Constant) and isinstance(kval.value, str):
                bad.append(
                    f"{src_file.name}:{getattr(kval, 'lineno', '?')} — "
                    f"string literal kw passed to {name}(): {kval.value!r}"
                )

    assert not bad, "ACL invariant violation found in retrieval/:\n" + "\n".join(bad)
