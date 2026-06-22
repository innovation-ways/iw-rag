"""Migration chain integrity guard — pure unit (no database required).

Uses Alembic's ``ScriptDirectory`` API to introspect the migration graph
without a live Postgres connection.  Catches two failure modes that have
bitten this repo (project memory: ``down_revision="PENDING"`` unresolved in
head, and undetected merge-induced forks).

Guard constants — **must be bumped when a migration merges**:
- ``_EXPECTED_HEAD``: the single revision id that ``ScriptDirectory.get_heads()``
  must return.  Update when a new migration is merged as the chain's tip.
- ``_EXPECTED_MIGRATION_COUNT``: total number of revision files in the
  graph.  Update on every merge of a new migration (±1 per file added).

See tests/CLAUDE.md gotcha line 47 and the project memory entry
"iw-rag document management" / "iw-ai-core migration pipeline orch-only".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.script import ScriptDirectory

from iw_rag.storage.migrations import make_alembic_config

# ---------------------------------------------------------------------------
# Pinned constants — MUST be updated when a migration merges
# ---------------------------------------------------------------------------

# The single revision id that must be the chain's head.
# Bump this whenever a new migration is added (becomes the new tip).
_EXPECTED_HEAD = "0010_conversations"

# Total number of revision files in the chain.
# Bump by +1 each time a new migration file is merged.
_EXPECTED_MIGRATION_COUNT = 10

# ---------------------------------------------------------------------------
# Shared fixture — ScriptDirectory loaded once for the module
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _REPO_ROOT / "alembic.ini"


def _script_directory() -> ScriptDirectory:
    """Build a ScriptDirectory from the real alembic.ini.

    Uses the repo-root-relative path so the test is independent of the
    caller's working directory.

    Returns:
        A configured :class:`alembic.script.ScriptDirectory` for the
        project's migration graph.
    """
    from alembic.config import Config

    cfg = Config(str(_ALEMBIC_INI))
    return ScriptDirectory.from_config(cfg)


# ===========================================================================
# 1. Exactly one head
# ===========================================================================


class TestSingleHead:
    """Guard against branch forks in the migration graph."""

    def test_exactly_one_head(self) -> None:
        """ScriptDirectory.get_heads() returns exactly one revision id.

        A fork (two heads) means a merge migration is needed.  A missing head
        means the graph is broken.  Either way is a bug.
        """
        sd = _script_directory()
        heads = sd.get_heads()
        assert len(heads) == 1, (
            f"Expected exactly 1 migration head; got {len(heads)}: {heads}. "
            "A fork means 'alembic merge' is required. "
            "Bump _EXPECTED_HEAD in this file after merging."
        )

    def test_head_matches_expected_constant(self) -> None:
        """The single head revision id equals the pinned _EXPECTED_HEAD constant.

        Fails when a new migration is merged and the constant is not updated,
        providing an explicit reminder to bump the guard.
        """
        sd = _script_directory()
        (head,) = sd.get_heads()
        assert head == _EXPECTED_HEAD, (
            f"Head is now {head!r} but _EXPECTED_HEAD is still {_EXPECTED_HEAD!r}. "
            "Update _EXPECTED_HEAD in this file to the new head after merging a migration."
        )


# ===========================================================================
# 2. Migration count
# ===========================================================================


class TestMigrationCount:
    """Guard against unintentional addition or removal of migration files."""

    def test_migration_count_matches_expected_constant(self) -> None:
        """Total number of revisions equals the pinned _EXPECTED_MIGRATION_COUNT.

        Fails when a new migration is merged and the constant is not bumped,
        or when a migration file is accidentally deleted.  Bump
        _EXPECTED_MIGRATION_COUNT by +1 (or -1) on each merge.
        """
        sd = _script_directory()
        actual = len(list(sd.walk_revisions()))
        assert actual == _EXPECTED_MIGRATION_COUNT, (
            f"Expected {_EXPECTED_MIGRATION_COUNT} migrations; got {actual}. "
            "Bump _EXPECTED_MIGRATION_COUNT in this file when a migration is added or removed."
        )


# ===========================================================================
# 3. No PENDING down_revision reaches the graph
# ===========================================================================


class TestNoPendingDownRevision:
    """Guard against the PENDING-sentinel bug that has bitten this repo.

    The engineering convention (alembic.ini comment, engineering-conventions §5)
    is that new migrations are authored with ``down_revision = "PENDING"`` and
    the IW daemon rewrites that sentinel to the real chain head at merge time.
    If the daemon fails (or a migration is merged manually without the daemon),
    the sentinel escapes into the graph and ``alembic upgrade head`` raises
    ``KeyError: 'PENDING'``.  This test prevents silent slip-through.
    """

    def test_no_pending_down_revision_in_chain(self) -> None:
        """No revision in walk_revisions() has down_revision == 'PENDING'.

        Walks every revision in the resolved graph and asserts none carry the
        unresolved sentinel.  If this test fails on a new migration that was
        just merged without daemon rebase, the fix is to hand-edit the
        migration's ``down_revision`` to the real predecessor (see project
        memory: iw-ai-core migration pipeline orch-only).
        """
        sd = _script_directory()
        pending_offenders: list[str] = []
        for rev in sd.walk_revisions():
            dr = rev.down_revision
            # down_revision can be None (base), a str, or a tuple (merge point).
            if isinstance(dr, str) and dr == "PENDING":
                pending_offenders.append(rev.revision)
            elif isinstance(dr, tuple):
                for item in dr:
                    if item == "PENDING":
                        pending_offenders.append(rev.revision)

        assert not pending_offenders, (
            f"Revisions with unresolved down_revision='PENDING': {pending_offenders}. "
            "The IW daemon rebase did not run (or ran partially). "
            "Hand-edit the down_revision to the real predecessor revision id "
            "(see project memory: iw-ai-core migration pipeline orch-only)."
        )

    def test_no_non_none_placeholder_down_revision(self) -> None:
        """No down_revision is a string that starts with 'PENDING' or is 'None' (string).

        Widens the guard to catch off-by-one variants of the sentinel
        (e.g. ``down_revision = "PENDING_0008"`` or the string literal ``"None"``
        instead of the Python ``None`` singleton).
        """
        sd = _script_directory()
        suspicious: list[tuple[str, str]] = []
        for rev in sd.walk_revisions():
            dr = rev.down_revision
            candidates = (
                [dr] if isinstance(dr, str) else (list(dr) if isinstance(dr, tuple) else [])
            )
            for item in candidates:
                if isinstance(item, str) and (item.upper().startswith("PENDING") or item == "None"):
                    suspicious.append((rev.revision, item))

        assert not suspicious, (
            f"Revisions with suspicious placeholder down_revision values: {suspicious}."
        )


# ===========================================================================
# 4. Linear, fully-resolvable chain
# ===========================================================================


class TestLinearChain:
    """Guard that walk_revisions() traverses the entire chain without errors."""

    def test_walk_revisions_resolves_fully(self) -> None:
        """walk_revisions() from head to base completes without raising.

        If any down_revision is unresolvable (e.g. refers to a revision id
        that does not exist in the versions directory), Alembic raises
        ResolutionError during the walk.  This test surfaces that immediately.
        """
        sd = _script_directory()
        # Materialise the entire generator — raises on the first unresolvable link.
        revisions = list(sd.walk_revisions())
        # A full head→base walk must yield every revision exactly once.
        assert len(revisions) == _EXPECTED_MIGRATION_COUNT
        # And the last walked revision is the base (down_revision is None).
        assert revisions[-1].down_revision is None

    def test_chain_has_exactly_one_base(self) -> None:
        """Exactly one revision has down_revision=None (the chain has a single root).

        Multiple bases indicate an unintended branch point created before
        the initial migration.
        """
        sd = _script_directory()
        bases = [rev.revision for rev in sd.walk_revisions() if rev.down_revision is None]
        assert len(bases) == 1, (
            f"Expected exactly 1 base revision (down_revision=None); got {len(bases)}: {bases}."
        )

    def test_every_non_base_down_revision_is_a_known_revision(self) -> None:
        """Every non-None down_revision points at a revision id in the graph.

        This is a belt-and-suspenders assertion that walk_revisions() can
        also pass (since Alembic itself would raise for unknown refs), but
        it produces a more informative failure message.
        """
        sd = _script_directory()
        all_revisions = {rev.revision for rev in sd.walk_revisions()}
        broken_links: list[tuple[str, str]] = []
        for rev in sd.walk_revisions():
            dr = rev.down_revision
            if dr is None:
                continue
            targets = [dr] if isinstance(dr, str) else list(dr)
            for target in targets:
                if target not in all_revisions:
                    broken_links.append((rev.revision, target))

        assert not broken_links, (
            f"Revisions whose down_revision targets an unknown revision id: {broken_links}. "
            "Check for typos or missing migration files."
        )


# ===========================================================================
# 5. make_alembic_config coverage
# ===========================================================================


class TestMakeAlembicConfig:
    """Coverage for :func:`iw_rag.storage.migrations.make_alembic_config`.

    The source is at 0% coverage; these tests exercise both call paths
    (explicit DSN and Settings fallback) and assert the returned Config
    has the expected options set.
    """

    _EXPLICIT_DSN = "postgresql+psycopg://unit_test_user:unit_test_pass@db-unit:5432/unit_test_db"

    def test_explicit_dsn_sets_sqlalchemy_url(self) -> None:
        """make_alembic_config(dsn=...) sets sqlalchemy.url to the supplied value."""
        cfg = make_alembic_config(dsn=self._EXPLICIT_DSN)
        assert cfg.get_main_option("sqlalchemy.url") == self._EXPLICIT_DSN

    def test_explicit_dsn_sets_script_location_to_alembic(self) -> None:
        """make_alembic_config(dsn=...) preserves script_location = 'alembic' from alembic.ini."""
        cfg = make_alembic_config(dsn=self._EXPLICIT_DSN)
        assert cfg.get_main_option("script_location") == "alembic"

    def test_none_dsn_falls_back_to_settings(self) -> None:
        """make_alembic_config(dsn=None) reads the DSN from Settings.database.dsn.

        The baseline env set by tests/conftest.py provides
        IW_RAG__DATABASE__DSN so get_settings() succeeds without a real DB.
        The returned sqlalchemy.url must match the baseline DSN.
        """
        from iw_rag.settings import get_settings

        expected_dsn = get_settings().database.dsn
        cfg = make_alembic_config(dsn=None)
        assert cfg.get_main_option("sqlalchemy.url") == expected_dsn

    def test_none_dsn_config_is_usable_by_script_directory(self) -> None:
        """Config from make_alembic_config(dsn=None) can build a valid ScriptDirectory.

        Ensures the script_location option is intact and ScriptDirectory
        resolves the migration graph without errors.
        """
        cfg = make_alembic_config(dsn=None)
        sd = ScriptDirectory.from_config(cfg)
        heads = sd.get_heads()
        # We don't re-pin the head here — that's TestSingleHead's job.
        # We just assert the config is wired well enough for ScriptDirectory.
        assert len(heads) == 1

    def test_explicit_dsn_config_is_usable_by_script_directory(self) -> None:
        """Config from make_alembic_config(dsn=...) builds a valid ScriptDirectory."""
        cfg = make_alembic_config(dsn=self._EXPLICIT_DSN)
        sd = ScriptDirectory.from_config(cfg)
        heads = sd.get_heads()
        assert len(heads) == 1

    def test_explicit_dsn_overrides_placeholder_in_alembic_ini(self) -> None:
        """Explicit DSN overrides the placeholder postgresql+psycopg://localhost/iw_rag.

        The alembic.ini contains a placeholder URL; make_alembic_config must
        replace it with the caller-supplied value, not leave the placeholder.
        """
        cfg = make_alembic_config(dsn=self._EXPLICIT_DSN)
        url = cfg.get_main_option("sqlalchemy.url")
        assert "localhost/iw_rag" not in url, (
            "make_alembic_config did not override the placeholder URL from alembic.ini."
        )
        assert url == self._EXPLICIT_DSN


# ===========================================================================
# Optional: xfail placeholder — remove if no PENDING is found
# ===========================================================================


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Sentinel: if a future migration with down_revision='PENDING' reaches head "
        "before the IW daemon rebases it, this test documents the bug. "
        "Currently the chain is clean (no PENDING). "
        # NOTE: this xfail exists to document the historical bug pattern; it is
        # not expected to trigger.  If it xpasses, a genuine PENDING bug has
        # slipped through — investigate immediately.
    ),
)
def test_pending_regression_sentinel() -> None:
    """Sentinel xfail for the PENDING-sentinel regression pattern (historical bug).

    This test intentionally fails only if a PENDING down_revision is found in
    the live chain.  Currently no such revision exists, so this test passes
    (and xpasses the xfail — marking it as unexpectedly passing, which is fine
    since strict=False).

    If a future PENDING escapes without daemon rebase, the walk will find it
    and the explicit assertion in TestNoPendingDownRevision will fail hard.
    This sentinel provides additional xfail documentation without blocking CI.
    """
    sd = _script_directory()
    found_pending = any(
        (isinstance(r.down_revision, str) and r.down_revision == "PENDING")
        or (isinstance(r.down_revision, tuple) and "PENDING" in r.down_revision)
        for r in sd.walk_revisions()
    )
    # Assert that NO pending exists — xfail means we expect this to fail
    # only when the PENDING bug is present.  Currently it should not fail.
    assert not found_pending, (
        "PRODUCTION BUG: down_revision='PENDING' found in the live migration chain. "
        "The IW daemon rebase did not run. "
        "Hand-edit the migration's down_revision to the real predecessor. "
        "See project memory: iw-ai-core migration pipeline orch-only."
    )
