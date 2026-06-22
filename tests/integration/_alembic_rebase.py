"""PENDING-rebase helper for integration tests.

The project convention is to author new Alembic migrations via
``make migration-pending MSG="..."`` so they emit
``down_revision = "PENDING"`` (F-00093 Feature_Design §"Migrations:
agents generate, daemon applies" — engineering-conventions §5).  The
IW daemon's ``migration_rebase`` rewrites that sentinel to the real
chain head at merge time.

The test environment has no daemon, so a migration declared with
``down_revision = "PENDING"`` breaks ``alembic upgrade head``: the
static chain resolver raises ``KeyError: 'PENDING'`` because no
revision file declares ``revision = "PENDING"``.  This helper does
the same rebase the daemon would do — but in an isolated temp
``alembic/versions/`` directory, so the source file is never
modified and the project contract (``down_revision = "PENDING"`` in
the file on disk) is preserved.

Usage::

    from tests.integration._alembic_rebase import rebased_alembic_config, PENDING_MIGRATION

    cfg, pending = rebased_alembic_config(dsn)
    command.upgrade(cfg, "head")
    if pending is not None:
        # Apply the rebased migration manually against the same DB.
        from alembic.operations import Operations
        from alembic.runtime.migration import MigrationContext
        import importlib.util
        spec = importlib.util.spec_from_file_location("m_pending", str(pending))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        with engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with Operations.context(ctx):
                mod.upgrade()

For tests that don't care about the PENDING migration's contents
(most of them), ``rebased_alembic_config`` returns
``pending_migration=None`` and the call site can skip the manual
apply — the chain through the prior revisions is what matters.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from alembic.config import Config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_DIR = _REPO_ROOT / "alembic"
_VERSIONS_DIR = _ALEMBIC_DIR / "versions"
_ALEMBIC_INI = _REPO_ROOT / "alembic.ini"

# Name of any in-flight migration that still uses the PENDING sentinel.
# The chain is PENDING-free after the F-00093 daemon rebase
# (``down_revision = "0007_ephemeral_sessions"`` on 0008, not "PENDING")
# + the F-00096 0009 hand-author (which deliberately uses a real
# ``down_revision = "0008_benchmark_reports"`` so the chain resolves).
# No migration needs stripping anymore; the tuple is empty by design.
# If a future migration in flight re-introduces the PENDING sentinel,
# add its filename here so the static chain resolver succeeds.
_PENDING_MIGRATION_FILES: tuple[str, ...] = ()


@dataclass(frozen=True)
class RebasedAlembic:
    """A rebased alembic script dir + the path of the PENDING migration (if any).

    Attributes:
        cfg: An ``alembic.config.Config`` whose ``script_location``
            points at a temp dir containing all real migrations.
        pending_migration: Absolute path to the PENDING-sentinel
            migration that was stripped from the rebased dir, or
            ``None`` when none was found.  Callers that need the
            PENDING migration's DDL applied must import this module
            and call its ``upgrade()`` against a live connection (see
            module docstring).
    """

    cfg: Config
    pending_migration: Path | None


@contextmanager
def rebased_alembic_config(dsn: str) -> Generator[RebasedAlembic, None, None]:
    """Yield an alembic ``Config`` whose versions dir is the real chain.

    Copies ``alembic/versions/`` to a temp dir, strips any migration
    that declares ``down_revision = "PENDING"`` (so the chain
    resolves), and configures a fresh ``Config`` pointed at the temp
    dir.  The PENDING migration is also yielded so callers that need
    its DDL can import + apply it manually.

    Args:
        dsn: The SQLAlchemy URL the alembic ``Config`` should use.

    Yields:
        A :class:`RebasedAlembic` with the rebased ``Config`` and the
        path to the PENDING migration (or ``None``).
    """
    with tempfile.TemporaryDirectory(prefix="iw-rag-alembic-rebase-") as tmp:
        tmp_path = Path(tmp)
        rebased_alembic_dir = tmp_path / "alembic"
        rebased_versions = rebased_alembic_dir / "versions"
        rebased_versions.mkdir(parents=True)
        # Copy every migration file.
        for src in _VERSIONS_DIR.glob("*.py"):
            shutil.copy(src, rebased_versions / src.name)
        # Strip the PENDING-sentinel migration(s) and remember the
        # first one we stripped (callers that need it can ask).
        pending_migration: Path | None = None
        for pending_name in _PENDING_MIGRATION_FILES:
            candidate = _VERSIONS_DIR / pending_name
            rebased = rebased_versions / pending_name
            if rebased.exists():
                if pending_migration is None:
                    pending_migration = candidate
                rebased.unlink()
        # The rebased dir must still look like a script_location —
        # copy env.py + script.py.mako so alembic can resolve things.
        shutil.copy(_ALEMBIC_DIR / "env.py", rebased_alembic_dir / "env.py")
        shutil.copy(
            _ALEMBIC_DIR / "script.py.mako",
            rebased_alembic_dir / "script.py.mako",
        )

        cfg = Config(str(_ALEMBIC_INI))
        cfg.set_main_option("script_location", str(rebased_alembic_dir))
        cfg.set_main_option("sqlalchemy.url", dsn)

        yield RebasedAlembic(cfg=cfg, pending_migration=pending_migration)
