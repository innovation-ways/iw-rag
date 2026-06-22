"""Resolve a benchmark report's opaque ``dataset_id`` uuid to a human label.

The persisted ``benchmark_reports`` row stores only the ``eval_datasets`` uuid
(a random FK), so every display surface that printed ``dataset_id`` showed an
unreadable uuid. This module is the single resolver shared by the report
read-service and the scoreboard service so both render ``benchmark_golden (v1)``
consistently.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from iw_rag.storage import schema


async def attach_dataset_labels(session: Any, rows: list[dict[str, Any]]) -> None:
    """Decorate report rows in place with ``dataset_name`` + ``dataset_version``.

    A single grouped ``SELECT`` over the distinct ``dataset_id`` values in *rows*
    resolves the readable name + version from ``eval_datasets``. Both keys are
    set to ``None`` for a dangling FK (legacy / deleted dataset) so the render
    sites can fall back to the uuid.

    Args:
        session: The open ``AsyncSession`` to read ``eval_datasets`` with.
        rows: Report row dicts; mutated in place with ``dataset_name`` /
            ``dataset_version``.
    """
    ids = {row.get("dataset_id") for row in rows if row.get("dataset_id")}
    label_by_id: dict[str, tuple[str, str]] = {}
    if ids:
        stmt = select(
            schema.eval_datasets.c.id,
            schema.eval_datasets.c.name,
            schema.eval_datasets.c.version,
        ).where(schema.eval_datasets.c.id.in_(ids))
        label_by_id = {str(r.id): (r.name, r.version) for r in (await session.execute(stmt)).all()}
    for row in rows:
        name, version = label_by_id.get(str(row.get("dataset_id")), (None, None))
        row["dataset_name"] = name
        row["dataset_version"] = version
