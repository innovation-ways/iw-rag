-- Bootstrap SQL for Apache AGE extension (GraphRAG mode, WI-028).
--
-- This file is sourced by the Postgres container at first-run
-- (docker/compose.yaml init volume, executed before the application starts).
-- It is a no-op when the 'age' shared library is not installed in the image —
-- the CREATE EXTENSION will fail silently and the graphrag mode stays disabled.
--
-- When to use:
--   Use an AGE-enabled Postgres image (e.g. apache/age or a custom build) and
--   mount this file into /docker-entrypoint-initdb.d/ to auto-bootstrap AGE.
--
-- Mode-enabled only:
--   This file is always mounted but the extension creation is guarded by
--   IF NOT EXISTS — it is a no-op when AGE is already installed (idempotent).
--   When modes.graphrag.enabled=false (the default) the extension simply sits
--   unused and the graph_chunk_backrefs table remains empty.
--
-- References:
--   - task-4a-graphrag-route §1 (store boundary + AGE bootstrap)
--   - lld-graphrag §7 (config)
--   - alembic/versions/0005_graphrag_age.py (migration that creates the table)

-- Best-effort AGE bootstrap: guarded so it is a TRUE no-op (not a fatal init
-- error) on a plain paradedb/paradedb image without the AGE build. A bare
-- `CREATE EXTENSION age` would raise and ABORT the whole init script (the
-- container entrypoint treats any error as fatal), so we pre-check
-- pg_available_extensions and only create when AGE is genuinely installable.
-- When AGE is absent the graphrag mode stays off (the default) and the spine /
-- other modes are unaffected.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'age') THEN
        CREATE EXTENSION IF NOT EXISTS age;
        PERFORM 1;  -- AGE present; the migration (0005) creates the graph + tables
        RAISE NOTICE 'AGE extension created (graphrag mode available).';
    ELSE
        RAISE NOTICE 'AGE extension not available in this image — graphrag mode disabled (default).';
    END IF;
END
$$;
