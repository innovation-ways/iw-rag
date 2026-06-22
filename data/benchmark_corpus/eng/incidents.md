# Incident Postmortems

Engineering incident log. Each entry has a distinct incident id and root cause.

## INC-2026-001 — search latency spike (P2)

Root cause: HNSW ef_search misconfiguration. Resolved by rolling back the offending change and adding a regression test.

## INC-2026-002 — ingestion backlog (P1)

Root cause: embedder OOM under batch size 256. Resolved by rolling back the offending change and adding a regression test.

## INC-2026-003 — citation mismatch (P3)

Root cause: datamark stripping regression. Resolved by rolling back the offending change and adding a regression test.

## INC-2026-004 — ACL leak false alarm (P2)

Root cause: stale principal_groups cache. Resolved by rolling back the offending change and adding a regression test.
