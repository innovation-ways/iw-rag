# API Reference (v2)

Internal service API. All endpoints require a bearer token; the principal's ACL
groups are server-injected and never read from the request body.

| Endpoint | Method | Rate Limit | Description |
| --- | --- | --- | --- |
| `/v2/search` | POST | 60 req/min | hybrid retrieval over the index |
| `/v2/ingest` | POST | 10 req/min | submit a document for offline ingestion |
| `/v2/documents` | GET | 120 req/min | list documents visible to the principal |
| `/v2/health` | GET | 600 req/min | liveness and resident-VRAM probe |

Authentication failures return 401; ACL-filtered empty results return 200 with an
empty list (never 403, to avoid existence oracles).
