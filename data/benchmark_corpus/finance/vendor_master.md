# Vendor Master File

Authoritative customer → payment-terms mapping (finance source of truth).

| Customer | Vendor ID | Payment Terms | Billing Contact |
| --- | --- | --- | --- |
| Globex Corp | VEND-01 | Net 30 | accounts-vend-01@example.invalid |
| Initech LLC | VEND-02 | Net 45 | accounts-vend-02@example.invalid |
| Umbrella Inc | VEND-03 | Net 60 | accounts-vend-03@example.invalid |
| Soylent Co | VEND-04 | Net 30 | accounts-vend-04@example.invalid |
| Hooli Group | VEND-05 | Net 45 | accounts-vend-05@example.invalid |
| Stark Industries | VEND-06 | Net 15 | accounts-vend-06@example.invalid |

Payment terms on every invoice MUST match this master. Disputes route to the
billing contact above.
