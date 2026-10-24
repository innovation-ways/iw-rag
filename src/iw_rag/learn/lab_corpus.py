# ruff: noqa: E501
"""The mini-corpus and queries behind the Learn section's retrieval labs.

Fictional, small, and chosen on purpose: six near-identical invoices (only the ID
tells them apart), a paraphrase target, exact error codes, and documents restricted to
one group.  ``scripts/learn_record_hybrid_trace.py`` runs the real retrieval
components over exactly this data and records the outcome; the labs replay that
recording.  Changing anything here means re-recording.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabChunk:
    """One chunk of the lab corpus.

    Attributes:
        chunk_id: Stable short id, referenced by the recorded traces.
        document: Source document label.
        acl_tags: Groups allowed to read the chunk.
        text: The chunk text.
    """

    chunk_id: str
    document: str
    acl_tags: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class LabQuery:
    """A question with the chunk that answers it.

    Attributes:
        query_id: Stable id used in the lab's HTTP query.
        text: The question.
        gold: The chunk that holds the answer.
        lesson: What this question is chosen to show.
    """

    query_id: str
    text: str
    gold: str
    lesson: str


@dataclass(frozen=True, slots=True)
class LabPrincipal:
    """A signed-in user the lab can be replayed as.

    Attributes:
        principal_id: Stable id used in the lab's HTTP query.
        label: Short description for the picker.
        groups: The user's groups (from authentication in the real system).
    """

    principal_id: str
    label: str
    groups: tuple[str, ...]


PRINCIPALS: tuple[LabPrincipal, ...] = (
    LabPrincipal("analyst", "Finance analyst", ("finance", "public")),
    LabPrincipal("hr", "HR partner", ("hr", "public")),
)

CHUNKS: tuple[LabChunk, ...] = (
    LabChunk(
        "f01",
        "finance/invoices.md",
        ("finance",),
        "Invoice INV-2026-0931 was issued to Contoso Ltd. on 12 September 2026. The total due is 15,000 USD, payable within 30 days.",
    ),
    LabChunk(
        "f02",
        "finance/invoices.md",
        ("finance",),
        "Invoice INV-2026-0877 was issued to Fabrikam Inc. on 2 September 2026. The total due is 8,200 USD, payable within 45 days.",
    ),
    LabChunk(
        "f03",
        "finance/invoices.md",
        ("finance",),
        "Invoice INV-2026-0912 was issued to Adventure Works on 9 September 2026. The total due is 23,750 USD, payable within 30 days.",
    ),
    LabChunk(
        "f04",
        "finance/invoices.md",
        ("finance",),
        "Invoice INV-2026-0954 was issued to Northwind Traders on 18 September 2026. The total due is 4,900 USD, payable within 60 days.",
    ),
    LabChunk(
        "f05",
        "finance/invoices.md",
        ("finance",),
        "Invoice INV-2026-0968 was issued to Tailspin Toys on 22 September 2026. The total due is 11,300 USD, payable within 30 days.",
    ),
    LabChunk(
        "f06",
        "finance/payment_terms.md",
        ("finance",),
        "Standard payment terms are 30 days from the invoice date. Late payments accrue interest of 1.5% per month, and accounts more than 90 days overdue are passed to collections.",
    ),
    LabChunk(
        "f07",
        "finance/expenses.md",
        ("finance",),
        "Travel expenses are reimbursed within 30 days of submission. Upload your receipts in the expenses tool and the money arrives with the next payroll run.",
    ),
    LabChunk(
        "f08",
        "finance/expenses.md",
        ("finance",),
        "Meals while travelling are covered up to 60 EUR per day. Alcohol is never reimbursed, and hotel stays above 180 EUR per night need prior approval.",
    ),
    LabChunk(
        "f09",
        "finance/vendors.md",
        ("finance",),
        "New suppliers must complete the vendor onboarding form, provide a tax certificate and be approved by the finance controller before the first purchase order.",
    ),
    LabChunk(
        "h01",
        "hr/leave_policy.md",
        ("hr",),
        "Full-time employees receive 25 days of paid annual leave per year, accrued monthly from the first day of employment.",
    ),
    LabChunk(
        "h02",
        "hr/leave_policy.md",
        ("hr",),
        "Employees may take up to 3 consecutive sick days with full pay and no medical certificate; a certificate is required from the fourth day.",
    ),
    LabChunk(
        "h03",
        "hr/working_hours.md",
        ("hr",),
        "Staff may work remotely up to three days a week, agreed with their manager. Core hours are 10:00 to 16:00 in the employee's local time.",
    ),
    LabChunk(
        "h04",
        "hr/salary.md",
        ("hr",),
        "Salary reviews take place every April. Managers submit their proposals to the People team by 15 March.",
    ),
    LabChunk(
        "p01",
        "public/error_codes.md",
        ("public",),
        "Error E4421 occurs when exporting a report with more than 50,000 rows. Split the export into smaller date ranges or schedule it to run overnight.",
    ),
    LabChunk(
        "p02",
        "public/error_codes.md",
        ("public",),
        "Error E1007 means the session token has expired. Sign out and sign in again to get a new token.",
    ),
    LabChunk(
        "p03",
        "public/it_help.md",
        ("public",),
        "To connect to the company VPN, install the client from the IT portal and sign in with your work account and a one-time code.",
    ),
    LabChunk(
        "p04",
        "public/office.md",
        ("public",),
        "The Lisbon office is open from 08:00 to 19:00. Visitors must register at reception and wear a badge at all times.",
    ),
    LabChunk(
        "p05",
        "public/it_help.md",
        ("public",),
        "If a printer shows offline, check the network cable, restart the printer, and then reinstall the driver from the IT portal.",
    ),
    LabChunk(
        "p06",
        "public/security.md",
        ("public",),
        "Report suspicious emails to the security team using the Report phishing button. Never forward them to colleagues.",
    ),
)

QUERIES: tuple[LabQuery, ...] = (
    LabQuery(
        "invoice-id",
        "Total due on INV-2026-0931",
        "f01",
        "An exact identifier. Five invoices read almost the same, so only the ID tells them apart.",
    ),
    LabQuery(
        "invoice-term",
        "Which invoice has a 60 day payment term?",
        "f04",
        "A generic topic (payment terms) mixed with one specific fact (60 days).",
    ),
    LabQuery(
        "paraphrase",
        "How long until I get paid back for my trip?",
        "f07",
        "A paraphrase: the question shares almost no words with the answer.",
    ),
    LabQuery(
        "error-code",
        "What does error E4421 mean?",
        "p01",
        "An exact error code in a corpus full of other short technical notes.",
    ),
    LabQuery(
        "remote",
        "Can I work from home on Fridays?",
        "h03",
        "Everyday wording ('work from home') for a policy that says 'work remotely'.",
    ),
    LabQuery(
        "sick",
        "How many sick days can I take without a doctor's note?",
        "h02",
        "An answer that only the HR group may read.",
    ),
)

__all__ = ["CHUNKS", "PRINCIPALS", "QUERIES", "LabChunk", "LabPrincipal", "LabQuery"]
