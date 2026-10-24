# ruff: noqa: E501
"""The documents and questions behind the Learn section's reranker lab.

Four markdown documents, chunked with the production chunker: a travel policy with
headings (so every section becomes a parent worth expanding to), an expenses FAQ whose
words overlap with it (so there are distractors), a list of near-identical invoices, and
a handbook with no headings at all (so its single parent is too big to expand).
``scripts/learn_record_rerank_trace.py`` ingests exactly this text, runs the real spine
and records the outcome.  Changing anything here means re-recording.
"""

from __future__ import annotations

from dataclasses import dataclass

LEAF_CHARS = 420


@dataclass(frozen=True, slots=True)
class RerankDoc:
    """One bundled document.

    Attributes:
        doc_id: Stable id, also the ``learn://`` source name.
        title: Label shown in the lab.
        text: The markdown text.
    """

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class RerankQuery:
    """A question with the passage that answers it.

    Attributes:
        query_id: Stable id used in the lab's HTTP query.
        text: The question.
        answer: An exact sentence of the document that answers it.
        context: A phrase the generator needs to read to understand the answer, which
            sits in the leaf's section but not in the leaf itself ("" when the leaf is
            self-contained).
        lesson: What this question is chosen to show.
    """

    query_id: str
    text: str
    answer: str
    context: str
    lesson: str


TRAVEL = RerankDoc(
    "travel_policy.md",
    "Travel policy",
    "# Northwind travel policy\n\n"
    "## Booking\n\n"
    "Book flights and hotels through the travel portal at least 14 days before departure. Fares above 600 EUR for a single journey need written approval from the department head before the booking is confirmed.\n\n"
    "Rail is preferred for trips under four hours. Economy class applies to every flight under six hours; premium economy may be used above that, and business class is never booked.\n\n"
    "## Meals and hotels\n\n"
    "Meals while travelling are covered up to 60 EUR per day, with receipts. Alcohol is never reimbursed, and neither are minibar items or room service ordered after 22:00.\n\n"
    "Hotel stays up to 180 EUR per night are approved automatically. Anything above that amount needs prior approval from the finance controller, and the approval must be attached to the claim.\n\n"
    "## Claims and deadlines\n\n"
    "Expense claims cover every cost paid personally during a business trip. Submit them in the expenses tool with a receipt for each item, grouped by trip.\n\n"
    "It must be filed within 30 days of the trip's last day. Claims filed after that date are paid only with the finance controller's written exception, and never later than 90 days.\n\n"
    "Approved claims are paid with the next payroll run. The money arrives within 30 days of submission, and the payslip shows each item separately.\n\n"
    "## Cancellations\n\n"
    "If a trip is cancelled, cancel the booking in the portal the same day. Non-refundable fares are charged to the department, not to the traveller, unless the cancellation was personal.\n\n"
    "A trip cancelled less than 48 hours before departure must be reported to the travel desk by phone, because the portal blocks changes inside that window.\n",
)

FAQ = RerankDoc(
    "expense_faq.md",
    "Expenses FAQ",
    "# Expenses FAQ\n\n"
    "## Common questions\n\n"
    "Can I claim a taxi to the airport? Yes, when public transport is unavailable or the journey starts before 06:00. Keep the receipt and write the reason on the claim.\n\n"
    "Are tips reimbursed? Tips up to 10% of a restaurant bill are reimbursed, with the receipt. Tips anywhere else are not.\n\n"
    "Who approves my expense claim? Your line manager approves claims up to 1,000 EUR, and the finance controller approves everything above that amount.\n\n"
    "How do I claim mileage for my own car? Record the distance in the expenses tool; the rate is 0.36 EUR per kilometre.\n\n"
    "## Corporate cards\n\n"
    "Corporate cards are issued to employees who travel more than six times a year. Card statements are reconciled monthly, and unexplained charges are deducted from the next payroll.\n\n"
    "Lost cards must be reported to the card provider at once and to the finance team within 24 hours.\n",
)

INVOICES = RerankDoc(
    "invoices.md",
    "Invoices",
    "# Invoices\n\n"
    "## September 2026\n\n"
    "Invoice INV-2026-0931 was issued to Contoso Ltd. on 12 September 2026. The total due is 15,000 USD, payable within 30 days.\n\n"
    "Invoice INV-2026-0877 was issued to Fabrikam Inc. on 2 September 2026. The total due is 8,200 USD, payable within 45 days.\n\n"
    "Invoice INV-2026-0912 was issued to Adventure Works on 9 September 2026. The total due is 23,750 USD, payable within 30 days.\n\n"
    "Invoice INV-2026-0954 was issued to Northwind Traders on 18 September 2026. The total due is 4,900 USD, payable within 60 days.\n\n"
    "Invoice INV-2026-0968 was issued to Tailspin Toys on 22 September 2026. The total due is 11,300 USD, payable within 30 days.\n",
)

HANDBOOK = RerankDoc(
    "handbook.md",
    "Handbook (no headings)",
    "Welcome to the Lisbon office. The building is open from 08:00 to 19:00 on weekdays, and the kitchen on the second floor is shared by every team. Please leave it as you would like to find it.\n\n"
    "Every employee receives an access badge on the first day. The badge opens the front door, the parking garage and the meeting rooms, and it also works at the printers, where it releases your documents.\n\n"
    "It must be handed back to reception on the last working day, and it stops working at midnight. A lost one is replaced for a fee of 15 EUR, which is waived for the first loss in a calendar year.\n\n"
    "Visitors must register at reception and wear a visitor sticker at all times. Hosts are responsible for collecting their guests and for making sure they leave before 19:00.\n\n"
    "Bicycles may be parked in the garage on level minus one. The racks are first come, first served, and unclaimed bicycles are removed after thirty days with a note on the handlebars.\n\n"
    "Meeting rooms are booked in the calendar and released automatically if nobody checks in within ten minutes. The large room on the third floor is reserved for the all-hands every second Thursday.\n\n"
    "Printers are shared. Colour printing is disabled by default and can be enabled per document, which is logged. Confidential documents should be collected immediately and never left in the tray.\n\n"
    "Lunch is not provided, but the second-floor kitchen has a fridge, two microwaves and a coffee machine that is cleaned every Friday afternoon by the facilities team, who also water the plants.\n\n"
    "The office is cleaned in the evenings. Please clear your desk of food and open drinks before you leave, and lock away anything valuable, because cleaners have access to every floor after 19:00.\n\n"
    "Fire exits are marked on every floor and tested on the first Monday of each quarter. When the alarm sounds, leave by the nearest stairs, never by the lift, and gather at the square opposite the main entrance.\n\n"
    "Questions about the building go to the facilities team on the first floor, and questions about equipment go to IT. Both teams answer messages within one working day and are available in person on weekdays.\n\n"
    "Post and parcels are delivered to the front desk between 09:00 and 17:00. Personal parcels are accepted for up to five working days, after which they are returned to the sender, so please collect them promptly.\n\n"
    "Laptops are issued by IT on the first day and remain the property of the company. Report a damaged or stolen laptop to IT within 24 hours, and never install software that has not been approved in the software catalogue.\n\n"
    "Remote work is possible up to three days a week by agreement with your manager. On days in the office, desks are shared and booked in the calendar, and personal items must be taken home in the evening.\n\n"
    "The quiet room on the fourth floor is for focused work and phone calls are not allowed there. The phone booths on the second floor can be used without booking, for a maximum of thirty minutes at a time.\n\n"
    "Plants, pictures and small personal items are welcome at shared desks, but nothing may be fixed to walls or glass. Facilities will remove anything that blocks a fire exit, a sprinkler or a camera.\n\n"
    "The company holds a garden party each June and a winter dinner each December. Invitations are sent by email four weeks ahead, and employees may bring one guest to each event.\n\n"
    "Feedback about the office is always welcome. Use the suggestion box in the kitchen or write to the facilities team, who review every suggestion at their monthly meeting and publish the outcome on the intranet.\n\n"
    "Charging points for electric cars are available on level minus two and are shared by everyone. A car may stay connected for at most four hours, after which it must be moved so that the next colleague can charge.\n\n"
    "First aid kits are kept in the kitchen and at reception, and trained first aiders are listed on the intranet. Any accident, however small, should be reported to facilities the same day so that it can be recorded.\n\n"
    "Smoking is not permitted anywhere in the building or within five metres of the entrance. A designated area is available in the courtyard behind the garage.\n",
)

DOCS: tuple[RerankDoc, ...] = (TRAVEL, FAQ, INVOICES, HANDBOOK)

QUERIES: tuple[RerankQuery, ...] = (
    RerankQuery(
        "alcohol",
        "Is alcohol reimbursed on business trips?",
        "Alcohol is never reimbursed",
        "",
        "A direct question with a clear answer inside one passage.",
    ),
    RerankQuery(
        "late-claim",
        "What happens if I file my expenses late?",
        "Claims filed after that date are paid only with the finance controller's written exception",
        "Expense claims cover every cost",
        "The answer leaf says 'It' without ever naming what it is; the section around it does.",
    ),
    RerankQuery(
        "paid-back",
        "How long until I get paid back for my trip?",
        "The money arrives within 30 days of submission",
        "",
        "Wording that differs from the passage, with look-alike chunks nearby.",
    ),
    RerankQuery(
        "invoice",
        "Which invoice is payable within 45 days?",
        "Invoice INV-2026-0877 was issued to Fabrikam Inc.",
        "",
        "Five near-identical invoices: only one detail tells them apart.",
    ),
    RerankQuery(
        "badge",
        "What happens to my access badge when I leave the company?",
        "It must be handed back to reception on the last working day",
        "Every employee receives an access badge",
        "A document with no headings: its one section is too big to expand.",
    ),
)

__all__ = ["DOCS", "LEAF_CHARS", "QUERIES", "RerankDoc", "RerankQuery"]
