# ruff: noqa: E501
"""Sample documents and probe questions for the Learn section's labs.

The text is fictional ("Northwind") and written so that a few answer sentences sit
inside long paragraphs: that is exactly where a chunk boundary can cut an answer in
half.  Two variants share the same words, one with headings and one without, to show
what the section structure is worth to the chunker.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Probe:
    """A question whose answer is one exact sentence of the sample text.

    Attributes:
        question: The question a user might ask.
        sentence: The exact sentence in the sample that answers it.
    """

    question: str
    sentence: str


@dataclass(frozen=True, slots=True)
class Sample:
    """A sample document for a lab.

    Attributes:
        sample_id: Stable id used in the lab's HTTP query.
        title: Short label for the picker.
        blurb: One line saying what the variant demonstrates.
        text: The markdown text.
        probes: Questions answered by exact sentences in ``text``.
    """

    sample_id: str
    title: str
    blurb: str
    text: str
    probes: tuple[Probe, ...]


_SICK = (
    "Sick leave exists so that nobody has to choose between their health and their pay. "
    "Any employee who is unable to work because of illness or injury should tell their "
    "manager as early as they reasonably can, ideally before the start of the working day, "
    "by phone or message. Employees on a shift pattern follow the same rule, and the deadline "
    "is then two hours before the start of their shift. There is no need to explain the "
    "diagnosis, and managers must never ask for one. Employees may take up to 3 consecutive sick days with full pay and no "
    "medical certificate. From the fourth consecutive day a certificate from a registered "
    "doctor is required, and it should be sent to the People team rather than to the line "
    "manager so that medical details stay confidential. Sick days do not reduce annual leave, "
    "and they are not counted against the carry-over allowance described below. If an illness "
    "begins during a period of annual leave and is certified by a doctor, the affected days "
    "are re-credited as sick leave. Repeated short absences are reviewed by the People team "
    "in a supportive conversation, never as a disciplinary matter, to check whether any "
    "adjustment to the role or the working pattern would help."
)

_ANNUAL_1 = (
    "Every full-time employee receives 25 days of paid annual leave per calendar year, plus "
    "the public holidays of the country in which they work. Part-time employees receive the "
    "same entitlement pro rata. Leave accrues monthly from the first day of employment, so "
    "someone who joins in July has accrued roughly half of the yearly allowance by the end "
    "of December. Requests should be made in the people system at least two weeks ahead for "
    "any absence longer than three days, and managers answer within five working days. "
    "Unused annual leave expires on 31 March of the following year. Because of this rule, "
    "employees are encouraged to plan at least two thirds of their leave before the end of "
    "September, so that the summer and the year-end are not overloaded with requests that "
    "the team cannot cover at the same time."
)

_ANNUAL_2 = (
    "During the two weeks around the end of the financial year the finance team may ask "
    "its members to keep leave to a minimum. Any such request is communicated at least "
    "three months in advance and is compensated with an extra day of leave that does not "
    "expire. Employees who leave the company are paid for any accrued but unused leave."
)

_ANNUAL_3 = (
    "Leave can be taken in half days. A request that spans a public holiday counts only the "
    "working days it contains. Employees who work on a public holiday because their role "
    "requires it receive a day in lieu, which must be taken within three months and is "
    "recorded separately from the annual allowance so that it never expires by accident. "
    "Managers should aim to approve requests in the order in which they arrive, and may "
    "decline one only when approving it would leave the team below its minimum cover. In "
    "that case they must propose alternative dates in the same reply."
)

_PARENTAL_2 = (
    "Before the leave begins, the employee and their manager agree a short plan for handing "
    "over current work and for staying in touch, if the employee wishes to. Contact during "
    "the leave is always optional. On return, there is a phased option of working 80% of "
    "the usual hours for the first month at full pay, and employees who are breastfeeding "
    "are given a private room and flexible breaks without any reduction in pay."
)

_PARENTAL = (
    "Primary carers are entitled to 16 weeks of parental leave on full pay, and secondary "
    "carers to 6 weeks. The leave can be taken in one block or in up to three separate "
    "blocks during the first 12 months after the birth or adoption. Employees keep their "
    "benefits throughout and return to the same role, or to an equivalent one at the same "
    "level and pay if the role has changed in the meantime."
)

_CARRY_TABLE = (
    "| Situation | Days that may be carried over |\n"
    "|---|---|\n"
    "| Ordinary year | 5 |\n"
    "| Leave blocked by the finance close | 10 |\n"
    "| Long-term illness | all unused days |"
)

_CLOSING = (
    "Carried-over days must be used by 30 June. Questions about this policy go to the "
    "People team, who also keep the list of public holidays per country. This policy is "
    "reviewed every year in January, and any change is announced to all employees at least "
    "one month before it takes effect."
)

_PROBES = (
    Probe(
        question="How many sick days can I take without a doctor's note?",
        sentence=(
            "Employees may take up to 3 consecutive sick days with full pay and no "
            "medical certificate."
        ),
    ),
    Probe(
        question="When does unused annual leave expire?",
        sentence="Unused annual leave expires on 31 March of the following year.",
    ),
)

_WITH_HEADINGS = (
    "# Northwind leave policy\n\n"
    "This policy explains the leave every Northwind employee is entitled to, how to request "
    "it, and what happens to leave that is not used.\n\n"
    f"## Annual leave\n\n{_ANNUAL_1}\n\n{_ANNUAL_2}\n\n{_ANNUAL_3}\n\n"
    f"## Sick leave\n\n{_SICK}\n\n"
    f"## Parental leave\n\n{_PARENTAL}\n\n{_PARENTAL_2}\n\n"
    f"## Carry-over\n\n{_CARRY_TABLE}\n\n{_CLOSING}\n"
)

_WITHOUT_HEADINGS = (
    "This policy explains the leave every Northwind employee is entitled to, how to request "
    "it, and what happens to leave that is not used.\n\n"
    f"{_ANNUAL_1}\n\n{_ANNUAL_2}\n\n{_ANNUAL_3}\n\n{_SICK}\n\n{_PARENTAL}\n\n{_PARENTAL_2}\n\n"
    f"{_CARRY_TABLE}\n\n{_CLOSING}\n"
)

SAMPLES: tuple[Sample, ...] = (
    Sample(
        sample_id="with-headings",
        title="Leave policy, with headings",
        blurb="Each section becomes its own parent, so the model reads one whole section.",
        text=_WITH_HEADINGS,
        probes=_PROBES,
    ),
    Sample(
        sample_id="no-headings",
        title="Same text, no headings",
        blurb="Everything lands in one giant section, so there is no useful parent to read.",
        text=_WITHOUT_HEADINGS,
        probes=_PROBES,
    ),
)

_BY_ID = {s.sample_id: s for s in SAMPLES}


def get_sample(sample_id: str) -> Sample | None:
    """Return the sample for ``sample_id``, or ``None`` if there is none."""
    return _BY_ID.get(sample_id)


__all__ = ["SAMPLES", "Probe", "Sample", "get_sample"]
