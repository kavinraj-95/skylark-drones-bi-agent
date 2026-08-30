"""Resolving time expressions into concrete date ranges.

Two things make this less trivial than it looks.

**Fiscal years.** This is an Indian business - amounts are in rupees with GST, and
invoice numbers read `SDPL/FY25-26/916`. So "Q3" means the third fiscal quarter
(Oct-Dec) by default, not the third calendar quarter. Both bases are supported and
every answer says which one it used.

**The data trails the calendar.** Relative expressions are resolved *literally*
first, against the dataset's as-of date. If that literal period turns out to contain
no records, we substitute the most recent period that does and report the
substitution. We never redefine what "this quarter" means - we answer the question
that was asked, discover it is empty, and say so.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Callable


class PeriodBasis(str, Enum):
    FISCAL = "fiscal"
    CALENDAR = "calendar"


class PeriodKind(str, Enum):
    """Granularity of a period.

    Tracked so that substituting an empty period keeps the same granularity - falling
    back from "2025" to a single quarter would answer a different question than the
    one asked.
    """

    QUARTER = "quarter"
    YEAR = "year"
    MONTH = "month"


@dataclass(frozen=True)
class Period:
    """A concrete, inclusive date range with a human-readable label."""

    start: date
    end: date
    label: str
    basis: PeriodBasis = PeriodBasis.FISCAL
    kind: PeriodKind = PeriodKind.QUARTER

    def contains(self, value: date | None) -> bool:
        return value is not None and self.start <= value <= self.end

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.label


@dataclass
class ResolvedTimeframe:
    """The outcome of interpreting a time expression.

    Carries enough detail for the answer to explain itself: what was asked, what was
    used, and whether the two differ.
    """

    period: Period | None
    #: The expression as the user phrased it ("this quarter").
    expression: str
    #: Assumptions worth stating in the answer.
    assumptions: list[str] = field(default_factory=list)
    #: True when the literal period was empty and a populated one was used instead.
    substituted: bool = False
    #: The literal period, when it differs from the one actually used.
    literal_period: Period | None = None

    @property
    def is_unbounded(self) -> bool:
        """True when no time filter applies - the question was about all data."""
        return self.period is None


def fiscal_year_of(value: date, start_month: int) -> int:
    """The fiscal year a date falls in, labelled by its starting calendar year.

    With an April start, 2025-06-01 and 2026-02-01 are both FY2025 (i.e. FY25-26).
    """
    return value.year if value.month >= start_month else value.year - 1


def fiscal_quarter_of(value: date, start_month: int) -> int:
    """Which fiscal quarter (1-4) a date falls in."""
    return ((value.month - start_month) % 12) // 3 + 1


def _month_start(year: int, month: int) -> date:
    # Normalise month overflow so callers can pass month 13+.
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return date(year, month, 1)


def _month_end(year: int, month: int) -> date:
    start = _month_start(year, month)
    return date(start.year, start.month, calendar.monthrange(start.year, start.month)[1])


def quarter_period(
    year: int, quarter: int, *, basis: PeriodBasis, fiscal_start_month: int
) -> Period:
    """Build a Period for a given quarter under the given basis."""
    if basis is PeriodBasis.FISCAL:
        start_month = fiscal_start_month + (quarter - 1) * 3
        start = _month_start(year, start_month)
        end = _month_end(year, start_month + 2)
        label = f"FY{str(year)[2:]}-{str(year + 1)[2:]} Q{quarter} (fiscal)"
    else:
        start = _month_start(year, (quarter - 1) * 3 + 1)
        end = _month_end(year, (quarter - 1) * 3 + 3)
        label = f"{year} Q{quarter} (calendar)"
    return Period(start=start, end=end, label=label, basis=basis, kind=PeriodKind.QUARTER)


def period_containing(value: date, *, basis: PeriodBasis, fiscal_start_month: int) -> Period:
    """The quarter that contains `value`."""
    if basis is PeriodBasis.FISCAL:
        return quarter_period(
            fiscal_year_of(value, fiscal_start_month),
            fiscal_quarter_of(value, fiscal_start_month),
            basis=basis,
            fiscal_start_month=fiscal_start_month,
        )
    return quarter_period(
        value.year, (value.month - 1) // 3 + 1, basis=basis, fiscal_start_month=fiscal_start_month
    )


def shift_quarters(period: Period, delta: int, *, fiscal_start_month: int) -> Period:
    """Move a quarter forward or backward by `delta` quarters."""
    anchor = _month_start(period.start.year, period.start.month + delta * 3)
    return period_containing(anchor, basis=period.basis, fiscal_start_month=fiscal_start_month)


def shift_years(period: Period, delta: int, *, fiscal_start_month: int) -> Period:
    """Move a year-period forward or backward by `delta` years."""
    base = (
        fiscal_year_of(period.start, fiscal_start_month)
        if period.basis is PeriodBasis.FISCAL
        else period.start.year
    )
    return year_period(base + delta, basis=period.basis, fiscal_start_month=fiscal_start_month)


def shift_period(period: Period, delta: int, *, fiscal_start_month: int) -> Period:
    """Step a period by `delta` units of its own granularity."""
    if period.kind is PeriodKind.YEAR:
        return shift_years(period, delta, fiscal_start_month=fiscal_start_month)
    if period.kind is PeriodKind.MONTH:
        anchor = _month_start(period.start.year, period.start.month + delta)
        return month_period(anchor.year, anchor.month)
    return shift_quarters(period, delta, fiscal_start_month=fiscal_start_month)


def year_period(year: int, *, basis: PeriodBasis, fiscal_start_month: int) -> Period:
    if basis is PeriodBasis.FISCAL:
        return Period(
            start=_month_start(year, fiscal_start_month),
            end=_month_end(year + 1, fiscal_start_month - 1)
            if fiscal_start_month > 1
            else _month_end(year, 12),
            label=f"FY{str(year)[2:]}-{str(year + 1)[2:]} (fiscal year)",
            basis=basis,
            kind=PeriodKind.YEAR,
        )
    return Period(
        start=date(year, 1, 1),
        end=date(year, 12, 31),
        label=f"{year} (calendar year)",
        basis=basis,
        kind=PeriodKind.YEAR,
    )


def month_period(year: int, month: int) -> Period:
    start = _month_start(year, month)
    return Period(
        start=start,
        end=_month_end(year, month),
        label=start.strftime("%B %Y"),
        basis=PeriodBasis.CALENDAR,
        kind=PeriodKind.MONTH,
    )


# --------------------------------------------------------------------------------
# Expression resolution
# --------------------------------------------------------------------------------

#: Relative expressions we understand, mapped to a quarter offset from the anchor.
_RELATIVE_QUARTERS = {
    "this quarter": 0,
    "current quarter": 0,
    "the quarter": 0,
    "last quarter": -1,
    "previous quarter": -1,
    "next quarter": 1,
}

_RELATIVE_YEARS = {
    "this year": 0,
    "current year": 0,
    "this fiscal year": 0,
    "last year": -1,
    "previous year": -1,
    "next year": 1,
}

_QUARTER_PATTERN = re.compile(
    r"\b(?:fy\s*)?(?P<q>q[1-4])\b(?:\s*(?:of\s*)?(?:fy\s*)?(?P<y>\d{2,4}))?", re.I
)
_YEAR_PATTERN = re.compile(r"\b(?:fy\s*)?(?P<y>20\d{2})\b", re.I)


def _normalise_year(raw: str) -> int:
    value = int(raw)
    return 2000 + value if value < 100 else value


def resolve_timeframe(
    expression: str | None,
    *,
    as_of: date | None,
    fiscal_start_month: int,
    basis: PeriodBasis = PeriodBasis.FISCAL,
    has_data: Callable[[Period], bool] | None = None,
    max_lookback: int = 12,
) -> ResolvedTimeframe:
    """Turn a natural time expression into a concrete `Period`.

    Resolution is anchored on `as_of` - the latest date on which the data shows
    something actually happened - rather than on wall-clock today. That is the whole
    point: the boards may trail the calendar by months, and anchoring on today would
    make every relative question return an empty period.

    When `has_data` is supplied, the literal period is checked first. If it is empty,
    we walk backwards to the most recent period that does contain records and mark the
    result `substituted`, keeping the literal period so the answer can explain the
    difference. The question asked is never quietly rewritten.
    """
    text = (expression or "").strip().lower()
    anchor = as_of or date.today()

    if not text or text in {"all", "all time", "overall", "to date", "ever"}:
        return ResolvedTimeframe(period=None, expression=expression or "all time")

    assumptions: list[str] = []
    period: Period | None = None

    # "Q3", "Q3 FY25", "Q1 2026"
    match = _QUARTER_PATTERN.search(text)
    if match:
        quarter = int(match.group("q")[1])
        if match.group("y"):
            year = _normalise_year(match.group("y"))
        else:
            year = (
                fiscal_year_of(anchor, fiscal_start_month)
                if basis is PeriodBasis.FISCAL
                else anchor.year
            )
            assumptions.append(
                f"No year was given, so {match.group('q').upper()} is read as the one in "
                f"the period the data currently reaches ({anchor.isoformat()})."
            )
        period = quarter_period(
            year, quarter, basis=basis, fiscal_start_month=fiscal_start_month
        )

    if period is None:
        for phrase, delta in _RELATIVE_QUARTERS.items():
            if phrase in text:
                current = period_containing(
                    anchor, basis=basis, fiscal_start_month=fiscal_start_month
                )
                period = shift_quarters(current, delta, fiscal_start_month=fiscal_start_month)
                assumptions.append(
                    f"'{phrase}' is measured from the most recent activity in the data "
                    f"({anchor.isoformat()}), not from today's date."
                )
                break

    if period is None:
        for phrase, delta in _RELATIVE_YEARS.items():
            if phrase in text:
                base_year = (
                    fiscal_year_of(anchor, fiscal_start_month)
                    if basis is PeriodBasis.FISCAL
                    else anchor.year
                )
                period = year_period(
                    base_year + delta, basis=basis, fiscal_start_month=fiscal_start_month
                )
                assumptions.append(
                    f"'{phrase}' is measured from the most recent activity in the data "
                    f"({anchor.isoformat()}), not from today's date."
                )
                break

    if period is None:
        match = _YEAR_PATTERN.search(text)
        if match:
            period = year_period(
                _normalise_year(match.group("y")),
                basis=basis,
                fiscal_start_month=fiscal_start_month,
            )

    if period is None:
        # Unrecognised expression. Better to analyse everything and say so than to
        # invent a range the user did not ask for.
        return ResolvedTimeframe(
            period=None,
            expression=expression or "",
            assumptions=[
                f"Could not interpret the period {expression!r}, so all available data "
                "is included."
            ],
        )

    resolved = ResolvedTimeframe(period=period, expression=expression or "", assumptions=assumptions)

    if has_data is None or has_data(period):
        return resolved

    # The literal period is empty. Walk back to the most recent one that is not.
    resolved.literal_period = period
    candidate = period
    for _ in range(max_lookback):
        candidate = shift_period(candidate, -1, fiscal_start_month=fiscal_start_month)
        if has_data(candidate):
            resolved.period = candidate
            resolved.substituted = True
            resolved.assumptions.append(
                f"{period.label} contains no records at all, so the most recent period "
                f"that does - {candidate.label} - is reported instead."
            )
            return resolved

    # Nothing anywhere in range. Keep the literal period and let the caller report zero.
    resolved.assumptions.append(
        f"{period.label} contains no records, and neither does any earlier period "
        "within the search window."
    )
    return resolved
