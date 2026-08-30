"""Canonical business entities and the field-state model.

The central idea: **a value and our confidence in that value travel together.**

Business analytics gets destroyed by `None -> 0`. A blank "billed value" and a real
zero are different facts, and this dataset encodes the *same* fact both ways
(`Billed Value (Excl GST)` has 63 blanks; `Billed Value (Incl GST)` has 63 zeros).
So every canonical field is a `Field`, carrying the parsed value, the raw string it
came from, and a `FieldState` saying how much to trust it. Metrics then decide
explicitly what to do with each state rather than inheriting whatever `float(x or 0)`
happens to produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class FieldState(str, Enum):
    """How a canonical field's value came to be.

    The distinctions are deliberate. "We have no value" and "we have a value we do not
    trust" and "we derived this value ourselves" are three different claims, and
    collapsing them is how a BI tool starts lying.
    """

    #: Parsed cleanly from a populated source value.
    OK = "OK"
    #: Source was blank. Not zero, not false - absent.
    MISSING = "MISSING"
    #: Source had content we could not parse (`"45days"` in a quantity column).
    MALFORMED = "MALFORMED"
    #: Parsed, but the value is outside the known vocabulary for this field.
    UNMAPPED = "UNMAPPED"
    #: Parsed, but genuinely underdetermined - e.g. a month name with no year.
    AMBIGUOUS = "AMBIGUOUS"
    #: We derived this rather than observing it (e.g. sector "energy" -> Renewables).
    #: Never interchangeable with OK: an inference is our claim, not the source's.
    INFERRED = "INFERRED"
    #: The field cannot apply to this record.
    NOT_APPLICABLE = "NOT_APPLICABLE"

    @property
    def is_usable(self) -> bool:
        """True when a metric may safely aggregate this value."""
        return self in (FieldState.OK, FieldState.INFERRED)


@dataclass(frozen=True)
class Field(Generic[T]):
    """A single canonical value plus its provenance."""

    value: T | None = None
    raw: str | None = None
    state: FieldState = FieldState.MISSING
    #: Human-readable reason, shown in data-quality reporting. Only set when the
    #: state is not OK, so it reads as an explanation rather than noise.
    note: str | None = None

    @property
    def ok(self) -> bool:
        return self.state.is_usable and self.value is not None

    def or_none(self) -> T | None:
        """The value if it is usable, else None. Never a silent default."""
        return self.value if self.ok else None

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.ok

    @classmethod
    def missing(cls, raw: str | None = None, note: str | None = None) -> Field[T]:
        return cls(None, raw, FieldState.MISSING, note)

    @classmethod
    def malformed(cls, raw: str | None, note: str) -> Field[T]:
        return cls(None, raw, FieldState.MALFORMED, note)

    @classmethod
    def good(cls, value: T, raw: str | None = None) -> Field[T]:
        return cls(value, raw, FieldState.OK, None)


@dataclass
class Quantity:
    """A quantity that may carry a unit, because this dataset's do.

    `Quantities as per PO` holds `1`, `1600`, `NA`, `45days`, `45 days`, `5360 HA` in
    one column. Units are kept rather than discarded so that nothing sums hectares
    with days.
    """

    amount: float
    unit: str | None = None

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.amount:g} {self.unit}".strip()


@dataclass
class Record:
    """Shared shape for anything sourced from a monday board item."""

    #: monday's item ID. The only globally unique identity we can rely on - the Deals
    #: board has no business key of its own (`Sakura` appears 27 times).
    item_id: str
    item_name: str
    #: Every canonical field, by name, for generic quality reporting.
    fields: dict[str, Field[Any]] = field(default_factory=dict)
    #: Raw column text as monday returned it, keyed by column title. Retained so the
    #: UI can always show what a value looked like before normalization.
    raw: dict[str, str] = field(default_factory=dict)
    #: Record-level problems (e.g. "this row is a repeated header").
    issues: list[str] = field(default_factory=list)
    #: True when the row is structurally not a business record and must be excluded
    #: from every metric. Counted and reported, never silently dropped.
    excluded: bool = False
    exclusion_reason: str | None = None

    def get(self, name: str) -> Field[Any]:
        return self.fields.get(name, Field())

    def value(self, name: str) -> Any:
        return self.get(name).or_none()

    def state(self, name: str) -> FieldState:
        return self.get(name).state


@dataclass
class Deal(Record):
    """A sales-pipeline record from the Deals board.

    `stage` is the trustworthy pipeline signal, not `status`: 70 rows are marked
    `Won` while sitting at stage `A. Lead Generated`, all created on one day with no
    value, probability or dates - a bulk backlog import with a defaulted status.
    Conflicts between the two are reported rather than silently resolved.
    """

    name: str = ""
    owner_code: Field[str] = field(default_factory=Field)
    client_code: Field[str] = field(default_factory=Field)
    status: Field[str] = field(default_factory=Field)
    stage: Field[str] = field(default_factory=Field)
    #: Ordinal position of `stage` in the A->O ladder, for pipeline ordering.
    stage_rank: Field[int] = field(default_factory=Field)
    closure_probability: Field[str] = field(default_factory=Field)
    value_inr: Field[float] = field(default_factory=Field)
    sector: Field[str] = field(default_factory=Field)
    product: Field[str] = field(default_factory=Field)
    created_date: Field[date] = field(default_factory=Field)
    tentative_close_date: Field[date] = field(default_factory=Field)
    actual_close_date: Field[date] = field(default_factory=Field)
    #: Set when `status` and `stage` tell different stories.
    status_stage_conflict: bool = False

    @property
    def is_open(self) -> bool:
        """Open on the evidence of *stage*, which is fully populated and ordered."""
        rank = self.stage_rank.or_none()
        return rank is not None and OPEN_STAGE_RANKS[0] <= rank <= OPEN_STAGE_RANKS[1]


@dataclass
class WorkOrder(Record):
    """A project-execution record from the Work Orders board."""

    deal_name: Field[str] = field(default_factory=Field)
    #: `Serial #` - a genuine business key, unique across all 176 rows.
    serial: Field[str] = field(default_factory=Field)
    customer_code: Field[str] = field(default_factory=Field)
    owner_code: Field[str] = field(default_factory=Field)
    sector: Field[str] = field(default_factory=Field)
    nature_of_work: Field[str] = field(default_factory=Field)
    type_of_work: Field[str] = field(default_factory=Field)
    execution_status: Field[str] = field(default_factory=Field)
    invoice_status: Field[str] = field(default_factory=Field)
    wo_status: Field[str] = field(default_factory=Field)
    amount_excl_gst: Field[float] = field(default_factory=Field)
    amount_incl_gst: Field[float] = field(default_factory=Field)
    billed_excl_gst: Field[float] = field(default_factory=Field)
    billed_incl_gst: Field[float] = field(default_factory=Field)
    collected_incl_gst: Field[float] = field(default_factory=Field)
    to_be_billed_excl_gst: Field[float] = field(default_factory=Field)
    receivable: Field[float] = field(default_factory=Field)
    quantity_po: Field[Quantity] = field(default_factory=Field)
    quantity_billed: Field[Quantity] = field(default_factory=Field)
    po_date: Field[date] = field(default_factory=Field)
    start_date: Field[date] = field(default_factory=Field)
    end_date: Field[date] = field(default_factory=Field)
    data_delivery_date: Field[date] = field(default_factory=Field)
    last_invoice_date: Field[date] = field(default_factory=Field)


#: Inclusive rank bounds for stages that represent live, unresolved pipeline.
#: A-F are pre-commitment; G onward the outcome is decided (see STAGE_LADDER).
OPEN_STAGE_RANKS = (1, 6)


@dataclass
class BoardProvenance:
    """Where a board's data came from, for the UI's LIVE/STALE banner."""

    board_id: str
    board_name: str
    item_count: int
    column_count: int


@dataclass
class Dataset:
    """Both boards, normalized, plus everything needed to talk about them honestly."""

    deals: list[Deal] = field(default_factory=list)
    work_orders: list[WorkOrder] = field(default_factory=list)
    fetched_at: datetime | None = None
    is_stale: bool = False
    stale_reason: str | None = None
    provenance: dict[str, BoardProvenance] = field(default_factory=dict)
    #: Column titles seen on each board that we could not map to a canonical field.
    #: Surfaced rather than ignored - an unexpected column may be a renamed one.
    unmapped_columns: dict[str, list[str]] = field(default_factory=dict)

    @property
    def active_deals(self) -> list[Deal]:
        """Deals usable for analysis - excludes header echoes and other non-records."""
        return [d for d in self.deals if not d.excluded]

    @property
    def active_work_orders(self) -> list[WorkOrder]:
        return [w for w in self.work_orders if not w.excluded]

    @property
    def as_of(self) -> date | None:
        """The latest date on which the data shows something actually *happened*.

        Deliberately built from observed events only - record creation, actual close,
        PO receipt, data delivery, invoicing. Forecast and planning fields
        (`tentative_close_date`, `start_date`, `end_date`) are excluded: a deal
        forecast to close next April says nothing about how current the data is, and
        letting a forecast define "now" would push the as-of date into the future and
        silently make recent periods look empty.

        This is what relative time expressions resolve against. Recomputed from live
        data on every fetch - never hardcoded - because the dataset's notion of "now"
        trails wall-clock time.
        """
        dates: list[date] = []
        for deal in self.active_deals:
            for f in (deal.created_date, deal.actual_close_date):
                if f.ok and f.value:
                    dates.append(f.value)
        for wo in self.active_work_orders:
            for f in (wo.po_date, wo.data_delivery_date, wo.last_invoice_date):
                if f.ok and f.value:
                    dates.append(f.value)
        return max(dates) if dates else None


#: The Deals board's stage ladder. The letter prefixes encode the intended order, and
#: the two unlettered values are placed by meaning: `Project Completed` sits past
#: `K. Amount Accrued` (delivery finished), and the terminal states keep their letters.
#:
#: Ranks 1-6 are live pipeline; 7+ the outcome is already determined. That boundary is
#: what `Deal.is_open` uses, and it is the reason stage cannot be used to *predict*
#: anything: `G. Project Won` and `L. Project Lost` are outcomes wearing stage labels.
STAGE_LADDER: dict[str, int] = {
    "A. Lead Generated": 1,
    "B. Sales Qualified Leads": 2,
    "C. Demo Done": 3,
    "D. Feasibility": 4,
    "E. Proposal/Commercials Sent": 5,
    "F. Negotiations": 6,
    "G. Project Won": 7,
    "H. Work Order Received": 8,
    "I. POC": 9,
    "J. Invoice sent": 10,
    "K. Amount Accrued": 11,
    "Project Completed": 12,
    "L. Project Lost": 13,
    "M. Projects On Hold": 14,
    "N. Not relevant at the moment": 15,
    "O. Not Relevant at all": 16,
}

#: Stages where the deal is definitively won (revenue is real).
WON_STAGES = {"G. Project Won", "H. Work Order Received", "J. Invoice sent",
              "K. Amount Accrued", "Project Completed"}
#: Stages where the deal is definitively lost or abandoned.
LOST_STAGES = {"L. Project Lost", "N. Not relevant at the moment", "O. Not Relevant at all"}
#: Parked - neither won nor lost, and excluded from win-rate denominators.
#:
#: `I. POC` is a judgement call we flag rather than hide. Its letter places it after
#: `H. Work Order Received` (implying post-win delivery), but a proof-of-concept is
#: just as plausibly a pre-commitment trial, and all 3 non-backlog POC rows ended up
#: dead. Too few records to decide, so it is treated as parked: counted, excluded from
#: both won and lost, and reported as an assumption.
HELD_STAGES = {"M. Projects On Hold", "I. POC"}
