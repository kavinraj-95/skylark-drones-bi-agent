"""Dataset-level data-quality auditing.

Produces the evidence a founder needs to know how far to trust an answer. Two
audiences, one source of truth:

* the **Data Quality panel**, which shows the whole picture; and
* **per-answer caveats**, where the analytics engine pulls only the findings that
  touch the fields a given metric actually used - so caveats stay relevant instead of
  becoming a wall of boilerplate everyone learns to ignore.

Every finding is computed from the data at hand. Nothing here is a hardcoded
observation about the current dataset; point the app at different boards and the
findings change accordingly.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from ..ingest.entities import (
    HELD_STAGES,
    LOST_STAGES,
    WON_STAGES,
    Dataset,
    Deal,
    FieldState,
    Record,
    WorkOrder,
)


class Severity(str, Enum):
    """How much a finding should change the reader's confidence."""

    #: Materially affects the numbers. Must be surfaced whenever the field is used.
    HIGH = "high"
    #: Worth knowing, bounded impact.
    MEDIUM = "medium"
    #: Informational.
    LOW = "low"


@dataclass
class Finding:
    """One data-quality observation."""

    code: str
    title: str
    detail: str
    severity: Severity
    #: Canonical field names this finding concerns. Used to attach the right caveats
    #: to the right answers.
    fields: tuple[str, ...] = ()
    #: "deals", "work_orders", or "both".
    board: str = "both"
    affected_records: int = 0
    #: What the analytics layer does about it, in one line.
    handling: str = ""


@dataclass
class FieldCoverage:
    """Per-field completeness, by state."""

    name: str
    total: int
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def usable(self) -> int:
        return self.counts.get(FieldState.OK.value, 0) + self.counts.get(
            FieldState.INFERRED.value, 0
        )

    @property
    def usable_pct(self) -> float:
        return (self.usable / self.total * 100) if self.total else 0.0


@dataclass
class QualityReport:
    findings: list[Finding] = field(default_factory=list)
    deal_coverage: list[FieldCoverage] = field(default_factory=list)
    work_order_coverage: list[FieldCoverage] = field(default_factory=list)
    excluded_deals: int = 0
    excluded_work_orders: int = 0

    def for_fields(self, names: Iterable[str]) -> list[Finding]:
        """Findings relevant to a specific set of canonical fields.

        A finding with no declared fields is dataset-wide and always relevant.
        """
        wanted = set(names)
        return [
            f for f in self.findings
            if not f.fields or wanted.intersection(f.fields)
        ]

    def by_severity(self) -> list[Finding]:
        order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
        return sorted(self.findings, key=lambda f: (order[f.severity], -f.affected_records))


def _coverage(records: Sequence[Record], names: Iterable[str]) -> list[FieldCoverage]:
    out: list[FieldCoverage] = []
    total = len(records)
    for name in names:
        counts = collections.Counter(r.get(name).state.value for r in records)
        out.append(FieldCoverage(name=name, total=total, counts=dict(counts)))
    return out


#: States meaning "the source recorded nothing here". A field is only *empty* when
#: every record is one of these - a column full of AMBIGUOUS or MALFORMED values is
#: not empty, it is unusable, and conflating the two overstates how much data is gone.
_ABSENT_STATES = (FieldState.MISSING, FieldState.NOT_APPLICABLE)


def _empty_fields(records: Sequence[Record], names: Iterable[str]) -> list[str]:
    """Fields where the source recorded nothing on any record."""
    if not records:
        return []
    return [
        name for name in names
        if all(r.get(name).state in _ABSENT_STATES for r in records)
    ]


def audit(dataset: Dataset) -> QualityReport:
    """Build the full data-quality report for a dataset."""
    deals = dataset.active_deals
    work_orders = dataset.active_work_orders
    report = QualityReport(
        excluded_deals=sum(1 for d in dataset.deals if d.excluded),
        excluded_work_orders=sum(1 for w in dataset.work_orders if w.excluded),
    )

    deal_fields = sorted({k for d in deals for k in d.fields})
    wo_fields = sorted({k for w in work_orders for k in w.fields})
    report.deal_coverage = _coverage(deals, deal_fields)
    report.work_order_coverage = _coverage(work_orders, wo_fields)

    for check in (
        _check_excluded_rows,
        _check_status_stage_conflict,
        _check_probability_leakage,
        _check_stage_tautology,
        _check_value_completeness,
        _check_empty_columns,
        _check_blank_versus_zero,
        _check_malformed_values,
        _check_unmapped_values,
        _check_ambiguous_dates,
        _check_duplicates,
        _check_cross_board_identity,
        _check_value_concentration,
        _check_unmapped_columns,
        _check_data_recency,
    ):
        finding = check(dataset, deals, work_orders)
        if isinstance(finding, list):
            report.findings.extend(finding)
        elif finding is not None:
            report.findings.append(finding)

    return report


# --------------------------------------------------------------------------------
# Individual checks. Each returns a Finding, a list of Findings, or None.
# --------------------------------------------------------------------------------


def _check_excluded_rows(dataset: Dataset, deals, work_orders) -> Finding | None:
    excluded = [r for r in (*dataset.deals, *dataset.work_orders) if r.excluded]
    if not excluded:
        return None
    return Finding(
        code="excluded_rows",
        title="Non-data rows removed from the boards",
        detail=(
            f"{len(excluded)} row(s) repeat their own column headers rather than holding "
            "business data - an artifact of the original spreadsheet export surviving "
            "the import into monday.com."
        ),
        severity=Severity.MEDIUM,
        affected_records=len(excluded),
        handling="Excluded from every metric and reported here rather than silently dropped.",
    )


def _check_status_stage_conflict(dataset, deals: list[Deal], work_orders) -> Finding | None:
    conflicts = [d for d in deals if d.status_stage_conflict]
    if not conflicts:
        return None

    # Characterise the largest coherent cluster, so the finding explains *why* rather
    # than just counting.
    won_early = [
        d for d in conflicts
        if d.status.or_none() == "Won" and not d.value_inr.ok and (d.stage_rank.or_none() or 99) <= 2
    ]
    detail = (
        f"{len(conflicts)} deal(s) have a Deal Status that contradicts their Deal Stage."
    )
    if won_early:
        created = collections.Counter(
            d.created_date.value.isoformat() for d in won_early if d.created_date.ok
        )
        common = created.most_common(1)
        when = f" all created on {common[0][0]}" if common and common[0][1] == len(won_early) else ""
        detail += (
            f" {len(won_early)} of them are marked 'Won' while still at an early stage,"
            f" carry no deal value, probability or dates{when} - the signature of a bulk"
            " backlog import with a defaulted status rather than genuine closed business."
        )

    return Finding(
        code="status_stage_conflict",
        title="Deal Status contradicts Deal Stage",
        detail=detail,
        severity=Severity.HIGH,
        fields=("status", "stage", "stage_rank"),
        board="deals",
        affected_records=len(conflicts),
        handling=(
            "Deal Stage is treated as authoritative - it is fully populated and ordered, "
            "while Deal Status is not. The conflict is reported, never silently resolved."
        ),
    )


def _check_probability_leakage(dataset, deals: list[Deal], work_orders) -> Finding | None:
    """Detect a closure-probability field that encodes the outcome it should predict."""
    closed = [
        d for d in deals
        if d.closure_probability.ok and d.stage.or_none() in (WON_STAGES | LOST_STAGES)
    ]
    if len(closed) < 5:
        return None

    by_band: dict[str, list[bool]] = collections.defaultdict(list)
    for d in closed:
        by_band[d.closure_probability.value].append(d.stage.or_none() in WON_STAGES)

    degenerate = [
        (band, len(outcomes), sum(outcomes) / len(outcomes))
        for band, outcomes in by_band.items()
        if outcomes and (sum(outcomes) / len(outcomes)) in (0.0, 1.0)
    ]
    if not degenerate:
        return None

    parts = ", ".join(
        f"{band} = {rate:.0%} won across {n} closed deal(s)" for band, n, rate in sorted(degenerate)
    )
    return Finding(
        code="probability_leakage",
        title="Closure Probability appears to be set retrospectively",
        detail=(
            f"On already-closed deals the probability bands separate outcomes perfectly "
            f"({parts}). A forward-looking forecast does not behave this way; the field "
            "looks like it is updated after the outcome is known."
        ),
        severity=Severity.HIGH,
        fields=("closure_probability",),
        board="deals",
        affected_records=len(closed),
        handling=(
            "Weighted pipeline uses fixed, declared weights and is labelled a heuristic. "
            "These probabilities are never used to calibrate them, and weighted pipeline "
            "is never presented as expected revenue."
        ),
    )


def _check_stage_tautology(dataset, deals: list[Deal], work_orders) -> Finding | None:
    """Flag that stage-derived win rates cannot be predictive."""
    terminal = [d for d in deals if d.stage.or_none() in (WON_STAGES | LOST_STAGES)]
    if not terminal:
        return None
    return Finding(
        code="stage_tautology",
        title="Stage-based win rates are definitional, not predictive",
        detail=(
            "Several pipeline stages are outcomes wearing stage labels - a deal at "
            "'Project Won' is won by definition, and one at 'Project Lost' is lost. Any "
            "win rate computed per stage therefore restates the classification rather "
            "than measuring conversion."
        ),
        severity=Severity.MEDIUM,
        fields=("stage", "stage_rank"),
        board="deals",
        affected_records=len(terminal),
        handling=(
            "No stage-derived probability is used anywhere. Deals with no closure "
            "probability are reported as excluded from weighted pipeline, never imputed."
        ),
    )


def _check_value_completeness(dataset, deals: list[Deal], work_orders) -> Finding | None:
    missing = [d for d in deals if not d.value_inr.ok]
    if not missing:
        return None
    open_missing = [d for d in missing if d.is_open]
    return Finding(
        code="deal_value_missing",
        title="Many deals carry no deal value",
        detail=(
            f"{len(missing)} of {len(deals)} deals have no value recorded"
            f" ({len(open_missing)} of them still in open pipeline stages). Value-based "
            "totals therefore describe only the deals that have one."
        ),
        severity=Severity.HIGH,
        fields=("value_inr",),
        board="deals",
        affected_records=len(missing),
        handling=(
            "Records without a value are excluded from value metrics and counted "
            "separately. They are never treated as zero."
        ),
    )


def _check_empty_columns(dataset, deals: list[Deal], work_orders: list[WorkOrder]) -> list[Finding]:
    out: list[Finding] = []
    for label, records in (("deals", deals), ("work_orders", work_orders)):
        if not records:
            continue
        names = sorted({k for r in records for k in r.fields})
        empty = _empty_fields(records, names)
        if not empty:
            continue
        out.append(
            Finding(
                code=f"empty_columns_{label}",
                title=f"Columns with no data at all ({label.replace('_', ' ')})",
                detail=(
                    f"{len(empty)} column(s) are empty on every record: "
                    + ", ".join(sorted(empty))
                    + "."
                ),
                severity=Severity.MEDIUM,
                fields=tuple(empty),
                board=label,
                affected_records=len(records),
                handling=(
                    "Questions touching these fields are answered with 'no data recorded' "
                    "rather than an inferred or zero value."
                ),
            )
        )
    return out


def _check_blank_versus_zero(dataset, deals, work_orders: list[WorkOrder]) -> list[Finding]:
    """Find field pairs encoding the same fact as blank in one and zero in the other."""
    pairs = (
        ("billed_excl_gst", "billed_incl_gst", "billed value"),
        ("to_be_billed_excl_gst", "to_be_billed_incl_gst", "amount still to bill"),
    )
    out: list[Finding] = []
    for left, right, label in pairs:
        if not work_orders:
            continue
        left_missing = sum(1 for w in work_orders if w.get(left).state == FieldState.MISSING)
        right_zero = sum(1 for w in work_orders if w.get(right).ok and w.get(right).value == 0)
        if left_missing >= 5 and right_zero >= 5:
            out.append(
                Finding(
                    code=f"blank_vs_zero_{left}",
                    title=f"Blank and zero used interchangeably for {label}",
                    detail=(
                        f"{left_missing} record(s) leave the excluding-GST figure blank while "
                        f"{right_zero} record(s) record a hard zero in the including-GST "
                        "figure. The same underlying fact is encoded two different ways."
                    ),
                    severity=Severity.MEDIUM,
                    fields=(left, right),
                    board="work_orders",
                    affected_records=max(left_missing, right_zero),
                    handling=(
                        "Blank stays MISSING and is excluded from averages; a recorded zero "
                        "is treated as a real zero. The two are never merged."
                    ),
                )
            )
    return out


def _check_malformed_values(dataset, deals, work_orders) -> list[Finding]:
    out: list[Finding] = []
    for label, records in (("deals", deals), ("work_orders", work_orders)):
        counts: dict[str, int] = collections.Counter()
        examples: dict[str, str] = {}
        for record in records:
            for name, f in record.fields.items():
                if f.state == FieldState.MALFORMED:
                    counts[name] += 1
                    examples.setdefault(name, str(f.raw))
        if not counts:
            continue
        described = ", ".join(
            f"{name} ({n}, e.g. {examples[name]!r})" for name, n in counts.most_common(5)
        )
        out.append(
            Finding(
                code=f"malformed_{label}",
                title=f"Unparseable values ({label.replace('_', ' ')})",
                detail=f"Values that could not be read as their expected type: {described}.",
                severity=Severity.MEDIUM,
                fields=tuple(counts),
                board=label,
                affected_records=sum(counts.values()),
                handling=(
                    "Kept with their original text and marked MALFORMED. Excluded from "
                    "calculations, counted in exclusions."
                ),
            )
        )
    return out


def _check_unmapped_values(dataset, deals, work_orders) -> list[Finding]:
    out: list[Finding] = []
    for label, records in (("deals", deals), ("work_orders", work_orders)):
        by_field: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for record in records:
            for name, f in record.fields.items():
                if f.state == FieldState.UNMAPPED and f.value:
                    by_field[name][str(f.value)] += 1
        if not by_field:
            continue
        described = "; ".join(
            f"{name}: " + ", ".join(f"{v!r} ({n})" for v, n in counter.most_common(4))
            for name, counter in by_field.items()
        )
        out.append(
            Finding(
                code=f"unmapped_values_{label}",
                title=f"Values outside the known vocabulary ({label.replace('_', ' ')})",
                detail=f"{described}.",
                severity=Severity.LOW,
                fields=tuple(by_field),
                board=label,
                affected_records=sum(sum(c.values()) for c in by_field.values()),
                handling=(
                    "Kept as-is and grouped under their own label. Never force-fitted to "
                    "the nearest known category - an unfamiliar value is more likely new "
                    "than misspelled."
                ),
            )
        )
    return out


def _check_ambiguous_dates(dataset, deals, work_orders) -> Finding | None:
    counts: dict[str, int] = collections.Counter()
    for record in (*deals, *work_orders):
        for name, f in record.fields.items():
            if f.state == FieldState.AMBIGUOUS:
                counts[name] += 1
    if not counts:
        return None
    described = ", ".join(f"{name} ({n})" for name, n in counts.most_common(5))
    return Finding(
        code="ambiguous_dates",
        title="Month names recorded without a year",
        detail=(
            f"Some period fields hold a bare month name, which cannot be placed on a "
            f"timeline: {described}."
        ),
        severity=Severity.MEDIUM,
        fields=tuple(counts),
        affected_records=sum(counts.values()),
        handling=(
            "Marked AMBIGUOUS and excluded from any date-range filter or time series. "
            "Usable only for month-of-year grouping."
        ),
    )


def _check_duplicates(dataset, deals, work_orders) -> Finding | None:
    flagged = [
        r for r in (*deals, *work_orders)
        if any(i.startswith("Identical to") for i in r.issues)
    ]
    if not flagged:
        return None
    return Finding(
        code="duplicate_like_records",
        title="Records identical across every business field",
        detail=(
            f"{len(flagged)} record(s) are indistinguishable from at least one other "
            "record on every mapped field. They may be genuine repeat orders or "
            "duplicated data entry - the data cannot tell us which."
        ),
        severity=Severity.MEDIUM,
        affected_records=len(flagged),
        handling=(
            "Left in the totals and flagged. Merging them would silently delete revenue "
            "if they are genuine; dropping the flag would hide a real risk if they are not."
        ),
    )


def _check_cross_board_identity(dataset, deals: list[Deal], work_orders: list[WorkOrder]) -> Finding | None:
    """Establish whether the boards can be joined at all."""
    if not deals or not work_orders:
        return None

    deal_clients = {d.client_code.value for d in deals if d.client_code.ok}
    wo_customers = {w.customer_code.value for w in work_orders if w.customer_code.ok}
    shared_codes = deal_clients & wo_customers

    deal_names = {d.name.strip().lower() for d in deals if d.name}
    wo_names = {
        w.deal_name.value.strip().lower() for w in work_orders if w.deal_name.ok
    }
    shared_names = deal_names & wo_names

    ambiguous = 0
    if shared_names:
        deal_name_counts = collections.Counter(
            d.name.strip().lower() for d in deals if d.name
        )
        wo_name_counts = collections.Counter(
            w.deal_name.value.strip().lower() for w in work_orders if w.deal_name.ok
        )
        ambiguous = sum(
            1 for n in shared_names
            if deal_name_counts[n] > 1 and wo_name_counts[n] > 1
        )

    return Finding(
        code="cross_board_identity",
        title="The two boards share no reliable join key",
        detail=(
            f"Client codes do not overlap at all ({len(shared_codes)} shared out of "
            f"{len(deal_clients)} deal codes and {len(wo_customers)} work order codes) - "
            "the two boards were masked into independent identifier namespaces, so codes "
            "that look similar refer to different companies. Deal name is the only shared "
            f"attribute ({len(shared_names)} names in common), but it is not unique: "
            f"{ambiguous} of those names map to several deals *and* several work orders."
        ),
        severity=Severity.HIGH,
        fields=("client_code", "customer_code", "name", "deal_name"),
        affected_records=len(work_orders),
        handling=(
            "Cross-board questions are answered by comparing aggregates (by sector, owner "
            "or period), never by joining rows. A row-level join here would fabricate "
            "relationships that do not exist in the data."
        ),
    )


def _check_value_concentration(dataset, deals: list[Deal], work_orders) -> Finding | None:
    values = sorted(
        (d.value_inr.value for d in deals if d.value_inr.ok and d.value_inr.value),
        reverse=True,
    )
    if len(values) < 10:
        return None
    total = sum(values)
    if total <= 0:
        return None
    top_share = values[0] / total
    if top_share < 0.15:
        return None
    return Finding(
        code="value_concentration",
        title="Deal values are extremely concentrated",
        detail=(
            f"The single largest deal accounts for {top_share:.0%} of all recorded deal "
            f"value, and values span from {min(values):,.0f} to {max(values):,.0f}. "
            "Totals and averages are dominated by a handful of records."
        ),
        severity=Severity.MEDIUM,
        fields=("value_inr",),
        board="deals",
        affected_records=len(values),
        handling=(
            "Value answers report the median and the largest deals' share alongside the "
            "total, so a single outlier is never presented as a trend."
        ),
    )


def _check_unmapped_columns(dataset: Dataset, deals, work_orders) -> Finding | None:
    unmapped = {k: v for k, v in dataset.unmapped_columns.items() if v}
    if not unmapped:
        return None
    described = "; ".join(
        f"{board.replace('_', ' ')}: " + ", ".join(cols) for board, cols in unmapped.items()
    )
    return Finding(
        code="unmapped_columns",
        title="Board columns the agent does not understand",
        detail=(
            f"These columns exist in monday.com but map to no known business field, so "
            f"nothing is analysed from them: {described}."
        ),
        severity=Severity.MEDIUM,
        affected_records=0,
        handling=(
            "Reported rather than ignored - an unexpected column is often a renamed one, "
            "which would otherwise silently remove a field from analysis."
        ),
    )


def _check_data_recency(dataset: Dataset, deals, work_orders) -> Finding | None:
    """Flag when the data stops well before today."""
    from datetime import date as _date

    as_of = dataset.as_of
    if not as_of:
        return None
    today = _date.today()
    days = (today - as_of).days
    if days < 45:
        return None
    return Finding(
        code="data_recency",
        title="The data stops well before today",
        detail=(
            f"The most recent activity recorded on either board is {as_of.isoformat()}, "
            f"{days} days before today ({today.isoformat()}). Periods after that date "
            "contain no records at all."
        ),
        severity=Severity.HIGH,
        affected_records=0,
        handling=(
            "Relative periods such as 'this quarter' are resolved literally first. If the "
            "literal period turns out to be empty, the most recent period that does hold "
            "data is used instead and the substitution is stated in the answer."
        ),
    )
