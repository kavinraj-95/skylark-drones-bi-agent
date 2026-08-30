"""The deterministic metric registry.

Every number this system reports is computed here, in Python, from normalized
records. The LLM never does arithmetic - it receives finished `MetricResult` objects
and explains them. That separation is the reason a founder can trust the figures.

Each metric declares, in one place:

* what it means, in a sentence a founder would accept;
* which canonical fields it reads (used to attach the right data-quality caveats);
* how it aggregates; and
* what it does about missing data.

That last point is the one that matters most. A metric never silently treats a
missing value as zero. It either excludes the record and says how many it excluded,
or it does not produce a number at all.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..ingest.entities import (
    HELD_STAGES,
    LOST_STAGES,
    WON_STAGES,
    Deal,
    WorkOrder,
)
from .provenance import Provenance


class Unit(str, Enum):
    INR = "inr"
    COUNT = "count"
    PERCENT = "percent"
    RATIO = "ratio"


#: Heuristic weights for `weighted_pipeline_value`.
#:
#: These are declared assumptions, not measurements. The dataset cannot calibrate
#: them: on already-closed deals the probability bands separate outcomes perfectly
#: (High = 100% won), which is the signature of a field updated after the fact rather
#: than a forecast. Stage-derived weights are no better - several stages *are* the
#: outcome, so a win rate per stage restates the classification instead of predicting
#: it. So we use fixed, conventional weights, label the metric a heuristic, and never
#: call the result expected revenue.
PROBABILITY_WEIGHTS: dict[str, float] = {"High": 0.8, "Medium": 0.5, "Low": 0.2}

#: How many of the largest deals to report when describing concentration.
CONCENTRATION_TOP_N = 5


@dataclass
class MetricResult:
    """One computed number, with everything needed to present it honestly."""

    name: str
    label: str
    value: float | None
    unit: Unit
    definition: str
    provenance: Provenance
    #: Supporting numbers that give the headline value context (median, top share).
    context: dict[str, Any] = field(default_factory=dict)
    #: Set when the metric could not be computed at all, explaining why.
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.value is not None and self.unavailable_reason is None

    def formatted(self) -> str:
        """Display string. Rupee amounts use Indian crore/lakh scaling."""
        if not self.available:
            return "not available"
        assert self.value is not None
        if self.unit is Unit.INR:
            return format_inr(self.value)
        if self.unit is Unit.COUNT:
            return f"{int(self.value):,}"
        if self.unit is Unit.PERCENT:
            return f"{self.value:.1f}%"
        return f"{self.value:,.2f}"


def format_inr(amount: float) -> str:
    """Format rupees using Indian scale words, which is how a founder here reads them."""
    sign = "-" if amount < 0 else ""
    value = abs(amount)
    if value >= 1e7:
        return f"{sign}₹{value / 1e7:,.2f} Cr"
    if value >= 1e5:
        return f"{sign}₹{value / 1e5:,.2f} L"
    return f"{sign}₹{value:,.0f}"


@dataclass(frozen=True)
class MetricDefinition:
    """Registry entry describing one metric."""

    name: str
    label: str
    definition: str
    unit: Unit
    #: Canonical fields read. Drives data-quality caveat selection.
    source_fields: tuple[str, ...]
    #: Which boards the metric needs: "deals", "work_orders", or both.
    boards: tuple[str, ...]
    #: (deals, work_orders) -> MetricResult
    compute: Callable[[Sequence[Deal], Sequence[WorkOrder]], MetricResult]
    missing_data_rule: str


# --------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------


def _sum_field(
    records: Sequence[Any],
    field_name: str,
    *,
    name: str,
    label: str,
    definition: str,
    unit: Unit,
    boards: tuple[str, ...],
    extra_assumptions: Sequence[str] = (),
) -> MetricResult:
    """Sum one numeric field, excluding - never zero-filling - unusable values.

    Records whose value is missing or malformed are counted into the provenance with
    their reason, so the answer can say how much of the population the total covers.
    """
    prov = Provenance(
        records_considered=len(records),
        source_fields=(field_name,),
        boards=boards,
        assumptions=list(extra_assumptions),
    )

    total = 0.0
    used = 0
    by_reason: dict[str, list[str]] = {}
    for record in records:
        f = record.get(field_name)
        if f.ok and f.value is not None:
            total += float(f.value)
            used += 1
        else:
            reason = f.note or f"No value recorded ({f.state.value.lower()})."
            by_reason.setdefault(reason, []).append(record.item_id)

    prov.records_used = used
    for reason, ids in by_reason.items():
        prov.add_exclusion(reason, len(ids), field_name=field_name, examples=ids[:3])

    if used == 0:
        return MetricResult(
            name=name, label=label, value=None, unit=unit, definition=definition,
            provenance=prov,
            unavailable_reason=(
                f"None of the {len(records)} matching record(s) have a usable "
                f"{field_name.replace('_', ' ')}."
            ),
        )

    values = [
        float(r.get(field_name).value)
        for r in records
        if r.get(field_name).ok and r.get(field_name).value is not None
    ]
    return MetricResult(
        name=name, label=label, value=total, unit=unit, definition=definition,
        provenance=prov,
        context={
            "median": statistics.median(values),
            "largest": max(values),
            "smallest": min(values),
            "records_with_value": used,
        },
    )


def _count(
    records: Sequence[Any], *, name: str, label: str, definition: str, boards: tuple[str, ...],
    source_fields: tuple[str, ...] = (),
) -> MetricResult:
    prov = Provenance(
        records_considered=len(records), records_used=len(records),
        source_fields=source_fields, boards=boards,
    )
    return MetricResult(
        name=name, label=label, value=float(len(records)), unit=Unit.COUNT,
        definition=definition, provenance=prov,
    )


def _distribution(
    records: Sequence[Any], field_name: str, *, name: str, label: str, definition: str,
    boards: tuple[str, ...],
) -> MetricResult:
    """Count records by a categorical field, keeping unusable values visible."""
    prov = Provenance(
        records_considered=len(records), source_fields=(field_name,), boards=boards,
    )
    counts: dict[str, int] = {}
    unusable = 0
    for record in records:
        f = record.get(field_name)
        if f.value:
            # UNMAPPED values keep their original text and are counted under it, so an
            # unfamiliar category shows up rather than vanishing.
            counts[str(f.value)] = counts.get(str(f.value), 0) + 1
        else:
            unusable += 1

    prov.records_used = len(records) - unusable
    prov.add_exclusion(
        f"No {field_name.replace('_', ' ')} recorded.", unusable, field_name=field_name
    )

    return MetricResult(
        name=name, label=label, value=float(len(counts)), unit=Unit.COUNT,
        definition=definition, provenance=prov,
        context={"breakdown": dict(sorted(counts.items(), key=lambda kv: -kv[1]))},
    )


def open_deals(deals: Sequence[Deal]) -> list[Deal]:
    """Deals still live in the sales pipeline, judged by stage.

    Stage is used rather than status because it is fully populated and ordered, while
    status is known to carry a defaulted value on a large block of records.
    """
    return [d for d in deals if d.is_open]


def won_deals(deals: Sequence[Deal]) -> list[Deal]:
    return [d for d in deals if d.stage.or_none() in WON_STAGES]


def lost_deals(deals: Sequence[Deal]) -> list[Deal]:
    return [d for d in deals if d.stage.or_none() in LOST_STAGES]


def held_deals(deals: Sequence[Deal]) -> list[Deal]:
    return [d for d in deals if d.stage.or_none() in HELD_STAGES]


# --------------------------------------------------------------------------------
# Metric implementations
# --------------------------------------------------------------------------------


def _open_pipeline_value(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _sum_field(
        open_deals(deals), "value_inr",
        name="open_pipeline_value", label="Open pipeline value",
        definition=(
            "Total recorded value of deals still in a live sales stage (stages A-F, "
            "before the outcome is decided)."
        ),
        unit=Unit.INR, boards=("deals",),
    )


def _weighted_pipeline_value(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    """Open pipeline scaled by declared heuristic probability weights.

    Deliberately *not* called expected revenue. Deals with no closure probability are
    excluded and counted - never assigned a default weight, because the data provides
    no defensible basis for one.
    """
    candidates = open_deals(deals)
    prov = Provenance(
        records_considered=len(candidates),
        source_fields=("value_inr", "closure_probability"),
        boards=("deals",),
        assumptions=[
            "Weighted pipeline applies fixed heuristic weights to the recorded closure "
            "probability ("
            + ", ".join(f"{k} = {v:.0%}" for k, v in PROBABILITY_WEIGHTS.items())
            + "). These are conventional assumptions, not win rates measured from this "
            "data - the probability field appears to be set after the outcome is known, "
            "so it cannot calibrate them.",
            "This is a weighting of open pipeline, not a revenue forecast.",
        ],
    )

    total = 0.0
    used = 0
    no_value: list[str] = []
    no_probability: list[str] = []
    unknown_band: list[str] = []

    for deal in candidates:
        if not deal.value_inr.ok or deal.value_inr.value is None:
            no_value.append(deal.item_id)
            continue
        band = deal.closure_probability.or_none()
        if not band:
            no_probability.append(deal.item_id)
            continue
        weight = PROBABILITY_WEIGHTS.get(band)
        if weight is None:
            unknown_band.append(deal.item_id)
            continue
        total += float(deal.value_inr.value) * weight
        used += 1

    prov.records_used = used
    prov.add_exclusion("No deal value recorded.", len(no_value), field_name="value_inr",
                       examples=no_value[:3])
    prov.add_exclusion(
        "No closure probability recorded, and no default is assumed.",
        len(no_probability), field_name="closure_probability", examples=no_probability[:3],
    )
    prov.add_exclusion(
        "Closure probability outside the known High/Medium/Low bands.",
        len(unknown_band), field_name="closure_probability", examples=unknown_band[:3],
    )

    if used == 0:
        return MetricResult(
            name="weighted_pipeline_value", label="Weighted pipeline (heuristic)",
            value=None, unit=Unit.INR,
            definition="Open pipeline value scaled by heuristic closure-probability weights.",
            provenance=prov,
            unavailable_reason=(
                "No open deal has both a value and a closure probability, so a weighted "
                "figure cannot be computed."
            ),
        )

    return MetricResult(
        name="weighted_pipeline_value", label="Weighted pipeline (heuristic)",
        value=total, unit=Unit.INR,
        definition=(
            "Open pipeline value scaled by heuristic closure-probability weights. A "
            "risk-adjusted view of open pipeline, not a revenue forecast."
        ),
        provenance=prov,
        context={"weights": dict(PROBABILITY_WEIGHTS)},
    )


def _deal_count(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _count(
        deals, name="deal_count", label="Deals",
        definition="Number of deal records matching the filters.", boards=("deals",),
    )


def _open_deal_count(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _count(
        open_deals(deals), name="open_deal_count", label="Open deals",
        definition="Number of deals in a live sales stage (A-F).",
        boards=("deals",), source_fields=("stage", "stage_rank"),
    )


def _stage_distribution(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _distribution(
        deals, "stage", name="stage_distribution", label="Deals by stage",
        definition="Count of deals at each pipeline stage.", boards=("deals",),
    )


def _pipeline_by_sector(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    """Open pipeline value grouped by sector."""
    candidates = open_deals(deals)
    prov = Provenance(
        records_considered=len(candidates), source_fields=("value_inr", "sector"),
        boards=("deals",),
    )
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    used = 0
    no_value = 0
    no_sector = 0

    for deal in candidates:
        sector = deal.sector.or_none()
        if not sector:
            no_sector += 1
            continue
        if not deal.value_inr.ok or deal.value_inr.value is None:
            no_value += 1
            counts[sector] = counts.get(sector, 0)
            continue
        totals[sector] = totals.get(sector, 0.0) + float(deal.value_inr.value)
        counts[sector] = counts.get(sector, 0) + 1
        used += 1

    prov.records_used = used
    prov.add_exclusion("No sector recorded.", no_sector, field_name="sector")
    prov.add_exclusion("No deal value recorded.", no_value, field_name="value_inr")

    if not totals:
        return MetricResult(
            name="pipeline_by_sector", label="Open pipeline by sector", value=None,
            unit=Unit.INR,
            definition="Open pipeline value grouped by sector.", provenance=prov,
            unavailable_reason="No open deal has both a sector and a value.",
        )

    return MetricResult(
        name="pipeline_by_sector", label="Open pipeline by sector",
        value=sum(totals.values()), unit=Unit.INR,
        definition="Open pipeline value grouped by sector.", provenance=prov,
        context={
            "breakdown": dict(sorted(totals.items(), key=lambda kv: -kv[1])),
            "deal_counts": counts,
        },
    )


def _won_value(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _sum_field(
        won_deals(deals), "value_inr",
        name="won_value", label="Closed-won deal value",
        definition=(
            "Total recorded value of deals whose stage indicates the deal was won "
            "(Project Won, Work Order Received, Invoice Sent, Amount Accrued, "
            "Project Completed)."
        ),
        unit=Unit.INR, boards=("deals",),
        extra_assumptions=[
            "Won/lost is determined by Deal Stage rather than Deal Status, because "
            "Status carries a defaulted value on a large block of records."
        ],
    )


def _win_rate(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    """Share of decided deals that were won.

    Only deals whose stage represents a settled outcome count. Open and parked deals
    are excluded from the denominator - counting them would understate the rate simply
    because they have not finished yet.
    """
    won = won_deals(deals)
    lost = lost_deals(deals)
    decided = len(won) + len(lost)

    prov = Provenance(
        records_considered=len(deals), records_used=decided,
        source_fields=("stage",), boards=("deals",),
        assumptions=[
            "Win rate counts only deals with a settled outcome. Deals still open or on "
            "hold are excluded from the denominator rather than counted as losses."
        ],
    )
    prov.add_exclusion(
        "Still open - outcome not yet decided.", len(open_deals(deals)), field_name="stage"
    )
    prov.add_exclusion("On hold or in POC - outcome not settled.", len(held_deals(deals)),
                       field_name="stage")

    if decided == 0:
        return MetricResult(
            name="win_rate", label="Win rate", value=None, unit=Unit.PERCENT,
            definition="Won deals as a share of deals with a settled outcome.",
            provenance=prov,
            unavailable_reason="No deal in the selection has reached a settled outcome.",
        )

    return MetricResult(
        name="win_rate", label="Win rate", value=len(won) / decided * 100, unit=Unit.PERCENT,
        definition="Won deals as a share of deals with a settled outcome (won + lost).",
        provenance=prov,
        context={"won": len(won), "lost": len(lost), "decided": decided},
    )


def _deal_concentration(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    """How much of open pipeline sits in the largest few deals.

    Reported alongside every value total, because in this data one deal can be a third
    of the book and a bare total would hide that entirely.
    """
    values = sorted(
        (
            (float(d.value_inr.value), d.name or d.item_id)
            for d in open_deals(deals)
            if d.value_inr.ok and d.value_inr.value
        ),
        key=lambda pair: -pair[0],
    )
    prov = Provenance(
        records_considered=len(open_deals(deals)), records_used=len(values),
        source_fields=("value_inr",), boards=("deals",),
    )
    prov.add_exclusion(
        "No deal value recorded.", len(open_deals(deals)) - len(values), field_name="value_inr"
    )

    if not values:
        return MetricResult(
            name="deal_concentration", label="Pipeline concentration", value=None,
            unit=Unit.PERCENT,
            definition="Share of open pipeline value held by the largest deals.",
            provenance=prov,
            unavailable_reason="No open deal has a recorded value.",
        )

    total = sum(v for v, _ in values)
    top = values[:CONCENTRATION_TOP_N]
    top_share = (sum(v for v, _ in top) / total * 100) if total else 0.0

    return MetricResult(
        name="deal_concentration", label="Pipeline concentration",
        value=top_share, unit=Unit.PERCENT,
        definition=(
            f"Share of open pipeline value held by the largest {CONCENTRATION_TOP_N} deals."
        ),
        provenance=prov,
        context={
            "largest_share_pct": (values[0][0] / total * 100) if total else 0.0,
            "largest_deal": values[0][1],
            "median_deal_value": statistics.median([v for v, _ in values]),
            "top_deals": [{"name": n, "value": v} for v, n in top],
        },
    )


def _work_order_count(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _count(
        work_orders, name="work_order_count", label="Work orders",
        definition="Number of work order records matching the filters.",
        boards=("work_orders",),
    )


def _active_work_order_count(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    """Work orders that are not finished.

    "Active" means execution has started or is pending but not complete. Records with
    no execution status are excluded and reported rather than assumed either way.
    """
    finished = {"Completed"}
    active = [
        w for w in work_orders
        if w.execution_status.ok and w.execution_status.value not in finished
    ]
    unknown = [w for w in work_orders if not w.execution_status.ok]

    prov = Provenance(
        records_considered=len(work_orders), records_used=len(work_orders) - len(unknown),
        source_fields=("execution_status",), boards=("work_orders",),
        assumptions=[
            "'Active' means any execution status other than Completed - including "
            "Ongoing, Not Started, Partial Completed and paused work."
        ],
    )
    prov.add_exclusion("No execution status recorded.", len(unknown),
                       field_name="execution_status")

    return MetricResult(
        name="active_work_order_count", label="Active work orders",
        value=float(len(active)), unit=Unit.COUNT,
        definition="Work orders whose execution status is anything other than Completed.",
        provenance=prov,
        context={
            "breakdown": {
                status: sum(1 for w in active if w.execution_status.value == status)
                for status in sorted({
                    w.execution_status.value for w in active if w.execution_status.ok
                })
            }
        },
    )


def _execution_status_distribution(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _distribution(
        work_orders, "execution_status",
        name="execution_status_distribution", label="Work orders by execution status",
        definition="Count of work orders at each execution status.", boards=("work_orders",),
    )


def _work_order_value(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _sum_field(
        work_orders, "amount_excl_gst",
        name="work_order_value", label="Work order value (excl GST)",
        definition="Total contracted value of matching work orders, excluding GST.",
        unit=Unit.INR, boards=("work_orders",),
    )


def _billed_value(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _sum_field(
        work_orders, "billed_excl_gst",
        name="billed_value", label="Billed value (excl GST)",
        definition="Total value invoiced to date across matching work orders, excluding GST.",
        unit=Unit.INR, boards=("work_orders",),
        extra_assumptions=[
            "Records that leave this field blank are excluded rather than counted as "
            "zero - the board uses blank and 0 interchangeably for the same fact."
        ],
    )


def _unbilled_value(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _sum_field(
        work_orders, "to_be_billed_excl_gst",
        name="unbilled_value", label="Still to bill (excl GST)",
        definition="Total value contracted but not yet invoiced, excluding GST.",
        unit=Unit.INR, boards=("work_orders",),
    )


def _receivables(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    return _sum_field(
        work_orders, "receivable",
        name="receivables", label="Amount receivable",
        definition="Total outstanding receivable across matching work orders.",
        unit=Unit.INR, boards=("work_orders",),
    )


def _wo_value_by_sector(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    prov = Provenance(
        records_considered=len(work_orders), source_fields=("amount_excl_gst", "sector"),
        boards=("work_orders",),
    )
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    used = 0
    skipped = 0
    for wo in work_orders:
        sector = wo.sector.or_none()
        if not sector or not wo.amount_excl_gst.ok or wo.amount_excl_gst.value is None:
            skipped += 1
            continue
        totals[sector] = totals.get(sector, 0.0) + float(wo.amount_excl_gst.value)
        counts[sector] = counts.get(sector, 0) + 1
        used += 1

    prov.records_used = used
    prov.add_exclusion("No sector or no order value recorded.", skipped)

    if not totals:
        return MetricResult(
            name="wo_value_by_sector", label="Work order value by sector", value=None,
            unit=Unit.INR, definition="Contracted work order value grouped by sector.",
            provenance=prov, unavailable_reason="No work order has both a sector and a value.",
        )

    return MetricResult(
        name="wo_value_by_sector", label="Work order value by sector",
        value=sum(totals.values()), unit=Unit.INR,
        definition="Contracted work order value grouped by sector, excluding GST.",
        provenance=prov,
        context={
            "breakdown": dict(sorted(totals.items(), key=lambda kv: -kv[1])),
            "work_order_counts": counts,
        },
    )


def _sector_sales_vs_ops(deals: Sequence[Deal], work_orders: Sequence[WorkOrder]) -> MetricResult:
    """Compare where sales effort sits against where delivery work sits.

    This is the cross-board metric, and it is deliberately an *aggregate* comparison.
    The two boards share no reliable join key - client codes live in independent
    masked namespaces and deal names are not unique - so joining rows would fabricate
    relationships. Comparing each board's own sector mix is sound, because sector is
    recorded natively on both.
    """
    pipeline: dict[str, float] = {}
    for deal in open_deals(deals):
        sector = deal.sector.or_none()
        if sector and deal.value_inr.ok and deal.value_inr.value:
            pipeline[sector] = pipeline.get(sector, 0.0) + float(deal.value_inr.value)

    ops_count: dict[str, int] = {}
    ops_value: dict[str, float] = {}
    for wo in work_orders:
        sector = wo.sector.or_none()
        if not sector:
            continue
        ops_count[sector] = ops_count.get(sector, 0) + 1
        if wo.amount_excl_gst.ok and wo.amount_excl_gst.value:
            ops_value[sector] = ops_value.get(sector, 0.0) + float(wo.amount_excl_gst.value)

    prov = Provenance(
        records_considered=len(deals) + len(work_orders),
        records_used=len(open_deals(deals)) + len(work_orders),
        source_fields=("sector", "value_inr", "amount_excl_gst", "execution_status"),
        boards=("deals", "work_orders"),
        assumptions=[
            "The two boards are compared at sector level, not joined record by record. "
            "They share no reliable key: client codes are in separate masked namespaces "
            "and deal names repeat across many records, so a row-level join would invent "
            "relationships that the data does not support.",
        ],
    )

    pipeline_total = sum(pipeline.values())
    ops_total = sum(ops_value.values())
    sectors = sorted(set(pipeline) | set(ops_count))

    comparison = []
    for sector in sectors:
        comparison.append({
            "sector": sector,
            "pipeline_value": pipeline.get(sector, 0.0),
            "pipeline_share_pct": (pipeline.get(sector, 0.0) / pipeline_total * 100)
            if pipeline_total else 0.0,
            "work_orders": ops_count.get(sector, 0),
            "work_order_value": ops_value.get(sector, 0.0),
            "work_order_share_pct": (ops_value.get(sector, 0.0) / ops_total * 100)
            if ops_total else 0.0,
        })
    comparison.sort(key=lambda row: -row["pipeline_value"])

    if not comparison:
        return MetricResult(
            name="sector_sales_vs_ops", label="Sales pipeline vs operational workload",
            value=None, unit=Unit.COUNT,
            definition="Sector-level comparison of open pipeline against work order load.",
            provenance=prov, unavailable_reason="Neither board has usable sector data.",
        )

    return MetricResult(
        name="sector_sales_vs_ops", label="Sales pipeline vs operational workload",
        value=float(len(comparison)), unit=Unit.COUNT,
        definition=(
            "Sector-level comparison of open sales pipeline against delivered/active "
            "work order load. Compared in aggregate, never joined record by record."
        ),
        provenance=prov,
        context={
            "comparison": comparison,
            "pipeline_total": pipeline_total,
            "work_order_total": ops_total,
        },
    )


# --------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------

_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "open_pipeline_value", "Open pipeline value",
        "Total recorded value of deals in live sales stages (A-F).",
        Unit.INR, ("value_inr", "stage"), ("deals",), _open_pipeline_value,
        "Deals with no recorded value are excluded and counted, never treated as zero.",
    ),
    MetricDefinition(
        "weighted_pipeline_value", "Weighted pipeline (heuristic)",
        "Open pipeline scaled by declared heuristic closure-probability weights.",
        Unit.INR, ("value_inr", "closure_probability", "stage"), ("deals",),
        _weighted_pipeline_value,
        "Deals lacking a value or a probability are excluded; no default weight is assumed.",
    ),
    MetricDefinition(
        "deal_count", "Deals", "Number of deal records matching the filters.",
        Unit.COUNT, (), ("deals",), _deal_count, "All matching records are counted.",
    ),
    MetricDefinition(
        "open_deal_count", "Open deals", "Number of deals in live sales stages (A-F).",
        Unit.COUNT, ("stage", "stage_rank"), ("deals",), _open_deal_count,
        "Deals with an unrecognised stage are not counted as open.",
    ),
    MetricDefinition(
        "stage_distribution", "Deals by stage", "Count of deals at each pipeline stage.",
        Unit.COUNT, ("stage",), ("deals",), _stage_distribution,
        "Deals with no stage are counted separately as unrecorded.",
    ),
    MetricDefinition(
        "pipeline_by_sector", "Open pipeline by sector",
        "Open pipeline value grouped by sector.",
        Unit.INR, ("value_inr", "sector", "stage"), ("deals",), _pipeline_by_sector,
        "Deals missing either sector or value are excluded and counted.",
    ),
    MetricDefinition(
        "won_value", "Closed-won deal value",
        "Total value of deals whose stage indicates a win.",
        Unit.INR, ("value_inr", "stage"), ("deals",), _won_value,
        "Won/lost comes from Deal Stage, not Deal Status. Valueless deals are excluded.",
    ),
    MetricDefinition(
        "win_rate", "Win rate", "Won deals as a share of deals with a settled outcome.",
        Unit.PERCENT, ("stage",), ("deals",), _win_rate,
        "Open and on-hold deals are excluded from the denominator, not counted as losses.",
    ),
    MetricDefinition(
        "deal_concentration", "Pipeline concentration",
        "Share of open pipeline value held by the largest deals.",
        Unit.PERCENT, ("value_inr", "stage"), ("deals",), _deal_concentration,
        "Computed only over deals that have a recorded value.",
    ),
    MetricDefinition(
        "work_order_count", "Work orders", "Number of work order records matching the filters.",
        Unit.COUNT, (), ("work_orders",), _work_order_count, "All matching records are counted.",
    ),
    MetricDefinition(
        "active_work_order_count", "Active work orders",
        "Work orders whose execution status is anything other than Completed.",
        Unit.COUNT, ("execution_status",), ("work_orders",), _active_work_order_count,
        "Work orders with no execution status are excluded and reported.",
    ),
    MetricDefinition(
        "execution_status_distribution", "Work orders by execution status",
        "Count of work orders at each execution status.",
        Unit.COUNT, ("execution_status",), ("work_orders",), _execution_status_distribution,
        "Work orders with no status are counted separately as unrecorded.",
    ),
    MetricDefinition(
        "work_order_value", "Work order value (excl GST)",
        "Total contracted value of matching work orders.",
        Unit.INR, ("amount_excl_gst",), ("work_orders",), _work_order_value,
        "Records with no amount are excluded and counted.",
    ),
    MetricDefinition(
        "billed_value", "Billed value (excl GST)",
        "Total value invoiced to date, excluding GST.",
        Unit.INR, ("billed_excl_gst",), ("work_orders",), _billed_value,
        "Blank is treated as unknown and excluded; a recorded 0 is treated as a real zero.",
    ),
    MetricDefinition(
        "unbilled_value", "Still to bill (excl GST)",
        "Total value contracted but not yet invoiced.",
        Unit.INR, ("to_be_billed_excl_gst",), ("work_orders",), _unbilled_value,
        "Records with no amount are excluded and counted.",
    ),
    MetricDefinition(
        "receivables", "Amount receivable", "Total outstanding receivable.",
        Unit.INR, ("receivable",), ("work_orders",), _receivables,
        "Records with no amount are excluded and counted.",
    ),
    MetricDefinition(
        "wo_value_by_sector", "Work order value by sector",
        "Contracted work order value grouped by sector.",
        Unit.INR, ("amount_excl_gst", "sector"), ("work_orders",), _wo_value_by_sector,
        "Records missing sector or value are excluded and counted.",
    ),
    MetricDefinition(
        "sector_sales_vs_ops", "Sales pipeline vs operational workload",
        "Sector-level comparison of open pipeline against work order load.",
        Unit.COUNT, ("sector", "value_inr", "amount_excl_gst"), ("deals", "work_orders"),
        _sector_sales_vs_ops,
        "Aggregate comparison only - the boards are never joined record by record.",
    ),
)

REGISTRY: dict[str, MetricDefinition] = {d.name: d for d in _DEFINITIONS}

#: Names the query planner is allowed to request. Anything outside this set is
#: rejected during plan validation rather than executed.
METRIC_NAMES: tuple[str, ...] = tuple(REGISTRY)


def compute(
    name: str, deals: Sequence[Deal], work_orders: Sequence[WorkOrder]
) -> MetricResult:
    """Run one registered metric. Unknown names are a programming error, not input."""
    definition = REGISTRY.get(name)
    if definition is None:
        raise KeyError(f"Unknown metric {name!r}.")
    return definition.compute(deals, work_orders)
