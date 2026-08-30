"""Executes a validated `QueryPlan` against a `Dataset`.

Everything here is deterministic. Given the same records and the same plan, the same
numbers come out - which is the property that lets the LLM be trusted with the prose
and nothing else.

The output, `AnalysisResult`, is the only thing the responder ever sees. It carries
the computed metrics, the assumptions made along the way, the data-quality findings
that touch the fields actually used, and a note of what was filtered out - so an
answer can always explain itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from ..agent.intent import Intent
from ..agent.plan import QueryPlan
from ..ingest.entities import Dataset, Deal, WorkOrder
from ..quality.audit import Finding, QualityReport
from . import metrics as M
from .timeframe import Period


@dataclass
class AnalysisResult:
    """The complete, self-describing outcome of one question."""

    plan: QueryPlan
    metrics: dict[str, M.MetricResult] = field(default_factory=dict)
    #: Sentences the answer should state as assumptions.
    assumptions: list[str] = field(default_factory=list)
    #: Data-quality findings relevant to the fields this answer actually used.
    data_quality: list[Finding] = field(default_factory=list)
    #: Record counts after filtering, per board.
    records_in_scope: dict[str, int] = field(default_factory=dict)
    #: Record counts before filtering, per board.
    records_total: dict[str, int] = field(default_factory=dict)
    #: When the underlying data was fetched, and whether it is live.
    fetched_at: str | None = None
    is_stale: bool = False
    as_of: date | None = None
    #: Set when the plan produced nothing to report.
    empty_reason: str | None = None

    @property
    def available_metrics(self) -> dict[str, M.MetricResult]:
        return {k: v for k, v in self.metrics.items() if v.available}

    def source_fields(self) -> set[str]:
        fields: set[str] = set()
        for result in self.metrics.values():
            fields.update(result.provenance.source_fields)
        return fields


def _filter_deals(
    deals: Sequence[Deal],
    plan: QueryPlan,
    period: Period | None,
    *,
    apply_status: bool = True,
) -> tuple[list[Deal], list[str]]:
    """Apply a plan's filters to deals, reporting what each one removed.

    `apply_status=False` keeps sector and period but drops the status filter, for
    metrics whose definition spans statuses.
    """
    notes: list[str] = []
    selected = list(deals)

    if plan.sectors:
        wanted = set(plan.sectors)
        selected = [d for d in selected if d.sector.or_none() in wanted]

    if plan.status and apply_status:
        by_status = {
            "open": M.open_deals, "won": M.won_deals,
            "lost": M.lost_deals, "held": M.held_deals,
        }[plan.status]
        selected = by_status(selected)

    if period is not None:
        # Deals are placed in time by creation date. Actual close dates are recorded
        # on only a small minority of records, so filtering on them would silently
        # drop most of the pipeline.
        dated = [d for d in selected if d.created_date.ok]
        undated = len(selected) - len(dated)
        selected = [d for d in dated if period.contains(d.created_date.value)]
        if undated:
            notes.append(
                f"{undated} deal(s) have no creation date and are excluded from the "
                "period filter."
            )
        notes.append("Deals are placed in a period by their creation date.")

    return selected, notes


def _filter_work_orders(
    work_orders: Sequence[WorkOrder], plan: QueryPlan, period: Period | None
) -> tuple[list[WorkOrder], list[str]]:
    notes: list[str] = []
    selected = list(work_orders)

    if plan.sectors:
        wanted = set(plan.sectors)
        selected = [w for w in selected if w.sector.or_none() in wanted]

    if period is not None:
        # PO date is the most reliably populated date on this board.
        dated = [w for w in selected if w.po_date.ok]
        undated = len(selected) - len(dated)
        selected = [w for w in dated if period.contains(w.po_date.value)]
        if undated:
            notes.append(
                f"{undated} work order(s) have no PO date and are excluded from the "
                "period filter."
            )
        notes.append("Work orders are placed in a period by their PO/LOI date.")

    return selected, notes


def has_data_in_period(dataset: Dataset, plan: QueryPlan) -> callable:
    """Build the emptiness test the timeframe resolver uses.

    Applies the plan's non-time filters, so "this quarter" is judged empty against the
    slice the user actually asked about - a quarter with plenty of mining deals but no
    energy deals is genuinely empty for an energy question.
    """

    def check(period: Period) -> bool:
        if "deals" in plan.boards:
            deals, _ = _filter_deals(dataset.active_deals, plan, period)
            if deals:
                return True
        if "work_orders" in plan.boards:
            work_orders, _ = _filter_work_orders(dataset.active_work_orders, plan, period)
            if work_orders:
                return True
        return False

    return check


def execute(plan: QueryPlan, dataset: Dataset, quality: QualityReport) -> AnalysisResult:
    """Run a validated plan and return everything needed to answer honestly."""
    plan.validate()

    period = plan.timeframe.period if plan.timeframe else None

    deals, deal_notes = _filter_deals(dataset.active_deals, plan, period)
    work_orders, wo_notes = _filter_work_orders(dataset.active_work_orders, plan, period)

    result = AnalysisResult(
        plan=plan,
        records_total={
            "deals": len(dataset.active_deals),
            "work_orders": len(dataset.active_work_orders),
        },
        records_in_scope={"deals": len(deals), "work_orders": len(work_orders)},
        fetched_at=dataset.fetched_at.isoformat() if dataset.fetched_at else None,
        is_stale=dataset.is_stale,
        as_of=dataset.as_of,
    )

    result.assumptions.extend(plan.assumptions)
    if plan.timeframe:
        result.assumptions.extend(plan.timeframe.assumptions)
    if "deals" in plan.boards:
        result.assumptions.extend(deal_notes)
    if "work_orders" in plan.boards:
        result.assumptions.extend(wo_notes)

    # Metrics whose meaning spans statuses see the sector/period slice without the
    # status filter - otherwise "what is our win rate?" answered about won deals would
    # be 100% by construction.
    spanning = [n for n in plan.metrics if M.REGISTRY[n].spans_statuses]
    deals_all_statuses = deals
    if plan.status and spanning:
        deals_all_statuses, _ = _filter_deals(
            dataset.active_deals, plan, period, apply_status=False
        )
        result.assumptions.append(
            f"Win rate and stage distribution are computed across all outcomes, "
            f"ignoring the '{plan.status}' filter - restricting them to one outcome "
            "would make the result true by definition."
        )

    for name in plan.metrics:
        source = deals_all_statuses if M.REGISTRY[name].spans_statuses else deals
        result.metrics[name] = M.compute(name, source, work_orders)

    # Pull in metric-level assumptions without repeating any.
    for metric in result.metrics.values():
        for assumption in metric.provenance.assumptions:
            if assumption not in result.assumptions:
                result.assumptions.append(assumption)

    # Only caveats touching fields this answer actually used - except when the
    # question *is* about data quality, where the whole report is the answer.
    if plan.intent is Intent.DATA_QUALITY:
        result.data_quality = quality.by_severity()
    else:
        result.data_quality = quality.for_fields(result.source_fields())

    if not result.available_metrics and plan.metrics:
        scope = " and ".join(
            f"{count} {board.replace('_', ' ')}"
            for board, count in result.records_in_scope.items()
            if board in plan.boards
        )
        result.empty_reason = (
            f"No metric could be computed. {scope or 'No records'} matched the filters "
            f"({plan.describe_filters()})."
        )

    return result
