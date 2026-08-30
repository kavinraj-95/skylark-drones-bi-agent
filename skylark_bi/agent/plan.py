"""`QueryPlan` - a validated, executable description of what to compute.

The plan is built by *code* from a `QueryIntent`, never by the LLM. By the time one
exists, every field is concrete and checked: metric names come from the registry,
sectors from the data, dates from the timeframe resolver. Anything the model asked
for that is not on those whitelists never reaches execution.

This is the boundary the whole design rests on. The LLM proposes; this module
disposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..analytics.metrics import METRIC_NAMES
from ..analytics.timeframe import ResolvedTimeframe
from .intent import Intent

#: Which metrics answer which kind of question. Keeping this in code rather than in a
#: prompt means the set of numbers behind an answer is reproducible and reviewable.
INTENT_METRICS: dict[Intent, tuple[str, ...]] = {
    Intent.PIPELINE_HEALTH: (
        "open_pipeline_value", "weighted_pipeline_value", "open_deal_count",
        "pipeline_by_sector", "deal_concentration", "stage_distribution",
    ),
    Intent.REVENUE: (
        "won_value", "win_rate", "billed_value", "unbilled_value", "receivables",
    ),
    Intent.SECTOR_PERFORMANCE: (
        "pipeline_by_sector", "wo_value_by_sector", "open_pipeline_value",
        "won_value", "deal_count",
    ),
    Intent.OPERATIONS: (
        "work_order_count", "active_work_order_count", "execution_status_distribution",
        "work_order_value",
    ),
    Intent.BILLING: (
        "billed_value", "unbilled_value", "receivables", "work_order_value",
    ),
    Intent.CROSS_BOARD: (
        "sector_sales_vs_ops", "open_pipeline_value", "work_order_value",
        "active_work_order_count",
    ),
    Intent.RISK: (
        "deal_concentration", "open_pipeline_value", "weighted_pipeline_value",
        "receivables", "unbilled_value", "win_rate",
    ),
    Intent.LEADERSHIP_UPDATE: (
        "open_pipeline_value", "weighted_pipeline_value", "won_value", "win_rate",
        "pipeline_by_sector", "deal_concentration", "active_work_order_count",
        "execution_status_distribution", "billed_value", "receivables",
        "sector_sales_vs_ops",
    ),
    Intent.DATA_QUALITY: (),
    Intent.CAPABILITIES: (),
    Intent.UNSUPPORTED: (),
}

#: Which boards each intent needs. Used to skip work and to tell the user which
#: sources an answer drew on.
INTENT_BOARDS: dict[Intent, tuple[str, ...]] = {
    Intent.PIPELINE_HEALTH: ("deals",),
    Intent.REVENUE: ("deals", "work_orders"),
    Intent.SECTOR_PERFORMANCE: ("deals", "work_orders"),
    Intent.OPERATIONS: ("work_orders",),
    Intent.BILLING: ("work_orders",),
    Intent.CROSS_BOARD: ("deals", "work_orders"),
    Intent.RISK: ("deals", "work_orders"),
    Intent.LEADERSHIP_UPDATE: ("deals", "work_orders"),
    Intent.DATA_QUALITY: ("deals", "work_orders"),
    Intent.CAPABILITIES: (),
    Intent.UNSUPPORTED: (),
}


class PlanValidationError(ValueError):
    """A plan referenced something outside the allowed vocabulary."""


@dataclass
class QueryPlan:
    """Everything needed to compute an answer, all of it already validated."""

    intent: Intent
    metrics: tuple[str, ...]
    boards: tuple[str, ...]
    #: Concrete sector names to filter on. Empty means no sector filter.
    sectors: tuple[str, ...] = ()
    #: "open" | "won" | "lost" | "held", or None for no status filter.
    status: str | None = None
    timeframe: ResolvedTimeframe | None = None
    #: Assumptions to state in the answer, accumulated during resolution.
    assumptions: list[str] = field(default_factory=list)
    #: The user's question, restated in business terms.
    restatement: str = ""
    #: Set when the plan cannot be executed and the user must be asked something.
    clarifying_question: str | None = None

    def validate(self) -> None:
        """Reject anything outside the registry or the known board names.

        Called immediately before execution. A failure here is a bug or an attack,
        never ordinary input - the resolver only ever builds plans from whitelists.
        """
        unknown_metrics = [m for m in self.metrics if m not in METRIC_NAMES]
        if unknown_metrics:
            raise PlanValidationError(
                f"Unknown metric(s): {', '.join(unknown_metrics)}."
            )

        unknown_boards = [b for b in self.boards if b not in ("deals", "work_orders")]
        if unknown_boards:
            raise PlanValidationError(f"Unknown board(s): {', '.join(unknown_boards)}.")

        if self.status is not None and self.status not in ("open", "won", "lost", "held"):
            raise PlanValidationError(f"Unknown status filter: {self.status!r}.")

    @property
    def needs_clarification(self) -> bool:
        return self.clarifying_question is not None

    def describe_filters(self) -> str:
        """Human-readable filter summary for the analysis panel."""
        parts: list[str] = []
        if self.sectors:
            parts.append("Sector: " + ", ".join(self.sectors))
        if self.status:
            parts.append(f"Status: {self.status}")
        if self.timeframe and self.timeframe.period:
            parts.append(f"Period: {self.timeframe.period.label}")
        elif self.timeframe and self.timeframe.is_unbounded:
            parts.append("Period: all available data")
        return " · ".join(parts) if parts else "No filters - all records"
