"""Turns a `QueryIntent` into a validated `QueryPlan`.

This is the deterministic middle of the pipeline, and the reason the LLM's output is
safe to act on. The model says what the user appears to want, in their words; this
module decides what that means in this dataset, using whitelists and the actual data:

    QueryIntent (LLM, raw terms)  ->  [ this module ]  ->  QueryPlan (concrete, checked)

Nothing the model wrote is passed through as-is. Sectors are resolved against the
sectors present in the data, time expressions against the dataset's as-of date, and
metric names come from the registry keyed by intent - never from the model at all.
"""

from __future__ import annotations

from ..analytics.engine import has_data_in_period
from ..analytics.timeframe import PeriodBasis, resolve_timeframe
from ..ingest.entities import Dataset
from .intent import Intent, QueryIntent
from .plan import INTENT_BOARDS, INTENT_METRICS, QueryPlan
from .semantic import resolve_sector, resolve_status

#: Intents that are implicitly about *open* deals unless the user says otherwise.
_OPEN_BY_DEFAULT = frozenset({Intent.PIPELINE_HEALTH, Intent.RISK})


def available_sectors(dataset: Dataset) -> tuple[str, ...]:
    """Sectors actually present in the current data, on either board."""
    sectors = {
        d.sector.value for d in dataset.active_deals if d.sector.value
    } | {
        w.sector.value for w in dataset.active_work_orders if w.sector.value
    }
    return tuple(sorted(str(s) for s in sectors))


def resolve(
    intent: QueryIntent,
    dataset: Dataset,
    *,
    fiscal_start_month: int,
    basis: PeriodBasis = PeriodBasis.FISCAL,
) -> QueryPlan:
    """Build an executable plan from the model's reading of the question."""
    plan = QueryPlan(
        intent=intent.intent,
        metrics=INTENT_METRICS.get(intent.intent, ()),
        boards=INTENT_BOARDS.get(intent.intent, ()),
        restatement=intent.restatement,
    )

    if intent.intent is Intent.UNSUPPORTED:
        plan.clarifying_question = (
            intent.clarifying_question
            or "That question is outside what this data can answer. Try asking about "
               "pipeline, revenue, sectors, work orders, billing or data quality."
        )
        return plan

    # The model may only *request* a clarification; whether one is warranted is
    # checked here, and a clarification with no question attached is ignored.
    if intent.needs_clarification and intent.clarifying_question:
        plan.clarifying_question = intent.clarifying_question
        return plan

    # -- sector ------------------------------------------------------------------
    sector_resolution = resolve_sector(
        intent.sector_term, available=available_sectors(dataset) or None
    )
    plan.sectors = sector_resolution.sectors
    plan.assumptions.extend(sector_resolution.notes)

    # An unresolvable sector materially changes the answer, so this is one of the few
    # cases where asking beats assuming.
    if sector_resolution.unresolved and sector_resolution.suggestions:
        plan.clarifying_question = (
            f"I could not find a sector called '{sector_resolution.term}'. "
            f"Did you mean one of: {', '.join(sector_resolution.suggestions)}?"
        )
        return plan

    # -- status ------------------------------------------------------------------
    status, status_notes = resolve_status(intent.status_term)
    plan.assumptions.extend(status_notes)

    if status is None and intent.intent in _OPEN_BY_DEFAULT:
        # "How's our pipeline looking" means open pipeline. Making that implicit
        # filter explicit matters for more than tidiness: the emptiness test below
        # uses the plan's filters, so without it a quarter containing only *closed*
        # deals would look populated and the substitution would never fire, leaving
        # the user with an answer full of "not available".
        status = "open"
        plan.assumptions.append(
            "'Pipeline' is read as deals still open (stages A-F); closed and lost "
            "deals are excluded."
        )
    plan.status = status

    # -- timeframe ---------------------------------------------------------------
    # Resolved after the other filters so that emptiness is judged against the slice
    # the user actually asked about.
    if intent.time_expression:
        plan.timeframe = resolve_timeframe(
            intent.time_expression,
            as_of=dataset.as_of,
            fiscal_start_month=fiscal_start_month,
            basis=basis,
            has_data=has_data_in_period(dataset, plan),
        )

    plan.validate()
    return plan
