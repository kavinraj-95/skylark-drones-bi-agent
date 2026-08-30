"""Metric correctness, asserted against values computed independently from the CSVs.

Every expected number here is derived in the test itself from the source data, not
copied from a previous run. If a metric silently changes behaviour - starts treating
blanks as zero, say - these fail.
"""

from __future__ import annotations

import csv
from datetime import date

import pytest

from skylark_bi.agent.intent import Intent, QueryIntent
from skylark_bi.agent.resolver import resolve
from skylark_bi.analytics import metrics as M
from skylark_bi.analytics.engine import execute
from skylark_bi.analytics.timeframe import (
    PeriodBasis,
    fiscal_quarter_of,
    fiscal_year_of,
    resolve_timeframe,
)
from skylark_bi.ingest.entities import WON_STAGES

from .conftest import IMPORT_DIR


def _raw_deal_rows():
    """Read the source CSV directly - an independent check on the whole pipeline."""
    with (IMPORT_DIR / "monday_deals.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Drop the header-echo rows the way the importer's consumer should.
    return [r for r in rows if (r.get("Deal Status") or "").strip() != "Deal Status"]


class TestOpenPipeline:
    def test_matches_an_independent_sum(self, dataset):
        open_deals = M.open_deals(dataset.active_deals)
        expected = sum(
            d.value_inr.value for d in open_deals if d.value_inr.ok and d.value_inr.value
        )
        result = M.compute("open_pipeline_value", dataset.active_deals, [])
        assert result.value == pytest.approx(expected)

    def test_missing_values_are_excluded_never_zero_filled(self, dataset):
        open_deals = M.open_deals(dataset.active_deals)
        with_value = [d for d in open_deals if d.value_inr.ok]
        result = M.compute("open_pipeline_value", dataset.active_deals, [])

        assert result.provenance.records_used == len(with_value)
        assert result.provenance.records_considered == len(open_deals)
        assert result.provenance.records_excluded == len(open_deals) - len(with_value)
        # The exclusion is reported, not swallowed.
        assert result.provenance.exclusions

    def test_coverage_is_reported_honestly(self, dataset):
        result = M.compute("open_pipeline_value", dataset.active_deals, [])
        assert 0 < result.provenance.coverage < 1
        assert "of" in result.provenance.summary()


class TestWeightedPipeline:
    def test_applies_declared_weights_only(self, dataset):
        open_deals = M.open_deals(dataset.active_deals)
        expected = sum(
            d.value_inr.value * M.PROBABILITY_WEIGHTS[d.closure_probability.value]
            for d in open_deals
            if d.value_inr.ok
            and d.closure_probability.ok
            and d.closure_probability.value in M.PROBABILITY_WEIGHTS
        )
        result = M.compute("weighted_pipeline_value", dataset.active_deals, [])
        assert result.value == pytest.approx(expected)

    def test_deals_without_probability_are_excluded_not_defaulted(self, dataset):
        """No default weight exists - the data cannot justify one."""
        result = M.compute("weighted_pipeline_value", dataset.active_deals, [])
        reasons = " ".join(e.reason for e in result.provenance.exclusions)
        assert "no default is assumed" in reasons

    def test_is_labelled_a_heuristic_not_a_forecast(self, dataset):
        result = M.compute("weighted_pipeline_value", dataset.active_deals, [])
        assert "heuristic" in result.label.lower()
        assert any("not a revenue forecast" in a for a in result.provenance.assumptions)

    def test_never_exceeds_unweighted_pipeline(self, dataset):
        weighted = M.compute("weighted_pipeline_value", dataset.active_deals, [])
        unweighted = M.compute("open_pipeline_value", dataset.active_deals, [])
        assert weighted.value <= unweighted.value


class TestWinRate:
    def test_open_deals_are_not_counted_as_losses(self, dataset):
        result = M.compute("win_rate", dataset.active_deals, [])
        won = len(M.won_deals(dataset.active_deals))
        lost = len(M.lost_deals(dataset.active_deals))
        assert result.context["decided"] == won + lost
        assert result.value == pytest.approx(won / (won + lost) * 100)

    def test_uses_stage_not_status(self, dataset):
        """Status is unreliable here; the metric must not depend on it."""
        by_stage = len([d for d in dataset.active_deals if d.stage.or_none() in WON_STAGES])
        by_status = len([d for d in dataset.active_deals if d.status.or_none() == "Won"])
        assert by_stage != by_status, "fixture no longer exercises the disagreement"

        result = M.compute("win_rate", dataset.active_deals, [])
        assert result.context["won"] == by_stage


class TestBilledValue:
    def test_blank_excluded_but_recorded_zero_counted(self, dataset):
        """The blank-vs-zero distinction, asserted on the field that mixes them."""
        work_orders = dataset.active_work_orders
        expected = sum(
            w.billed_excl_gst.value
            for w in work_orders
            if w.billed_excl_gst.ok and w.billed_excl_gst.value is not None
        )
        result = M.compute("billed_value", [], work_orders)
        assert result.value == pytest.approx(expected)

        zeros = [w for w in work_orders if w.billed_incl_gst.ok and w.billed_incl_gst.value == 0]
        assert zeros, "fixture no longer contains recorded zeros"
        incl = M.compute("work_order_value", [], work_orders)
        assert incl.available


class TestConcentration:
    def test_reports_share_and_median_together(self, dataset):
        result = M.compute("deal_concentration", dataset.active_deals, [])
        assert result.available
        assert 0 < result.value <= 100
        assert "median_deal_value" in result.context
        assert "largest_share_pct" in result.context
        assert len(result.context["top_deals"]) <= M.CONCENTRATION_TOP_N


class TestCrossBoard:
    def test_compares_aggregates_and_says_so(self, dataset):
        result = M.compute(
            "sector_sales_vs_ops", dataset.active_deals, dataset.active_work_orders
        )
        assert result.available
        assert any("never joined" in a or "not joined" in a
                   for a in result.provenance.assumptions)
        assert set(result.provenance.boards) == {"deals", "work_orders"}

    def test_every_sector_row_carries_both_sides(self, dataset):
        result = M.compute(
            "sector_sales_vs_ops", dataset.active_deals, dataset.active_work_orders
        )
        for row in result.context["comparison"]:
            assert {"sector", "pipeline_value", "work_orders", "work_order_value"} <= set(row)


class TestUnavailableMetrics:
    def test_no_data_yields_unavailable_not_zero(self):
        """An absent number must read as absent, never as zero."""
        result = M.compute("open_pipeline_value", [], [])
        assert not result.available
        assert result.value is None
        assert result.formatted() == "not available"
        assert result.unavailable_reason


class TestTimeframe:
    def test_indian_fiscal_year_boundaries(self):
        assert fiscal_year_of(date(2025, 6, 1), 4) == 2025
        assert fiscal_year_of(date(2026, 2, 1), 4) == 2025   # still FY25-26
        assert fiscal_year_of(date(2026, 4, 1), 4) == 2026
        assert fiscal_quarter_of(date(2026, 1, 15), 4) == 4

    def test_relative_period_anchors_on_data_not_today(self, dataset):
        """The whole point: the data trails the calendar."""
        resolved = resolve_timeframe(
            "this quarter", as_of=dataset.as_of, fiscal_start_month=4,
            has_data=lambda p: True,
        )
        assert resolved.period.contains(dataset.as_of)
        assert any("not from today" in a for a in resolved.assumptions)

    def test_empty_period_substitutes_and_reports_it(self):
        populated = {"FY25-26 Q2 (fiscal)"}
        resolved = resolve_timeframe(
            "this quarter", as_of=date(2026, 1, 15), fiscal_start_month=4,
            has_data=lambda p: p.label in populated,
        )
        assert resolved.substituted
        assert resolved.period.label == "FY25-26 Q2 (fiscal)"
        assert resolved.literal_period.label == "FY25-26 Q4 (fiscal)"
        assert any("contains no records" in a for a in resolved.assumptions)

    def test_substitution_preserves_granularity(self):
        """An empty year must fall back to a year, not narrow to a quarter."""
        resolved = resolve_timeframe(
            "2027", as_of=date(2026, 1, 15), fiscal_start_month=4,
            has_data=lambda p: p.label == "FY24-25 (fiscal year)",
        )
        assert resolved.period.kind.value == "year"

    def test_uninterpretable_period_falls_back_to_all_data(self):
        resolved = resolve_timeframe(
            "since the dawn of time", as_of=date(2026, 1, 15), fiscal_start_month=4
        )
        assert resolved.is_unbounded
        assert resolved.assumptions


class TestEndToEnd:
    """The full path: question -> intent -> plan -> data -> metrics."""

    def test_energy_this_quarter(self, dataset, quality_report):
        intent = QueryIntent(
            intent=Intent.PIPELINE_HEALTH,
            sector_term="energy sector",
            time_expression="this quarter",
            restatement="Energy pipeline this quarter",
        )
        plan = resolve(intent, dataset, fiscal_start_month=4)
        assert not plan.needs_clarification
        # "energy" is an inference, and both sectors are applied.
        assert set(plan.sectors) == {"Renewables", "Powerline"}
        assert plan.status == "open"

        result = execute(plan, dataset, quality_report)
        assert result.available_metrics
        joined = " ".join(result.assumptions)
        assert "energy" in joined            # mapping disclosed
        assert "not from today" in joined    # anchoring disclosed

    def test_cross_board_question_uses_both_boards(self, dataset, quality_report):
        intent = QueryIntent(intent=Intent.CROSS_BOARD, restatement="sales vs ops")
        plan = resolve(intent, dataset, fiscal_start_month=4)
        result = execute(plan, dataset, quality_report)
        assert set(plan.boards) == {"deals", "work_orders"}
        assert result.records_in_scope["deals"] > 0
        assert result.records_in_scope["work_orders"] > 0

    def test_unknown_sector_asks_rather_than_guessing(self, dataset):
        intent = QueryIntent(
            intent=Intent.PIPELINE_HEALTH, sector_term="fintech", restatement="fintech"
        )
        plan = resolve(intent, dataset, fiscal_start_month=4)
        assert plan.needs_clarification
        assert "fintech" in plan.clarifying_question

    def test_unsupported_question_is_declined_clearly(self, dataset):
        intent = QueryIntent(intent=Intent.UNSUPPORTED, restatement="headcount")
        plan = resolve(intent, dataset, fiscal_start_month=4)
        assert plan.needs_clarification

    def test_answer_carries_only_relevant_caveats(self, dataset, quality_report):
        intent = QueryIntent(intent=Intent.OPERATIONS, restatement="work orders")
        plan = resolve(intent, dataset, fiscal_start_month=4)
        result = execute(plan, dataset, quality_report)
        assert len(result.data_quality) < len(quality_report.findings)


class TestPipelineMatchesSource:
    """Independent reconciliation against the raw CSV."""

    def test_deal_count_reconciles(self, dataset):
        assert len(dataset.active_deals) == len(_raw_deal_rows())

    def test_header_echo_rows_are_excluded_and_counted(self, dataset):
        excluded = [d for d in dataset.deals if d.excluded]
        assert excluded
        assert all("column headers" in (d.exclusion_reason or "") for d in excluded)

    def test_as_of_ignores_forecast_dates(self, dataset):
        """A forecast must never define how current the data is."""
        forecasts = [
            d.tentative_close_date.value
            for d in dataset.active_deals
            if d.tentative_close_date.ok and d.tentative_close_date.value
        ]
        assert max(forecasts) > dataset.as_of


class TestStatusFilterInteraction:
    """A status filter must not make a status-spanning metric true by construction."""

    def test_win_rate_ignores_a_won_filter(self, dataset, quality_report):
        """"What's our win rate?" once returned 100% because the planner read "win" as
        a status filter, leaving only won deals in the denominator."""
        from skylark_bi.agent.plan import QueryPlan

        plan = QueryPlan(
            intent=Intent.REVENUE, metrics=("win_rate",), boards=("deals",), status="won"
        )
        result = execute(plan, dataset, quality_report)
        win_rate = result.metrics["win_rate"]

        assert win_rate.available
        assert win_rate.value < 100.0
        assert win_rate.context["lost"] > 0
        assert any("true by definition" in a for a in result.assumptions)

    def test_unfiltered_win_rate_is_identical(self, dataset, quality_report):
        from skylark_bi.agent.plan import QueryPlan

        filtered = execute(
            QueryPlan(Intent.REVENUE, ("win_rate",), ("deals",), status="won"),
            dataset, quality_report,
        ).metrics["win_rate"]
        unfiltered = execute(
            QueryPlan(Intent.REVENUE, ("win_rate",), ("deals",)),
            dataset, quality_report,
        ).metrics["win_rate"]
        assert filtered.value == pytest.approx(unfiltered.value)

    def test_sector_filter_still_applies_to_spanning_metrics(self, dataset, quality_report):
        """Only the status filter is dropped - sector and period must still bite."""
        from skylark_bi.agent.plan import QueryPlan

        scoped = execute(
            QueryPlan(Intent.REVENUE, ("win_rate",), ("deals",),
                      sectors=("Mining",), status="won"),
            dataset, quality_report,
        ).metrics["win_rate"]
        overall = execute(
            QueryPlan(Intent.REVENUE, ("win_rate",), ("deals",)),
            dataset, quality_report,
        ).metrics["win_rate"]
        assert scoped.context["decided"] < overall.context["decided"]

    def test_value_metrics_still_respect_status(self, dataset, quality_report):
        from skylark_bi.agent.plan import QueryPlan

        won_only = execute(
            QueryPlan(Intent.REVENUE, ("deal_count",), ("deals",), status="won"),
            dataset, quality_report,
        ).metrics["deal_count"]
        everything = execute(
            QueryPlan(Intent.REVENUE, ("deal_count",), ("deals",)),
            dataset, quality_report,
        ).metrics["deal_count"]
        assert won_only.value < everything.value
