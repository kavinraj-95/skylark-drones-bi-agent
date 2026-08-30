"""Data-quality auditing.

Findings must be *derived*, not asserted. Each test therefore checks that the audit
discovers a property the source data genuinely has - and, where it matters, that it
does not fire on data that lacks it.
"""

from __future__ import annotations

import copy

import pytest

from skylark_bi.ingest.builder import build_dataset
from skylark_bi.ingest.entities import FieldState
from skylark_bi.monday.client import BoardSnapshot
from skylark_bi.quality.audit import Severity, audit


def codes(report) -> set[str]:
    return {f.code for f in report.findings}


class TestFindingsAreDiscovered:
    def test_status_stage_conflict_is_characterised_not_just_counted(self, quality_report):
        finding = next(f for f in quality_report.findings if f.code == "status_stage_conflict")
        assert finding.severity is Severity.HIGH
        assert finding.affected_records > 0
        # It should explain the pattern, not merely report a number.
        assert "backlog" in finding.detail.lower()
        assert "Stage is treated as authoritative" in finding.handling

    def test_probability_leakage_is_detected(self, quality_report):
        """The band that separates closed outcomes perfectly is the tell."""
        finding = next(f for f in quality_report.findings if f.code == "probability_leakage")
        assert finding.severity is Severity.HIGH
        assert "100% won" in finding.detail
        assert "retrospectively" in finding.title.lower()

    def test_cross_board_join_is_reported_as_impossible(self, quality_report):
        finding = next(f for f in quality_report.findings if f.code == "cross_board_identity")
        assert "0 shared" in finding.detail
        assert "never by joining rows" in finding.handling

    def test_empty_columns_exclude_merely_ambiguous_ones(self, quality_report, dataset):
        """A column full of unusable values is not the same as an empty column."""
        finding = next(
            f for f in quality_report.findings if f.code.startswith("empty_columns_work")
        )
        for name in finding.fields:
            states = {w.get(name).state for w in dataset.active_work_orders}
            assert states <= {FieldState.MISSING, FieldState.NOT_APPLICABLE}
            assert FieldState.AMBIGUOUS not in states

    def test_blank_versus_zero_is_flagged(self, quality_report):
        assert any(f.code.startswith("blank_vs_zero") for f in quality_report.findings)

    def test_header_echo_rows_are_reported(self, quality_report):
        finding = next(f for f in quality_report.findings if f.code == "excluded_rows")
        assert finding.affected_records > 0

    def test_malformed_values_carry_an_example(self, quality_report):
        finding = next(
            f for f in quality_report.findings if f.code.startswith("malformed_work")
        )
        assert "e.g." in finding.detail

    def test_recency_gap_is_flagged(self, quality_report):
        finding = next(f for f in quality_report.findings if f.code == "data_recency")
        assert finding.severity is Severity.HIGH
        assert "no records" in finding.detail

    def test_concentration_is_flagged(self, quality_report):
        assert "value_concentration" in codes(quality_report)

    def test_every_finding_explains_its_handling(self, quality_report):
        """A finding without a handling note tells a founder nothing actionable."""
        for finding in quality_report.findings:
            assert finding.handling, f"{finding.code} has no handling note"
            assert finding.detail


class TestFindingsAreDataDriven:
    """The audit must describe the data in front of it, not this dataset specifically."""

    def test_clean_data_produces_far_fewer_findings(
        self, deals_snapshot, work_orders_snapshot, quality_report
    ):
        clean_deals = BoardSnapshot(
            board_id="1", board_name="Deals",
            columns=deals_snapshot.columns, items=deals_snapshot.items[:1],
        )
        clean_wos = BoardSnapshot(
            board_id="2", board_name="Work Orders",
            columns=work_orders_snapshot.columns, items=work_orders_snapshot.items[:1],
        )
        small = audit(build_dataset(clean_deals, clean_wos))
        assert len(small.findings) < len(quality_report.findings)

    def test_no_findings_on_empty_boards_beyond_structure(
        self, deals_snapshot, work_orders_snapshot
    ):
        dataset = build_dataset(
            BoardSnapshot("1", "Deals", [], deals_snapshot.columns),
            BoardSnapshot("2", "Work Orders", [], work_orders_snapshot.columns),
        )
        report = audit(dataset)
        assert "probability_leakage" not in codes(report)
        assert "value_concentration" not in codes(report)

    def test_counts_track_the_data(self, deals_snapshot, work_orders_snapshot, quality_report):
        """Halving the deals must change what the audit reports."""
        half = BoardSnapshot(
            board_id="1", board_name="Deals",
            columns=deals_snapshot.columns,
            items=deals_snapshot.items[: len(deals_snapshot.items) // 2],
        )
        report = audit(build_dataset(half, work_orders_snapshot))
        full_conflicts = next(
            f for f in quality_report.findings if f.code == "status_stage_conflict"
        )
        half_conflicts = next(
            (f for f in report.findings if f.code == "status_stage_conflict"), None
        )
        assert half_conflicts is None or (
            half_conflicts.affected_records < full_conflicts.affected_records
        )


class TestCaveatTargeting:
    def test_only_relevant_findings_attach_to_a_metric(self, quality_report):
        """Caveats must be scoped, or people learn to ignore them."""
        pipeline = quality_report.for_fields({"value_inr", "stage"})
        ops = quality_report.for_fields({"execution_status"})
        assert len(pipeline) != len(ops)
        assert any(f.code == "deal_value_missing" for f in pipeline)
        assert not any(f.code == "deal_value_missing" for f in ops)

    def test_dataset_wide_findings_always_apply(self, quality_report):
        """A finding with no declared fields is relevant to every answer."""
        anything = quality_report.for_fields({"execution_status"})
        assert any(f.code == "data_recency" for f in anything)

    def test_severity_ordering(self, quality_report):
        ordered = quality_report.by_severity()
        severities = [f.severity for f in ordered]
        assert severities == sorted(
            severities, key=lambda s: {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}[s]
        )


class TestCoverage:
    def test_coverage_percentages_are_sane(self, quality_report):
        for coverage in quality_report.deal_coverage + quality_report.work_order_coverage:
            assert 0 <= coverage.usable_pct <= 100
            assert sum(coverage.counts.values()) == coverage.total

    def test_fully_populated_field_reports_full_coverage(self, quality_report):
        stage = next(c for c in quality_report.deal_coverage if c.name == "stage")
        assert stage.usable_pct == pytest.approx(100.0)
