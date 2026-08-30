"""Behaviour when the boards are not what we expect.

Column renames, missing columns, unfamiliar values and empty boards are all normal in
a real monday workspace. None of them should produce a stack trace, and none should
produce a confidently wrong number.
"""

from __future__ import annotations

import copy

import pytest

from skylark_bi.ingest.builder import build_dataset, build_deals
from skylark_bi.ingest.entities import FieldState
from skylark_bi.monday.client import BoardSnapshot
from skylark_bi.monday.errors import SchemaMismatchError
from skylark_bi.quality.audit import audit


def _rename_column(snapshot: BoardSnapshot, title: str, new_title: str) -> BoardSnapshot:
    clone = copy.deepcopy(snapshot)
    for column in clone.columns:
        if column["title"] == title:
            column["title"] = new_title
    return clone


def _drop_column(snapshot: BoardSnapshot, title: str) -> BoardSnapshot:
    clone = copy.deepcopy(snapshot)
    dropped = {c["id"] for c in clone.columns if c["title"] == title}
    clone.columns = [c for c in clone.columns if c["id"] not in dropped]
    for item in clone.items:
        item["column_values"] = [
            cv for cv in item["column_values"] if cv["id"] not in dropped
        ]
    return clone


class TestColumnDrift:
    def test_tolerates_case_and_punctuation_changes(self, deals_snapshot):
        """A human tidying a column title must not break the agent."""
        renamed = _rename_column(deals_snapshot, "Deal Name", "deal-name")
        deals, resolution = build_deals(renamed)
        assert "name" in resolution.field_to_column
        assert any(d.name for d in deals)

    def test_one_missing_column_degrades_gracefully(self, deals_snapshot):
        """The field goes MISSING and is reported - the app keeps working."""
        without_value = _drop_column(deals_snapshot, "Masked Deal value")
        deals, resolution = build_deals(without_value)

        assert "value_inr" in resolution.missing_fields
        assert all(d.value_inr.state is FieldState.MISSING for d in deals)
        assert all("no column supplying" in (d.value_inr.note or "") for d in deals[:5])

    def test_metrics_report_unavailable_rather_than_zero(self, deals_snapshot):
        from skylark_bi.analytics import metrics as M

        deals, _ = build_deals(_drop_column(deals_snapshot, "Masked Deal value"))
        result = M.compute("open_pipeline_value", deals, [])
        assert not result.available
        assert result.value is None          # not 0.0
        assert result.unavailable_reason

    def test_wholesale_schema_mismatch_fails_loudly(self, deals_snapshot):
        """The failure mode we actually hit: an import that lost its header row.

        Reporting every metric as "no data" would look like an empty business rather
        than a broken import, so this must raise something actionable instead.
        """
        broken = copy.deepcopy(deals_snapshot)
        for index, column in enumerate(broken.columns):
            column["title"] = f"column_{index}"

        with pytest.raises(SchemaMismatchError) as exc:
            build_deals(broken)

        message = exc.value.user_message
        assert "header row" in message
        assert "column_0" in message   # shows what it actually found

    def test_unexpected_extra_column_is_reported_not_ignored(self, deals_snapshot):
        """An unknown column may be a renamed one, so it must surface."""
        clone = copy.deepcopy(deals_snapshot)
        clone.columns.append(
            {"id": "extra_1", "title": "Some New Field", "type": "text", "settings_str": "{}"}
        )
        for item in clone.items:
            item["column_values"].append({"id": "extra_1", "type": "text", "text": "x"})

        _, resolution = build_deals(clone)
        assert "Some New Field" in resolution.unmapped_columns

    def test_duplicate_titles_are_flagged_not_picked_at_random(self, deals_snapshot):
        """monday does not enforce unique column titles."""
        clone = copy.deepcopy(deals_snapshot)
        clone.columns.append(
            {"id": "dupe_1", "title": "Deal Name", "type": "text", "settings_str": "{}"}
        )
        _, resolution = build_deals(clone)
        assert "Deal Name" in resolution.duplicate_titles


class TestEmptyAndOddBoards:
    def test_empty_board_does_not_crash(self, deals_snapshot, work_orders_snapshot):
        empty = BoardSnapshot(
            board_id="1", board_name="Deals", items=[], columns=deals_snapshot.columns
        )
        dataset = build_dataset(empty, work_orders_snapshot)
        assert dataset.active_deals == []
        assert dataset.as_of is not None       # still derivable from work orders
        assert audit(dataset) is not None

    def test_both_boards_empty(self, deals_snapshot, work_orders_snapshot):
        dataset = build_dataset(
            BoardSnapshot("1", "Deals", [], deals_snapshot.columns),
            BoardSnapshot("2", "Work Orders", [], work_orders_snapshot.columns),
        )
        assert dataset.as_of is None
        report = audit(dataset)
        assert isinstance(report.findings, list)

    def test_item_with_no_column_values(self, deals_snapshot):
        clone = copy.deepcopy(deals_snapshot)
        clone.items.append({"id": "999", "name": "Bare", "column_values": []})
        deals, _ = build_deals(clone)
        bare = next(d for d in deals if d.item_id == "999")
        assert bare.name == "Bare"             # falls back to the item name
        assert not bare.value_inr.ok

    def test_null_text_values_are_handled(self, deals_snapshot):
        clone = copy.deepcopy(deals_snapshot)
        for cv in clone.items[0]["column_values"]:
            cv["text"] = None
        deals, _ = build_deals(clone)
        assert deals[0] is not None


class TestUnknownValues:
    def test_unfamiliar_sector_is_kept_and_surfaced(self, deals_snapshot):
        """A new sector must not be snapped onto an existing one."""
        clone = copy.deepcopy(deals_snapshot)
        sector_col = next(c["id"] for c in clone.columns if c["title"] == "Sector/service")
        for cv in clone.items[0]["column_values"]:
            if cv["id"] == sector_col:
                cv["text"] = "Quantum Computing"

        deals, _ = build_deals(clone)
        changed = deals[0]
        assert changed.sector.state is FieldState.UNMAPPED
        assert changed.sector.value == "Quantum Computing"

    def test_unknown_stage_does_not_sort_to_the_front(self, deals_snapshot):
        clone = copy.deepcopy(deals_snapshot)
        stage_col = next(c["id"] for c in clone.columns if c["title"] == "Deal Stage")
        for cv in clone.items[0]["column_values"]:
            if cv["id"] == stage_col:
                cv["text"] = "Z. Brand New Stage"

        deals, _ = build_deals(clone)
        changed = deals[0]
        assert changed.stage_rank.state is FieldState.UNMAPPED
        assert changed.stage_rank.value is None
        assert not changed.is_open          # never assumed to be early-stage
