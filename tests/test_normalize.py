"""Value normalization, exercised on the exact messy values in the source data."""

from __future__ import annotations

from datetime import date

import pytest

from skylark_bi.ingest.entities import FieldState
from skylark_bi.ingest.normalize import (
    canonicalize,
    parse_date,
    parse_month_name,
    parse_number,
    parse_quantity,
    parse_text,
)
from skylark_bi.ingest.mapping import INVOICE_STATUSES, SECTORS


class TestNumbers:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("489360", 489360.0),
            ("7832265.523", 7832265.523),
            ("1.2332", 1.2332),          # masking artifact, still a real value
            ("₹1,234.5", 1234.5),
            ("(500)", -500.0),
        ],
    )
    def test_parses_real_values(self, raw, expected):
        field = parse_number(raw)
        assert field.state is FieldState.OK
        assert field.value == pytest.approx(expected)

    def test_zero_is_a_value_not_a_blank(self):
        """The single most important rule: a recorded 0 is data, a blank is not."""
        zero = parse_number("0")
        assert zero.state is FieldState.OK
        assert zero.value == 0.0
        assert zero.ok

        blank = parse_number("")
        assert blank.state is FieldState.MISSING
        assert blank.value is None
        assert not blank.ok

    def test_unparseable_keeps_its_raw_text(self):
        field = parse_number("Rate based on MW slabs")
        assert field.state is FieldState.MALFORMED
        assert field.raw == "Rate based on MW slabs"
        assert field.value is None


class TestDates:
    def test_iso(self):
        assert parse_date("2025-11-28").value == date(2025, 11, 28)

    def test_day_first(self):
        assert parse_date("12-01-2025").value == date(2025, 1, 12)

    def test_monday_json_blob(self):
        assert parse_date('{"date":"2025-03-04"}').value == date(2025, 3, 4)

    def test_bare_month_is_ambiguous_not_guessed(self):
        """A month with no year must never be assigned one."""
        field = parse_date("Dec")
        assert field.state is FieldState.AMBIGUOUS
        assert field.value is None
        assert not field.ok

    def test_month_and_year_is_still_not_a_day(self):
        assert parse_date("July 2025").state is FieldState.AMBIGUOUS

    def test_garbage_is_malformed(self):
        assert parse_date("45days").state is FieldState.MALFORMED


class TestQuantities:
    @pytest.mark.parametrize(
        "raw,amount,unit",
        [("1", 1.0, None), ("1600", 1600.0, None), ("45days", 45.0, "days"),
         ("45 days", 45.0, "days"), ("5360 HA", 5360.0, "HA")],
    )
    def test_units_are_kept(self, raw, amount, unit):
        field = parse_quantity(raw)
        assert field.state is FieldState.OK
        assert field.value.amount == amount
        assert field.value.unit == unit

    def test_explicit_na_differs_from_blank(self):
        """`NA` asserts the field does not apply; blank says we simply do not know."""
        assert parse_quantity("NA").state is FieldState.NOT_APPLICABLE
        assert parse_quantity("").state is FieldState.MISSING


class TestCanonicalisation:
    def test_known_variants_collapse(self):
        assert canonicalize("renewable", SECTORS, field_name="sector").value == "Renewables"
        assert canonicalize("BIlled", INVOICE_STATUSES, field_name="s").value == "Fully Billed"

    def test_unknown_value_is_kept_not_forced(self):
        """An unfamiliar category is more likely new than misspelled."""
        field = canonicalize("Billed- Visit 7", INVOICE_STATUSES, field_name="invoice status")
        assert field.state is FieldState.UNMAPPED
        assert field.value == "Billed- Visit 7"   # preserved, not snapped to a neighbour

    def test_blank_is_missing(self):
        assert canonicalize("", SECTORS, field_name="sector").state is FieldState.MISSING


class TestMonths:
    def test_month_name_is_ambiguous_even_when_valid(self):
        field = parse_month_name("Dec")
        assert field.value == 12
        assert field.state is FieldState.AMBIGUOUS   # no year: not a point in time
        assert not field.ok

    def test_abbreviation_and_full_name_agree(self):
        assert parse_month_name("Dec").value == parse_month_name("December").value


def test_text_strips_but_preserves():
    field = parse_text("  Pure   Service ")
    assert field.value == "Pure Service"
    assert field.state is FieldState.OK
