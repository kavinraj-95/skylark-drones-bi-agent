"""Maps monday board columns onto canonical fields.

This is the seam that keeps business logic independent of monday's structure. Boards
are matched by **column title**, resolved at runtime, so nothing here depends on
monday's opaque column IDs - and a board rebuilt from the same CSV still works even
though every ID changed.

It also means we do not depend on monday's *column types*. That matters concretely:
monday's CSV importer only creates a Status column when a column has 9 or fewer
distinct values, so `Deal Stage` (17 values), `Sector/service` (13) and `Type of Work`
(36) all land as free text. Reading display strings and canonicalising them here is
therefore the only approach that survives however the boards were set up.

Matching is tolerant of case, whitespace and punctuation drift, because a human
renaming "Deal Name" to "Deal name" should not break the agent. It is *not* fuzzy
beyond that: an unrecognised column is reported as unmapped rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Callable

from .entities import Field
from .normalize import (
    build_alias_map,
    canonicalize,
    parse_date,
    parse_month_name,
    parse_number,
    parse_quantity,
    parse_text,
)

# --------------------------------------------------------------------------------
# Controlled vocabularies.
#
# Each entry is {canonical spelling: [variants seen or plausible]}. Variants exist to
# absorb real inconsistencies in this data ("BIlled", "Dec" vs "December") without
# merging genuinely different categories. Values outside a vocabulary are surfaced as
# UNMAPPED, never coerced.
# --------------------------------------------------------------------------------

SECTORS = build_alias_map({
    "Mining": ["mining", "mines"],
    "Renewables": ["renewable", "renewables", "solar", "renewable energy"],
    "Railways": ["railway", "railways", "rail"],
    "Powerline": ["power line", "powerlines", "power-line", "transmission"],
    "Construction": ["construction", "infra", "infrastructure"],
    "Manufacturing": ["manufacturing"],
    "Aviation": ["aviation"],
    "Security and Surveillance": ["security & surveillance", "security and surveillance"],
    "DSP": ["dsp"],
    "Tender": ["tender", "tenders"],
    "Others": ["other", "others", "misc", "miscellaneous"],
})

DEAL_STATUSES = build_alias_map({
    "Won": ["won", "win", "closed won", "closed-won"],
    "Dead": ["dead", "lost", "closed lost", "closed-lost"],
    "Open": ["open", "active", "in progress"],
    "On Hold": ["on hold", "onhold", "hold", "paused"],
})

CLOSURE_PROBABILITIES = build_alias_map({
    "High": ["high", "h"],
    "Medium": ["medium", "med", "m"],
    "Low": ["low", "l"],
})

EXECUTION_STATUSES = build_alias_map({
    "Completed": ["completed", "complete", "done"],
    "Ongoing": ["ongoing", "in progress", "in-progress"],
    "Executed until current month": ["executed until current month"],
    "Not Started": ["not started", "notstarted", "yet to start"],
    "Partial Completed": ["partial completed", "partially completed", "partial"],
    "Pause / struck": ["pause / struck", "pause/struck", "paused", "struck"],
    "Details pending from Client": ["details pending from client", "details pending"],
})

INVOICE_STATUSES = build_alias_map({
    # "BIlled" is a real casing typo in the source; "Billed- Visit 3/7" are ad-hoc
    # per-visit annotations that we normalise to Partially Billed only where the text
    # says so. Anything unrecognised stays UNMAPPED and is reported.
    "Fully Billed": ["fully billed", "billed", "billed fully"],
    "Partially Billed": ["partially billed", "partial billed", "partly billed"],
    "Not billed yet": ["not billed yet", "not billed", "unbilled"],
    "Not Billable": ["not billable", "non billable"],
    "Stuck": ["stuck"],
    "Update Required": ["update required", "update reqd"],
})

WO_STATUSES = build_alias_map({
    "Open": ["open"],
    "Closed": ["closed", "close"],
})

NATURE_OF_WORK = build_alias_map({
    "One time Project": ["one time project", "one-time project", "onetime project"],
    "Proof of Concept": ["proof of concept", "poc"],
    "Annual Rate Contract": ["annual rate contract", "arc"],
    "Monthly Contract": ["monthly contract"],
})

DOCUMENT_TYPES = build_alias_map({
    "Purchase Order": ["purchase order", "po"],
    "LOA/LOI": ["loa/loi", "loa", "loi"],
    "Email Confirmation": ["email confirmation", "email"],
})


def _norm_title(title: str) -> str:
    """Normalise a column title for tolerant matching.

    Lowercases, strips punctuation and collapses whitespace, so "Deal Name",
    "deal name" and "Deal-Name" all agree - but "Deal Value" still does not match
    "Deal Name".
    """
    cleaned = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass(frozen=True)
class ColumnSpec:
    """One canonical field and the source column titles that can supply it."""

    #: Canonical field name, matching the attribute on Deal / WorkOrder.
    name: str
    #: Accepted source column titles, most preferred first.
    titles: tuple[str, ...]
    #: Parser turning the raw display string into a `Field`.
    parser: Callable[[str | None], Field]
    #: Human-readable description, surfaced in metric provenance.
    description: str = ""

    @property
    def normalised_titles(self) -> tuple[str, ...]:
        return tuple(_norm_title(t) for t in self.titles)


def _categorical(aliases: dict[str, str], field_name: str) -> Callable[[str | None], Field]:
    """Build a parser that canonicalises against a controlled vocabulary."""

    def parse(raw: str | None) -> Field:
        return canonicalize(raw, aliases, field_name=field_name)

    return parse


DEAL_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("name", ("Deal Name", "Deal name masked", "Name", "Item"), parse_text,
               "Masked deal name. Not unique - repeats across records."),
    ColumnSpec("owner_code", ("Owner code", "Owner", "BD/KAM Personnel code"), parse_text,
               "Masked code for the deal owner."),
    ColumnSpec("client_code", ("Client Code", "Customer Code"), parse_text,
               "Masked client identifier. Namespaced separately from Work Order codes."),
    ColumnSpec("status", ("Deal Status", "Status"), _categorical(DEAL_STATUSES, "deal status"),
               "Coarse won/dead/open flag. Known to be unreliable - see data quality."),
    ColumnSpec("stage", ("Deal Stage", "Stage"), parse_text,
               "Lettered pipeline stage (A-O). The authoritative pipeline signal."),
    ColumnSpec("closure_probability", ("Closure Probability", "Probability"),
               _categorical(CLOSURE_PROBABILITIES, "closure probability"),
               "Subjective High/Medium/Low likelihood. Appears to be set retrospectively."),
    ColumnSpec("value_inr", ("Masked Deal value", "Deal Value", "Value"), parse_number,
               "Masked deal value in rupees."),
    ColumnSpec("sector", ("Sector/service", "Sector"), _categorical(SECTORS, "sector"),
               "Business sector."),
    ColumnSpec("product", ("Product deal", "Product"), parse_text,
               "Product/service composition of the deal."),
    ColumnSpec("created_date", ("Created Date", "Created"), parse_date,
               "Date the deal record was created."),
    ColumnSpec("tentative_close_date", ("Tentative Close Date", "Expected Close Date"),
               parse_date, "Forecast close date."),
    ColumnSpec("actual_close_date", ("Close Date (A)", "Actual Close Date"), parse_date,
               "Actual close date. Populated on only a small minority of records."),
)

WORK_ORDER_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("deal_name", ("Deal name masked", "Deal Name", "Name", "Item"), parse_text,
               "Masked deal name. The only link to the Deals board, and not unique."),
    ColumnSpec("serial", ("Serial #", "Serial", "Serial No"), parse_text,
               "SDPLDEAL-NNN work order key. Unique across the board."),
    ColumnSpec("customer_code", ("Customer Name Code", "Customer Code"), parse_text,
               "Masked customer code. A separate namespace from the Deals board."),
    ColumnSpec("owner_code", ("BD/KAM Personnel code", "Owner code"), parse_text,
               "Masked code for the responsible BD/KAM person."),
    ColumnSpec("sector", ("Sector", "Sector/service"), _categorical(SECTORS, "sector"),
               "Business sector."),
    ColumnSpec("nature_of_work", ("Nature of Work",),
               _categorical(NATURE_OF_WORK, "nature of work"),
               "Engagement shape: one-time, POC, contract."),
    ColumnSpec("type_of_work", ("Type of Work",), parse_text,
               "Survey/service type. Free text with many multi-value combinations."),
    ColumnSpec("execution_status", ("Execution Status",),
               _categorical(EXECUTION_STATUSES, "execution status"),
               "Operational delivery state."),
    ColumnSpec("invoice_status", ("Invoice Status",),
               _categorical(INVOICE_STATUSES, "invoice status"), "Billing progress."),
    ColumnSpec("billing_status", ("Billing Status",),
               _categorical(INVOICE_STATUSES, "billing status"),
               "Second, overlapping billing field. Retained separately, not merged."),
    ColumnSpec("wo_status", ("WO Status (billed)", "WO Status"),
               _categorical(WO_STATUSES, "work order status"), "Open/closed work order flag."),
    ColumnSpec("document_type", ("Document Type",),
               _categorical(DOCUMENT_TYPES, "document type"), "Contracting instrument."),
    ColumnSpec("amount_excl_gst", ("Amount in Rupees (Excl of GST) (Masked)",), parse_number,
               "Order value excluding GST."),
    ColumnSpec("amount_incl_gst", ("Amount in Rupees (Incl of GST) (Masked)",), parse_number,
               "Order value including GST."),
    ColumnSpec("billed_excl_gst", ("Billed Value in Rupees (Excl of GST.) (Masked)",),
               parse_number, "Value invoiced to date, excluding GST."),
    ColumnSpec("billed_incl_gst", ("Billed Value in Rupees (Incl of GST.) (Masked)",),
               parse_number, "Value invoiced to date, including GST."),
    ColumnSpec("collected_incl_gst", ("Collected Amount in Rupees (Incl of GST.) (Masked)",),
               parse_number, "Cash collected, including GST."),
    ColumnSpec("to_be_billed_excl_gst", ("Amount to be billed in Rs. (Exl. of GST) (Masked)",),
               parse_number, "Backlog still to invoice, excluding GST."),
    ColumnSpec("to_be_billed_incl_gst", ("Amount to be billed in Rs. (Incl. of GST) (Masked)",),
               parse_number, "Backlog still to invoice, including GST."),
    ColumnSpec("receivable", ("Amount Receivable (Masked)",), parse_number,
               "Outstanding receivable."),
    ColumnSpec("quantity_po", ("Quantities as per PO",), parse_quantity,
               "Contracted quantity. Mixed units and free text in the source."),
    ColumnSpec("quantity_ops", ("Quantity by Ops",), parse_quantity,
               "Quantity recorded by operations."),
    ColumnSpec("quantity_billed", ("Quantity billed (till date)",), parse_quantity,
               "Quantity invoiced to date."),
    ColumnSpec("quantity_balance", ("Balance in quantity",), parse_quantity,
               "Quantity remaining."),
    ColumnSpec("po_date", ("Date of PO/LOI",), parse_date, "Date of the purchase order or LOI."),
    ColumnSpec("start_date", ("Probable Start Date",), parse_date, "Planned start."),
    ColumnSpec("end_date", ("Probable End Date",), parse_date, "Planned end."),
    ColumnSpec("data_delivery_date", ("Data Delivery Date",), parse_date, "Data handover date."),
    ColumnSpec("last_invoice_date", ("Last invoice date",), parse_date, "Most recent invoice date."),
    ColumnSpec("latest_invoice_no", ("latest invoice no.", "Latest Invoice No"), parse_text,
               "Most recent invoice number (SDPL/FY..)."),
    ColumnSpec("ar_priority", ("AR Priority account",), parse_text,
               "Marks accounts prioritised for receivables chasing."),
    ColumnSpec("software_platform", (
        "Is any Skylark software platform part of the client deliverables in this deal?",
        "Software Platform",
    ), parse_text, "Whether a Skylark platform (Spectra/DMO) is in scope."),
    ColumnSpec("last_executed_month", ("Last executed month of recurring project",),
               parse_month_name, "Month name only - carries no year."),
    ColumnSpec("expected_billing_month", ("Expected Billing Month",), parse_month_name,
               "Month name only. Empty across every record in the source."),
    ColumnSpec("actual_billing_month", ("Actual Billing Month",), parse_month_name,
               "Month name only - carries no year."),
    ColumnSpec("actual_collection_month", ("Actual Collection Month",), parse_month_name,
               "Month name only. Empty across every record in the source."),
    ColumnSpec("collection_status", ("Collection status",), parse_text,
               "Empty across every record in the source."),
    ColumnSpec("collection_date", ("Collection Date",), parse_date,
               "Empty across every record in the source."),
)


#: Canonical fields a board must supply for analysis to mean anything. Chosen as the
#: minimum set without which every metric would be empty - not a wish list.
REQUIRED_DEAL_FIELDS = ("name", "stage", "sector", "value_inr")
REQUIRED_WORK_ORDER_FIELDS = ("serial", "sector", "execution_status", "amount_excl_gst")


@dataclass
class ColumnResolution:
    """The outcome of matching a board's real columns against our specs."""

    #: canonical field name -> monday column ID
    field_to_column: dict[str, str]
    #: monday column ID -> its title, for raw-value display
    column_titles: dict[str, str]
    #: Canonical fields we expected but the board does not have.
    missing_fields: list[str]
    #: Board columns we could not map. Reported, not ignored - an unexpected column
    #: may be a renamed one, and silently dropping it would hide real data.
    unmapped_columns: list[str]
    #: Titles that appear on more than one column. monday does not enforce unique
    #: titles, so we bind the first and flag the rest rather than picking at random.
    duplicate_titles: list[str]
    #: Every column title the board actually has, for diagnostics.
    board_titles: list[str] = dc_field(default_factory=list)

    def check_required(self, board_label: str, required: tuple[str, ...]) -> None:
        """Raise `SchemaMismatchError` when too few required fields resolved."""
        from ..monday.errors import SchemaMismatchError

        missing = [f for f in required if f not in self.field_to_column]
        # Tolerate one missing field - a board can legitimately lack a column. Losing
        # most of them means the import is wrong, not the data.
        if len(missing) > 1:
            raise SchemaMismatchError(
                board_label,
                found_titles=self.board_titles,
                matched=len(required) - len(missing),
                required=len(required),
                missing=missing,
            )


def resolve_columns(
    columns: list[dict[str, str]], specs: tuple[ColumnSpec, ...]
) -> ColumnResolution:
    """Match a board's live columns against the canonical specs.

    `columns` is monday's own column list (id, title, type). Resolution is by title
    and happens on every fetch, so renaming a board column is a recoverable,
    *reported* condition rather than a crash.
    """
    by_title: dict[str, list[tuple[str, str]]] = {}
    for column in columns:
        column_id = str(column.get("id", ""))
        title = str(column.get("title", ""))
        if not column_id:
            continue
        by_title.setdefault(_norm_title(title), []).append((column_id, title))

    duplicates = [
        titles[0][1] for titles in by_title.values() if len(titles) > 1
    ]

    field_to_column: dict[str, str] = {}
    column_titles: dict[str, str] = {}
    missing_fields: list[str] = []
    claimed: set[str] = set()

    for spec in specs:
        for candidate in spec.normalised_titles:
            matches = by_title.get(candidate)
            if matches:
                column_id, title = matches[0]
                field_to_column[spec.name] = column_id
                column_titles[column_id] = title
                claimed.add(column_id)
                break
        else:
            missing_fields.append(spec.name)

    # monday synthesises a "name" column that never appears in `columns`.
    unmapped = [
        str(c.get("title", ""))
        for c in columns
        if str(c.get("id", "")) not in claimed and str(c.get("id", "")) != "name"
    ]

    return ColumnResolution(
        field_to_column=field_to_column,
        column_titles=column_titles,
        missing_fields=missing_fields,
        unmapped_columns=sorted(t for t in unmapped if t),
        duplicate_titles=sorted(set(duplicates)),
        board_titles=[str(c.get("title", "")) for c in columns],
    )
