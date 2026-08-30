"""Turns raw monday board items into canonical `Deal` / `WorkOrder` records.

Two structural problems are handled here rather than in the parsers, because they are
properties of a *row*, not of a value:

**Header echoes.** The Deals export contains rows whose cells hold the column titles
themselves (a row where `Deal Status` literally reads "Deal Status"). Imported into
monday they become ordinary items and would otherwise be counted as deals. They are
detected, excluded from every metric, and *counted* - a silent drop would be its own
kind of dishonesty.

**Duplicate-like rows.** Several records are identical across every business field
(`Stewie Griffin` appears 5 times). We do not merge them: monday gives each a distinct
item ID, and we cannot tell a data-entry duplicate from five genuine repeat orders.
They are flagged and reported so a founder can judge, and left in the totals.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from ..monday.client import BoardSnapshot
from .entities import (
    STAGE_LADDER,
    BoardProvenance,
    Dataset,
    Deal,
    Field,
    FieldState,
    Record,
    WorkOrder,
)
from .mapping import (
    DEAL_COLUMNS,
    REQUIRED_DEAL_FIELDS,
    REQUIRED_WORK_ORDER_FIELDS,
    WORK_ORDER_COLUMNS,
    ColumnResolution,
    ColumnSpec,
    resolve_columns,
)
from .normalize import clean_text, parse_text

#: A row is treated as a header echo when at least this many of its cells repeat the
#: title of the column they sit in. Two independent cells agreeing is already far
#: beyond coincidence, and requiring two avoids misfiring on a single legitimate
#: value that happens to equal its column name.
_HEADER_ECHO_THRESHOLD = 2


def _display_value(column_value: dict[str, Any]) -> str:
    """Read one monday column value as a display string.

    Mirror and formula columns return null/"" for `text` and must be read from
    `display_value`, so we prefer it when present.
    """
    for key in ("display_value", "text"):
        raw = column_value.get(key)
        if raw not in (None, ""):
            return clean_text(str(raw))
    return ""


def _raw_by_column(item: dict[str, Any]) -> dict[str, str]:
    return {
        str(cv.get("id", "")): _display_value(cv)
        for cv in (item.get("column_values") or [])
    }


def _is_header_echo(raw: dict[str, str], resolution: ColumnResolution) -> bool:
    """True when a row's cells repeat their own column titles."""
    hits = 0
    for column_id, title in resolution.column_titles.items():
        value = raw.get(column_id, "")
        if value and value.strip().lower() == title.strip().lower():
            hits += 1
            if hits >= _HEADER_ECHO_THRESHOLD:
                return True
    return False


def _apply_fields(
    record: Record,
    raw: dict[str, str],
    resolution: ColumnResolution,
    specs: Sequence[ColumnSpec],
) -> None:
    """Parse every mapped column onto the record.

    Values land in `record.fields` unconditionally, and additionally on the matching
    dataclass attribute when one exists. The dict is authoritative - it covers every
    column, including the many Work Order columns that have no dedicated attribute -
    while the attributes give the analytics layer readable access to the common ones.
    """
    for spec in specs:
        column_id = resolution.field_to_column.get(spec.name)
        if column_id is None:
            # The board has no such column. That is a MISSING value with a *reason* -
            # distinct from a blank cell - and it must reach the dataclass attribute
            # too, or metrics reading `record.value_inr` would see a bare default and
            # lose the explanation.
            parsed = Field(
                None, None, FieldState.MISSING,
                f"The board has no column supplying {spec.name!r}.",
            )
        else:
            parsed = spec.parser(raw.get(column_id))

        record.fields[spec.name] = parsed
        if hasattr(record, spec.name):
            setattr(record, spec.name, parsed)


def _fingerprint(record: Record, spec_names: Iterable[str]) -> str:
    """Stable hash of a record's business content, ignoring monday's item ID.

    Used only to *detect* duplicate-like rows, never to merge them.
    """
    parts = [str(record.fields.get(name, Field()).raw or "") for name in spec_names]
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def _flag_duplicates(records: Sequence[Record], spec_names: Sequence[str]) -> None:
    seen: dict[str, list[Record]] = {}
    for record in records:
        seen.setdefault(_fingerprint(record, spec_names), []).append(record)

    for group in seen.values():
        if len(group) > 1:
            for record in group:
                record.issues.append(
                    f"Identical to {len(group) - 1} other record(s) across every business "
                    "field. Kept in totals - repeat orders and data-entry duplicates are "
                    "indistinguishable here."
                )


def build_deals(snapshot: BoardSnapshot) -> tuple[list[Deal], ColumnResolution]:
    """Normalize the Deals board into canonical `Deal` records."""
    resolution = resolve_columns(snapshot.columns, DEAL_COLUMNS)
    resolution.check_required(snapshot.board_name or "Deals", REQUIRED_DEAL_FIELDS)
    deals: list[Deal] = []

    for item in snapshot.items:
        raw = _raw_by_column(item)
        deal = Deal(
            item_id=str(item.get("id", "")),
            item_name=clean_text(str(item.get("name", ""))),
        )
        deal.raw = {
            resolution.column_titles.get(cid, cid): value for cid, value in raw.items()
        }
        _apply_fields(deal, raw, resolution, DEAL_COLUMNS)

        # The board's item name is the deal name; fall back to it if no column maps.
        name_field = deal.fields.get("name", Field())
        deal.name = name_field.or_none() or deal.item_name
        deal.fields["name"] = (
            name_field if name_field.ok else parse_text(deal.item_name)
        )

        if _is_header_echo(raw, resolution):
            deal.excluded = True
            deal.exclusion_reason = (
                "Row repeats the board's column headers rather than holding deal data "
                "(an artifact of the source export). Excluded from all metrics."
            )

        _rank_stage(deal)
        _flag_status_stage_conflict(deal)
        deals.append(deal)

    _flag_duplicates(
        [d for d in deals if not d.excluded], [s.name for s in DEAL_COLUMNS]
    )
    return deals, resolution


def _rank_stage(deal: Deal) -> None:
    """Place the deal's stage on the A->O ladder.

    A stage outside the known ladder is `UNMAPPED`, not rank 0 - an unrecognised stage
    must not silently sort to the front of the pipeline.
    """
    stage = deal.stage.or_none()
    if not stage:
        deal.stage_rank = Field(None, deal.stage.raw, deal.stage.state, deal.stage.note)
        deal.fields["stage_rank"] = deal.stage_rank
        return

    rank = STAGE_LADDER.get(stage)
    if rank is None:
        deal.stage_rank = Field(
            None, stage, FieldState.UNMAPPED,
            f"Stage {stage!r} is not part of the known A-O pipeline ladder.",
        )
    else:
        deal.stage_rank = Field.good(rank, stage)
    deal.fields["stage_rank"] = deal.stage_rank


def _flag_status_stage_conflict(deal: Deal) -> None:
    """Record disagreement between `Deal Status` and `Deal Stage`.

    The conflict is flagged, not resolved. Stage is what the metrics trust, but the
    founder is told the two fields disagree and on how many records.
    """
    from .entities import LOST_STAGES, WON_STAGES

    status = deal.status.or_none()
    stage = deal.stage.or_none()
    if not status or not stage:
        return

    won_by_stage = stage in WON_STAGES
    lost_by_stage = stage in LOST_STAGES

    if (status == "Won" and not won_by_stage) or (status == "Dead" and not lost_by_stage):
        deal.status_stage_conflict = True
        deal.issues.append(
            f"Deal Status says {status!r} but Deal Stage says {stage!r}. Stage is used "
            "for analysis; see data quality for why."
        )


def build_work_orders(snapshot: BoardSnapshot) -> tuple[list[WorkOrder], ColumnResolution]:
    """Normalize the Work Orders board into canonical `WorkOrder` records."""
    resolution = resolve_columns(snapshot.columns, WORK_ORDER_COLUMNS)
    resolution.check_required(
        snapshot.board_name or "Work Orders", REQUIRED_WORK_ORDER_FIELDS
    )
    work_orders: list[WorkOrder] = []

    for item in snapshot.items:
        raw = _raw_by_column(item)
        wo = WorkOrder(
            item_id=str(item.get("id", "")),
            item_name=clean_text(str(item.get("name", ""))),
        )
        wo.raw = {
            resolution.column_titles.get(cid, cid): value for cid, value in raw.items()
        }
        _apply_fields(wo, raw, resolution, WORK_ORDER_COLUMNS)

        name_field = wo.fields.get("deal_name", Field())
        if not name_field.ok and wo.item_name:
            wo.deal_name = parse_text(wo.item_name)
            wo.fields["deal_name"] = wo.deal_name

        if _is_header_echo(raw, resolution):
            wo.excluded = True
            wo.exclusion_reason = (
                "Row repeats the board's column headers rather than holding work order "
                "data. Excluded from all metrics."
            )

        work_orders.append(wo)

    _flag_duplicates(
        [w for w in work_orders if not w.excluded], [s.name for s in WORK_ORDER_COLUMNS]
    )
    return work_orders, resolution


def build_dataset(
    deals_snapshot: BoardSnapshot,
    work_orders_snapshot: BoardSnapshot,
    *,
    fetched_at: datetime | None = None,
    is_stale: bool = False,
    stale_reason: str | None = None,
) -> Dataset:
    """Assemble both boards into a single normalized `Dataset`."""
    deals, deal_resolution = build_deals(deals_snapshot)
    work_orders, wo_resolution = build_work_orders(work_orders_snapshot)

    return Dataset(
        deals=deals,
        work_orders=work_orders,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        is_stale=is_stale,
        stale_reason=stale_reason,
        provenance={
            "deals": BoardProvenance(
                board_id=deals_snapshot.board_id,
                board_name=deals_snapshot.board_name,
                item_count=len(deals_snapshot.items),
                column_count=len(deals_snapshot.columns),
            ),
            "work_orders": BoardProvenance(
                board_id=work_orders_snapshot.board_id,
                board_name=work_orders_snapshot.board_name,
                item_count=len(work_orders_snapshot.items),
                column_count=len(work_orders_snapshot.columns),
            ),
        },
        unmapped_columns={
            "deals": deal_resolution.unmapped_columns,
            "work_orders": wo_resolution.unmapped_columns,
        },
    )
