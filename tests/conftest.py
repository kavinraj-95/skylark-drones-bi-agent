"""Test fixtures.

The fixtures here synthesise monday.com API responses from the original CSVs, in
exactly the shape `MondayClient` returns. That lets the full pipeline - normalization,
quality audit, analytics - be tested against the *real* messy data deterministically
and offline.

To be explicit about what this is and is not: these fixtures exist only for tests.
Nothing under `src/` reads them, or reads the CSVs, and `test_no_hardcoded_data.py`
enforces that. In production the identical code path is fed by live monday.com
responses.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_DIR = REPO_ROOT / "setup" / "import"


def _column_id(title: str, index: int) -> str:
    """Mimic monday's opaque column IDs.

    Deliberately unlike the human title, so that any code accidentally depending on
    column IDs looking like titles fails in tests.
    """
    return f"col_{index:02d}"


def _as_monday_board(
    csv_path: Path, board_id: str, board_name: str
) -> dict[str, Any]:
    """Read a CSV and emit it as a monday board payload (columns + items)."""
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    header = rows[0]
    data_rows = [r for r in rows[1:] if any(cell.strip() for cell in r)]

    columns = [
        {"id": _column_id(title, i), "title": title, "type": "text", "settings_str": "{}"}
        for i, title in enumerate(header)
    ]

    items: list[dict[str, Any]] = []
    for row_index, row in enumerate(data_rows, start=1):
        cells = list(row) + [""] * (len(header) - len(row))
        items.append(
            {
                "id": f"{board_id}{row_index:04d}",
                # monday uses the first column as the item name.
                "name": cells[0],
                "column_values": [
                    {
                        "id": columns[i]["id"],
                        "type": "text",
                        "text": cells[i],
                        "value": None,
                    }
                    for i in range(len(header))
                ],
            }
        )

    return {"id": board_id, "name": board_name, "columns": columns, "items": items}


def _snapshot(payload: dict[str, Any]):
    from skylark_bi.monday.client import BoardSnapshot

    return BoardSnapshot(
        board_id=payload["id"],
        board_name=payload["name"],
        items=payload["items"],
        columns=payload["columns"],
    )


@pytest.fixture(scope="session")
def deals_snapshot():
    return _snapshot(_as_monday_board(IMPORT_DIR / "monday_deals.csv", "9001", "Deals"))


@pytest.fixture(scope="session")
def work_orders_snapshot():
    return _snapshot(
        _as_monday_board(IMPORT_DIR / "monday_work_orders.csv", "9002", "Work Orders")
    )


@pytest.fixture(scope="session")
def dataset(deals_snapshot, work_orders_snapshot):
    """The full normalized dataset built from the real source data."""
    from skylark_bi.ingest.builder import build_dataset

    return build_dataset(deals_snapshot, work_orders_snapshot)


@pytest.fixture(scope="session")
def quality_report(dataset):
    """The data-quality audit for the full fixture dataset."""
    from skylark_bi.quality.audit import audit

    return audit(dataset)
