"""Connection check: `python -m skylark_bi.verify`.

Confirms the token works, resolves both board IDs, pages through every item, and
reports what came back. Reads only - it cannot modify anything, and says so.

Kept as a module rather than a test because it is the first thing to run after
configuring monday.com, and the first thing to run when something breaks in
production.
"""

from __future__ import annotations

import sys

from .config import ConfigError, load_settings
from .ingest.builder import build_dataset
from .monday.client import MondayClient
from .monday.errors import MondayError
from .quality.audit import audit


def main() -> int:
    try:
        settings = load_settings(require_llm=False)
    except ConfigError as exc:
        print(f"Configuration problem:\n  {exc}", file=sys.stderr)
        return 2

    print(f"monday.com API version: {settings.monday.api_version}")

    try:
        with MondayClient(settings.monday) as client:
            identity = client.verify_token()
            account = identity.get("account") or {}
            print(
                f"Authenticated as {identity.get('name', '?')} "
                f"(account: {account.get('name', '?')})"
            )

            deals_id, work_orders_id = client.resolve_board_ids()
            print(f"Deals board id:       {deals_id}")
            print(f"Work Orders board id: {work_orders_id}")

            deals = client.get_deals(deals_id)
            work_orders = client.get_work_orders(work_orders_id)
    except MondayError as exc:
        print(f"\nmonday.com error: {exc}\n  {exc.user_message}", file=sys.stderr)
        return 1

    print(f"\n{deals.board_name:<14} {len(deals):>4} items, {len(deals.columns)} columns")
    print(
        f"{work_orders.board_name:<14} {len(work_orders):>4} items, "
        f"{len(work_orders.columns)} columns"
    )

    try:
        dataset = build_dataset(deals, work_orders)
    except MondayError as exc:
        print(f"\nSchema problem:\n{exc.user_message}", file=sys.stderr)
        return 1

    report = audit(dataset)

    print(f"\nUsable deals:       {len(dataset.active_deals)} of {len(dataset.deals)}")
    print(f"Usable work orders: {len(dataset.active_work_orders)} of {len(dataset.work_orders)}")

    # Explain any gap between imported and usable, so the difference never has to be
    # guessed at. Excluded rows are a finding, not an error.
    excluded = [r for r in (*dataset.deals, *dataset.work_orders) if r.excluded]
    if excluded:
        reasons: dict[str, int] = {}
        for record in excluded:
            reasons[record.exclusion_reason or "unspecified"] = (
                reasons.get(record.exclusion_reason or "unspecified", 0) + 1
            )
        print("\nExcluded from analysis:")
        for reason, count in reasons.items():
            print(f"  {count:>3}  {reason}")

    print(f"\nData as of:         {dataset.as_of}")
    print(f"Quality findings:   {len(report.findings)}")

    for board, columns in dataset.unmapped_columns.items():
        if columns:
            print(f"\nUnmapped {board} columns (not analysed): {', '.join(columns)}")

    expected = {"Deals": 346, "Work Orders": 176}
    for name, count in (
        (deals.board_name, len(deals)),
        (work_orders.board_name, len(work_orders)),
    ):
        target = expected.get(name)
        if target and count < target:
            print(
                f"\nWarning: {name} has {count} items but the source CSV has {target}. "
                "A Free-plan account caps at 200 items in total - see setup/MONDAY_SETUP.md.",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
