"""Read-only monday.com GraphQL client.

Design notes that matter more than the code:

**Read-only is structural.** The public surface is four read methods. There is no
public `execute(query)`, so neither the LLM nor any calling layer has a path to
construct a GraphQL document. Every query is a module-level constant using GraphQL
*variables* - board IDs are never interpolated into query text. `_assert_read_only`
is a tripwire behind that, checked on every send.

Note that monday personal API tokens cannot be scoped per-application; they inherit
the owning user's permissions. So this is application-level enforcement and
defence-in-depth, not a server-side security boundary. See README.

**HTTP 200 does not mean success.** monday returns 200 with an `errors` array for
application-level failures, including GraphQL parse errors, often alongside a
partially-populated `data`. Treating 200 as success is the easiest way to build a
client that is silently wrong, so `_post` checks both.

**Pagination.** First page uses `items_page(limit:)`; subsequent pages use the
top-level `next_items_page(cursor:)` rather than re-nesting under `boards`, which
keeps complexity cost down. `cursor` is null when exhausted.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import MONDAY_API_URL, MondayConfig
from .errors import (
    BoardNotFoundError,
    MondayAuthError,
    MondayError,
    MondayQueryError,
    MondayRateLimitError,
    MondayUnavailableError,
    ReadOnlyViolationError,
)

log = logging.getLogger(__name__)

#: monday caps `items_page` at 500 items per request.
PAGE_LIMIT = 500
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_ATTEMPTS = 4

# --------------------------------------------------------------------------------
# Query documents. Module-level constants, parameterised only via GraphQL variables.
# --------------------------------------------------------------------------------

_COMPLEXITY = "complexity { before query after reset_in_x_seconds }"

Q_ME = """
query Me {
  me { id name email account { id name slug } }
}
"""

Q_BOARDS = """
query ListBoards($limit: Int!, $page: Int!) {
  boards(limit: $limit, page: $page, state: active, order_by: used_at) {
    id
    name
    board_kind
    state
    items_count
    workspace { id name }
  }
}
"""

Q_COLUMNS = """
query BoardColumns($boardId: ID!) {
  boards(ids: [$boardId]) {
    id
    name
    items_count
    columns { id title type settings_str }
  }
}
"""

# `display_value` is required for mirror/formula columns, whose `text` is null or "".
_ITEM_FIELDS = """
    id
    name
    column_values {
      id
      type
      text
      value
      ... on MirrorValue { display_value }
      ... on FormulaValue { display_value }
      ... on BoardRelationValue { display_value }
      ... on DependencyValue { display_value }
    }
"""

Q_ITEMS_FIRST = f"""
query BoardItems($boardId: ID!, $limit: Int!) {{
  {_COMPLEXITY}
  boards(ids: [$boardId]) {{
    id
    name
    items_page(limit: $limit) {{
      cursor
      items {{ {_ITEM_FIELDS} }}
    }}
  }}
}}
"""

Q_ITEMS_NEXT = f"""
query BoardItemsNext($cursor: String!, $limit: Int!) {{
  {_COMPLEXITY}
  next_items_page(cursor: $cursor, limit: $limit) {{
    cursor
    items {{ {_ITEM_FIELDS} }}
  }}
}}
"""

#: Any GraphQL operation keyword that is not a read.
_FORBIDDEN_OPERATION = re.compile(r"(?<![A-Za-z0-9_])(mutation|subscription)(?![A-Za-z0-9_])")
_COMMENTS = re.compile(r"#[^\n]*")


def assert_read_only(document: str) -> None:
    """Raise `ReadOnlyViolationError` unless `document` is a pure read.

    Strips comments first so a commented-out mutation does not trip the wire, then
    requires the document to declare `query` and to contain no mutation or
    subscription operation. Anonymous (`{ ... }`) documents are rejected outright -
    every query in this module is named, so an unnamed one means something unexpected
    built it.
    """
    stripped = _COMMENTS.sub("", document).strip()
    if _FORBIDDEN_OPERATION.search(stripped):
        raise ReadOnlyViolationError(
            "Refusing to send a non-read GraphQL operation to monday.com."
        )
    if not stripped.lstrip().startswith("query"):
        raise ReadOnlyViolationError(
            "Refusing to send a GraphQL document that does not declare a named query."
        )


@dataclass
class BoardSnapshot:
    """Raw items for one board, exactly as monday returned them.

    Deliberately dumb: no normalization happens here. Keeping the raw payload intact
    means the normalization layer can always show what a value looked like before we
    touched it.
    """

    board_id: str
    board_name: str
    items: list[dict[str, Any]] = field(default_factory=list)
    columns: list[dict[str, Any]] = field(default_factory=list)
    api_calls: int = 0

    def __len__(self) -> int:
        return len(self.items)


class MondayClient:
    """Read-only access to monday.com boards.

    Public surface is intentionally four methods. Adding a method that accepts a
    caller-supplied query string would defeat the structural read-only guarantee.
    """

    def __init__(self, config: MondayConfig, *, transport: httpx.BaseTransport | None = None):
        self._config = config
        self._client = httpx.Client(
            timeout=REQUEST_TIMEOUT_SECONDS,
            transport=transport,
            headers={
                # monday expects a bare token - no "Bearer " prefix.
                "Authorization": config.api_token,
                "Content-Type": "application/json",
                "API-Version": config.api_version,
            },
        )
        self.api_calls = 0
        self.last_complexity: dict[str, Any] | None = None

    # -- lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MondayClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public read API ---------------------------------------------------------

    def verify_token(self) -> dict[str, Any]:
        """Confirm the token works and return the identity behind it."""
        return self._post(Q_ME, {})["me"]

    def get_boards(self) -> list[dict[str, Any]]:
        """List every active board the token can see (offset-paginated)."""
        boards: list[dict[str, Any]] = []
        page = 1
        while True:
            batch = self._post(Q_BOARDS, {"limit": 100, "page": page}).get("boards") or []
            boards.extend(batch)
            if len(batch) < 100:
                return boards
            page += 1

    def get_columns(self, board_id: str) -> list[dict[str, Any]]:
        """Return a board's column definitions (id, title, type, settings_str)."""
        return self._board_meta(board_id)["columns"]

    def get_deals(self, board_id: str) -> BoardSnapshot:
        """Fetch every item from the Deals board."""
        return self._fetch_board(board_id)

    def get_work_orders(self, board_id: str) -> BoardSnapshot:
        """Fetch every item from the Work Orders board."""
        return self._fetch_board(board_id)

    def resolve_board_ids(self) -> tuple[str, str]:
        """Return (deals_board_id, work_orders_board_id).

        Uses configured IDs when present, otherwise matches board names
        case-insensitively. Raises `BoardNotFoundError` naming the boards it did see,
        so a misconfiguration is diagnosable from the error alone.
        """
        cfg = self._config
        if not cfg.needs_board_discovery:
            return cfg.deals_board_id, cfg.work_orders_board_id

        boards = self.get_boards()
        by_name = {str(b.get("name", "")).strip().lower(): str(b["id"]) for b in boards}

        deals = cfg.deals_board_id or by_name.get(cfg.deals_board_name.strip().lower(), "")
        work_orders = cfg.work_orders_board_id or by_name.get(
            cfg.work_orders_board_name.strip().lower(), ""
        )

        if not deals or not work_orders:
            seen = ", ".join(sorted(str(b.get("name", "?")) for b in boards)) or "(none)"
            missing = []
            if not deals:
                missing.append(f"deals board named {cfg.deals_board_name!r}")
            if not work_orders:
                missing.append(f"work orders board named {cfg.work_orders_board_name!r}")
            raise BoardNotFoundError(
                f"Could not find {' and '.join(missing)}. Boards visible to this token: {seen}."
            )
        return deals, work_orders

    # -- internals ---------------------------------------------------------------

    def _board_meta(self, board_id: str) -> dict[str, Any]:
        payload = self._post(Q_COLUMNS, {"boardId": str(board_id)})
        boards = payload.get("boards") or []
        if not boards:
            raise BoardNotFoundError(
                f"monday.com returned no board for id {board_id!r}. Check the ID and "
                "that the token's user has access to it."
            )
        return boards[0]

    def _fetch_board(self, board_id: str) -> BoardSnapshot:
        """Page through a board until the cursor is exhausted."""
        meta = self._board_meta(board_id)
        snapshot = BoardSnapshot(
            board_id=str(meta["id"]),
            board_name=str(meta.get("name") or ""),
            columns=meta.get("columns") or [],
        )

        payload = self._post(Q_ITEMS_FIRST, {"boardId": str(board_id), "limit": PAGE_LIMIT})
        boards = payload.get("boards") or []
        if not boards:
            raise BoardNotFoundError(f"monday.com returned no board for id {board_id!r}.")

        page = boards[0].get("items_page") or {}
        snapshot.items.extend(page.get("items") or [])
        cursor = page.get("cursor")

        # Bounded to keep a malformed cursor from looping forever.
        max_pages = 200
        while cursor and max_pages > 0:
            payload = self._post(Q_ITEMS_NEXT, {"cursor": cursor, "limit": PAGE_LIMIT})
            page = payload.get("next_items_page") or {}
            batch = page.get("items") or []
            snapshot.items.extend(batch)
            cursor = page.get("cursor")
            max_pages -= 1

        snapshot.api_calls = self.api_calls
        log.info("Fetched %d items from board %s", len(snapshot.items), board_id)
        return snapshot

    def _post(self, document: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Send one GraphQL read, with retries. Returns the `data` object."""
        assert_read_only(document)

        last_error: MondayError | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.post(
                    MONDAY_API_URL, json={"query": document, "variables": variables}
                )
            except httpx.TimeoutException as exc:
                last_error = MondayUnavailableError(f"monday.com timed out: {exc}")
            except httpx.HTTPError as exc:
                last_error = MondayUnavailableError(f"Could not reach monday.com: {exc}")
            else:
                try:
                    return self._parse(response)
                except MondayRateLimitError as exc:
                    if exc.daily_limit:
                        raise  # Resets at midnight UTC; retrying today is pointless.
                    last_error = exc
                    if exc.retry_after_seconds:
                        time.sleep(min(exc.retry_after_seconds, 30))
                        continue
                except (MondayAuthError, MondayQueryError):
                    raise  # Deterministic - a retry changes nothing.

            self.api_calls += 1
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2 ** (attempt - 1), 8) + random.uniform(0, 0.4))

        raise last_error or MondayUnavailableError("monday.com request failed.")

    def _parse(self, response: httpx.Response) -> dict[str, Any]:
        """Turn one HTTP response into `data`, or raise a typed error.

        Success requires HTTP 200 *and* the absence of an `errors` array - monday
        returns 200 for application-level errors.
        """
        self.api_calls += 1
        status = response.status_code

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            raise MondayQueryError(
                f"monday.com returned a non-JSON response (HTTP {status})."
            ) from None

        if status in (401, 403):
            raise MondayAuthError(f"monday.com rejected the token (HTTP {status}).")
        if status == 429:
            raise self._rate_limit_error(response, body)

        errors = body.get("errors") or body.get("error_message")
        if errors:
            message = self._error_text(errors)
            lowered = message.lower()
            if "unauthor" in lowered or "not authenticated" in lowered:
                raise MondayAuthError(f"monday.com rejected the token: {message}")
            if "complexity" in lowered or "rate limit" in lowered or "daily limit" in lowered:
                raise self._rate_limit_error(response, body)
            raise MondayQueryError(f"monday.com query error: {message}")

        if status >= 500:
            raise MondayUnavailableError(f"monday.com server error (HTTP {status}).")
        if status != 200:
            raise MondayQueryError(f"Unexpected response from monday.com (HTTP {status}).")

        data = body.get("data")
        if data is None:
            raise MondayQueryError("monday.com returned no data.")

        if isinstance(data, dict) and isinstance(data.get("complexity"), dict):
            self.last_complexity = data["complexity"]
        return data

    @staticmethod
    def _error_text(errors: Any) -> str:
        if isinstance(errors, str):
            return errors
        if isinstance(errors, list):
            parts = [
                str(e.get("message", e)) if isinstance(e, dict) else str(e) for e in errors
            ]
            return "; ".join(p for p in parts if p)
        return str(errors)

    def _rate_limit_error(self, response: httpx.Response, body: dict[str, Any]) -> MondayRateLimitError:
        text = self._error_text(body.get("errors") or "").lower()
        retry_after = None
        for source in (response.headers.get("Retry-After"), body.get("retry_in_seconds")):
            try:
                retry_after = int(str(source))
                break
            except (TypeError, ValueError):
                continue

        if "daily" in text:
            return MondayRateLimitError(
                "monday.com daily API call limit exhausted.",
                daily_limit=True,
                user_message=(
                    "monday.com's daily API limit for this account has been reached. "
                    "It resets at midnight UTC."
                ),
            )
        return MondayRateLimitError(
            "monday.com rate limit or complexity budget exhausted.",
            retry_after_seconds=retry_after,
        )
