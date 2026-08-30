"""monday.com client behaviour, driven by mocked HTTP responses.

The cases here are the ones that produce *silently wrong* clients if handled badly -
above all monday's habit of returning HTTP 200 with an `errors` array.
"""

from __future__ import annotations

import httpx
import pytest

from skylark_bi.config import MondayConfig
from skylark_bi.monday.client import MondayClient
from skylark_bi.monday.errors import (
    BoardNotFoundError,
    MondayAuthError,
    MondayQueryError,
    MondayRateLimitError,
    MondayUnavailableError,
)

CONFIG = MondayConfig(api_token="test-token", api_version="2026-07")


def client_with(handler) -> MondayClient:
    return MondayClient(CONFIG, transport=httpx.MockTransport(handler))


def json_response(payload: dict, status: int = 200, headers: dict | None = None):
    return httpx.Response(status, json=payload, headers=headers or {})


class TestAuthHeaders:
    def test_token_is_sent_bare_without_bearer_prefix(self):
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return json_response({"data": {"me": {"id": "1", "name": "Test"}}})

        with client_with(handler) as client:
            client.verify_token()

        assert seen["authorization"] == "test-token"
        assert not seen["authorization"].lower().startswith("bearer")
        assert seen["api-version"] == "2026-07"


class TestErrorHandling:
    def test_http_200_with_errors_array_is_a_failure(self):
        """monday returns 200 for application errors. Treating that as success is the
        single easiest way to build a client that quietly returns nothing."""

        def handler(request):
            return json_response({
                "errors": [{"message": "Parse error on \"boards\" (line 1)"}],
                "data": None,
            })

        with client_with(handler) as client, pytest.raises(MondayQueryError):
            client.verify_token()

    def test_200_with_errors_and_partial_data_still_fails(self):
        def handler(request):
            return json_response({
                "errors": [{"message": "Field 'nope' doesn't exist"}],
                "data": {"boards": [{"id": "1"}]},
            })

        with client_with(handler) as client, pytest.raises(MondayQueryError):
            client.get_boards()

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failures_are_typed(self, status):
        def handler(request):
            return json_response({"errors": [{"message": "Not Authenticated"}]}, status)

        with client_with(handler) as client, pytest.raises(MondayAuthError):
            client.verify_token()

    def test_auth_failure_signalled_only_in_the_body(self):
        def handler(request):
            return json_response({"errors": [{"message": "Unauthorized access"}]})

        with client_with(handler) as client, pytest.raises(MondayAuthError):
            client.verify_token()

    def test_daily_limit_is_not_retried(self):
        """It resets at midnight UTC - retrying inside the same day only burns budget."""
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return json_response(
                {"errors": [{"message": "DAILY_LIMIT_EXCEEDED"}]}, 429
            )

        with client_with(handler) as client, pytest.raises(MondayRateLimitError) as exc:
            client.verify_token()

        assert exc.value.daily_limit
        assert calls["n"] == 1

    def test_network_failure_is_typed(self):
        def handler(request):
            raise httpx.ConnectError("connection refused")

        with client_with(handler) as client, pytest.raises(MondayUnavailableError):
            client.verify_token()

    def test_non_json_response_is_handled(self):
        def handler(request):
            return httpx.Response(200, text="<html>maintenance</html>")

        with client_with(handler) as client, pytest.raises(MondayQueryError):
            client.verify_token()

    def test_errors_never_contain_the_token(self):
        def handler(request):
            return json_response({"errors": [{"message": "Not Authenticated"}]}, 401)

        with client_with(handler) as client:
            with pytest.raises(MondayAuthError) as exc:
                client.verify_token()
        assert "test-token" not in str(exc.value)
        assert "test-token" not in exc.value.user_message


class TestPagination:
    def test_pages_until_the_cursor_is_exhausted(self):
        """Two pages, then a null cursor. All items must arrive, in order."""
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            body = request.content.decode()
            if "BoardColumns" in body:
                return json_response({"data": {"boards": [{
                    "id": "1", "name": "Deals", "items_count": 3,
                    "columns": [{"id": "c1", "title": "Deal Name", "type": "text"}],
                }]}})
            if "BoardItemsNext" in body:
                return json_response({"data": {"next_items_page": {
                    "cursor": None,
                    "items": [{"id": "3", "name": "third", "column_values": []}],
                }}})
            return json_response({"data": {"boards": [{
                "id": "1", "name": "Deals",
                "items_page": {
                    "cursor": "CURSOR_1",
                    "items": [
                        {"id": "1", "name": "first", "column_values": []},
                        {"id": "2", "name": "second", "column_values": []},
                    ],
                },
            }]}})

        with client_with(handler) as client:
            snapshot = client.get_deals("1")

        assert [i["id"] for i in snapshot.items] == ["1", "2", "3"]
        assert calls["n"] == 3  # columns + page 1 + page 2

    def test_single_page_makes_no_extra_call(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            body = request.content.decode()
            if "BoardColumns" in body:
                return json_response({"data": {"boards": [
                    {"id": "1", "name": "Deals", "columns": []}
                ]}})
            return json_response({"data": {"boards": [{
                "id": "1", "name": "Deals",
                "items_page": {"cursor": None, "items": [
                    {"id": "1", "name": "only", "column_values": []}
                ]},
            }]}})

        with client_with(handler) as client:
            snapshot = client.get_deals("1")

        assert len(snapshot) == 1
        assert calls["n"] == 2


class TestBoardResolution:
    def _boards_handler(self, boards):
        def handler(request):
            return json_response({"data": {"boards": boards}})

        return handler

    def test_resolves_board_ids_by_name(self):
        handler = self._boards_handler([
            {"id": "111", "name": "Deals"},
            {"id": "222", "name": "Work Orders"},
            {"id": "333", "name": "Something else"},
        ])
        with client_with(handler) as client:
            assert client.resolve_board_ids() == ("111", "222")

    def test_name_matching_is_case_insensitive(self):
        handler = self._boards_handler([
            {"id": "111", "name": "deals"}, {"id": "222", "name": "WORK ORDERS"},
        ])
        with client_with(handler) as client:
            assert client.resolve_board_ids() == ("111", "222")

    def test_missing_board_names_what_it_did_find(self):
        handler = self._boards_handler([{"id": "111", "name": "Deals"}])
        with client_with(handler) as client:
            with pytest.raises(BoardNotFoundError) as exc:
                client.resolve_board_ids()
        assert "Work Orders" in str(exc.value)
        assert "Deals" in str(exc.value)

    def test_configured_ids_skip_discovery(self):
        def handler(request):
            raise AssertionError("should not have called the API")

        config = MondayConfig(
            api_token="t", api_version="2026-07",
            deals_board_id="1", work_orders_board_id="2",
        )
        client = MondayClient(config, transport=httpx.MockTransport(handler))
        assert client.resolve_board_ids() == ("1", "2")


class TestColumnValueReading:
    def test_prefers_display_value_for_mirror_columns(self):
        """Mirror and formula columns return null/"" for `text`."""
        from skylark_bi.ingest.builder import _display_value

        assert _display_value({"text": None, "display_value": "From mirror"}) == "From mirror"
        assert _display_value({"text": "", "display_value": "From formula"}) == "From formula"
        assert _display_value({"text": "Plain", "display_value": None}) == "Plain"
        assert _display_value({"text": None, "value": None}) == ""


class TestMalformedResponses:
    def test_board_missing_from_response(self):
        def handler(request):
            return json_response({"data": {"boards": []}})

        with client_with(handler) as client, pytest.raises(BoardNotFoundError):
            client.get_deals("999")

    def test_null_data_is_rejected(self):
        def handler(request):
            return json_response({"data": None})

        with client_with(handler) as client, pytest.raises(MondayQueryError):
            client.verify_token()
