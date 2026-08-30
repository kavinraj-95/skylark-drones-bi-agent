"""Typed errors for the monday.com integration.

Each error carries a `user_message` that is safe and useful to show in the UI. The
distinction matters: `MondayAuthError` should tell the operator to check their token,
while `MondayRateLimitError` should tell the user to wait - and neither should ever
surface a raw traceback or leak the token.
"""

from __future__ import annotations


class MondayError(RuntimeError):
    """Base class for every monday.com failure."""

    user_message = "Could not reach monday.com."

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        if user_message:
            self.user_message = user_message


class MondayAuthError(MondayError):
    user_message = (
        "monday.com rejected the API token. Check that MONDAY_API_TOKEN is set and "
        "still valid, and that the token's user can see both boards."
    )


class MondayRateLimitError(MondayError):
    """Rate limit, complexity budget, or daily call limit exhausted."""

    user_message = "monday.com rate limit reached. Please retry shortly."

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: int | None = None,
        daily_limit: bool = False,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message, user_message=user_message)
        self.retry_after_seconds = retry_after_seconds
        #: Daily limits reset at midnight UTC - retrying within the same day is futile.
        self.daily_limit = daily_limit


class MondayQueryError(MondayError):
    """A GraphQL-level error: parse failure, unknown field, bad board ID."""

    user_message = "monday.com could not run the data query."


class MondayUnavailableError(MondayError):
    """Network failure or timeout - monday.com could not be contacted at all."""

    user_message = "monday.com is unreachable right now."


class BoardNotFoundError(MondayError):
    user_message = (
        "Could not find the expected boards in monday.com. Set MONDAY_DEALS_BOARD_ID "
        "and MONDAY_WORK_ORDERS_BOARD_ID explicitly, or rename the boards to match "
        "MONDAY_DEALS_BOARD_NAME / MONDAY_WORK_ORDERS_BOARD_NAME."
    )


class SchemaMismatchError(MondayError):
    """A board's columns do not look like the schema we expect.

    Raised when so few canonical fields resolve that any analysis would be vacuous.
    Failing loudly here is deliberate: the alternative is an app that cheerfully
    reports every metric as "no data", which looks like an empty business rather than
    a broken import.
    """

    def __init__(
        self,
        board_label: str,
        *,
        found_titles: list[str],
        matched: int,
        required: int,
        missing: list[str],
    ) -> None:
        self.board_label = board_label
        self.found_titles = found_titles
        self.matched = matched
        self.required = required
        self.missing = missing
        preview = ", ".join(repr(t) for t in found_titles[:8])
        super().__init__(
            f"{board_label} board matched only {matched} of {required} required columns.",
            user_message=(
                f"The **{board_label}** board in monday.com does not have the expected "
                f"columns - only {matched} of {required} required fields could be "
                f"matched.\n\nColumn titles found: {preview}"
                f"{' ...' if len(found_titles) > 8 else ''}\n\n"
                "This usually means the CSV was imported without treating the first row "
                "as a header row, so monday named each column after one of its values. "
                "Re-import the board and make sure the header row is mapped as headers "
                "(see setup/MONDAY_SETUP.md)."
            ),
        )


class ReadOnlyViolationError(MondayError):
    """A non-read GraphQL operation was about to be sent.

    This is a tripwire, not the primary control. The primary control is structural:
    `MondayClient` exposes no public method that accepts a caller-supplied query, and
    every query document is a module-level constant using GraphQL variables. If this
    error is ever raised, a code change has broken that invariant.
    """

    user_message = "Blocked a non-read operation against monday.com."
