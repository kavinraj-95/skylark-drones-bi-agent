"""Safety properties: read-only enforcement, injection resistance, LLM containment.

These are the tests that matter most if the rest of the system is working. Each one
asserts a property the design claims, rather than an implementation detail.
"""

from __future__ import annotations

import inspect

import pytest

from skylark_bi.agent.intent import Intent, QueryIntent
from skylark_bi.agent.plan import PlanValidationError, QueryPlan
from skylark_bi.agent.resolver import resolve
from skylark_bi.agent.responder import build_prompt, render_plain
from skylark_bi.analytics.engine import execute
from skylark_bi.monday.client import (
    Q_BOARDS,
    Q_COLUMNS,
    Q_ITEMS_FIRST,
    Q_ITEMS_NEXT,
    Q_ME,
    MondayClient,
    assert_read_only,
)
from skylark_bi.monday.errors import ReadOnlyViolationError


class TestReadOnly:
    """Read-only is structural first, and a tripwire second. Both are asserted."""

    @pytest.mark.parametrize("document", [Q_ME, Q_BOARDS, Q_COLUMNS, Q_ITEMS_FIRST, Q_ITEMS_NEXT])
    def test_every_shipped_query_passes(self, document):
        assert_read_only(document)

    @pytest.mark.parametrize(
        "document",
        [
            "mutation M { create_item(board_id: 1) { id } }",
            "mutation { delete_item(item_id: 5) { id } }",
            "query Q { boards { id } } mutation M { change_column_value { id } }",
            "subscription S { events { id } }",
            "{ boards { id } }",  # anonymous: nothing in this codebase builds one
        ],
    )
    def test_non_read_operations_are_refused(self, document):
        with pytest.raises(ReadOnlyViolationError):
            assert_read_only(document)

    def test_commented_out_mutation_does_not_false_positive(self):
        assert_read_only("query Q {\n  # mutation goes here one day\n  boards { id }\n}")

    def test_client_exposes_no_arbitrary_query_method(self):
        """The primary control: there is no public entry point taking a query string.

        If this fails, someone has added a method that lets a caller - or an LLM
        upstream of one - choose the GraphQL sent to monday.com.
        """
        public = [
            name for name, _ in inspect.getmembers(MondayClient, inspect.isfunction)
            if not name.startswith("_")
        ]
        assert set(public) == {
            "close", "verify_token", "get_boards", "get_columns",
            "get_deals", "get_work_orders", "resolve_board_ids",
        }

        for name in public:
            params = set(inspect.signature(getattr(MondayClient, name)).parameters)
            assert not params & {"query", "document", "graphql", "gql"}, (
                f"{name}() accepts a caller-supplied query"
            )

    def test_no_mutation_text_anywhere_in_the_package(self):
        """No code path can even name a monday mutation."""
        from pathlib import Path

        package = Path(__file__).resolve().parents[1] / "skylark_bi"
        offenders = []
        for path in package.rglob("*.py"):
            if path.name == "errors.py":
                continue  # defines the guard's own error type
            text = path.read_text(encoding="utf-8")
            for builder in ("create_item", "change_column_value", "delete_item",
                            "duplicate_item", "archive_item", "create_board"):
                if builder in text:
                    offenders.append(f"{path.name}: {builder}")
        assert not offenders, f"Mutation-building code present: {offenders}"


class TestPromptInjection:
    """The LLM has no field capable of carrying an instruction into execution."""

    def test_intent_schema_cannot_carry_a_query(self):
        """A model persuaded to emit GraphQL has nowhere to put it."""
        fields = set(QueryIntent.model_fields)
        assert not fields & {"query", "graphql", "sql", "code", "metric", "metrics", "boards"}

    def test_unknown_fields_from_the_model_are_dropped(self):
        intent = QueryIntent.model_validate({
            "intent": "pipeline_health",
            "restatement": "pipeline",
            "graphql": "mutation { delete_item(item_id: 1) { id } }",
            "metrics": ["arbitrary_metric"],
        })
        assert not hasattr(intent, "graphql")
        assert not hasattr(intent, "metrics")

    def test_injected_terms_cannot_reach_execution(self, dataset):
        """Hostile free text survives only as a filter term, and resolves to nothing."""
        intent = QueryIntent(
            intent=Intent.PIPELINE_HEALTH,
            sector_term="'; mutation { delete_item(item_id: 1) { id } } #",
            restatement="injection attempt",
        )
        plan = resolve(intent, dataset, fiscal_start_month=4)
        # It cannot be resolved to a sector, so the agent asks rather than acting.
        assert plan.needs_clarification
        assert plan.sectors == ()

    def test_metrics_never_come_from_the_model(self, dataset):
        """Metric selection is keyed off intent, from a registry - not model output."""
        intent = QueryIntent(intent=Intent.OPERATIONS, restatement="work orders")
        plan = resolve(intent, dataset, fiscal_start_month=4)
        from skylark_bi.agent.plan import INTENT_METRICS

        assert plan.metrics == INTENT_METRICS[Intent.OPERATIONS]

    def test_plan_validation_rejects_unknown_metrics(self):
        plan = QueryPlan(
            intent=Intent.PIPELINE_HEALTH,
            metrics=("open_pipeline_value", "drop_all_tables"),
            boards=("deals",),
        )
        with pytest.raises(PlanValidationError, match="drop_all_tables"):
            plan.validate()

    def test_plan_validation_rejects_unknown_boards(self):
        plan = QueryPlan(intent=Intent.PIPELINE_HEALTH, metrics=(), boards=("payroll",))
        with pytest.raises(PlanValidationError, match="payroll"):
            plan.validate()


class TestLLMContainment:
    """The model interprets numbers; it cannot originate or override them."""

    def test_responder_never_receives_raw_records(self, dataset, quality_report):
        """No board row - masked or not - reaches the model."""
        intent = QueryIntent(intent=Intent.PIPELINE_HEALTH, restatement="pipeline")
        plan = resolve(intent, dataset, fiscal_start_month=4)
        result = execute(plan, dataset, quality_report)
        prompt = build_prompt("How is pipeline?", result)

        # Identifiers that exist only on individual records must not appear.
        sample = dataset.active_deals[0]
        assert sample.item_id not in prompt
        for deal in dataset.active_deals[:40]:
            client = deal.client_code.or_none()
            if client:
                assert client not in prompt

    def test_deterministic_answer_holds_the_real_number(self, dataset, quality_report):
        """The fallback renderer proves the figures exist independently of any LLM.

        This is the containment guarantee in its strongest form: with no model in the
        loop at all, the same numbers are produced.
        """
        intent = QueryIntent(intent=Intent.PIPELINE_HEALTH, restatement="pipeline")
        plan = resolve(intent, dataset, fiscal_start_month=4)
        result = execute(plan, dataset, quality_report)

        metric = result.metrics["open_pipeline_value"]
        assert metric.available
        assert metric.formatted() in render_plain(result)

    def test_prompt_states_figures_are_precomputed(self, dataset, quality_report):
        intent = QueryIntent(intent=Intent.REVENUE, restatement="revenue")
        plan = resolve(intent, dataset, fiscal_start_month=4)
        result = execute(plan, dataset, quality_report)
        prompt = build_prompt("What is our revenue?", result)
        assert "already computed" in prompt

        from skylark_bi.agent.responder import SYSTEM_PROMPT

        assert "Never compute" in SYSTEM_PROMPT


class TestSecrets:
    def test_no_credentials_in_the_responder_prompt(self, dataset, quality_report, monkeypatch):
        monkeypatch.setenv("MONDAY_API_TOKEN", "super-secret-token-value")
        intent = QueryIntent(intent=Intent.OPERATIONS, restatement="ops")
        plan = resolve(intent, dataset, fiscal_start_month=4)
        result = execute(plan, dataset, quality_report)
        assert "super-secret-token-value" not in build_prompt("ops?", result)

    def test_errors_do_not_leak_the_token(self):
        from skylark_bi.monday.errors import MondayAuthError

        error = MondayAuthError("monday.com rejected the token (HTTP 401).")
        assert "token" in error.user_message.lower()
        assert "Authorization" not in str(error)
