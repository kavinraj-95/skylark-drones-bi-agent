"""`QueryIntent` - the only thing the LLM is allowed to produce from a question.

This is deliberately a *narrow* surface. The planner LLM reports what the user
appears to be asking, in the user's own vocabulary: `sector_term: "energy"`,
`time_expression: "this quarter"`. It does not decide what those terms mean in this
dataset, and it has no field capable of carrying a query, an API parameter, or a
computed number.

That last point is the security property. There is no string here that reaches
monday.com or an evaluator, so a prompt-injection attempt that persuades the model to
emit `mutation { ... }` has nowhere to put it - the field simply does not exist, and
anything unrecognised fails validation before execution.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Intent(str, Enum):
    """What kind of question this is. Drives which metrics get selected."""

    PIPELINE_HEALTH = "pipeline_health"
    REVENUE = "revenue"
    SECTOR_PERFORMANCE = "sector_performance"
    OPERATIONS = "operations"
    BILLING = "billing"
    CROSS_BOARD = "cross_board"
    RISK = "risk"
    LEADERSHIP_UPDATE = "leadership_update"
    DATA_QUALITY = "data_quality"
    #: The question is about this dataset but not something we can compute.
    UNSUPPORTED = "unsupported"


#: Maximum characters accepted in any free-text field the model returns. Long strings
#: are a smell - they are either hallucinated prose or an injection attempt - and
#: nothing legitimate here needs more.
MAX_TERM_LENGTH = 120


class QueryIntent(BaseModel):
    """The LLM's reading of a founder's question."""

    # "ignore" rather than "forbid": pydantic emits `additionalProperties: false` for
    # forbid, which Gemini's schema dialect rejects outright. Ignoring is equally safe
    # here - unknown keys are dropped during validation, so a model that invents a
    # field still cannot get it into a QueryPlan.
    model_config = {"extra": "ignore"}

    intent: Intent = Field(
        description="The kind of business question being asked."
    )
    #: The sector the user named, in their own words. Not resolved to a real sector -
    #: that is done deterministically by the semantic resolver.
    sector_term: str | None = Field(
        default=None,
        description="Sector or industry the user mentioned, verbatim. Null if none.",
    )
    #: The period the user named, in their own words ("this quarter", "Q3", "2025").
    time_expression: str | None = Field(
        default=None,
        description="Time period the user mentioned, verbatim. Null if none.",
    )
    #: Free-text hints about other filters (owner, status). Applied only where the
    #: resolver recognises them.
    status_term: str | None = Field(
        default=None,
        description="Deal or execution status the user mentioned, verbatim. Null if none.",
    )
    #: True when the question genuinely cannot be answered without more information.
    #: The bar is high on purpose: a clarifying question the user did not need is a
    #: worse experience than a stated assumption.
    needs_clarification: bool = Field(
        default=False,
        description=(
            "True ONLY when the question is so ambiguous that different readings give "
            "materially different answers. Prefer stating an assumption instead."
        ),
    )
    clarifying_question: str | None = Field(
        default=None,
        description="A single, specific question to ask. Null unless needs_clarification.",
    )
    #: One line explaining what the user seems to want. Shown in the analysis panel.
    restatement: str = Field(
        default="",
        description="One sentence restating the question in business terms.",
    )

    @field_validator("sector_term", "time_expression", "status_term", "clarifying_question")
    @classmethod
    def _bound_length(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        return text[:MAX_TERM_LENGTH]

    @field_validator("restatement")
    @classmethod
    def _bound_restatement(cls, value: str) -> str:
        return (value or "").strip()[:400]
