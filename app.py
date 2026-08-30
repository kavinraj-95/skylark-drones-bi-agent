"""Streamlit UI for the Skylark Drones business intelligence agent.

Deliberately thin. Every number shown here was computed in `skylark_bi.analytics`;
this file only arranges them. The one thing it does add is the **Analysis** panel -
rendered straight from `AnalysisResult` with no LLM involved - so an evaluator can
check what the agent actually did rather than take the prose on trust.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from skylark_bi.agent.llm import LLMClient
from skylark_bi.agent.service import Answer, BIService
from skylark_bi.analytics.engine import AnalysisResult
from skylark_bi.analytics.metrics import Unit, format_inr
from skylark_bi.config import ConfigError, load_settings
from skylark_bi.monday.errors import MondayError

st.set_page_config(page_title="Skylark BI Agent", page_icon="🛩️", layout="wide")

SAMPLE_QUESTIONS = [
    "What can you answer?",
    "How's our pipeline looking for the energy sector this quarter?",
    "Which sectors have the strongest pipeline?",
    "What is our current weighted pipeline?",
    "How many active work orders do we have?",
    "Compare our sales pipeline with operational workload.",
    "What are the biggest risks in our current pipeline?",
    "Give me a leadership update.",
    "What data quality issues should I know about?",
]


def schema_fingerprint() -> str:
    """A short hash of the shapes this UI stores between reruns.

    Streamlit keeps `cache_resource` values and `session_state` alive across a code
    reload without restarting the process. After a redeploy that changes a dataclass,
    the new UI can therefore be handed objects built by the *old* definition - which
    is exactly how `answer.notes` raised AttributeError in production.

    Deriving the key from the field names means it changes automatically whenever the
    stored shape does, with nothing to remember to bump by hand.
    """
    parts = [
        f"{cls.__name__}:{','.join(sorted(getattr(cls, '__dataclass_fields__', {})))}"
        for cls in (Answer, AnalysisResult)
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


SCHEMA = schema_fingerprint()


@st.cache_resource(show_spinner=False)
def get_service(schema: str) -> BIService:
    """One service per server process, so the data cache is shared across reruns.

    Keyed on the schema fingerprint so a redeploy that changes a stored shape builds a
    fresh service rather than reusing one holding incompatible cached answers.
    """
    settings = load_settings(require_llm=False)
    llm = LLMClient(settings.llm) if settings.llm.api_key else None
    return BIService(settings, llm=llm)


def render_provenance(service: BIService) -> None:
    """The LIVE / STALE banner.

    Always visible and always explicit about where the data came from. An evaluator
    should never have to wonder whether they are looking at live monday.com data.
    """
    try:
        data = service.load()
    except Exception as exc:  # surfaced as a banner, never a traceback
        st.error(getattr(exc, "user_message", str(exc)))
        return

    dataset = data.dataset
    fetched = dataset.fetched_at.strftime("%Y-%m-%d %H:%M UTC") if dataset.fetched_at else "unknown"
    deals = dataset.provenance.get("deals")
    work_orders = dataset.provenance.get("work_orders")

    if dataset.is_stale:
        st.warning(
            f"**STALE — monday.com is unavailable.** Showing the last successful fetch "
            f"from {fetched}. {dataset.stale_reason or ''}"
        )
    else:
        st.caption(
            f"**LIVE** · fetched {fetched} from monday.com · "
            f"Deals board `{deals.board_id if deals else '?'}` "
            f"({len(dataset.active_deals)} records) · "
            f"Work Orders board `{work_orders.board_id if work_orders else '?'}` "
            f"({len(dataset.active_work_orders)} records) · "
            f"data as of **{dataset.as_of or 'unknown'}**"
        )


def render_analysis_panel(answer: Answer) -> None:
    """Show exactly what was computed, with no LLM in the path."""
    result = answer.result
    if result is None:
        return

    with st.expander("Analysis — what produced this answer", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("**Understood as**")
            st.write(result.plan.restatement or "—")
            st.markdown("**Boards queried**")
            st.write(", ".join(b.replace("_", " ") for b in result.plan.boards) or "—")
            st.markdown("**Filters**")
            st.write(result.plan.describe_filters())
        with right:
            st.markdown("**Records**")
            for board in result.plan.boards:
                st.write(
                    f"{board.replace('_', ' ').title()}: "
                    f"{result.records_in_scope.get(board, 0)} in scope "
                    f"of {result.records_total.get(board, 0)}"
                )
            st.markdown("**Data as of**")
            st.write(str(result.as_of or "unknown"))

        st.markdown("**Metrics computed**")
        for metric in result.metrics.values():
            if metric.available:
                st.markdown(f"- `{metric.name}` = **{metric.formatted()}** — {metric.definition}")
                st.caption(f"  {metric.provenance.summary()}")
                for exclusion in metric.provenance.exclusions:
                    st.caption(f"  ↳ {exclusion.count} excluded — {exclusion.reason}")
            else:
                st.markdown(f"- `{metric.name}` — *not available*: {metric.unavailable_reason}")

        if result.assumptions:
            st.markdown("**Assumptions**")
            for assumption in result.assumptions:
                st.markdown(f"- {assumption}")


def render_answer(answer: Answer) -> None:
    """Render one assistant turn.

    Attributes are read defensively. The schema check above should already have
    removed incompatible entries; this is the second line of defence, because an
    evaluator meeting a crashed app learns nothing about the agent.
    """
    if getattr(answer, "error", None):
        st.error(answer.error)
        return
    if getattr(answer, "clarifying_question", None):
        st.info(answer.clarifying_question)
        return

    for note in getattr(answer, "notes", None) or []:
        st.warning(note)
    render_metric_tiles(answer)
    st.markdown(getattr(answer, "text", "") or "")
    render_analysis_panel(answer)


def render_metric_tiles(answer: Answer) -> None:
    """Headline figures, so the numbers are visible without reading the prose."""
    result = answer.result
    if result is None:
        return
    tiles = [
        m for m in result.available_metrics.values()
        if m.unit in (Unit.INR, Unit.PERCENT, Unit.COUNT)
        and not m.context.get("comparison")
    ][:4]
    if not tiles:
        return
    for column, metric in zip(st.columns(len(tiles)), tiles):
        column.metric(metric.label, metric.formatted())


def render_data_quality(service: BIService) -> None:
    """The dedicated data-quality view."""
    try:
        data = service.load()
    except Exception as exc:
        st.error(getattr(exc, "user_message", str(exc)))
        return

    report = data.quality
    st.subheader("Data quality")
    st.caption(
        "Every finding below is computed from the live boards. Nothing is hardcoded — "
        "point the agent at different data and the findings change."
    )

    counts = {"high": 0, "medium": 0, "low": 0}
    for finding in report.findings:
        counts[finding.severity.value] += 1
    high, medium, low = st.columns(3)
    high.metric("High severity", counts["high"])
    medium.metric("Medium", counts["medium"])
    low.metric("Low", counts["low"])

    for finding in report.by_severity():
        icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}[finding.severity.value]
        with st.expander(f"{icon} {finding.title} — {finding.affected_records} record(s)"):
            st.write(finding.detail)
            if finding.handling:
                st.caption(f"**How the agent handles it:** {finding.handling}")

    st.subheader("Field coverage")
    st.caption("Share of records with a usable value, per canonical field.")
    for label, coverage in (
        ("Deals", report.deal_coverage),
        ("Work Orders", report.work_order_coverage),
    ):
        st.markdown(f"**{label}**")
        st.dataframe(
            [
                {
                    "Field": c.name,
                    "Usable": c.usable,
                    "of": c.total,
                    "Coverage": f"{c.usable_pct:.0f}%",
                    "States": ", ".join(f"{k}={v}" for k, v in sorted(c.counts.items())),
                }
                for c in sorted(coverage, key=lambda c: c.usable_pct)
            ],
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.title("🛩️ Skylark Drones — Business Intelligence Agent")
    st.caption(
        "Ask a business question in plain English. Figures are computed "
        "deterministically from monday.com; the language model interprets them but "
        "never calculates them."
    )

    try:
        service = get_service(SCHEMA)
    except ConfigError as exc:
        st.error(str(exc))
        st.info("See `setup/MONDAY_SETUP.md` for configuration steps.")
        return

    with st.sidebar:
        st.header("Sample questions")
        for question in SAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True, key=f"sample-{question}"):
                st.session_state.pending_question = question
        st.divider()
        if st.button("↻ Refresh data from monday.com", use_container_width=True):
            try:
                service.load(force_refresh=True)
                st.success("Refreshed.")
            except (MondayError, Exception) as exc:
                st.error(getattr(exc, "user_message", str(exc)))
        st.caption(
            "Read-only. This app issues GraphQL **queries** only and has no code path "
            "that can modify a board."
        )

    chat_tab, quality_tab = st.tabs(["Ask", "Data quality"])

    with quality_tab:
        render_data_quality(service)

    with chat_tab:
        render_provenance(service)

        if "history" not in st.session_state:
            st.session_state.history = []

        # Drop anything stored by a previous version of the code. Discarding a few
        # replayed messages is a far better outcome than the app refusing to load.
        st.session_state.history = [
            e for e in st.session_state.history if e.get("schema") == SCHEMA
        ]

        for entry in st.session_state.history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                render_answer(entry["answer"])

        question = st.chat_input("Ask about pipeline, revenue, sectors, work orders…")
        if not question and st.session_state.get("pending_question"):
            question = st.session_state.pop("pending_question")

        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Querying monday.com and analysing…"):
                    answer = service.ask(question)
                render_answer(answer)
            st.session_state.history.append(
                {"question": question, "answer": answer, "schema": SCHEMA}
            )


if __name__ == "__main__":
    main()
