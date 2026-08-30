"""Describing what the agent can actually answer.

"What can you do?" is one of the first things anyone types into a new agent, and
answering it with pipeline figures - as this app did - is worse than useless. It is
also the right response when a question cannot be placed at all: telling someone what
*is* possible beats a bare refusal.

The description is generated from the metric registry and the live dataset rather than
written out by hand, so it cannot drift from what the system actually does. Add a
metric or point the agent at boards with different sectors, and this text follows.
"""

from __future__ import annotations

from ..analytics.metrics import REGISTRY
from ..ingest.entities import Dataset

#: The question categories a founder can ask about, each with an example phrased the
#: way someone would actually say it. Ordered by how often they are likely to be used.
QUESTION_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("Pipeline health", "How's our pipeline looking for the energy sector this quarter?"),
    ("Sector performance", "Which sectors have the strongest pipeline?"),
    ("Revenue and win rate", "What's our win rate?"),
    ("Operations", "How many active work orders do we have?"),
    ("Billing and receivables", "How much have we billed and what's still outstanding?"),
    ("Sales vs delivery", "Compare our sales pipeline with operational workload."),
    ("Risk", "What are the biggest risks in our current pipeline?"),
    ("Leadership update", "Give me a leadership update."),
    ("Data quality", "What data quality issues should I know about?"),
)


def describe(dataset: Dataset | None = None) -> str:
    """Build the capability description, grounded in the current data."""
    lines: list[str] = [
        "I answer business questions from your two monday.com boards — **Deals** "
        "(sales pipeline) and **Work Orders** (project execution).",
        "",
        "Every figure I give you is calculated in code from the live boards. The "
        "language model reads your question and explains the result; it never does "
        "the arithmetic. I also tell you what I excluded and why, so you can see how "
        "much of the data a number actually covers.",
        "",
        "**What you can ask**",
        "",
    ]
    lines.extend(f"- **{name}** — *\"{example}\"*" for name, example in QUESTION_CATEGORIES)

    if dataset is not None:
        lines.extend(["", "**What I'm currently looking at**", ""])
        deals = len(dataset.active_deals)
        work_orders = len(dataset.active_work_orders)
        lines.append(f"- {deals} deals and {work_orders} work orders")
        if dataset.as_of:
            lines.append(
                f"- Most recent activity in the data: **{dataset.as_of.isoformat()}** — "
                "relative periods like \"this quarter\" are measured from there, not "
                "from today's date, and I say so when the two differ"
            )
        sectors = sorted(
            {d.sector.value for d in dataset.active_deals if d.sector.value}
            | {w.sector.value for w in dataset.active_work_orders if w.sector.value}
        )
        if sectors:
            lines.append(f"- Sectors on record: {', '.join(str(s) for s in sectors)}")

    lines.extend([
        "",
        "**What I can't do**",
        "",
        "- Anything outside these two boards — headcount, costs, marketing, individual "
        "performance. I'll say so rather than guess.",
        "- Link a specific deal to a specific work order. The boards share no reliable "
        "key, so I compare them by sector instead of inventing a join.",
        "- Change anything. Access is read-only.",
        "",
        f"_{len(REGISTRY)} metrics available. Ask \"what data quality issues should I "
        "know about?\" to see what I think of your data._",
    ])
    return "\n".join(lines)
