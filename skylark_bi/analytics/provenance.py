"""Provenance: what a number was built from, and what it left out.

Every metric this system produces can answer four questions:

* which records went into it,
* which records were left out and why,
* which board fields it read, and
* what we assumed along the way.

That record is what makes the difference between "pipeline is 732M" and a number a
founder can actually act on. It also feeds the UI's analysis panel verbatim, with no
LLM in the path, so the evaluator can check the arithmetic rather than trust prose.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field


@dataclass
class Exclusion:
    """A group of records left out of a metric, with the reason."""

    reason: str
    count: int
    #: Canonical field responsible, when there is one.
    field_name: str | None = None
    #: A few example record identifiers, for spot-checking.
    examples: list[str] = field(default_factory=list)


@dataclass
class Provenance:
    """The audit trail behind one metric."""

    #: Records that met the filters and were candidates for this metric.
    records_considered: int = 0
    #: Records that actually contributed a value.
    records_used: int = 0
    exclusions: list[Exclusion] = field(default_factory=list)
    #: Canonical field names read, for attaching the right data-quality caveats.
    source_fields: tuple[str, ...] = ()
    #: monday boards touched.
    boards: tuple[str, ...] = ()
    assumptions: list[str] = field(default_factory=list)

    @property
    def records_excluded(self) -> int:
        return sum(e.count for e in self.exclusions)

    @property
    def coverage(self) -> float:
        """Fraction of candidate records that contributed. 1.0 when none were dropped."""
        if not self.records_considered:
            return 0.0
        return self.records_used / self.records_considered

    def add_exclusion(
        self, reason: str, count: int, *, field_name: str | None = None, examples: list[str] | None = None
    ) -> None:
        if count > 0:
            self.exclusions.append(
                Exclusion(reason=reason, count=count, field_name=field_name, examples=examples or [])
            )

    def summary(self) -> str:
        """One line describing coverage, for use directly in an answer."""
        if not self.records_considered:
            return "No records matched the filters."
        if not self.exclusions:
            return f"Based on all {self.records_used} matching record(s)."
        return (
            f"Based on {self.records_used} of {self.records_considered} matching record(s); "
            f"{self.records_excluded} excluded."
        )


def summarise_field_states(records, field_name: str) -> dict[str, int]:
    """Count the field states present for one canonical field."""
    return dict(collections.Counter(r.get(field_name).state.value for r in records))
