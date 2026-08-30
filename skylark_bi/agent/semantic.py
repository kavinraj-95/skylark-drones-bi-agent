"""Deterministic resolution of business vocabulary onto this dataset.

This is where "energy" becomes a concrete list of sectors, and it is deliberately
*code*, not a prompt. Two reasons:

1. **It is a business-semantic inference, not normalization.** "Energy" is not a
   sector in this data. Deciding it means Renewables plus Powerline is a judgement
   about the business, and judgements like that belong somewhere a human can read,
   review and change - not buried in an LLM prompt where they drift between calls.
2. **It must be auditable.** Every inference made here is recorded and surfaced in
   the answer, so the founder sees "I read energy as Renewables + Powerline" rather
   than silently getting two sectors they did not ask for.

Anything the resolver cannot place is reported as unresolved rather than guessed.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from ..ingest.mapping import SECTORS

#: Business synonyms that map a colloquial term onto one or more real sectors.
#:
#: These are *inferences*, and each is flagged as such in the answer. `energy` is the
#: important one: a founder asking about energy almost certainly means the generation
#: and transmission work, which lives under Renewables and Powerline here.
SECTOR_ALIASES: dict[str, tuple[str, ...]] = {
    "energy": ("Renewables", "Powerline"),
    "power": ("Renewables", "Powerline"),
    "utilities": ("Renewables", "Powerline"),
    "clean energy": ("Renewables",),
    "green energy": ("Renewables",),
    "solar": ("Renewables",),
    "wind": ("Renewables",),
    "transmission": ("Powerline",),
    "grid": ("Powerline",),
    "rail": ("Railways",),
    "metro": ("Railways",),
    "infra": ("Construction",),
    "infrastructure": ("Construction",),
    "minerals": ("Mining",),
    "coal": ("Mining",),
}

#: How close a term must be to a real sector name to be accepted as a typo of it.
#: High on purpose - loose matching here would quietly answer about the wrong sector.
_FUZZY_CUTOFF = 0.86

#: Category nouns a founder attaches to a sector name ("the energy sector", "mining
#: vertical"). Stripped before lookup so the resolver matches on the sector itself.
#: This belongs here rather than in the planner prompt: the LLM's job is to report the
#: user's words faithfully, and normalising them is deterministic work.
_CATEGORY_NOUNS = (
    "sector", "sectors", "industry", "industries", "vertical", "verticals",
    "segment", "segments", "space", "business", "market",
)


def _strip_category_noun(term: str) -> str:
    """Remove a leading/trailing category noun from a sector phrase."""
    words = [w for w in term.replace("-", " ").split() if w]
    while words and words[-1] in _CATEGORY_NOUNS:
        words.pop()
    while words and words[0] in ("the", "our"):
        words.pop(0)
    return " ".join(words) if words else term


@dataclass
class SectorResolution:
    """The outcome of interpreting a sector term."""

    #: Concrete sector names to filter on. Empty means "no sector filter".
    sectors: tuple[str, ...] = ()
    #: The term as the user wrote it.
    term: str | None = None
    #: True when the mapping is our inference rather than a literal match.
    inferred: bool = False
    #: True when we could not place the term at all.
    unresolved: bool = False
    #: Sentences to surface in the answer.
    notes: list[str] = field(default_factory=list)
    #: Alternatives to offer when the term is unresolved.
    suggestions: tuple[str, ...] = ()


def known_sectors() -> tuple[str, ...]:
    """Every canonical sector name, for prompts, validation and suggestions."""
    return tuple(sorted(set(SECTORS.values())))


def resolve_sector(term: str | None, *, available: tuple[str, ...] | None = None) -> SectorResolution:
    """Turn a user's sector word into concrete sector names.

    `available` is the set of sectors actually present in the current data. Passing it
    lets the resolver tell "this sector does not exist" apart from "this sector exists
    but has no records right now" - a distinction that matters when explaining an
    empty answer.
    """
    if not term:
        return SectorResolution()

    cleaned = _strip_category_noun(term.strip().lower())
    catalogue = available or known_sectors()
    if not cleaned:
        return SectorResolution(term=term)

    # 1. Exact match against a real sector, via the same alias table normalization
    #    uses (so "renewable" and "Renewables" agree).
    canonical = SECTORS.get(cleaned)
    if canonical:
        return SectorResolution(sectors=(canonical,), term=term)

    # 2. A declared business synonym. This is an inference and is labelled as one.
    alias = SECTOR_ALIASES.get(cleaned)
    if alias:
        present = tuple(s for s in alias if s in catalogue)
        target = present or alias
        joined = " and ".join(target)
        return SectorResolution(
            sectors=target,
            term=term,
            inferred=True,
            notes=[
                f"'{term}' is not a sector recorded in this data, so it is read as "
                f"{joined}."
            ],
        )

    # 3. A close spelling of a real sector. Deliberately strict.
    close = difflib.get_close_matches(cleaned, [s.lower() for s in catalogue], n=1, cutoff=_FUZZY_CUTOFF)
    if close:
        match = next(s for s in catalogue if s.lower() == close[0])
        return SectorResolution(
            sectors=(match,),
            term=term,
            inferred=True,
            notes=[f"'{term}' is read as the {match} sector."],
        )

    # 4. Unplaceable. Say so and offer what does exist, rather than answering about
    #    something the user did not ask for.
    return SectorResolution(
        term=term,
        unresolved=True,
        suggestions=catalogue,
        notes=[
            f"'{term}' does not match any sector in this data, so no sector filter was "
            f"applied. Sectors present: {', '.join(catalogue)}."
        ],
    )


#: Status words a founder might use, mapped onto how the analytics layer thinks.
#: `open`/`won`/`lost` are stage-derived, because Deal Status is unreliable here.
_STATUS_TERMS: dict[str, str] = {
    "open": "open", "active": "open", "live": "open", "in play": "open",
    "won": "won", "closed won": "won", "closed-won": "won", "converted": "won",
    "lost": "lost", "dead": "lost", "closed lost": "lost", "closed-lost": "lost",
    "on hold": "held", "held": "held", "paused": "held", "parked": "held",
}


def resolve_status(term: str | None) -> tuple[str | None, list[str]]:
    """Map a status word onto `open` / `won` / `lost` / `held`, or None."""
    if not term:
        return None, []
    cleaned = term.strip().lower()
    resolved = _STATUS_TERMS.get(cleaned)
    if resolved:
        return resolved, []
    return None, [
        f"'{term}' was not recognised as a deal status, so no status filter was applied."
    ]
