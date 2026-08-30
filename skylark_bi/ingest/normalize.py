"""Value-level normalization: raw monday strings -> typed canonical values.

Every parser here returns a `Field`, never a bare value, so the reason a value is
absent survives into the analytics layer. Three rules govern this module:

1. **Never invent.** A blank stays `MISSING`. Unparseable content stays `MALFORMED`
   with its raw text attached. No imputation, no defaults, no `or 0`.
2. **Never merge distinct things.** Canonicalization only collapses values that are
   genuinely the same thing spelled differently (`BIlled` -> `Billed`). Anything
   outside the known vocabulary is `UNMAPPED` and passed through, because an
   unrecognised sector is more likely a new sector than a typo.
3. **Preserve uncertainty.** A month name with no year is `AMBIGUOUS`, not a guess at
   which year was meant.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

from .entities import Field, FieldState, Quantity

# Values that carry no information. Split in two because the difference is real:
# a blank cell is an absence of data, whereas someone typing "NA" is an assertion
# that the field does not apply to this record. `Quantities as per PO` contains
# both, and the FieldState taxonomy exists precisely to keep them apart.
_BLANK_TOKENS = {"", "-", "--", "?", "#n/a", "#value!", "#ref!"}
_NOT_APPLICABLE_TOKENS = {
    "n/a", "na", "n.a.", "none", "nil", "null", "not applicable", "unknown",
    "tbd", "tba",
}
_NULL_TOKENS = _BLANK_TOKENS | _NOT_APPLICABLE_TOKENS

_DATE_FORMATS = (
    "%Y-%m-%d",       # ISO - what monday date columns return
    "%Y/%m/%d",
    "%d-%m-%Y",       # common Indian convention
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d-%b-%Y",       # 12-Jan-2025
    "%d %b %Y",
    "%d-%B-%Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

#: Trailing units seen in quantity columns. Kept, never discarded - summing hectares
#: with days would be nonsense.
_UNIT_PATTERN = re.compile(
    r"^(?P<num>-?[\d,]*\.?\d+)\s*(?P<unit>[A-Za-z][A-Za-z\s./%-]*)?$"
)


def clean_text(raw: str | None) -> str:
    """Collapse whitespace and normalise unicode. Purely cosmetic - no semantics."""
    if raw is None:
        return ""
    text = unicodedata.normalize("NFKC", str(raw))
    text = text.replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_null_token(raw: str | None) -> bool:
    """True when a value is a recognised way of writing "nothing here"."""
    return clean_text(raw).lower() in _NULL_TOKENS


def _empty_field(text: str, note: str | None = None) -> Field:
    """Build the right kind of empty `Field` for a null-ish token.

    An explicit "NA" is `NOT_APPLICABLE` - the record is telling us the field does not
    apply. A blank is `MISSING` - we simply do not know. Metrics treat these
    identically today, but data-quality reporting does not, and conflating them would
    overstate how much data is actually absent.
    """
    if text.lower() in _NOT_APPLICABLE_TOKENS:
        return Field(
            None, text, FieldState.NOT_APPLICABLE,
            note or f"Explicitly marked {text!r} in the source.",
        )
    return Field.missing(text or None, note)


def parse_text(raw: str | None, *, note: str | None = None) -> Field[str]:
    """A free-text field, with null-ish tokens treated as missing."""
    text = clean_text(raw)
    if is_null_token(text):
        return _empty_field(text, note)
    return Field.good(text, text)


def parse_number(raw: str | None) -> Field[float]:
    """A numeric field.

    Handles thousands separators, currency symbols, parenthesised negatives and
    percentage signs. A real `0` parses to `0.0` with state OK - it is a value, and
    must never be conflated with a blank.
    """
    text = clean_text(raw)
    if is_null_token(text):
        return _empty_field(text)

    candidate = text
    negative = candidate.startswith("(") and candidate.endswith(")")
    if negative:
        candidate = candidate[1:-1]

    candidate = re.sub(r"[₹$€£,\s]", "", candidate)
    candidate = candidate.rstrip("%")

    try:
        value = float(candidate)
    except ValueError:
        return Field.malformed(text, f"Could not read {text!r} as a number.")

    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return Field.malformed(text, f"{text!r} is not a finite number.")

    return Field.good(-value if negative else value, text)


def parse_date(raw: str | None) -> Field[date]:
    """A date field.

    Accepts the formats this dataset and monday actually produce. Anything that looks
    like a bare month name is `AMBIGUOUS` rather than a guess: `"Dec"` with no year
    cannot be placed on a timeline, and picking one silently would corrupt every
    period comparison downstream.
    """
    text = clean_text(raw)
    if is_null_token(text):
        return _empty_field(text)

    # monday date columns can return a JSON blob rather than a plain string.
    if text.startswith("{"):
        match = re.search(r'"date"\s*:\s*"([^"]+)"', text)
        if match:
            text = match.group(1)

    for fmt in _DATE_FORMATS:
        try:
            return Field.good(datetime.strptime(text, fmt).date(), clean_text(raw))
        except ValueError:
            continue

    if text.lower().rstrip(".") in _MONTHS:
        return Field(
            None, text, FieldState.AMBIGUOUS,
            f"{text!r} names a month but carries no year, so it cannot be placed on a timeline.",
        )

    # "July 2025" / "2025-07" style month precision - real, but not a day.
    month_year = re.match(r"^([A-Za-z]+)[\s-]+(\d{4})$", text)
    if month_year and month_year.group(1).lower() in _MONTHS:
        return Field(
            None, text, FieldState.AMBIGUOUS,
            f"{text!r} is month-precision only; day-level analysis excludes it.",
        )

    return Field.malformed(text, f"Could not read {text!r} as a date.")


def parse_month_name(raw: str | None) -> Field[int]:
    """A bare month name -> month number, explicitly without a year.

    Returns `AMBIGUOUS` rather than OK even on success: the dataset spans multiple
    years, so `"Dec"` identifies a month but not a point in time. Callers may count
    by month; they must not build a time series from it.
    """
    text = clean_text(raw)
    if is_null_token(text):
        return _empty_field(text)

    month = _MONTHS.get(text.lower().rstrip("."))
    if month is None:
        return Field.malformed(text, f"{text!r} is not a recognisable month.")

    return Field(
        month, text, FieldState.AMBIGUOUS,
        f"{text!r} has no year; usable for month-of-year grouping only.",
    )


def parse_quantity(raw: str | None) -> Field[Quantity]:
    """A quantity that may carry a unit.

    `Quantities as per PO` mixes `1`, `1600`, `NA`, `45days`, `45 days` and `5360 HA`
    in one column. Bare numbers parse with `unit=None`; numbers with a trailing unit
    keep it; anything else is `MALFORMED` with the raw text retained.
    """
    text = clean_text(raw)
    if is_null_token(text):
        return _empty_field(text)

    match = _UNIT_PATTERN.match(text)
    if not match:
        return Field.malformed(text, f"Could not read {text!r} as a quantity.")

    try:
        amount = float(match.group("num").replace(",", ""))
    except ValueError:
        return Field.malformed(text, f"Could not read {text!r} as a quantity.")

    unit = clean_text(match.group("unit")) or None
    return Field.good(Quantity(amount, unit), text)


def canonicalize(
    raw: str | None,
    aliases: dict[str, str],
    *,
    field_name: str,
) -> Field[str]:
    """Map a categorical value onto a known vocabulary.

    `aliases` maps a lowercased, whitespace-collapsed form to its canonical spelling.
    A value that is present but absent from the vocabulary returns `UNMAPPED` with the
    original text as its value - it is kept and surfaced, never dropped and never
    force-fitted to the nearest known label. An unrecognised sector is far more likely
    to be a genuinely new sector than a misspelling of an existing one.
    """
    text = clean_text(raw)
    if is_null_token(text):
        return _empty_field(text)

    canonical = aliases.get(text.lower())
    if canonical is not None:
        return Field.good(canonical, text)

    return Field(
        text, text, FieldState.UNMAPPED,
        f"{text!r} is not a known value for {field_name}; counted separately.",
    )


def build_alias_map(canonical_values: dict[str, list[str]]) -> dict[str, str]:
    """Build a lookup from {canonical: [variants]}, including each canonical itself."""
    aliases: dict[str, str] = {}
    for canonical, variants in canonical_values.items():
        aliases[canonical.lower()] = canonical
        for variant in variants:
            aliases[clean_text(variant).lower()] = canonical
    return aliases
