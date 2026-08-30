# Decision Log

Skylark Drones — monday.com Business Intelligence Agent. Two pages.

---

## Assumptions

**What was unclear, and what I assumed.**

| Ambiguity | Assumption | Why |
|---|---|---|
| "Energy sector" appears in the brief but not in the data (sectors are Renewables, Mining, Railways, Powerline, Construction, Others, DSP, Tender, Manufacturing, Security & Surveillance, Aviation). | `energy` → **Renewables + Powerline**, declared in a `SECTOR_ALIASES` table, marked `INFERRED`, and stated in every answer that uses it. | This is a *business inference*, not normalisation, so it belongs in reviewable code — not buried in a prompt where it drifts between calls. |
| Which field describes a deal's true state — `Deal Status` or `Deal Stage`? They disagree on **120 of 344** records. | **`Deal Stage` is authoritative.** | Evidence, not preference: 70 rows read `Won` while sitting at stage `A. Lead Generated`, *all* created 2025-11-27, *all* with no value, probability or dates. That is a bulk backlog import with a defaulted status. Stage is 100% populated and ordered A→O. The conflict is reported, never silently resolved. |
| Fiscal or calendar quarters? | **Fiscal, April–March**, with calendar supported and the basis always stated. | The data says so: invoice numbers read `SDPL/FY25-26/916`, amounts are ₹ with GST. |
| What is "open pipeline"? | Stages **A–F** — before the outcome is decided. `Pipeline` questions imply open deals unless stated otherwise. | G onward (`Project Won`, `Work Order Received`…) are outcomes, not pipeline. |
| Is `I. POC` pre- or post-win? | Treated as **parked** — counted, excluded from both won and lost. | Its letter implies post-win; a proof-of-concept implies pre-commitment. Three records is too few to decide, so the judgement is flagged rather than hidden. |
| Blank vs `NA` vs `0`. | Three distinct states: `MISSING`, `NOT_APPLICABLE`, and a real `0`. | `Billed Value (Excl GST)` has 63 blanks while `(Incl GST)` has 63 zeros — the same fact, encoded two ways. Collapsing them would silently invent ₹0 of billing. |

---

## Architecture

**One Python process, three hard layers.** No queues, no vector DB, no agent
framework — nothing in the requirements justifies them, and each would be deployment
risk bought with time that the data work needed.

The decision that carries the most weight:

> **The LLM reads intent and explains results. Code decides meaning, fetches, and calculates.**

Concretely, the LLM produces a small validated `QueryIntent` — *"the user said
'energy', said 'this quarter', and seems to be asking about pipeline health"*. It
never selects metrics, resolves terms, or performs arithmetic. Deterministic code
turns that into a whitelisted `QueryPlan`, executes it, and returns finished numbers;
the LLM's second and final job is prose.

I split intent extraction from semantic resolution deliberately. An earlier design had
the model emit resolved sectors and date ranges directly. That is fewer moving parts
but strictly worse: term resolution becomes non-reproducible, and unauditable. Now
`"energy"` reaches the resolver verbatim and the mapping is a reviewable table.

**Trade-off accepted:** intent classification is coarser than free-form planning. A
question that does not fit one of ten intents is declined rather than improvised. For
a BI tool answering to founders, a clean decline beats a confident fabrication.

---

## Integration: API, not MCP

**Direct monday GraphQL v2**, `API-Version` pinned to `2026-07`, `items_page(limit:500)`
then top-level `next_items_page` cursors.

I evaluated monday's official Platform MCP and rejected it, for reasons specific to
this build rather than a general preference:

- It exposes **60+ tools including writers, with no documented read-only mode** — the
  opposite of what a read-only requirement wants.
- Tool selection is **model-mediated**, so call counts are unpredictable against a
  finite daily API budget. MCP calls consume the *same* budget as direct requests, so
  it is not a rate-limit escape hatch either.
- It would require MCP client/session management inside a hosted Streamlit process.

A ~250-line client with explicit pagination, retries and a read-only guard wins on
control, testability and deployment risk. MCP would be the better choice for
open-ended exploration of a whole workspace; this is two known boards.

**The single highest-value implementation detail:** monday returns **HTTP 200 with an
`errors` array** for application-level failures, sometimes alongside partial `data`.
Treating 200 as success is the easiest way to build a client that is quietly wrong.
There is a test for it.

---

## Data quality

Every canonical field carries `{value, raw, state}` where state is one of `OK`,
`MISSING`, `NOT_APPLICABLE`, `MALFORMED`, `UNMAPPED`, `AMBIGUOUS`, `INFERRED`. Metrics
then decide explicitly what to do with each — rather than inheriting whatever
`float(x or 0)` happens to produce.

Three rules the code holds to:

1. **Never invent.** No imputation, no defaults. Excluded records are counted and
   reported with a reason.
2. **Never merge distinct things.** Canonicalisation collapses only genuine spelling
   variants (`BIlled` → `Fully Billed`). Anything unfamiliar stays `UNMAPPED` and
   visible — an unrecognised sector is far likelier to be *new* than misspelled.
3. **Preserve uncertainty.** `"Dec"` with no year is `AMBIGUOUS`, excluded from every
   time series, never assigned a year.

The audit derives 14 findings from live data. Two I would highlight:

- **`Closure Probability` leaks its own outcome.** On closed deals, `High` → 100% won
  (n=25), `Medium` → 0% (n=2), `Low` → 0% (n=5). Forecasts do not behave that way; the
  field is updated after the fact. This is *why* weighted-pipeline weights are fixed
  and declared rather than fitted.
- **The boards cannot be joined.** Client codes overlap on **zero** records — `COMPANY089`
  and `WOCOMPANY_089` are independent masked namespaces, and the ~50 numeric
  coincidences are exactly that. Deal name is the only shared attribute and is
  many-to-many (`Sakura`: 27 deals ↔ 9 work orders). So cross-board questions compare
  *aggregates*. Joining rows here would fabricate relationships, and would look
  plausible while doing it.

---

## Analytics

18 metrics in one registry, each declaring its definition, source fields, aggregation,
and missing-data rule. Provenance travels with every result: records considered, used,
excluded, and why.

**`weighted_pipeline_value` is labelled a heuristic and never called expected revenue.**
Weights are High/Medium/Low → 0.8/0.5/0.2, stated in every answer that uses it. Deals
without a probability are **excluded and counted, never given a default weight.**

I tried to do better and measure real weights from history. It is not possible here,
and the reason is worth recording: the probability field leaks outcomes (above), and
stage-derived rates are *tautological* — `G. Project Won` converts at 100% because the
stage **is** the outcome. So there is **no stage-derived fallback at all**. An
arbitrary constant that is honestly labelled beats a fitted number that launders a
circular definition.

**Value answers always pair the total with median and top-N concentration**, because
one deal is 33% of recorded value here and a bare total would hide that.

---

## LLM design

| The LLM does | Code does |
|---|---|
| Classify the question into one of ten intents | Select metrics (registry keyed by intent) |
| Report the user's terms verbatim | Resolve terms against the actual data |
| Explain finished numbers | Fetch, filter, aggregate, exclude |
| Decide *whether* to ask for clarification | Decide *whether that request is warranted* |

The model receives **only** an `AnalysisResult` — formatted metrics, assumptions,
caveats. Never a board row. That is both a correctness property and a privacy one:
Gemini's free tier may train on prompts, so only masked aggregates ever leave the
process. A test asserts no record identifier reaches the prompt.

Both LLM calls degrade: keyword intent detection and a deterministic renderer. With no
model at all, every number is identical — only the prose gets plainer.

**Gemini over Claude** because a Claude Pro *subscription* cannot authenticate a hosted
server app (Anthropic's docs direct shared automation to a Platform API key), and I did
not want a paid dependency in a prototype. The provider is a one-file adapter;
`LLM_PROVIDER=anthropic` switches it.

---

## Leadership updates — my interpretation

The brief says only *"the agent should help prepare data for leadership updates"*, and
leaves interpretation open. I read it as:

> **Convert current board data into a briefing a founder could take into a board
> meeting: headline, commercial position, operational position, sector picture, risks,
> and — crucially — what in the data would change these numbers.**

Two deliberate choices:

1. **It is composed from the same metric registry, not a separate code path.** A
   figure can therefore never differ between the leadership update and the question
   that produced it. Only the presentation prompt changes.
2. **It includes a data-caveats section.** An update that overstates confidence is
   worse than useless to leadership. The most valuable thing this feature does is tell
   a founder that ₹73 Cr of pipeline rests on 55 of 142 deals having a recorded value.

---

## Trade-offs — what I deliberately did not build

| Not built | Why |
|---|---|
| Charts and visualisations | A beautiful dashboard over weak intelligence fails the brief. Intelligence first. |
| Conversational memory | Real value, but multi-turn reference resolution is its own correctness problem. Each question stands alone. |
| Row-level cross-board join | The data cannot support it. Building it would be the single worst decision available. |
| OAuth `boards:read` app | The only true server-side read-only guarantee, but ~1h of the budget. Enforced in-process instead, and the README says plainly that this is defence-in-depth, not a security boundary. |
| Fuzzy record deduplication | 20 records are identical across every field. Merging could delete genuine repeat orders; the data cannot distinguish. Flagged and left in totals. |
| A runtime multi-agent system | Ten agents to answer ten question types is theatre. One linear pipeline. |

---

## Time constraints — what six hours prioritised

Roughly: data understanding and integration first, UI last. I profiled both datasets
**before** designing anything, which is why the architecture anticipates the
`Deal Status` contradiction and the unjoinable boards rather than discovering them
late.

The largest single cost was **not** code — it was the two live monday.com problems the
schedule did not budget for: the Free plan's 200-item cap (dataset is 522, hence the
Pro trial), and an import that lost its header row, which produced a board whose
column titles were its own values. The second turned into a feature: the agent now
detects a wholesale schema mismatch and reports it actionably rather than reporting
every metric as "no data", which would look like an empty business instead of a broken
import.

What I would not cut if I did it again: the field-state model. It took time early and
paid for itself in every metric afterwards.

---

## What I would do with another day, and another week

**A day**
- Conversational memory, so "and for mining?" resolves against the previous turn.
- An OAuth `boards:read` app for genuine server-side read-only.
- Period-over-period comparison ("pipeline vs last quarter") — the timeframe layer
  supports it; the metrics do not yet.
- Charts for sector breakdown and stage distribution, now that the numbers are trustworthy.

**A week**
- **Fix the data at source.** The most valuable output of this exercise is the audit
  itself: the `Deal Status` defaulting, the retrospective probability field, and the
  missing values on 52% of deals are business problems, not engineering ones. A
  weekly data-quality digest into Slack would be worth more than any query feature.
- **Empirical win rates** from a stage-transition history rather than a point-in-time
  snapshot — which would make weighted pipeline a real forecast instead of a heuristic.
- **A proper entity-resolution layer** for cross-board linkage, with explicit
  confidence scores, so row-level questions become answerable with stated uncertainty.
- Incremental fetching via monday's `updated_at`, instead of full board reads.
