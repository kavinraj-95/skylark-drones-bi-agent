# Skylark Drones — monday.com Business Intelligence Agent

A conversational agent that answers founder-level business questions by querying two
monday.com boards live — **Deals** (sales pipeline) and **Work Orders** (project
execution) — normalising deliberately messy data, computing metrics deterministically,
and explaining what the numbers mean along with what they cannot tell you.

```
"How's our pipeline looking for the energy sector this quarter?"
        │
        ├─ "energy" is not a sector here → read as Renewables + Powerline (stated)
        ├─ "this quarter" (FY25-26 Q4) has no records → most recent populated quarter used (stated)
        └─ ₹2.60 Cr open pipeline, from 8 of 27 deals — the other 19 have no value recorded
```

---

## The one design decision that matters

> **The LLM reads intent and explains results. Code decides meaning, fetches, and calculates.**

A language model that adds up 37 deal values will eventually get it wrong, and you
will not know which time. So it never does. The model produces a small, validated
`QueryIntent` describing what the user *appears* to be asking; deterministic Python
resolves that into a whitelisted `QueryPlan`, executes it, and hands back finished
numbers. The model's second and final job is to explain those numbers in prose.

There is a test asserting the figures exist with no model in the loop at all
(`tests/test_security.py::TestLLMContainment`).

---

## Architecture

```
Streamlit UI (app.py)  ─ thin: arranges output, computes nothing
        │
        ▼
┌──────────────────┐  LLM #1 → QueryIntent: raw terms only
│     PLANNER      │  {intent, sector_term:"energy", time_expression:"this quarter"}
└──────────────────┘  agent/planner.py · agent/intent.py
        │             (deterministic keyword fallback if the LLM is unavailable)
        ▼
┌──────────────────┐  DETERMINISTIC — no LLM
│ SEMANTIC RESOLVER│  "energy"       → ["Renewables","Powerline"]   (declared alias, flagged)
│   + TIMEFRAME    │  "this quarter" → FY quarter; if empty, most recent populated one
└──────────────────┘  agent/semantic.py · analytics/timeframe.py
        │
        ▼
┌──────────────────┐  metrics from a registry keyed by intent — never from the model
│  PLAN VALIDATOR  │  unknown metric / board / status ⇒ rejected before execution
└──────────────────┘  agent/plan.py · agent/resolver.py
        │
        ▼
┌──────────────────┐  GraphQL v2 · cursor pagination · retry/backoff · complexity
│  MONDAY CLIENT   │  four read methods, no arbitrary-query entry point
└──────────────────┘  monday/client.py
        │  raw items
        ▼
┌──────────────────┐  column titles resolved at runtime → canonical fields
│  NORMALIZATION   │  every field → Field{value, raw, state}
└──────────────────┘  ingest/ — state ∈ OK · MISSING · MALFORMED · UNMAPPED ·
        │                              AMBIGUOUS · INFERRED · NOT_APPLICABLE
        ▼
  Deal[] / WorkOrder[] ──▶ ┌──────────────┐
        │   (+ as-of date) │ QUALITY AUDIT│ quality/audit.py
        ▼                  └──────────────┘
┌──────────────────┐  18 metrics, each declaring source fields, aggregation
│ ANALYTICS ENGINE │  and its missing-data rule
└──────────────────┘  analytics/metrics.py · engine.py · provenance.py
        │
        ▼
  AnalysisResult { metrics · assumptions · data-quality caveats · provenance }
        │
        ├──────────────────────────▶ "Analysis" panel — rendered verbatim, no LLM
        ▼
┌──────────────────┐  LLM #2 — receives ONLY this struct. No rows. No arithmetic.
│    RESPONDER     │  agent/responder.py (deterministic renderer as fallback)
└──────────────────┘
```

### Component responsibilities

| Module | Owns |
|---|---|
| `monday/client.py` | GraphQL transport, pagination, retries, read-only guard |
| `monday/errors.py` | Typed failures, each with a user-safe message |
| `ingest/mapping.py` | Column title → canonical field; controlled vocabularies |
| `ingest/normalize.py` | Dates, currency, quantities-with-units, categorical values |
| `ingest/entities.py` | `Field` state model, `Deal`, `WorkOrder`, `Dataset`, stage ladder |
| `ingest/builder.py` | Raw monday items → canonical records; header-echo and duplicate detection |
| `quality/audit.py` | 14 data-quality checks, all data-derived |
| `analytics/timeframe.py` | Fiscal/calendar periods, as-of anchoring, empty-period substitution |
| `analytics/metrics.py` | The metric registry — the only place numbers are produced |
| `analytics/provenance.py` | Records used/excluded, source fields, assumptions |
| `analytics/engine.py` | `QueryPlan` → `AnalysisResult` |
| `agent/` | Intent extraction, semantic resolution, plan validation, response |
| `app.py` | Streamlit chat, data-quality panel, analysis panel |

---

## Tech stack, and why

| Choice | Reason |
|---|---|
| **Python 3.12** | One pinned version. Streamlit Community Cloud's default. |
| **Streamlit** | Fastest credible path to a hosted, no-setup prototype. The core package has zero Streamlit imports, so it stays unit-testable and the UI is disposable. |
| **monday GraphQL API v2** (not MCP) | See Decision Log. Short version: monday's MCP exposes 60+ tools including writers, has no read-only mode, is model-mediated (unpredictable call counts against a finite daily budget), and would need MCP session management inside a hosted web app. |
| **Google Gemini** (Flash class) | Free tier, no card. A Claude Pro *subscription* cannot authenticate a hosted server app — Anthropic's docs direct shared automation to a Platform API key. Provider sits behind a one-file adapter; `LLM_PROVIDER=anthropic` switches it. |
| **httpx + pydantic** | HTTP with a mockable transport; schema validation at the LLM boundary. |
| **uv** | Fast, lockfile-based, reproducible. |

No queues, no vector database, no agent framework. Nothing in the requirements
justifies them.

---

## Setup

Full walkthrough: **[`setup/MONDAY_SETUP.md`](setup/MONDAY_SETUP.md)**. In brief:

1. **Start monday.com's 14-day Pro trial before importing.** The Free plan caps an
   account at **200 items total** and this dataset is 522 — a Free import silently
   truncates.
2. Import `setup/import/monday_deals.csv` as a board named **`Deals`** (346 rows) and
   `setup/import/monday_work_orders.csv` as **`Work Orders`** (176 rows). **Ensure the
   first row is mapped as the header row** — if it is not, monday names each column
   after one of its values and the agent will (correctly) refuse to analyse the board.
3. Do not clean anything up in monday. The mess is the exercise.
4. Get a monday API token (avatar → Developers → My access tokens) and a free Gemini
   key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```bash
git clone <repo> && cd skylark_drones
uv sync --extra dev
cp .env.example .env      # fill in MONDAY_API_TOKEN and GEMINI_API_KEY

uv run python -m skylark_bi.verify   # reads only; prints row counts and as-of date
uv run streamlit run app.py
```

### Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `MONDAY_API_TOKEN` | ✅ | — | Personal API token |
| `MONDAY_DEALS_BOARD_ID` | | auto | Blank ⇒ discovered by board name |
| `MONDAY_WORK_ORDERS_BOARD_ID` | | auto | " |
| `MONDAY_DEALS_BOARD_NAME` | | `Deals` | Used only for discovery |
| `MONDAY_WORK_ORDERS_BOARD_NAME` | | `Work Orders` | " |
| `MONDAY_API_VERSION` | | `2026-07` | Pinned; monday rotates quarterly |
| `LLM_PROVIDER` | | `gemini` | or `anthropic` |
| `GEMINI_API_KEY` | ✅¹ | — | Free tier is sufficient |
| `GEMINI_MODEL` | | `gemini-flash-lite-latest` | |
| `ANTHROPIC_API_KEY` | ¹ | — | Platform key, **not** a Pro subscription |
| `DATA_TTL_SECONDS` | | `900` | Reuse window before re-querying monday |
| `FISCAL_YEAR_START_MONTH` | | `4` | April — Indian FY, evidenced by `SDPL/FY25-26` invoice numbers |

¹ One LLM key is needed. Without it the app still runs and still computes every
number; answers render in a plainer deterministic format.

### Deployment (Streamlit Community Cloud)

Point it at this repo with `app.py` as the entry point and Python 3.12, then add
`MONDAY_API_TOKEN` and `GEMINI_API_KEY` under **Secrets**. Config is read from the
environment first and Streamlit secrets second, so the same code runs both places.

---

## Example questions

| Question | What it demonstrates |
|---|---|
| *How's our pipeline looking for the energy sector this quarter?* | Sector inference **and** empty-period substitution, both disclosed |
| *Which sectors have the strongest pipeline?* | Grouped aggregation with coverage reporting |
| *What is our current weighted pipeline?* | A metric labelled a heuristic, with its weights stated |
| *How many active work orders do we have?* | Work Orders board, explicit definition of "active" |
| *Compare our sales pipeline with operational workload.* | Cross-board analysis **without** a fabricated join |
| *What are the biggest risks in our current pipeline?* | Concentration analysis |
| *Give me a leadership update.* | Multi-metric brief from the same registry |
| *What data quality issues should I know about?* | The full audit, in business language |
| *How many people work in our Bangalore office?* | Declines cleanly — out of scope |
| *Pipeline for fintech?* | Asks a clarifying question instead of guessing |

---

## How messy data is handled

Every canonical field carries `{value, raw, state}`. **A blank is never zero.**

| State | Meaning |
|---|---|
| `OK` | Parsed cleanly from a populated value |
| `MISSING` | Source was blank — absent, not zero |
| `NOT_APPLICABLE` | Source explicitly said `NA` — a different claim from blank |
| `MALFORMED` | Present but unparseable (`"Rate based on MW slabs"` in a quantity column) |
| `UNMAPPED` | Outside the known vocabulary — kept, never snapped to a neighbour |
| `AMBIGUOUS` | Real but underdetermined (`"Dec"` with no year) |
| `INFERRED` | Derived by us, not observed — never interchangeable with `OK` |

The audit surfaces 14 findings from the live data, including:

- **`Deal Status` contradicts `Deal Stage` on 120 records** — 70 marked `Won` while at
  stage `A. Lead Generated`, all created the same day with no value, probability or
  dates. Stage is treated as authoritative; the conflict is reported, not resolved.
- **`Closure Probability` leaks the outcome** — on closed deals, `High` = 100% won.
  A forecast does not behave that way. It is therefore never used to calibrate weights.
- **The boards share no reliable join key** — client codes overlap on *zero* records
  (two independent masked namespaces), and deal names repeat. Cross-board questions
  compare aggregates; a row-level join would invent relationships.
- **Blank and zero used interchangeably** for the same billing fact.
- **The data ends ~Jan 2026** — so relative periods resolve against the data's own
  as-of date, computed live from observed events (never forecast dates).

---

## Testing

```bash
uv run pytest            # 135 tests
```

| Suite | Covers |
|---|---|
| `test_normalize.py` | Parsers, on the exact messy values in the source |
| `test_monday_client.py` | Pagination, auth, **HTTP 200 + `errors` array**, rate limits, malformed responses |
| `test_analytics.py` | Metrics reconciled against independent sums; fiscal periods; substitution |
| `test_quality.py` | Findings are *discovered*, and do not fire on data lacking the property |
| `test_resilience.py` | Renamed/missing/extra/duplicate columns, empty boards, unknown values |
| `test_security.py` | Read-only enforcement, prompt injection, LLM containment, secret leakage |
| `test_no_hardcoded_data.py` | Mechanical proof no business data is embedded |

---

## Read-only

The assignment requires read-only access. Stated precisely:

monday **personal API tokens cannot be scoped per application** — they inherit the
owning user's permissions. So read-only here is **application-level enforcement and
defence-in-depth, not a server-side security boundary.** Three layers:

1. **Structural (primary).** `MondayClient` exposes four read methods. There is no
   public `execute(query)`, so no caller — and no LLM upstream of one — can choose the
   GraphQL that gets sent.
2. **No mutation vocabulary.** A test asserts no monday mutation name appears anywhere
   in the package.
3. **Tripwire.** Every outgoing document is checked and rejected unless it is a named
   `query`. Board IDs travel as GraphQL variables; query text is never interpolated.

For a genuine server-side guarantee, use a view-only monday user's token, or an OAuth
app scoped to `boards:read`. Both are noted in the Decision Log as future work.

---

## Data provenance

The UI always shows where data came from:

- `LIVE · fetched <timestamp> · Deals board <id> (N records) · data as of <date>`
- `STALE — monday.com is unavailable. Showing the last successful fetch from <ts>`

The stale path deserves a note, because it sits close to something the assignment
forbids. The snapshot is written **only** from a successful live monday.com response,
at runtime, to a gitignored path. It is never committed and never seeds a first run —
if monday.com has never answered, there is nothing to serve and the app says so. Tests
enforce all of that (`test_no_hardcoded_data.py::TestSnapshotIsNotCommitted`).

`setup/import/*.csv` are one-time import artifacts. The application never reads them,
contains no CSV parsing at all, and a test enforces both. They are the masked sample
data supplied with the assignment — deal names are cartoon characters, companies and
owners are codes, and values are scaled — and they are kept here only so the monday.com
boards can be reproduced and so the test suite can exercise the parsers against the
real inconsistencies rather than invented ones.

---

## Known limitations

- **Cross-board analysis is aggregate-only.** The data supports nothing finer. Stated
  in every cross-board answer rather than hidden.
- **Weighted pipeline is a declared heuristic** (High/Medium/Low → 0.8/0.5/0.2), never
  a forecast. The dataset cannot calibrate real weights — see Decision Log.
- **No conversational memory.** Each question is answered independently; follow-ups
  like "and for mining?" are not resolved against the previous turn.
- **Period filtering uses one date per board** — deal creation date, work order PO
  date. Records missing that date are excluded from period filters and counted.
- **`I. POC` is classified as parked**, neither won nor lost. Documented as a judgement
  call; too few records to decide.
- **No charts.** Deliberate — intelligence over decoration within the time budget.
- **monday Free-plan item cap (200)** truncates this dataset. The hosted prototype runs
  on a Pro trial; see the Decision Log for what happens when it lapses.
