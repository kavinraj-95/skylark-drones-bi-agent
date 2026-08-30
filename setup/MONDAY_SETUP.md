# monday.com setup

One-time setup to create the two boards the agent reads. Takes about 15 minutes.

The two CSVs in `setup/import/` are import artifacts only — the application never
reads them. It queries monday.com over the GraphQL API at runtime.

---

## 1. Start a Pro trial

The **Free plan caps an account at 200 items in total**, and this dataset is 522
(346 deals + 176 work orders). A Free-plan import will silently truncate.

Every new monday.com account gets a **14-day Pro trial** with that cap lifted. If your
trial has not been used: go to **Administration → Billing** and start it before
importing. If it has already lapsed, tell me and I will switch the plan to a
documented ~200-item subset instead.

Note the date you start the trial — it goes in the README so an evaluator knows the
window in which the hosted link is guaranteed live.

---

## 2. Import the Deals board

1. In your workspace: **+ Add → Import data → CSV/Excel**.
2. Upload `setup/import/monday_deals.csv`.
3. Name the board exactly **`Deals`**.
4. On the column-mapping screen, leave monday's suggested types **as they are** and
   click through. The agent does not depend on monday's column types — it reads
   display strings and normalises them itself. (You will see `Deal Stage` land as
   free text rather than a Status column: monday only creates a Status column when a
   field has 9 or fewer distinct values, and this one has 17. That is expected and
   handled.)
5. Confirm the board shows **346 items**.

## 3. Import the Work Orders board

Same flow with `setup/import/monday_work_orders.csv`, named exactly
**`Work Orders`**. Confirm **176 items**.

> **Do not clean anything up in monday.** The messiness — blank cells, the two rows
> that repeat the column headers, `45days` in a quantity column — is the point of the
> exercise. The agent detects and reports all of it.

---

## 4. Create an API token

**Avatar (bottom-left) → Developers → My access tokens → Show / Copy.**

A note on read-only, because the assignment requires it: monday personal tokens
**cannot be scoped per application** — they inherit your own permissions. The app
enforces read-only itself (it exposes only read methods, has no arbitrary-query entry
point, and refuses any non-`query` GraphQL operation). That is application-level
defence-in-depth, not a server-side guarantee, and the README says so plainly.

If you want a genuine server-side guarantee later, the options are a monday user with
view-only board permission, or an OAuth app scoped to `boards:read`.

---

## 5. Configure the app

```bash
cp .env.example .env
```

Fill in:

```
MONDAY_API_TOKEN=<the token you just copied>
GEMINI_API_KEY=<from https://aistudio.google.com/apikey — free, no card>
```

Board IDs can stay blank: the app discovers them by name (`Deals`, `Work Orders`). To
pin them explicitly, open each board and copy the number from its URL
(`.../boards/1234567890`) into `MONDAY_DEALS_BOARD_ID` / `MONDAY_WORK_ORDERS_BOARD_ID`.

---

## 6. Verify the connection

```bash
.venv/bin/python -m skylark_bi.verify
```

This confirms the token works, resolves both board IDs, pages through every item, and
prints the row counts and the dataset's as-of date. It performs **reads only**.

Expected:

```
Deals        346 items
Work Orders  176 items
```

If the counts are short, the Free-plan item cap truncated the import — see step 1.
