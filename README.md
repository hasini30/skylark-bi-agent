# Founder BI Assistant — monday.com

A conversational BI agent over two monday.com boards. Ask a founder-level
question, get an answer with the rows behind it and a caveat naming what it
excludes.

Real business data is messy in ways you can't anticipate, so the app **profiles
the boards before answering anything**: it measures every column, infers its
actual type, assigns each column a business role, generates its own
documentation, and records what's wrong. That profile is what the agent queries
against, and it's all visible in the **Data Review** tab.

- [Live app](#live-app) · [monday.com setup](#mondaycom-setup)
- [Architecture](#architecture) — [data review](#the-data-review) · [models](#models)
- [Leadership updates](#leadership-updates) · [Decision log](#decision-log)
- [Limitations](#limitations) · [Layout](#layout)

---

## Live app

**→ [Open the hosted app](https://monday-data-agent.streamlit.app/)**

Nothing to install. Ask a question in the chat, or open the Data Review tab to
see what the app worked out about the data. It runs against a demo pair of
boards — to reproduce it on your own account, see
[monday.com setup](#mondaycom-setup).

The first request after a quiet period takes about fifteen seconds while the
app wakes up and re-reads the boards.

---

## monday.com setup

The hosted link runs against a fixed pair of boards. To reproduce the setup on
your own account:

**1. Create the boards.** Import each spreadsheet into monday.com as its own
board — one for Work Orders, one for Deals. Let the importer pick column types;
the app verifies them against the actual values rather than trusting them, so
you don't need to get this right.

**2. Find the board IDs.** They're in each board's web address:
`https://your-domain.monday.com/boards/1234567890` → `1234567890`.

**3. Generate an API token.** Profile → Developers → Developer Center → Access
Tokens. **A read-only token is enough** — the app cannot write to monday.

**4. Get an OpenRouter key** from [openrouter.ai](https://openrouter.ai) for
model access.

**5. Supply the values.** As a `.env` file next to the code, or through App
settings → Secrets (in TOML form) on a Streamlit deployment:

```env
MONDAY_API_TOKEN=...
MONDAY_WORK_ORDERS_BOARD_ID=1234567890
MONDAY_DEALS_BOARD_ID=0987654321
OPENROUTER_API_KEY=...
MODEL_CHAIN=z-ai/glm-5.3-flash,openai/gpt-5.6-sol,z-ai/glm-5.2:free
```

**6. Run it.**

```bash
pip install -r requirements.txt
streamlit run app.py
```

No code changes are needed for different boards — the app profiles whatever
columns it finds and knows nothing about ours. If the board IDs are missing it
lists the boards your token can see, so you can find them.

`MODEL_CHAIN` is an ordered fallback list handled server-side by OpenRouter: if
the first model is rate-limited, down, or refuses, the next is used. Switching
models — or between free and paid — is this one line. See [Models](#models).

Column interpretation runs at **low reasoning effort** while the chat agent runs
at maximum. Measured on the same model: naming columns is twice as fast at low
effort with identical output, while the agent falls from 4 of 4 correct figures
to 1 of 4 when its effort is reduced. Choosing which of eight money columns to
sum needs deliberation; naming one does not.

`INTERPRET_MODEL` is optional. The one-off column interpretation is a far simpler
task than multi-round SQL, so it can run on a cheaper, faster model; unset, it
uses `MODEL_CHAIN`.

---

## Architecture

```
monday.com  ──►  profile  ──►  DuckDB   ──►  agent   ──►  answer
 (read-only)     measure       typed        tools       + caveat
                 type          tables       + SQL       + source rows
                 describe
                 check
```

Python, Streamlit for the UI, in-memory DuckDB as the query engine, OpenRouter
for model access.

**Why a local analytical database.** monday's GraphQL API returns every cell as
loosely-typed text with weak filtering and no aggregation. Loading both boards
into DuckDB gives the agent real SQL over typed columns, so its effort goes into
the question rather than into string parsing. Both boards are fetched from the
API at runtime — nothing is read from a checked-in file.

**Writes are structurally impossible.** Every monday call goes through a single
gateway that inspects the GraphQL operation and rejects anything that isn't a
query. Read-only is enforced by the code, not by convention — an edit that
introduced a mutation would fail immediately.

### The data review

monday boards carry no column documentation, so an agent would see bare column
names — no way to choose between several similarly-named money columns, or
between a forecast date and an actual one. So the app profiles both boards
before answering anything, and the agent's system prompt is built from the
result.

It runs on first load, cached against a hash of the board's structure, so
different boards are profiled automatically.

**Per column it derives:** the real storage and semantic type — verified against
the values rather than trusted from monday's declaration — fill and non-zero
rates, a business role (which column *is* revenue, the close date, the client), a
written description, and any quality problems.

**The checks go past null counting:** zeros used as absence markers, the same gap
encoded two ways across paired columns, spreadsheet headers that monday
registered as valid labels, columns failing their declared type, units glued to
numbers, and vocabularies that diverge between the boards.

**It works out whether the boards can be joined at all** — and often they can't.
A column shared by both may still repeat on each side, so joining on it multiplies
rows and inflates every total with no error to warn you. The app measures whether
a candidate key is unique on each side, reports the row fan-out an inner join
would cause, and where it isn't safe tells the agent to aggregate each board
separately and compare on a shared dimension instead.

**And which columns only look related.** Two independently anonymised code
schemes can both run 001, 002, 003 without a single code meaning the same entity.
Where two identifier columns look confusable but share no values, the app states
that explicitly, and the agent is forbidden from reshaping one column to make it
match another.

**Everything it finds is reported live in the Data Review tab.**

**Nulls get a verdict, not just a count.** One percentage misleads in both
directions — a deal that was never won has no close date, which is correct,
while the same absence on a *won* deal is a hole in the record. So each null rate
is broken down by record status and judged per group. Forecast fields (a
predicted date, a probability) are recognised as *supposed* to empty out once a
record closes. This regularly inverts the picture: a column that looks badly
incomplete overall is often near-complete on exactly the subset being asked
about.

**Severity is proportional** to how widespread a problem is, and nothing is
discarded — unparseable values are kept and flagged.

**Caveats describe the rows a query scanned, not the row it returned.** An
aggregate comes back as a single row with nothing null in it, so checking the
result would report no problems at all — precisely when the warning matters most.

### Models

All model access goes through OpenRouter, so the chain is a config value rather
than a code change. Primary `z-ai/glm-5.3-flash`, falling back to
`openai/gpt-5.6-sol` — a different vendor, so one provider's bad hour doesn't
take out both — then a free model as a floor. Roughly $0.003 a question.

---

## Leadership updates

The brief asks the agent to *help prepare data* for leadership updates. That puts
you in charge — you write the update, you need figures to drop into slides — so
the agent assembles tables against a stated requirement rather than producing a
fixed report, which would just be another dashboard.

```
/leadership-update board meeting Thursday, cover energy performance and why Q3 delivery slipped
```

Natural phrasing works too. It inspects the data, then returns a **plan** —
tables as plain-English questions, plus a clarifying question if two readings
would give materially different numbers — and **stops for your approval**. That
pause is enforced in the loop, not requested in the prompt, so it can't build six
tables you didn't want.

You confirm or amend (*"yes, drop 5, break 1 down by owner"*), and it builds them
one at a time, reading each result before composing the next query. Each table
carries its data, a measured caveat, source rows and a CSV export.

The plan holds **questions, not SQL** — you're confirming intent, and a founder
can't audit a query.

---

## Decision log

### What I assumed

- **"Revenue" isn't one column.** Work Orders carries several money columns —
  what was contracted, what's been invoiced, what's been collected, what's still
  owed, each with and without tax. The agent picks the contracted, pre-tax
  figure by default, and names the column it used in every answer rather than
  asking you to choose.
- **Dates come from whichever field is actually filled in.** Where a board's
  "actual close date" is mostly empty, the agent uses the creation date for
  timing and says so, rather than answering from a handful of records.
- **The boards are joined on whichever columns genuinely overlap.** Codes on the
  two boards use different schemes and share nothing, so the app measures the
  values and picks the pair that actually matches — reporting how many records
  fall outside it.
- **Boards are small enough to hold in memory** — a few thousand records.
- **Whoever reviews this may set their boards up differently**, so nothing
  depends on our particular column names or IDs. The app re-derives everything.

### Trade-offs

| Choice | Why | What it costs |
|---|---|---|
| Local analytical database over querying monday live | typed columns and real SQL; monday returns loosely-typed text with weak filtering and no aggregation | a sync step; data is as fresh as the last refresh |
| Deterministic rules mutate data, never the model | the same question returns the same number every time | no inference on genuinely ambiguous values |
| No model-driven anomaly detection | no ground truth to check against, so it returns findings whether or not any exist. One false positive costs more trust than ten true nulls earn | a genuine outlier goes unflagged unless a rule catches it |
| Roles resolved by rules, not a model call | reproducible, and free | blunter than model inference |
| Cache the profile in memory, regenerate on cold start | a rebuild costs a few seconds and a fraction of a cent; an external store means another dependency and credential | rebuilt after the app sleeps |
| Streamlit over a hand-built UI | the brief's bar on the interface is functional, not aesthetic | it looks like Streamlit |

**What makes caching safe:** measured facts are never cached and recompute every
sync, so no figure goes stale. Only descriptions and role assignments persist,
invalidated when the structure hash changes or fill rates drift more than 20
points.

### Tech stack

**DuckDB** needs no setup and is built for analytical queries over dataframes.
**Pydantic** validates tool arguments before they execute, and generates the tool
schemas from the same declaration. monday is reached through its **GraphQL API**
rather than the off-the-shelf MCP connector, for control over pagination,
retries, and reading board schema live.

### How I read "leadership updates"

The brief says the agent should *help prepare data* for leadership updates, and
leaves the interpretation open. I read the emphasis as being on **help** and on
**data**.

A fixed monthly report would be a dashboard, and the founder in the problem
statement is already surrounded by those — the complaint is not a shortage of
numbers, it is the work of assembling the right ones for a specific conversation.
So the agent does not produce an update. **You state what the update has to
cover, and it assembles the tables for it.**

Three consequences follow:

**It proposes before it builds.** The agent returns a plan — each table written
as a plain-English question — and stops for approval. That pause is enforced in
the loop rather than requested in the prompt, so it cannot spend your time on six
tables you did not want. It is also where a clarifying question belongs: better
to ask once what "delivery slipped" means than to answer it wrongly six ways.

**The plan holds questions, not SQL.** You are confirming intent, and nobody
should have to read a query to do that.

**The output is tables, not prose.** You are writing the update; the agent has no
idea what argument you are making. What it can do is hand you figures you can
paste into a slide, each with its caveat and the rows behind it, so you can
defend them in the room.

---

### With more time

**Latency.** The agent's loop involves multiple synchronous model calls, each waiting for the previous to finish. Optimising the query loop or adopting faster inference endpoints would reduce the wait time between asking a question and getting an answer.

**Persist the conversation and saved update plans.** Chat history lives in
session state, so a reload loses it and every leadership update is composed from
nothing. Storing conversations, and the table plans behind an update, would make
"the same update as last month" a one-click job — and make the figures directly
comparable month to month, which is the point of a recurring update.

**Charts.** A funnel or a quarter-on-quarter trend reads better as a picture than
a table, and a leadership update usually needs at least one.

**Persist the review, and let the user override it.** Two related gaps. The
review is cached in memory and rebuilt whenever the host restarts — moving it to
an object store, keyed by the same structure hash it already uses, would make it
computed once rather than once per cold start. And its conclusions are read-only:
when it picks the wrong column for revenue there is no way to say so. Overrides
stored alongside the review would let it be corrected once and stay corrected.

**An evaluation set for the agent.** The deterministic layer has tests; the
agent has none, because its output is prose. Twenty questions with verified
answers, checked on every change, would catch a model upgrade or a prompt edit
quietly degrading accuracy — the failure mode nothing currently detects.

---

## Limitations

**The profile rebuilds after the app sleeps.** Streamlit Community Cloud idles
the app out and clears its filesystem, so the first request afterwards waits
about fifteen seconds. A rebuilt description may be worded slightly differently;
the figures and role assignments don't change.

**Two boards.** The profile adapts to any column structure, but the app itself is
wired for two boards.

---

## Layout

| File | What it does |
|---|---|
| `app.py` | the interface — Chat and Data Review tabs |
| `agent.py` | how the agent thinks, and what it's told |
| `board_review.py` | studies the boards: measure, type, describe, check |
| `interpret.py` | the one model pass that names columns and assigns roles |
| `monday_api.py` | the read-only connection to monday |
| `test_review.py` | tests for the deterministic layer |

The deterministic half — type inference, quality checks, join safety, null
verdicts — is covered by tests, because it is the half that must never change
its answer:

```bash
python -m pytest test_review.py -q
```
