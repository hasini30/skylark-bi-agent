"""The BI agent: tool loop, prompt, and measured caveats."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import duckdb
from openai import OpenAI
from pydantic import BaseModel, Field, TypeAdapter

from board_review import quote_ident

DEFAULT_MODEL_CHAIN = "z-ai/glm-5.3-flash,openai/gpt-5.6-sol,z-ai/glm-5.2:free"
MAX_LOOPS = 12
MAX_HISTORY_MESSAGES = 24
ROW_LIMIT = 100


class ReportTable(BaseModel):
    title: str = Field(description="Short title for the table.")
    question: str = Field(description="The question this table answers, in plain English.")
    boards_needed: List[str] = Field(description="Which tables this needs.")


class LeadershipUpdatePlan(BaseModel):
    framing_note: str = Field(description="One short paragraph framing the plan. "
                                          "Include a clarifying question here if one is needed.")
    tables: List[ReportTable]


class SQLArgs(BaseModel):
    query: str


class InspectArgs(BaseModel):
    table_name: str
    column_name: str


def _plan_schema() -> Dict[str, Any]:
    """Inline $defs — several providers handle nested $ref poorly in tools."""
    schema = LeadershipUpdatePlan.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].split("/")[-1]])
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return inline(schema)


class BIAgent:
    def __init__(self, db_conn: duckdb.DuckDBPyConnection, workspace=None):
        self.db_conn = db_conn
        self.workspace = workspace
        self.executed: List[Dict[str, Any]] = []
        self.sql_error_count = 0
        self.status_container = None
        self.last_model: Optional[str] = None

        self.model_chain = [m.strip() for m in
                            os.environ.get("MODEL_CHAIN", DEFAULT_MODEL_CHAIN).split(",")
                            if m.strip()]

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set in .env")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=api_key, timeout=60.0)

        self.tools = self._build_tools()
        self.messages = [{"role": "system", "content": self._system_prompt()}]

    # ------------------------------------------------------------------ setup

    def _column_index(self) -> Dict[str, Any]:
        """Every known column, for matching against query text."""
        index = {}
        if self.workspace:
            for board in self.workspace.boards:
                for name, profile in board.profiles.items():
                    index[name.lower()] = (board, name, profile)
        return index

    def _build_tools(self) -> List[Dict[str, Any]]:
        return [
            {"type": "function", "function": {
                "name": "propose_leadership_update",
                "description": ("Propose a multi-table plan for a leadership update and "
                                "stop for the user's approval. Use for broad report "
                                "requests, never for a single question."),
                "parameters": _plan_schema()}},
            {"type": "function", "function": {
                "name": "execute_sql",
                "description": "Run a read-only SELECT against the DuckDB database.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "The SELECT query."}},
                    "required": ["query"]}}},
            {"type": "function", "function": {
                "name": "inspect_column",
                "description": "Distinct values, counts and fill rate for one column.",
                "parameters": {"type": "object", "properties": {
                    "table_name": {"type": "string"},
                    "column_name": {"type": "string"}},
                    "required": ["table_name", "column_name"]}}},
            {"type": "function", "function": {
                "name": "get_quality_report",
                "description": "Summary of data quality findings from the review.",
                "parameters": {"type": "object", "properties": {}}}},
        ]

    def _system_prompt(self) -> str:
        from board_review import agent_context
        from datetime import date
        context = agent_context(self.workspace) if self.workspace else "(no review available)"
        today = date.today()
        quarter = (today.month - 1) // 3 + 1
        return f"""You are a business intelligence agent for a founder.

TODAY IS {today.isoformat()} — calendar Q{quarter} {today.year}.
Each date column's actual range is given in the schema below. If the data does
not reach the period the user asked about, say so and name the period you used
instead. Never silently substitute one period for another. You answer
questions by querying monday.com data held in an in-memory DuckDB database.

The schema below comes from an automated review of the boards. Fill rates,
warnings and descriptions are measured from the actual data — trust them.

{context}

WRITING SQL
- Quote every column name in double quotes. Many contain spaces and brackets.
- Columns marked CAST NEEDED are stored as text; use the cast given in their
  description before doing arithmetic.
- Some rows are spreadsheet headers imported as data — a row whose category
  equals the column's own name. Exclude them.
- Join the boards ONLY if the schema above states a join is safe, and report how
  many rows it drops. If it says not to join, aggregate each table separately
  and line the results up on a shared dimension.

ANSWERING — a number on its own is not an answer
Every number must come from a query. Never state a figure you did not retrieve.

A founder can already see totals. What they cannot see is what the total means,
so every answer carries four things:

1. THE NUMBER — the figure they asked for.
2. THE COMPARISON — against the prior period, or against the rest of the
   population (other sectors, other stages, the average). A figure with nothing
   beside it cannot be judged. Run the second query to get it.
3. THE DRIVER — what is actually moving it. The largest contributor, the deal
   or client behind the change, the stage where value is stuck. Name it.
4. WHAT TO WATCH — the one thing that follows from this. A concentration risk,
   a slipping deal, a stage that is not clearing.

DATA QUALITY AND CAVEATS
The facts returned with the query result describe what you scanned. Apply these rules strictly:

1. Conclusions, not just counts: A caveat must state what the gap does to THIS answer. Do not just quote a flat percentage. Use the conditional-null breakdown (e.g. "The gap is concentrated in won deals - 61% of them have no value - so pipeline figures here are reliable and won-revenue figures are not.").
2. Refuse near-empty columns: If a question depends on a column above 90% empty, DO NOT produce a number. State that the field isn't tracked and name the nearest thing that is.
3. Position by severity: If the answer rests on a column between 50% and 90% empty, state the coverage BEFORE the number. For columns less than 50% empty, put the caveat at the end of your answer.

NEVER INVENT A LINK BETWEEN TABLES
If two columns do not share values exactly as they are stored, they do not refer
to the same things. Stripping a prefix, changing case, reformatting or renumbering
one column so it matches another is FORBIDDEN — two independently anonymised code
schemes can run 001, 002, 003 without a single code meaning the same entity.

If a question needs a link the data does not have, say the data cannot answer it
and offer the nearest thing it can: aggregate each table separately and compare on
a dimension they genuinely share. Never present a manufactured match as a result.

BREAKING A TOTAL DOWN — the parts must belong to the whole
When you split a figure into components — by stage, sector, status, owner, month
— every component query must carry THE SAME filters as the total. A breakdown of
open pipeline filters to open records in every part, not only in the headline.
Stage and status are different columns: filtering by stage alone silently
includes won and dead records.

Before you report a breakdown, add the parts up. If they exceed the total you
just stated, you have mixed filters — rewrite the queries rather than presenting
them. If the parts legitimately fall short (nulls excluded, a residual "other"),
say so in a clause. Never present components that contradict the total they sit
under.

Keep it short — four or five sentences beats a report. Do not pad with
restatements of the question, and do not editorialise beyond what the data
supports. If a comparison is not available in the data, say so in a clause
rather than skipping the point.

WHEN TO ASK A CLARIFYING QUESTION
Ask at most one, and only about what the user meant.

ASK when two readings would give materially different numbers:
- The period is ambiguous — "this quarter" could be calendar or fiscal.
- The measure is ambiguous — "revenue" here could mean contracted value,
  invoiced value, or cash collected. These are different columns and different
  numbers.
- The population is ambiguous — a plural noun may mean the open subset or all records.
Ask once, in one short sentence, then proceed on a stated assumption if the
user does not narrow it.

NEVER ASK about the data itself. Which column is more reliable, whether to
trust a field, how to handle blanks — the user would have to go and open
monday.com to answer, which is the work this tool exists to remove. Decide it
yourself from the fill rates and warnings above, then say which column you used
and why, in one clause. For example: "using contracted value, since invoiced
value is blank on 63 of 176 work orders".

LEADERSHIP UPDATES
For broad report requests — "prepare an update", "numbers for the board",
"put together something for the leadership review" — call
propose_leadership_update with the questions each table will answer, in plain
English. Do not write SQL at that point. Put any clarifying question in the
framing note. After the user approves, build the tables one at a time with
execute_sql, checking each result before writing the next query.
Do NOT use this tool for a single question such as "what is our energy pipeline".
"""

    # ------------------------------------------------------------------ tools

    def _note(self, text: str) -> None:
        if self.status_container:
            self.status_container.write(text)

    def _caveat_facts(self, query: str, df) -> str:
        """Facts about the rows the query SCANNED, not the row it returned.

        An aggregate returns one row with no nulls, so measuring the result
        says nothing. The profile of the columns referenced is what matters.
        """
        facts: List[str] = []
        index = self._column_index()
        lowered = query.lower()

        referenced = [entry for key, entry in index.items()
                      if key in lowered and len(key) > 2]

        for _board, name, profile in referenced[:8]:
            if profile.null_pct >= 0.2:
                facts.append(f'- "{name}" is empty on {profile.nulls} of '
                             f'{profile.rows} source rows ({profile.null_pct:.0%}).')
            if getattr(profile, "zero_pct", 0) >= 0.2:
                facts.append(f'- "{name}" is zero on {profile.zeros} of '
                             f'{profile.rows} source rows; only {profile.real} '
                             "carry a real figure.")

        if self.workspace:
            names = {n for _b, n, _p in referenced}
            for finding in self.workspace.all_findings:
                if finding["severity"] in ("Critical", "High") and \
                        any(n in str(finding["column"]) for n in names):
                    facts.append(f'- {finding["issue_type"]} on '
                                 f'{finding["column"]}: {finding["description"]}')

        if not facts:
            return ""
        return ("\n\nMEASURED FACTS about the source data behind this result "
                "(use them to write an accurate caveat; do not invent others):\n"
                + "\n".join(dict.fromkeys(facts)))

    def execute_sql(self, query: str) -> str:
        self._note("Running a query...")
        forbidden = ("insert ", "update ", "delete ", "drop ", "create ", "alter ",
                     "attach ", "detach ", "install ", "load ", "copy ",
                     "read_csv", "read_parquet", "read_json")
        if any(k in query.lower() for k in forbidden):
            return "Error: only SELECT queries are allowed."
        try:
            df = self.db_conn.execute(query).df()
        except Exception as exc:
            self.sql_error_count += 1
            self._note(f"Query failed (attempt {self.sql_error_count}/3)")
            if self.sql_error_count >= 3:
                return f"FATAL: query failed three times. Stop and explain. Last error: {exc}"
            return (f"Error: {exc}\n\nFix the SQL using the schema in your instructions "
                    "— check quoting and column names — then call execute_sql again.")

        self.sql_error_count = 0
        self.executed.append({"query": query, "df": df.copy()})
        self._note(f"Query returned {len(df)} rows")

        body = df.head(ROW_LIMIT).to_json(orient="records")
        if len(df) > ROW_LIMIT:
            body += f"\n\n(showing {ROW_LIMIT} of {len(df)} rows)"
        return body + self._caveat_facts(query, df)

    def inspect_column(self, table_name: str, column_name: str) -> str:
        self._note(f"Inspecting {column_name}...")
        col, table = quote_ident(column_name), quote_ident(table_name)
        try:
            total = self.db_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            filled = self.db_conn.execute(f"SELECT COUNT({col}) FROM {table}").fetchone()[0]
            top = self.db_conn.execute(
                f"SELECT {col} AS value, COUNT(*) AS n FROM {table} "
                f"GROUP BY 1 ORDER BY n DESC LIMIT 15").df()
            rare = self.db_conn.execute(
                f"SELECT {col} AS value, COUNT(*) AS n FROM {table} "
                f"WHERE {col} IS NOT NULL GROUP BY 1 ORDER BY n ASC LIMIT 10").df()
        except Exception as exc:
            return f"Error inspecting column: {exc}"

        rate = filled / total if total else 0
        return (f"{filled} of {total} rows populated ({rate:.0%}).\n\n"
                f"Most common:\n{top.to_string(index=False)}\n\n"
                f"Rarest (where inconsistencies hide):\n{rare.to_string(index=False)}")

    def get_quality_report(self) -> str:
        self._note("Reading the quality review...")
        if not self.workspace:
            return "No review available."
        rows = sorted(self.workspace.all_findings,
                      key=lambda f: {"Critical": 0, "High": 1, "Medium": 2,
                                     "Low": 3}.get(f["severity"], 4))
        return "\n".join(f'[{f["severity"]}] {f["board"]} . {f["column"]} — '
                         f'{f["issue_type"]}: {f["description"]}' for f in rows[:40])

    def run_tools(self, tool_calls) -> List[Dict[str, Any]]:
        results = []
        for call in tool_calls:
            name = call.function.name
            raw = call.function.arguments or "{}"
            if raw.strip().startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
                raw = re.sub(r"```$", "", raw).strip()
            try:
                args = json.loads(raw)
                if name == "execute_sql":
                    result = self.execute_sql(TypeAdapter(SQLArgs).validate_python(args).query)
                elif name == "inspect_column":
                    a = TypeAdapter(InspectArgs).validate_python(args)
                    result = self.inspect_column(a.table_name, a.column_name)
                elif name == "get_quality_report":
                    result = self.get_quality_report()
                else:
                    result = f"Unknown tool: {name}"
            except Exception as exc:
                result = (f"Your tool arguments were rejected: {exc}. "
                          "Return JSON matching the tool's parameter schema exactly.")
            results.append({"tool_call_id": call.id, "role": "tool",
                            "name": name, "content": result})
        return results

    # ------------------------------------------------------------------ loop

    def _trim_history(self) -> None:
        """Trim to a safe boundary: never cut a tool result away from its call."""
        if len(self.messages) <= MAX_HISTORY_MESSAGES:
            return
        head, tail = self.messages[0], self.messages[1:]
        cut = len(tail) - (MAX_HISTORY_MESSAGES - 1)
        while cut < len(tail) and tail[cut].get("role") == "tool":
            cut += 1
        self.messages = [head] + tail[cut:]

    def send_message(self, prompt: str, force_tool: Optional[str] = None):
        self.messages.append({"role": "user", "content": prompt})
        self._trim_history()

        rate_limited = 0
        loops = 0
        while loops < MAX_LOOPS:
            loops += 1
            kwargs: Dict[str, Any] = {"model": self.model_chain[0],
                                      "messages": self.messages,
                                      "tools": self.tools,
                                      "temperature": 0.0}
            if len(self.model_chain) > 1:
                kwargs["extra_body"] = {"models": self.model_chain}
            if force_tool and loops == 1:
                kwargs["tool_choice"] = {"type": "function",
                                         "function": {"name": force_tool}}
            try:
                response = self.client.chat.completions.create(**kwargs)
                rate_limited = 0
            except Exception as exc:
                if "429" in str(exc) or "rate" in str(exc).lower():
                    rate_limited += 1
                    if rate_limited >= 3:
                        return ("The model provider is rate limiting every request. "
                                "Wait a minute, or add credit to the OpenRouter account.")
                    self._note(f"Rate limited; waiting {5 * rate_limited}s...")
                    time.sleep(5 * rate_limited)
                    loops -= 1
                    continue
                raise

            self.last_model = getattr(response, "model", None)
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            calls = getattr(message, "tool_calls", None)
            if not calls:
                return message.content or ("The model returned nothing. "
                                           "Try rephrasing the question.")

            plan_call = next((c for c in calls
                              if c.function.name == "propose_leadership_update"), None)
            if plan_call:
                raw = plan_call.function.arguments or "{}"
                if raw.strip().startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
                    raw = re.sub(r"```$", "", raw).strip()
                try:
                    plan = LeadershipUpdatePlan.model_validate_json(raw).model_dump()
                except Exception as exc:
                    self.messages.append({"tool_call_id": plan_call.id, "role": "tool",
                                          "name": plan_call.function.name,
                                          "content": f"Plan rejected: {exc}. Retry."})
                    continue
                self.messages.append({"tool_call_id": plan_call.id, "role": "tool",
                                      "name": plan_call.function.name,
                                      "content": "Plan shown to the user, awaiting approval."})
                return {"type": "plan", "plan": plan}

            self.messages.extend(self.run_tools(calls))

        return ("I ran out of steps working this out. Try narrowing the question.")

    def update_connection(self, db_conn, workspace) -> None:
        """Point at refreshed data without losing the conversation."""
        self.db_conn = db_conn
        self.workspace = workspace
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = self._system_prompt()
