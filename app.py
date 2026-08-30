"""Founder BI Assistant — chat over monday.com boards, with a visible data review."""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import board_review as br
import interpret
from agent import BIAgent
from monday_api import fetch_board_items, fetch_board_metadata, list_boards

load_dotenv()
st.set_page_config(page_title="Founder BI Assistant", layout="wide")

# Streamlit Community Cloud supplies configuration through its Secrets manager
# rather than a .env file. Top-level secrets usually reach os.environ, but not
# on every runtime version, so mirror them across explicitly.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:
    pass  # no secrets configured (local run)

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
BOARD_SPECS = [("work_orders", "MONDAY_WORK_ORDERS_BOARD_ID", "Work Orders"),
               ("deals", "MONDAY_DEALS_BOARD_ID", "Deals")]


class DataStore:
    """Fetches both boards, reviews them, and builds the query database.

    Held in a process-wide cache: one review shared by every visitor, rebuilt
    only on restart or explicit refresh. The two boards are fetched and
    interpreted concurrently — both are I/O bound, so the wall-clock cost is
    one board's, not two.
    """

    def __init__(self, progress=None):
        self.log = []
        self.conn = None
        self.workspace = None
        self.last_synced = None
        self.error = None
        self.available_boards = []
        self.refresh(progress)

    @property
    def findings(self):
        return self.workspace.all_findings if self.workspace else []

    def refresh(self, progress=None):
        self.log = []

        def say(msg):
            self.log.append(msg)
            if progress:
                progress(msg)
        try:
            missing = [env for _t, env, _l in BOARD_SPECS if not os.environ.get(env)]
            if missing:
                try:
                    self.available_boards = list_boards()
                except Exception:
                    self.available_boards = []
                self.error = ("Board IDs not configured: " + ", ".join(missing) +
                              ". Set them in .env — see the boards listed below.")
                return

            say("Fetching both boards from monday.com…")
            def pull(spec):
                table, env, label = spec
                board_id = os.environ[env]
                return table, label, board_id, fetch_board_metadata(board_id), \
                    fetch_board_items(board_id)

            with ThreadPoolExecutor(max_workers=len(BOARD_SPECS)) as pool:
                pulled = list(pool.map(pull, BOARD_SPECS))
            rows = sum(len(p[4]) for p in pulled)
            say(f"Fetched {rows} records. Profiling columns…")

            def review(p):
                table, label, board_id, metadata, items = p
                return br.review_board(
                    board_id, metadata, items, table, label=label,
                    interpret_with=interpret.read_board,
                    merge_reading=interpret.merge)

            say("Working out what each column means — this is the slow part…")
            with ThreadPoolExecutor(max_workers=len(pulled)) as pool:
                results = list(pool.map(review, pulled))

            reviews = [r for r, _f in results]
            frames = {p[0]: f for p, (_r, f) in zip(pulled, results)}

            say("Checking how the boards relate…")
            workspace = br.compare_all(reviews, frames)
            new_conn = br.build_database(frames, workspace.all_findings)

            if self.conn:
                self.conn.close()
            self.conn, self.workspace = new_conn, workspace
            self.last_synced = datetime.now().strftime("%H:%M:%S")
            self.error = None
            say(f"Ready — {len(workspace.all_findings)} data issues catalogued.")
        except Exception as exc:
            # Keep whatever was already loaded rather than dropping to a broken state.
            self.error = f"Sync failed: {exc}"

@st.cache_resource(show_spinner="Reviewing your monday.com boards — "
                               "about 15 seconds on the first load…")
def get_store():
    return DataStore()


st.title("Founder BI Assistant")
st.caption("Ask questions about your monday.com boards. "
           "Every answer shows the rows behind it and what it left out.")

store = get_store()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state and store.conn:
    st.session_state.agent = BIAgent(store.conn, workspace=store.workspace)

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("Founder BI")

    if store.error:
        st.error(store.error)
        if store.available_boards:
            st.caption("Boards this token can see:")
            st.dataframe(pd.DataFrame(store.available_boards),
                         use_container_width=True, hide_index=True)

    if st.button("Refresh data", type="primary", use_container_width=True):
        with st.status("Refetching from monday.com…", expanded=True) as _r:
            store.refresh(progress=_r.write)
            if store.error:
                _r.write(store.error)
                _r.update(label="Refresh failed", state="error", expanded=True)
            else:
                _r.update(label="Refreshed", state="complete", expanded=False)
            if not store.error and "agent" in st.session_state:
                st.session_state.agent.update_connection(store.conn, store.workspace)
                st.session_state.messages.append(
                    {"role": "assistant", "content": "_— data refreshed from monday.com —_",
                     "divider": True})

    if store.workspace:
        st.caption(f"Last synced {store.last_synced}")
        cols = st.columns(len(store.workspace.boards))
        for col, board in zip(cols, store.workspace.boards):
            col.metric(board.board_name, board.rows)

        counts = {}
        for finding in store.findings:
            counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
        serious = counts.get("Critical", 0) + counts.get("High", 0)
        if serious:
            st.warning(f"{serious} serious data issues — see the Data Review tab")
        st.caption(" · ".join(f"{k}: {v}" for k, v in
                              sorted(counts.items(), key=lambda kv: SEVERITY_ORDER.get(kv[0], 9))))

    st.divider()
    model_used = getattr(st.session_state.get("agent"), "last_model", None)
    st.caption(f"Model: {model_used or os.environ.get('MODEL_CHAIN', 'z-ai/glm-5.3-flash').split(',')[0]}")

# ------------------------------------------------------------------ main

if not store.conn:
    st.info("No data loaded. Check the configuration in the sidebar.")
    st.stop()

tab_chat, tab_review = st.tabs(["Chat", "Data Review"])


def render_plan(plan):
    st.markdown(plan.get("framing_note", ""))
    for i, table in enumerate(plan.get("tables", []), start=1):
        st.markdown(f"**{i}. {table['title']}** — {table['question']}  \n"
                    f"<span style='opacity:.6'>{', '.join(table.get('boards_needed', []))}</span>",
                    unsafe_allow_html=True)
    st.caption("Reply to approve, or say what to add, drop or change.")


def render_results(results, key_prefix):
    for i, item in enumerate(results):
        with st.expander(f"Data {i + 1} — {len(item['df'])} rows"):
            st.dataframe(item["df"], use_container_width=True)
            st.code(item["query"], language="sql")
            st.download_button("Download CSV",
                               item["df"].to_csv(index=False).encode("utf-8"),
                               file_name=f"table_{i + 1}.csv", mime="text/csv",
                               key=f"{key_prefix}_dl_{i}")


with tab_chat:
    # Streamlit renders in the order containers are DECLARED, not the order they
    # are filled. Reserving three slots up front lets the composer stay the last
    # element on the page while still being read before the turn is processed.
    history_box = st.container()
    working_box = st.container()
    input_box = st.container()

    with input_box:
        prompt = st.chat_input(
            "Ask a question, or /leadership-update <what the update must cover>")

    with history_box:
        if not st.session_state.messages:
            counts = ", ".join(f"{b.rows} {b.board_name.lower()}"
                               for b in store.workspace.boards) if store.workspace else ""
            st.markdown(
                f"<div style='padding:1.5rem 0 .5rem 0;opacity:.75'>"
                f"<p style='margin:0 0 .75rem 0'>Reading <b>{counts}</b>. Everything "
                f"below is derived from the data — see the <b>Data Review</b> tab for "
                f"what was found.</p>"
                "<p style='margin:0 0 .35rem 0'><b>Try asking</b></p>"
                "<ul style='margin:0'>"
                "<li>How is our pipeline looking this quarter?</li>"
                "<li>Which sectors have the strongest pipeline?</li>"
                "<li>What data quality issues should leadership know about?</li>"
                "<li><code>/leadership-update</code> board meeting Thursday, "
                "cover sector performance and delivery</li>"
                "</ul></div>", unsafe_allow_html=True)

        for idx, message in enumerate(st.session_state.messages):
            if message.get("divider"):
                st.markdown(message["content"])
                continue
            with st.chat_message(message["role"]):
                content = message["content"]
                if isinstance(content, dict) and content.get("type") == "plan":
                    render_plan(content["plan"])
                else:
                    st.markdown(content)
                if message.get("results"):
                    render_results(message["results"], f"hist{idx}")

    if prompt:
        with working_box:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            agent = st.session_state.agent
            with st.chat_message("assistant"):
                with st.status("Working...", expanded=True) as status:
                    agent.status_container = status
                    agent.executed = []
                    agent.sql_error_count = 0

                    force_tool, to_send = None, prompt
                    if prompt.strip().startswith("/leadership-update"):
                        requirement = prompt.split("/leadership-update", 1)[1].strip()
                        to_send = ("Prepare a leadership update. Requirement: "
                                   + (requirement or "general business review"))
                        force_tool = "propose_leadership_update"

                    try:
                        reply = agent.send_message(to_send, force_tool=force_tool)
                        status.update(label="Done", state="complete", expanded=False)
                    except Exception as exc:
                        status.update(label="Failed", state="error")
                        st.error(f"Could not answer: {exc}")
                        reply = None

            if reply is not None:
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply,
                     "results": list(agent.executed)})
                st.rerun()

with tab_review:
    ws = store.workspace
    st.caption("Generated automatically from the boards. monday.com carries no "
               "column documentation, so every description below is derived from "
               "the data — column types are verified against the values rather "
               "than trusting what monday declares.")

    if ws.join:
        j = ws.join
        if j.get("kind") == "many-to-many":
            st.error(
                f"**These boards cannot be joined.** The closest shared column is "
                f"`{j['left']}` = `{j['right']}`, but it repeats on both sides — an "
                f"inner join turns {j['left_rows']} rows into {j['joined_rows']}, "
                f"inflating every total by about {j['fanout']}x. Compare them by "
                "aggregating each board separately on a shared dimension instead.")
        else:
            st.info(f"**Join between boards:** `{j['left']}` = `{j['right']}` — "
                    f"{j['kind']}, {j['matched']} of {j['left_total']} match "
                    f"({j['coverage']:.0%}). {j['left_total'] - j['matched']} rows are "
                    "dropped by an inner join. Discovered by value overlap, not assumed.")

    st.subheader("Which column to use for what")
    st.caption("Resolved from the data — cue match, then the best-populated "
               "candidate wins. This is what stops the agent guessing between "
               "eight similarly-named money columns.")
    role_cols = st.columns(len(ws.boards))
    for col, board in zip(role_cols, ws.boards):
        with col:
            st.markdown(f"**{board.board_name}**")
            if not board.roles:
                st.caption("No roles resolved.")
            for role, info in board.roles.items():
                src = info.get("source", "rules")
                tag = "" if src == "rules" else " · inferred"
                alt = (f"  \n<span style='opacity:.55'>over {len(info['alternatives'])} "
                       f"other candidate(s){tag}</span>" if info["alternatives"]
                       else (f"  \n<span style='opacity:.55'>{src}</span>" if tag else ""))
                st.markdown(
                    f"`{role}` → **{info['column']}**  \n"
                    f"<span style='opacity:.7'>{info['reason']}</span>{alt}",
                    unsafe_allow_html=True)

    st.divider()
    st.subheader("Data dictionary")
    for board in ws.boards:
        rows = []
        for name, p in board.profiles.items():
            rows.append({
                "Column": name,
                "Type": f"{p.storage_type} / {p.semantic_type}",
                "Filled": f"{1 - p.null_pct:.0%}",
                "Non-zero": f"{p.real}/{p.rows}" if p.storage_type == "number" else "",
                "Role": p.role or "",
                "Type from": p.semantic_source,
                "Notes from": p.description_source,
                "Description": p.description,
            })
        with st.expander(f"{board.board_name} — {len(rows)} columns, {board.rows} rows"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, height=420)

    st.subheader("Quality findings")
    findings = sorted(store.findings,
                      key=lambda f: (SEVERITY_ORDER.get(f["severity"], 9), f["board"]))
    if not findings:
        st.success("No issues found.")
    else:
        chosen = st.multiselect("Severity", list(SEVERITY_ORDER),
                                default=["Critical", "High"])
        shown = [f for f in findings if f["severity"] in chosen] if chosen else findings
        st.dataframe(pd.DataFrame(shown)[
            ["severity", "board", "column", "issue_type", "description"]],
            use_container_width=True, hide_index=True, height=460)
