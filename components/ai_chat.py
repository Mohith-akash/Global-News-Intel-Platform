"""
AI chat with two modes:
- SQL: the question picks a tested query template (src/sql_templates.py) and
  the LLM writes the answer from the rows it returns
- RAG: semantic search over headline embeddings (src/rag_engine.py)

Answers are cached for an hour, so the example questions people try most
don't spend LLM credit every time.
"""

import html
import logging

import pandas as pd
import streamlit as st

from src.ai_engine import LLMUnavailable, get_cerebras_llm
from src.data_processing import extract_headline
from src.database import WarehouseUnavailable, safe_query
from src.headline_utils import clean_headline
from src.rag_engine import get_voyage_api_key, rag_query
from src.sql_templates import build_sql, date_bounds
from src.utils import detect_query_type, get_country, get_dates, get_intensity_label

logger = logging.getLogger("gdelt")

SQL_EXAMPLES = ['What happened in India this week?', 'Crisis events in Middle East',
                'Top 5 countries by events', 'How many events in October?']
RAG_EXAMPLES = ['What are the tensions in Asia about?', 'Tell me about military conflicts',
                'What is happening between US and China?', 'Summarize news about Russia and Ukraine']


# ---------------------------------------------------------------- helpers

def _country_names(codes):
    return codes.apply(lambda x: get_country(x) or x if x and str(x) not in ('None', 'nan', '') else '—')


def _short_dates(dates):
    try:
        return pd.to_datetime(dates.astype(str), format='%Y%m%d').dt.strftime('%b %d')
    except (ValueError, TypeError):
        return dates


def _llm():
    llm = get_cerebras_llm()
    if llm is None:
        raise LLMUnavailable("The AI service is not configured.")
    return llm


def _ask(prompt):
    return _llm().complete(prompt) or "No response from AI."


def _say(text):
    st.markdown(text.replace('$', r'\$'))  # a bare $ would start LaTeX


def _show_sql(sql):
    with st.expander("Show the SQL"):
        st.code(sql, language='sql')


def _examples(label, examples):
    items = "<br>".join(f'• "{e}"' for e in examples)
    st.markdown(f'<div class="ai-info-card"><div class="ai-example-label">{label}</div>'
                f'<div class="ai-examples">{items}</div></div>', unsafe_allow_html=True)


def _remember(question, answer, sql, mode):
    history = st.session_state.qa_history
    history.append({"question": question, "answer": answer, "sql": sql, "mode": mode})
    del history[:-50]  # keep the last 50


def _run_safely(fn):
    """Show a readable message instead of a stack trace."""
    try:
        fn()
    except LLMUnavailable as e:
        st.warning(str(e))  # written for visitors: out of credit, rate limit...
    except WarehouseUnavailable:
        st.warning("The data warehouse is not reachable right now. Try again in a few minutes.")
    except Exception:
        logger.exception("chat failed")
        st.error("Something went wrong with that question. Try rephrasing it.")


# ---------------------------------------------------------------- page

def render_ai_chat(tbl="events_dagster"):
    """Mode picker, examples, previous conversations, and the chat itself."""
    st.session_state.setdefault("qa_history", [])
    _render_history()

    rag_available = get_voyage_api_key() is not None
    modes = ["SQL mode", "RAG mode" if rag_available else "RAG mode (needs a Voyage API key)"]
    mode = st.radio("Select Mode", modes, horizontal=True, label_visibility="collapsed", key="mode_selector")

    if "SQL" in mode:
        _examples("SQL mode: precise counts and lists. Try:", SQL_EXAMPLES)
        prompt = st.chat_input("Ask about global events (SQL)...", key="sql_chat")
        if prompt:
            _chat_turn(prompt, lambda: _render_sql_answer(prompt, tbl))
    elif not rag_available:
        st.warning("RAG mode needs VOYAGE_API_KEY in the environment or Streamlit secrets.")
    else:
        _examples("RAG mode: semantic search over headlines. Try:", RAG_EXAMPLES)
        prompt = st.chat_input("Ask about events semantically...", key="rag_chat")
        if prompt:
            _chat_turn(prompt, lambda: _render_rag_answer(prompt, tbl))


def _render_history():
    past = st.session_state.qa_history[-5:]
    if not past:
        return
    with st.expander("Previous questions", expanded=False):
        idx = st.selectbox("Select", range(len(past)), key="prev_select",
                           format_func=lambda i: past[i]["question"][:70] + ("…" if len(past[i]["question"]) > 70 else ""))
        sel = past[idx]
        badge = "SQL" if sel.get("mode") == "sql" else "RAG"
        # escaped: these are visitor-typed questions and model output
        st.markdown(f'''<div class="prev-convo-card">
            <div class="prev-convo-label">{badge} mode</div>
            <div class="prev-convo-q"><b>Q:</b></div><div class="prev-convo-text">{html.escape(sel['question'])}</div>
            <div class="prev-convo-q" style="margin-top:0.5rem;"><b>A:</b></div><div class="prev-convo-text">{html.escape(sel['answer'])}</div>
        </div>''', unsafe_allow_html=True)
        if sel.get("sql"):
            with st.expander("Query details"):
                st.code(sel["sql"], language="sql")


def _chat_turn(prompt, answer_fn):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        _run_safely(answer_fn)


# ---------------------------------------------------------------- RAG mode

@st.cache_data(ttl=3600, show_spinner=False)
def _rag_result(prompt, tbl, min_date, max_date):
    return rag_query(prompt, _llm(), top_k=15, min_date=min_date, max_date=max_date, table_name=tbl)


def _render_rag_answer(prompt, tbl):
    qi = detect_query_type(prompt)
    min_date, max_date = date_bounds(qi, get_dates())  # same reading of "this week" as SQL mode
    st.caption(f"Searching {qi['period_label']}")
    with st.spinner("Searching..."):
        result = _rag_result(prompt, tbl, min_date, max_date)
    _say(result["answer"])

    df = result["headlines"].copy()
    if not df.empty:
        if 'ACTOR_COUNTRY_CODE' in df.columns:
            df['COUNTRY'] = _country_names(df['ACTOR_COUNTRY_CODE'])
        if 'IMPACT_SCORE' in df.columns:
            df['SEVERITY'] = df['IMPACT_SCORE'].apply(get_intensity_label)
        if 'DATE' in df.columns:
            df['DATE'] = _short_dates(df['DATE'])
        if 'similarity' in df.columns:
            df['RELEVANCE'] = (df['similarity'] * 100).round(1).astype(str) + '%'
        cols = [c for c in ['DATE', 'HEADLINE', 'COUNTRY', 'SEVERITY', 'RELEVANCE', 'NEWS_LINK'] if c in df.columns]
        with st.expander("Retrieved headlines", expanded=True):
            st.dataframe(df[cols], hide_index=True, column_config={
                "DATE": st.column_config.TextColumn("Date", width=70),
                "HEADLINE": st.column_config.TextColumn("Event", width=None),
                "COUNTRY": st.column_config.TextColumn("Country", width=90),
                "SEVERITY": st.column_config.TextColumn("Severity", width=100),
                "RELEVANCE": st.column_config.TextColumn("Match", width=55),
                "NEWS_LINK": st.column_config.LinkColumn("Link", width=50, display_text="open"),
            })
    _remember(prompt, result["answer"], result.get("sql"), "rag")


# ---------------------------------------------------------------- SQL mode

def _event_rows(dd):
    """Readable headline per row (DB headline, else rebuilt from the URL), near-duplicates dropped."""
    dd['COUNTRY'] = _country_names(dd['ACTOR_COUNTRY_CODE'])
    headlines = [clean_headline(h) if isinstance(h, str) and len(h.strip()) > 15 else extract_headline(url or '')
                 for h, url in zip(dd['HEADLINE'], dd['NEWS_LINK'])]
    dd = dd.assign(HEADLINE=headlines)
    dd = dd[dd['HEADLINE'].map(bool)].reset_index(drop=True)
    # same story, slightly different first word: "Snow Expected Across Much Uk" and
    # "Snowfall Expected Across Much Uk" share words 2-6
    dd = dd.loc[~dd['HEADLINE'].map(lambda h: ' '.join(h.lower().split()[1:6]) if len(h.split()) > 1
                                    else h.lower()).duplicated()]
    dd['SEVERITY'] = dd['IMPACT_SCORE'].apply(get_intensity_label)
    dd['DATE'] = _short_dates(dd['DATE'])
    return dd


@st.cache_data(ttl=3600, show_spinner=False)
def _sql_result(prompt, tbl, today):
    """Run the template query and write the answer. `today` keys the cache by day."""
    qi = detect_query_type(prompt)
    plan = build_sql(prompt, qi, get_dates(), tbl)
    res = {"sql": plan.display_sql(), "period": qi['period_label'], "kind": "none", "answer": "No results found."}
    data = safe_query(plan.sql, plan.params)
    if data.empty:
        return res
    dd = data.copy()
    dd.columns = [c.upper() for c in dd.columns]

    if plan.is_count_aggregate:
        total = dd.iloc[0]['TOTAL_EVENTS']
        location = f"in {plan.country_filter_name}" if plan.country_filter_name else "globally"
        res.update(kind="count", total=int(total), country=plan.country_filter_name, answer=_ask(
            f"""Database query result: {total:,} events recorded {location} during {qi['period_label']}.

Question: {prompt}

Provide a brief, factual answer using ONLY this data. State the count clearly."""))
        return res

    if plan.is_country_aggregate:
        dd['COUNTRY'] = _country_names(dd['ACTOR_COUNTRY_CODE'])
        summary = "\n".join(f"- {c}: {n:,} events" for c, n in zip(dd['COUNTRY'], dd['EVENT_COUNT']))
        res.update(kind="countries", table=dd[['COUNTRY', 'EVENT_COUNT']], answer=_ask(
            f"""You are a news analyst. Below is real data from the GDELT database showing event counts by country ({qi['period_label']}). Use ONLY this data — do not invent reasons or add facts not shown here.

{summary}

Question: {prompt}

Based solely on the event counts above, briefly describe which countries dominate and any notable patterns in the numbers. Keep response concise and factual."""))
        return res

    dd = _event_rows(dd).head(plan.limit)
    if dd.empty:
        res.update(kind="no_events", answer="No events found.")
        return res
    summary = "\n".join(
        f"- {r['HEADLINE']} | {r['COUNTRY']} | {r['DATE']} | Severity: {r['SEVERITY']} ({r['IMPACT_SCORE']})"
        for _, r in dd.iterrows())
    res.update(kind="events", table=dd[['DATE', 'HEADLINE', 'COUNTRY', 'SEVERITY', 'NEWS_LINK']], answer=_ask(
        f"""You are a news analyst. Below are {len(dd)} events from the GDELT database ({qi['period_label']}).

{summary}

Question: {prompt}

Write a concise analytical answer (3-5 sentences) that directly answers the question. Group related events by theme or country. Then list the events as brief one-line bullet points. Do not create verbose per-event breakdowns."""))
    return res


def _render_sql_answer(prompt, tbl):
    with st.spinner("Querying..."):
        res = _sql_result(prompt, tbl, get_dates()['today'])
    kind = res["kind"]
    if kind == "none":
        st.warning("No results for this question.")
        st.caption('Try a country name or a time period, e.g. "events in Germany this week". '
                   'Data starts in October 2025.')
    elif kind == "no_events":
        st.warning("No events found for this question.")
    else:
        _say(res["answer"])
    if kind == "count":
        where = f"in {res['country']}" if res.get("country") else ""
        st.markdown(f'''<div class="count-card"><div class="count-label">Total Events {html.escape(where)}</div>
            <div class="count-value">{res["total"]:,}</div><div class="count-period">{res["period"]}</div></div>''',
                    unsafe_allow_html=True)
    elif kind == "countries":
        st.dataframe(res["table"], hide_index=True, width='stretch', column_config={
            "COUNTRY": st.column_config.TextColumn("Country", width=200),
            "EVENT_COUNT": st.column_config.NumberColumn("Event Count", width=120)})
    elif kind == "events":
        st.dataframe(res["table"], hide_index=True, width='stretch', column_config={
            "DATE": st.column_config.TextColumn("Date", width=70),
            "HEADLINE": st.column_config.TextColumn("Event", width=None),
            "COUNTRY": st.column_config.TextColumn("Country", width=100),
            "SEVERITY": st.column_config.TextColumn("Severity", width=120),
            "NEWS_LINK": st.column_config.LinkColumn("Link", width=50, display_text="open")})
    _show_sql(res["sql"])
    _remember(prompt, res["answer"], res["sql"], "sql")
