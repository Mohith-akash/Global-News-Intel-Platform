"""
AI Chat component for GDELT dashboard.
Supports two modes: SQL (precise queries) and RAG (semantic search).
"""

import re
import pandas as pd
import streamlit as st

from src.database import safe_query
from src.ai_engine import get_query_engine, get_cerebras_llm, AI_AVAILABLE
from src.data_processing import extract_headline
from src.headline_utils import clean_headline
from src.utils import get_dates, get_country, get_country_code, get_impact_label, detect_query_type
from src.rag_engine import rag_query, get_voyage_api_key


def render_ai_chat(c, sql_db, tbl="events_dagster"):
    """Main AI Chat component with SQL and RAG modes."""
    # Check if AI features are available
    if not AI_AVAILABLE:
        st.markdown('''<div class="card">
            <h3 style="color:#06b6d4;">🔧 AI Features Temporarily Unavailable</h3>
            <p class="text-muted">The AI chat interface requires additional dependencies that are currently unavailable.</p>
            <p class="text-muted">Don't worry! All other dashboard features (Home, Feed, Emotions) are working normally.</p>
            <div style="background:#1e293b;border-radius:8px;padding:1rem;margin-top:1rem;">
                <div style="color:#64748b;font-size:0.85rem;">
                    <b>📊 What you can still do:</b><br>
                    • View real-time event metrics and trends<br>
                    • Browse the news feed and timeseries charts<br>
                    • Analyze emotion data from GKG<br>
                    • Explore geographic distribution and actors
                </div>
            </div>
        </div>''', unsafe_allow_html=True)
        return
    
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "ai_mode" not in st.session_state:
        st.session_state.ai_mode = "sql"

    # Check if RAG is available
    rag_available = get_voyage_api_key() is not None
    
    # Previous conversations (shared between modes)
    if st.session_state.qa_history:
        past = st.session_state.qa_history[-5:]
        with st.expander("🕒 Previous Conversations", expanded=False):
            idx = st.selectbox("Select", range(len(past)), format_func=lambda i: (past[i]["question"][:70] + "…") if len(past[i]["question"]) > 70 else past[i]["question"], key="prev_select")
            sel = past[idx]
            mode_badge = "🔍 SQL" if sel.get("mode") == "sql" else "🧠 RAG"
            st.markdown(f'''<div class="prev-convo-card">
                <div class="prev-convo-label">💬 Previous Conversation ({mode_badge})</div>
                <div class="prev-convo-q"><b>Q:</b></div><div class="prev-convo-text">{sel['question']}</div>
                <div class="prev-convo-q" style="margin-top:0.5rem;"><b>A:</b></div><div class="prev-convo-text">{sel['answer']}</div>
            </div>''', unsafe_allow_html=True)
            if sel.get("sql"):
                with st.expander("🔍 Query Details"):
                    st.code(sel["sql"], language="sql")

    # Mode selector (radio buttons work better with chat_input than tabs)
    mode_options = ["🔍 SQL Mode", "🧠 RAG Mode"]
    if not rag_available:
        mode_options = ["🔍 SQL Mode", "🧠 RAG Mode (requires API key)"]
    
    selected_mode = st.radio(
        "Select Mode",
        mode_options,
        horizontal=True,
        label_visibility="collapsed",
        key="mode_selector"
    )
    
    is_sql_mode = "SQL" in selected_mode
    
    # Show mode-specific info
    if is_sql_mode:
        st.markdown('''<div class="ai-info-card">
            <div class="ai-example-label">💡 SQL MODE - Precise Queries:</div>
            <div class="ai-examples">• "What happened in India this week?"<br>• "Crisis events in Middle East"<br>• "Top 5 countries by events"<br>• "How many events in October?"</div>
        </div>''', unsafe_allow_html=True)
        render_sql_chat(c, sql_db, tbl)
    else:
        if not rag_available:
            st.warning("⚠️ RAG Mode requires VOYAGE_API_KEY. Add it to your environment or Streamlit secrets.")
            st.markdown('''<div class="ai-info-card">
                <div class="ai-example-label">🔑 To enable RAG Mode:</div>
                <div class="ai-examples">1. Get free API key at <a href="https://www.voyageai.com/" target="_blank">voyageai.com</a><br>2. Add VOYAGE_API_KEY to your .env file or Streamlit secrets</div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''<div class="ai-info-card">
                <div class="ai-example-label">💡 RAG MODE - Semantic Search:</div>
                <div class="ai-examples">• "What are the tensions in Asia about?"<br>• "Tell me about military conflicts"<br>• "What is happening between US and China?"<br>• "Summarize news about Russia and Ukraine"</div>
            </div>''', unsafe_allow_html=True)
            render_rag_chat(c, tbl)


def render_rag_chat(c, tbl="events_dagster"):
    """RAG-based semantic search chat."""
    from src.ai_engine import get_cerebras_llm
    from src.rag_engine import rag_query
    from src.utils import get_country, get_impact_label, get_dates
    
    prompt = st.chat_input("Ask about events semantically...", key="rag_chat")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            llm = get_cerebras_llm()
            if not llm:
                st.error("❌ Cerebras AI not available")
                return
            
            with st.spinner("🧠 Searching semantically..."):
                try:
                    # Get date range - use last 30 days like SQL mode
                    dates = get_dates()
                    result = rag_query(prompt, c, llm, top_k=15, min_date=dates['month_ago'], table_name=tbl)
                    
                    # Display answer (escape $ to prevent LaTeX rendering)
                    st.markdown(result["answer"].replace('$', r'\$'))
                    
                    # Display retrieved headlines
                    if not result["headlines"].empty:
                        df = result["headlines"].copy()
                        
                        # Format for display
                        if 'ACTOR_COUNTRY_CODE' in df.columns:
                            df['COUNTRY'] = df['ACTOR_COUNTRY_CODE'].apply(
                                lambda x: get_country(x) or x if x and str(x) not in ('None', 'nan', '') else '—'
                            )
                        if 'IMPACT_SCORE' in df.columns:
                            df['SEVERITY'] = df['IMPACT_SCORE'].apply(get_impact_label)
                        if 'DATE' in df.columns:
                            try:
                                df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%b %d')
                            except:
                                pass
                        if 'similarity' in df.columns:
                            df['RELEVANCE'] = (df['similarity'] * 100).round(1).astype(str) + '%'
                        
                        with st.expander("📋 Retrieved Headlines", expanded=True):
                            display_cols = ['DATE', 'HEADLINE', 'COUNTRY', 'SEVERITY', 'RELEVANCE', 'NEWS_LINK']
                            display_cols = [col for col in display_cols if col in df.columns]
                            st.dataframe(
                                df[display_cols],
                                hide_index=True,
                                column_config={
                                    "DATE": st.column_config.TextColumn("Date", width=70),
                                    "HEADLINE": st.column_config.TextColumn("Event", width=None),
                                    "COUNTRY": st.column_config.TextColumn("Country", width=90),
                                    "SEVERITY": st.column_config.TextColumn("Severity", width=100),
                                    "RELEVANCE": st.column_config.TextColumn("Match", width=55),
                                    "NEWS_LINK": st.column_config.LinkColumn("Link", width=50, display_text="🔗"),
                                }
                            )
                    
                    # Save to history
                    st.session_state.qa_history.append({
                        "question": prompt,
                        "answer": result["answer"],
                        "sql": result.get("sql"),
                        "mode": "rag"
                    })
                    if len(st.session_state.qa_history) > 50:
                        st.session_state.qa_history = st.session_state.qa_history[-50:]
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:100]}")


def render_sql_chat(c, sql_db, tbl="events_dagster"):
    """SQL-based precise query chat (original implementation)."""
    prompt = st.chat_input("Ask about global events (SQL)...", key="sql_chat")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            qe = get_query_engine(sql_db) if sql_db else None
            llm = get_cerebras_llm()
            if not llm:
                st.error("❌ Cerebras AI not available")
                return
            try:
                dates = get_dates()
                qi = detect_query_type(prompt)
                
                if qi['is_specific_date'] and qi['specific_date']:
                    date_filter = f"DATE = '{qi['specific_date']}'"
                elif qi.get('is_week_range') and qi.get('week_start') and qi.get('week_end'):
                    # Handle "last week" or "this week" queries
                    date_filter = f"DATE >= '{qi['week_start']}' AND DATE <= '{qi['week_end']}'"
                elif qi.get('is_month_range') and qi.get('month_start') and qi.get('month_end'):
                    # Handle month-only queries like "events in october"
                    date_filter = f"DATE >= '{qi['month_start']}' AND DATE <= '{qi['month_end']}'"
                elif qi['time_period'] == 'all' or qi['is_aggregate']:
                    date_filter = f"DATE >= '{dates['three_months_ago']}'"
                elif qi['time_period'] == 'month':
                    date_filter = f"DATE >= '{dates['month_ago']}'"
                elif qi['time_period'] == 'day':
                    date_filter = f"DATE = '{dates['today']}'"
                else:
                    date_filter = f"DATE >= '{dates['week_ago']}'"

                sql = None
                answer = ""
                is_country_aggregate = False
                is_count_aggregate = False
                country_filter_name = None
                with st.spinner("🔍 Querying..."):
                    # Determine display limit from query (default 5, max 10)
                    limit = 5
                    m = re.search(r'(\d+)\s*(events?|results?|items?)', prompt.lower())
                    if m: 
                        limit = min(int(m.group(1)), 10)
                    m2 = re.search(r'top\s+(\d+)', prompt.lower())
                    if m2:
                        limit = min(int(m2.group(1)), 10)
                    
                    # For month-range queries, cap at 5 to keep responses focused
                    if qi.get('is_month_range'):
                        limit = min(limit, 5)
                    
                    # Fetch more rows to account for filtering/deduplication (more = better quality)
                    fetch_limit = min(limit * 50, 500)  # 50x multiplier, capped at 500
                    
                    # Helper: detect country codes in prompt
                    def get_country_codes_from_prompt(text):
                        codes = []
                        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
                        
                        # Check for region aliases first (e.g., "middle east" -> multiple country codes)
                        try:
                            from src.config import REGION_ALIASES
                            for region, region_codes in REGION_ALIASES.items():
                                if region in clean_text:
                                    codes.extend(region_codes)
                                    return codes  # Return region codes immediately
                        except ImportError:
                            pass
                        
                        # Check for multi-word phrases first
                        multi_word_regions = [
                            'middle east', 'united states', 'united kingdom', 'great britain',
                            'south korea', 'north korea', 'saudi arabia', 'south africa',
                            'new zealand'
                        ]
                        for phrase in multi_word_regions:
                            if phrase in clean_text:
                                code = get_country_code(phrase)
                                if code and code not in codes:
                                    codes.append(code)
                        
                        # Then check individual words
                        for w in clean_text.split():
                            if len(w) >= 2:
                                code = get_country_code(w)
                                if code and code not in codes: 
                                    codes.append(code)
                        return codes
                    
                    prompt_lower = prompt.lower()
                    has_crisis = 'crisis' in prompt_lower or 'severe' in prompt_lower
                    has_country_word = 'countr' in prompt_lower
                    has_major = 'major' in prompt_lower or 'important' in prompt_lower or 'significant' in prompt_lower or 'biggest' in prompt_lower or 'trending' in prompt_lower
                    
                    # Check for specific query types (ORDER MATTERS!)
                    
                    # 1. COUNTRIES WITH CRISIS - must come before plain crisis
                    if has_crisis and has_country_word:
                        is_country_aggregate = True
                        sql = f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as EVENT_COUNT FROM {tbl} WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND IMPACT_SCORE < -3 AND {date_filter} GROUP BY ACTOR_COUNTRY_CODE ORDER BY EVENT_COUNT DESC LIMIT {limit}"
                    
                    # 2. Plain crisis events (require 10+ articles for quality headlines)
                    elif has_crisis:
                        # Check if user specified a country for crisis events
                        crisis_codes = get_country_codes_from_prompt(prompt)
                        if crisis_codes:
                            if len(crisis_codes) == 1:
                                crisis_country_filter = f"ACTOR_COUNTRY_CODE = '{crisis_codes[0]}'"
                            else:
                                codes_str = "', '".join(crisis_codes)
                                crisis_country_filter = f"ACTOR_COUNTRY_CODE IN ('{codes_str}')"
                            sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM {tbl} WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {crisis_country_filter} AND ARTICLE_COUNT >= 10 AND IMPACT_SCORE < -3 AND {date_filter} ORDER BY ARTICLE_COUNT DESC, IMPACT_SCORE ASC LIMIT {fetch_limit}"
                        else:
                            sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM {tbl} WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND ARTICLE_COUNT >= 10 AND IMPACT_SCORE < -3 AND {date_filter} ORDER BY ARTICLE_COUNT DESC, IMPACT_SCORE ASC LIMIT {fetch_limit}"
                    
                    # 3. MAJOR/IMPORTANT events - high article count (trending stories)
                    elif has_major:
                        sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM {tbl} WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND ARTICLE_COUNT > 50 AND {date_filter} ORDER BY ARTICLE_COUNT DESC LIMIT {fetch_limit}"
                    
                    # 4. TOP COUNTRIES - check this BEFORE is_aggregate
                    elif 'top' in prompt_lower and has_country_word:
                        is_country_aggregate = True
                        sql = f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as EVENT_COUNT FROM {tbl} WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {date_filter} GROUP BY ACTOR_COUNTRY_CODE ORDER BY EVENT_COUNT DESC LIMIT {limit}"
                    
                    # 5. Aggregate queries (how many, count, total) - now with country support
                    elif qi['is_aggregate']:
                        is_count_aggregate = True
                        codes = get_country_codes_from_prompt(prompt)
                        if codes:
                            cf = f"ACTOR_COUNTRY_CODE = '{codes[0]}'"
                            country_filter_name = get_country(codes[0]) or codes[0]
                            sql = f"SELECT COUNT(*) as TOTAL_EVENTS FROM {tbl} WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {cf} AND {date_filter}"
                        else:
                            sql = f"SELECT COUNT(*) as TOTAL_EVENTS FROM {tbl} WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {date_filter}"
                    
                    # 6. Default: specific events query
                    else:
                        codes = get_country_codes_from_prompt(prompt)

                        # Require at least 10 articles for quality headlines
                        article_threshold = 10

                        if codes:
                            if len(codes) == 1:
                                cf = f"ACTOR_COUNTRY_CODE = '{codes[0]}'"
                            else:
                                codes_str = "', '".join(codes)
                                cf = f"ACTOR_COUNTRY_CODE IN ('{codes_str}')"
                            sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM {tbl} WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {cf} AND ARTICLE_COUNT > {article_threshold} AND {date_filter} ORDER BY ARTICLE_COUNT DESC, DATE DESC LIMIT {fetch_limit}"
                        else:
                            sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM {tbl} WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND ARTICLE_COUNT > {article_threshold} AND {date_filter} ORDER BY ARTICLE_COUNT DESC LIMIT {fetch_limit}"
                    
                    # Enforce LIMIT on aggregate queries only (event queries need more rows for filtering)
                    if sql and (is_count_aggregate or is_country_aggregate):
                        if 'LIMIT' not in sql.upper():
                            sql = sql.rstrip(';') + f' LIMIT {limit}'
                    
                    if sql:
                        data = safe_query(c, sql)
                        if not data.empty:
                            dd = data.copy()
                            dd.columns = [col.upper() for col in dd.columns]
                            
                            # Handle COUNT(*) aggregate queries
                            if is_count_aggregate:
                                total = dd.iloc[0]['TOTAL_EVENTS']
                                location = f"in {country_filter_name}" if country_filter_name else "globally"
                                
                                ai_prompt = f"""Database query result: {total:,} events recorded {location} during {qi['period_label']}.

Question: {prompt}

Provide a brief, factual answer using ONLY this data. State the count clearly."""

                                _resp = llm.complete(ai_prompt)
                                answer = str(_resp) if _resp is not None else "No response from AI."
                                st.markdown(answer.replace('$', r'\$'))

                                st.markdown(f"""
                                <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;text-align:center;margin:1rem 0;">
                                    <div style="font-size:0.8rem;color:#64748b;text-transform:uppercase;">Total Events {f"in {country_filter_name}" if country_filter_name else ""}</div>
                                    <div style="font-size:2.5rem;font-weight:700;color:#06b6d4;">{total:,}</div>
                                    <div style="font-size:0.75rem;color:#94a3b8;">{qi['period_label']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                with st.expander("🔍 SQL"):
                                    st.code(sql, language='sql')
                            
                            # Handle country aggregate query differently
                            elif is_country_aggregate:
                                # Convert country codes to names
                                dd['COUNTRY'] = dd['ACTOR_COUNTRY_CODE'].apply(
                                    lambda x: get_country(x) or x if x and str(x) not in ('None', 'nan', '') else '—'
                                )
                                
                                # Build summary for AI
                                country_list = []
                                for _, row in dd.iterrows():
                                    country_list.append(f"- {row['COUNTRY']}: {row['EVENT_COUNT']:,} events")
                                summary_text = "\n".join(country_list)
                                
                                ai_prompt = f"""You are a news analyst. Below is real data from the GDELT database showing event counts by country ({qi['period_label']}). Use ONLY this data — do not invent reasons or add facts not shown here.

{summary_text}

Question: {prompt}

Based solely on the event counts above, briefly describe which countries dominate and any notable patterns in the numbers. Keep response concise and factual."""

                                _resp = llm.complete(ai_prompt)
                                answer = str(_resp) if _resp is not None else "No response from AI."
                                st.markdown(answer.replace('$', r'\$'))

                                st.dataframe(
                                    dd[['COUNTRY', 'EVENT_COUNT']],
                                    hide_index=True,
                                    width='stretch',
                                    column_config={
                                        "COUNTRY": st.column_config.TextColumn("Country", width=200),
                                        "EVENT_COUNT": st.column_config.NumberColumn("Event Count", width=120)
                                    }
                                )
                                
                                with st.expander("🔍 SQL"):
                                    st.code(sql, language='sql')
                            else:
                                # Regular event query - extract headlines, show details
                                # Convert country code to full name
                                if 'ACTOR_COUNTRY_CODE' in dd.columns:
                                    dd['COUNTRY'] = dd['ACTOR_COUNTRY_CODE'].apply(
                                        lambda x: get_country(x) or x if x and str(x) not in ('None', 'nan', '') else '—'
                                    )
                                
                                if 'NEWS_LINK' in dd.columns:
                                    headlines = []
                                    valid_indices = []
                                    for idx, row in dd.iterrows():
                                        # Use DB headline first, fallback to URL extraction
                                        db_headline = row.get('HEADLINE')
                                        if db_headline and isinstance(db_headline, str) and len(db_headline.strip()) > 15:
                                            headline = clean_headline(db_headline)
                                        else:
                                            # Extract from URL if DB headline missing/bad
                                            headline = extract_headline(row.get('NEWS_LINK', ''), None, row.get('IMPACT_SCORE', None))
                                        
                                        if headline:
                                            headlines.append(headline)
                                            valid_indices.append(idx)
                                    
                                    # Filter to only rows with valid headlines
                                    if valid_indices:
                                        dd = dd.loc[valid_indices].copy()
                                        dd = dd.reset_index(drop=True)  # Reset index for proper list assignment
                                        dd['HEADLINE'] = headlines
                                        
                                        # Smart deduplication: skip first word, use words 2-6
                                        # "Snow Expected Across Much Uk" -> "expected across much uk"
                                        # "Snowfall Expected Across Much Uk" -> "expected across much uk" (SAME!)
                                        dd['_dedup_key'] = dd['HEADLINE'].apply(
                                            lambda x: ' '.join(x.lower().split()[1:6]) if x and len(x.split()) > 1 else x.lower() if x else ''
                                        )
                                        dd = dd.drop_duplicates(subset=['_dedup_key'])
                                        dd = dd.drop(columns=['_dedup_key'])
                                    else:
                                        dd = dd.head(0)  # Empty dataframe
                                
                                # Add severity label
                                if 'IMPACT_SCORE' in dd.columns:
                                    dd['SEVERITY'] = dd['IMPACT_SCORE'].apply(get_impact_label)
                                
                                # Format date
                                if 'DATE' in dd.columns:
                                    try: 
                                        dd['DATE'] = pd.to_datetime(dd['DATE'].astype(str), format='%Y%m%d').dt.strftime('%b %d')
                                    except Exception: pass
                                
                                # Check if we have any valid data after filtering
                                if dd.empty:
                                    st.warning("📭 No events found for this query")
                                    answer = "No events found."
                                else:
                                    # Prepare data for AI summary (include headlines)
                                    summary_data = []
                                    for _, row in dd.head(limit).iterrows():
                                        headline = row.get('HEADLINE', 'Event')
                                        country = row.get('COUNTRY', row.get('ACTOR_COUNTRY_CODE', 'Global'))
                                        date = row.get('DATE', '')
                                        severity = row.get('SEVERITY', '')
                                        score = row.get('IMPACT_SCORE', 0)
                                        summary_data.append(f"- {headline} | {country} | {date} | Severity: {severity} ({score})")
                                    
                                    summary_text = "\n".join(summary_data)
                                    
                                    ai_prompt = f"""You are a news analyst. Below are {len(summary_data)} events from the GDELT database ({qi['period_label']}).

{summary_text}

Question: {prompt}

Write a concise analytical answer (3-5 sentences) that directly answers the question. Group related events by theme or country. Then list the events as brief one-line bullet points. Do not create verbose per-event breakdowns."""

                                    _resp = llm.complete(ai_prompt)
                                    answer = str(_resp) if _resp is not None else "No response from AI."
                                    st.markdown(answer.replace('$', r'\$'))
                                    
                                    display_cols = ['DATE', 'HEADLINE', 'COUNTRY', 'SEVERITY']
                                    if 'NEWS_LINK' in dd.columns:
                                        display_cols.append('NEWS_LINK')
                                    
                                    display_cols = [col for col in display_cols if col in dd.columns]
                                    
                                    st.dataframe(
                                        dd[display_cols].head(limit),
                                        hide_index=True,
                                        width='stretch',
                                        column_config={
                                            "DATE": st.column_config.TextColumn("Date", width=70),
                                            "HEADLINE": st.column_config.TextColumn("Event", width=None),
                                            "COUNTRY": st.column_config.TextColumn("Country", width=100),
                                            "SEVERITY": st.column_config.TextColumn("Severity", width=120),
                                            "NEWS_LINK": st.column_config.LinkColumn("🔗", width=40)
                                        }
                                    )
                                
                                with st.expander("🔍 SQL"):
                                    st.code(sql, language='sql')
                        else:
                            st.warning("📭 No results found for this query")
                            st.markdown('''<div style="background:#1e293b;border-radius:8px;padding:1rem;margin-top:0.5rem;">
                                <div style="color:#94a3b8;font-size:0.85rem;">💡 Try rephrasing with a country name or time period (e.g., "events in Germany this week"). Data available: Oct 2025 – Present.</div>
                            </div>''', unsafe_allow_html=True)
                            answer = "No results found."
                            with st.expander("🔍 SQL"):
                                st.code(sql, language='sql')
                    else:
                        st.warning("⚠️ Could not generate query")
                        answer = "Could not process query."
                
                st.session_state.qa_history.append({"question": prompt, "answer": answer, "sql": sql, "mode": "sql"})
                # Limit history to 50 messages to prevent unbounded memory growth
                if len(st.session_state.qa_history) > 50:
                    st.session_state.qa_history = st.session_state.qa_history[-50:]
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:100]}")