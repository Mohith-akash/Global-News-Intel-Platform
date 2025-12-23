"""
AI Chat component for GDELT dashboard.
"""

import re
import pandas as pd
import streamlit as st

from src.database import safe_query
from src.ai_engine import get_query_engine, get_cerebras_llm
from src.data_processing import clean_headline, enhance_headline, extract_headline
from src.utils import get_dates, get_country, get_country_code, get_impact_label, detect_query_type


def render_ai_chat(c, sql_db):
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    if st.session_state.qa_history:
        past = st.session_state.qa_history[-5:]
        with st.expander("🕒 Previous Conversations", expanded=False):
            idx = st.selectbox("Select", range(len(past)), format_func=lambda i: (past[i]["question"][:70] + "…") if len(past[i]["question"]) > 70 else past[i]["question"], key="prev_select")
            sel = past[idx]
            st.markdown(f'''<div class="prev-convo-card">
                <div class="prev-convo-label">💬 Previous Conversation</div>
                <div class="prev-convo-q"><b>Q:</b></div><div class="prev-convo-text">{sel['question']}</div>
                <div class="prev-convo-q" style="margin-top:0.5rem;"><b>A:</b></div><div class="prev-convo-text">{sel['answer']}</div>
            </div>''', unsafe_allow_html=True)
            if sel.get("sql"):
                with st.expander("🔍 SQL Query"):
                    st.code(sel["sql"], language="sql")

    st.markdown('''<div class="ai-info-card">
        <div class="ai-example-label">💡 EXAMPLE QUESTIONS:</div>
        <div class="ai-examples">• "What major events happened this week?"<br>• "Top 5 countries by event count"<br>• "Show crisis-level events"</div>
    </div>''', unsafe_allow_html=True)

    prompt = st.chat_input("Ask about global events...", key="chat")
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
                    
                    # Fetch more rows to account for filtering and deduplication
                    fetch_limit = limit * 20  # Fetch 20x more since many headlines will be filtered or duplicated
                    
                    # Helper: detect country codes in prompt
                    def get_country_codes_from_prompt(text):
                        codes = []
                        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
                        
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
                        sql = f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as EVENT_COUNT FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND IMPACT_SCORE < -3 AND {date_filter} GROUP BY ACTOR_COUNTRY_CODE ORDER BY EVENT_COUNT DESC LIMIT {limit}"
                    
                    # 2. Plain crisis events
                    elif has_crisis:
                        sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND IMPACT_SCORE < -3 AND {date_filter} ORDER BY IMPACT_SCORE ASC LIMIT {fetch_limit}"
                    
                    # 3. MAJOR/IMPORTANT events - high article count (trending stories)
                    elif has_major:
                        sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND ARTICLE_COUNT > 50 AND {date_filter} ORDER BY ARTICLE_COUNT DESC LIMIT {fetch_limit}"
                    
                    # 4. TOP COUNTRIES - check this BEFORE is_aggregate
                    elif 'top' in prompt_lower and has_country_word:
                        is_country_aggregate = True
                        sql = f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as EVENT_COUNT FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {date_filter} GROUP BY ACTOR_COUNTRY_CODE ORDER BY EVENT_COUNT DESC LIMIT {limit}"
                    
                    # 5. Aggregate queries (how many, count, total) - now with country support
                    elif qi['is_aggregate']:
                        is_count_aggregate = True
                        codes = get_country_codes_from_prompt(prompt)
                        if codes:
                            cf = f"ACTOR_COUNTRY_CODE = '{codes[0]}'"
                            country_filter_name = get_country(codes[0]) or codes[0]
                            sql = f"SELECT COUNT(*) as TOTAL_EVENTS FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {cf} AND {date_filter}"
                        else:
                            sql = f"SELECT COUNT(*) as TOTAL_EVENTS FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {date_filter}"
                    
                    # 6. Default: specific events query - prioritize high article count
                    else:
                        if qe:
                            try:
                                resp = qe.query(prompt)
                                sql = resp.metadata.get('sql_query')
                            except Exception: pass
                        if not sql:
                            codes = get_country_codes_from_prompt(prompt)
                            if codes:
                                if len(codes) == 1:
                                    cf = f"ACTOR_COUNTRY_CODE = '{codes[0]}'"
                                else:
                                    codes_str = "', '".join(codes)
                                    cf = f"ACTOR_COUNTRY_CODE IN ('{codes_str}')"
                                sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {cf} AND {date_filter} ORDER BY ARTICLE_COUNT DESC, DATE DESC LIMIT {fetch_limit}"
                            else:
                                # Default: get high article count events (most covered stories)
                                sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND ARTICLE_COUNT > 20 AND {date_filter} ORDER BY ARTICLE_COUNT DESC LIMIT {fetch_limit}"
                    
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

                                answer = str(llm.complete(ai_prompt))
                                st.markdown(answer)
                                
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
                                dd['COUNTRY'] = dd['ACTOR_COUNTRY_CODE'].apply(lambda x: get_country(x) or x)
                                
                                # Build summary for AI
                                country_list = []
                                for _, row in dd.iterrows():
                                    country_list.append(f"- {row['COUNTRY']}: {row['EVENT_COUNT']:,} events")
                                summary_text = "\n".join(country_list)
                                
                                ai_prompt = f"""Top {len(dd)} countries by events ({qi['period_label']}):

{summary_text}

Question: {prompt}

Briefly explain why these countries lead and any notable patterns. Keep response concise."""

                                answer = str(llm.complete(ai_prompt))
                                st.markdown(answer)
                                
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
                                    dd['COUNTRY'] = dd['ACTOR_COUNTRY_CODE'].apply(lambda x: get_country(x) or x)
                                
                                if 'NEWS_LINK' in dd.columns:
                                    headlines = []
                                    for _, row in dd.iterrows():
                                        headline = None
                                        
                                        # First try database HEADLINE
                                        db_headline = row.get('HEADLINE')
                                        if db_headline and isinstance(db_headline, str) and len(db_headline.strip()) > 25:
                                            if not (db_headline.isupper() and len(db_headline.split()) <= 3):
                                                cleaned = clean_headline(db_headline)
                                                if cleaned and len(cleaned.split()) >= 4:
                                                    headline = enhance_headline(cleaned)
                                        
                                        # Fall back to URL extraction
                                        if not headline:
                                            headline = extract_headline(
                                                row.get('NEWS_LINK', ''),
                                                None,
                                                row.get('IMPACT_SCORE', None)
                                            )
                                            if headline and len(headline.split()) < 4:
                                                headline = None
                                        
                                        headlines.append(headline)
                                    dd['HEADLINE'] = headlines
                                    
                                    # Filter out rows with no valid headline
                                    dd = dd[dd['HEADLINE'].notna()]
                                    
                                    # Deduplicate by headline to avoid showing same story multiple times
                                    dd = dd.drop_duplicates(subset=['HEADLINE'])
                                
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
                                    st.warning("📭 No events with proper headlines found for this query")
                                    answer = "No events with valid headlines were found."
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
                                    
                                    ai_prompt = f"""Events from {qi['period_label']}:

{summary_text}

Question: {prompt}

Give 2-3 sentences about each event - what happened, who's involved, why it matters."""

                                    answer = str(llm.complete(ai_prompt))
                                    st.markdown(answer)
                                    
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
                            answer = "No results found."
                    else:
                        st.warning("⚠️ Could not generate query")
                        answer = "Could not process query."
                
                st.session_state.qa_history.append({"question": prompt, "answer": answer, "sql": sql})
                # Limit history to 50 messages to prevent unbounded memory growth
                if len(st.session_state.qa_history) > 50:
                    st.session_state.qa_history = st.session_state.qa_history[-50:]
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:100]}")
