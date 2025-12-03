import streamlit as st
import os
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, text, inspect
import datetime
import pycountry
import logging
import streamlit.components.v1 as components
import re
from urllib.parse import urlparse, unquote
import duckdb

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Global Intelligence Platform", 
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gip")

# Validation
REQUIRED_ENVS = ["MOTHERDUCK_TOKEN", "GOOGLE_API_KEY"]
missing = [k for k in REQUIRED_ENVS if not os.getenv(k)]
if missing:
    st.error(f"❌ CRITICAL ERROR: Missing env vars: {', '.join(missing)}")
    st.stop()

# Constants
GEMINI_MODEL = "models/gemini-2.5-flash-preview-09-2025"
GEMINI_EMBED_MODEL = "models/embedding-001"

# Calculate date strings once at startup (as quoted strings for VARCHAR comparison)
TODAY = f"'{datetime.datetime.now().strftime('%Y%m%d')}'"
YESTERDAY = f"'{(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d')}'"
TWO_DAYS_AGO = f"'{(datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y%m%d')}'"
WEEK_AGO = f"'{(datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y%m%d')}'"

# --- 2. STYLING ---
def style_app():
    st.markdown("""
    <style>
        .stApp { background-color: #0b0f19; }
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem; }
        div[data-testid="stMetric"] { background-color: #111827; border: 1px solid #374151; border-radius: 8px; padding: 15px; }
        div[data-testid="stMetric"] label { color: #9ca3af; font-size: 0.9rem; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #f3f4f6; font-size: 1.8rem; }
        div[data-testid="stChatMessage"] { background-color: #1f2937; border: 1px solid #374151; border-radius: 12px; }
        div[data-testid="stChatMessageUser"] { background-color: #2563eb; color: white; }
        .report-box { background-color: #1e293b; padding: 25px; border-radius: 10px; border: 1px solid #475569; margin-bottom: 25px; }
        .example-box { background-color: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 20px; }
        .example-item { color: #94a3b8; font-size: 0.95em; margin-bottom: 8px; cursor: pointer; }
        .example-item:hover { color: #cbd5e1; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BACKEND (MOTHERDUCK) ---

@st.cache_resource
def get_db_connection():
    token = os.getenv("MOTHERDUCK_TOKEN")
    return duckdb.connect(f'md:gdelt_db?motherduck_token={token}', read_only=True)

@st.cache_resource
def get_sql_engine():
    token = os.getenv("MOTHERDUCK_TOKEN")
    return create_engine(f'duckdb:///md:gdelt_db?motherduck_token={token}')

def safe_read_sql(conn, query):
    try:
        return conn.execute(query).df()
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return pd.DataFrame()

def is_safe_sql(sql: str) -> bool:
    if not sql: return False
    low = sql.lower()
    banned = ["delete ", "update ", "drop ", "alter ", "insert ", "grant ", "revoke ", "--"]
    return not any(b in low for b in banned)

# Quick data check function
def check_data_availability(conn):
    """Check what data is actually available"""
    try:
        # Check total records
        total = safe_read_sql(conn, "SELECT COUNT(*) as c FROM EVENTS_DAGSTER")
        
        # Check date range
        date_range = safe_read_sql(conn, "SELECT MIN(DATE) as min_date, MAX(DATE) as max_date FROM EVENTS_DAGSTER")
        
        # Check recent data (last 7 days)
        week_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y%m%d')
        recent = safe_read_sql(conn, f"SELECT COUNT(*) as c FROM EVENTS_DAGSTER WHERE DATE >= '{week_ago}'")
        
        # Check sample countries
        countries = safe_read_sql(conn, "SELECT ACTOR_COUNTRY_CODE, COUNT(*) as c FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 5")
        
        return {
            'total': total.iloc[0,0] if not total.empty else 0,
            'date_range': date_range if not date_range.empty else None,
            'recent_count': recent.iloc[0,0] if not recent.empty else 0,
            'top_countries': countries if not countries.empty else None
        }
    except Exception as e:
        logger.error(f"Data check failed: {e}")
        return None

# [HEADLINE CLEANER]
def format_headline(url, actor=None):
    fallback = "Global Incident Report"
    if not url: return fallback
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        segments = [s for s in path.split('/') if s]
        if not segments: return fallback

        candidates = segments[-3:] 
        raw_text = ""
        for seg in reversed(candidates):
            seg = re.sub(r'\.(html|htm|php|asp|aspx|jsp|ece|cms)$', '', seg, flags=re.IGNORECASE)
            if seg.isdigit() or re.search(r'\d{4}', seg): continue
            if seg.lower() in ['index', 'default', 'article', 'news', 'story']: continue
            if len(seg) > 5:
                raw_text = seg; break
        
        if not raw_text: return fallback
        
        text = raw_text.replace('-', ' ').replace('_', ' ').replace('+', ' ')
        words = text.split()
        clean_words = []
        for w in words:
            if any(char.isdigit() for char in w) or len(w) > 14: continue
            if w.lower() in ['html', 'php', 'story', 'id', 'page']: continue
            clean_words.append(w)
            
        final_text = " ".join(clean_words).title()
        
        if len(final_text) < 10 or len(clean_words) < 2: return fallback
        if not re.search(r'[a-zA-Z]', final_text): return fallback
        
        return final_text
    except Exception: return fallback

# --- SIMPLIFIED AI QUERY ENGINE ---

@st.cache_resource
def get_query_engine(_engine):
    api_key = os.getenv("GOOGLE_API_KEY")
    
    llm = Gemini(
        model=GEMINI_MODEL, 
        api_key=api_key,
        temperature=0.0,  # Zero temperature for consistent SQL
    )
    embed_model = GeminiEmbedding(model_name=GEMINI_EMBED_MODEL, api_key=api_key)
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    try:
        inspector = inspect(_engine)
        combined_names = inspector.get_table_names() + inspector.get_view_names()
        target_table = next((t for t in combined_names if t.upper() == "EVENTS_DAGSTER"), None)
        
        if not target_table:
            st.error(f"❌ Table 'EVENTS_DAGSTER' not found")
            return None
        
        sql_database = SQLDatabase(_engine, include_tables=[target_table])
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database, 
            llm=llm,
            synthesize_response=True
        )
        
        # SIMPLIFIED PROMPT - Focus on basic SQL patterns
        enhanced_prompt = f"""You are a SQL query generator for geopolitical intelligence data.

**TABLE: EVENTS_DAGSTER**

**COLUMNS:**
- DATE (VARCHAR/text, format: 'YYYYMMDD', e.g., '20250104')
- MAIN_ACTOR (text)
- ACTOR_COUNTRY_CODE (text, ISO-2 codes like 'US', 'RU', 'CN')
- IMPACT_SCORE (float, -10 to +10, negative=conflict)
- ARTICLE_COUNT (integer)
- NEWS_LINK (text)
- AVG_TONE (float)
- EVENT_BASE_CODE (text)

**CURRENT DATE INFO:**
- Today: {TODAY}
- Yesterday: {YESTERDAY}
- 2 days ago: {TWO_DAYS_AGO}
- 7 days ago: {WEEK_AGO}

**CRITICAL SQL RULES:**
1. DATE is stored as VARCHAR (text in YYYYMMDD format)
2. ALWAYS use quotes in comparisons: DATE >= '20241127'
3. NEVER use date functions - just string comparison
4. Use DATE >= {WEEK_AGO} for "recent"
5. Use DATE = {TODAY} for "today"
6. Use DATE >= {TWO_DAYS_AGO} for "last 48 hours"
6. Always include: DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, NEWS_LINK
7. Always add: WHERE IMPACT_SCORE IS NOT NULL AND NEWS_LINK IS NOT NULL
8. Default LIMIT: 10
9. Use ORDER BY DATE DESC, ABS(IMPACT_SCORE) DESC

**QUERY PATTERNS:**

Pattern 1 - Recent events:
SELECT DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, NEWS_LINK, ARTICLE_COUNT
FROM EVENTS_DAGSTER
WHERE DATE >= {WEEK_AGO}
AND IMPACT_SCORE IS NOT NULL
AND NEWS_LINK IS NOT NULL
ORDER BY DATE DESC, ABS(IMPACT_SCORE) DESC
LIMIT 10

Pattern 2 - Crisis events (last 48 hours):
SELECT DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, NEWS_LINK, ARTICLE_COUNT
FROM EVENTS_DAGSTER
WHERE DATE >= {TWO_DAYS_AGO}
AND IMPACT_SCORE < -5
AND NEWS_LINK IS NOT NULL
ORDER BY IMPACT_SCORE ASC
LIMIT 10

Pattern 3 - Country comparison:
SELECT ACTOR_COUNTRY_CODE, COUNT(*) as events, AVG(IMPACT_SCORE) as avg_impact
FROM EVENTS_DAGSTER
WHERE DATE >= {WEEK_AGO}
AND ACTOR_COUNTRY_CODE IN ('RU', 'CN')
AND IMPACT_SCORE IS NOT NULL
GROUP BY ACTOR_COUNTRY_CODE

Pattern 4 - Regional conflicts:
SELECT DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, NEWS_LINK
FROM EVENTS_DAGSTER
WHERE DATE >= {WEEK_AGO}
AND ACTOR_COUNTRY_CODE IN ('IL', 'PS', 'SY', 'IQ', 'IR', 'SA', 'YE')
AND IMPACT_SCORE < -3
AND NEWS_LINK IS NOT NULL
ORDER BY DATE DESC, IMPACT_SCORE ASC
LIMIT 15

**INTERPRETATION RULES:**
- "crisis" → IMPACT_SCORE < -5
- "conflict" → IMPACT_SCORE < -3
- "recent" → DATE >= {WEEK_AGO}
- "today" → DATE = {TODAY}
- "last 48 hours" → DATE >= {TWO_DAYS_AGO}
- "this week" → DATE >= {WEEK_AGO}
- "trending" → ORDER BY ARTICLE_COUNT DESC

**COUNTRY CODE MAP:**
- United States: US
- Russia: RU
- China: CN
- Israel: IL
- Palestine: PS
- Syria: SY
- Iraq: IQ
- Iran: IR
- Ukraine: UA

Return ONLY valid SQL. No markdown, no explanations in the SQL."""

        query_engine.update_prompts({"text_to_sql_prompt": enhanced_prompt})
        return query_engine

    except Exception as e:
        st.error(f"🔥 AI Engine Error: {str(e)}")
        logger.exception("Query engine initialization failed")
        return None


# --- ENHANCED QUERY EXECUTION ---

def execute_ai_query(query_engine, prompt, conn_ui, max_retries=2):
    """Execute query with retry logic"""
    
    for attempt in range(max_retries):
        try:
            resp = query_engine.query(prompt)
            
            if hasattr(resp, 'metadata') and 'sql_query' in resp.metadata:
                sql = resp.metadata['sql_query']
                
                # Clean up SQL
                sql = sql.strip()
                if sql.startswith('```sql'):
                    sql = sql.replace('```sql', '').replace('```', '').strip()
                
                if not is_safe_sql(sql):
                    return {
                        'success': False,
                        'error': 'Generated SQL contains forbidden operations',
                        'response': None
                    }
                
                try:
                    df_context = safe_read_sql(conn_ui, sql)
                    return {
                        'success': True,
                        'response': resp.response,
                        'sql': sql,
                        'data': df_context,
                        'metadata': resp.metadata
                    }
                except Exception as sql_error:
                    logger.error(f"SQL execution failed: {sql_error}")
                    if attempt < max_retries - 1:
                        error_msg = str(sql_error)
                        refined_prompt = f"{prompt}\n\nPrevious query failed. Error: {error_msg}\nGenerate simpler SQL using DATE >= {WEEK_AGO} format (no date functions)."
                        continue
                    else:
                        return {
                            'success': False,
                            'error': f'SQL failed: {sql_error}',
                            'response': resp.response,
                            'sql': sql
                        }
            else:
                return {
                    'success': True,
                    'response': resp.response,
                    'sql': None,
                    'data': None
                }
                
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return {
                    'success': False,
                    'error': str(e),
                    'response': None
                }
    
    return {
        'success': False,
        'error': 'Max retries exceeded',
        'response': None
    }


# --- 4. LOGIC MODULES ---

def generate_briefing(engine):
    sql = f"""
        SELECT DATE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, NEWS_LINK 
        FROM EVENTS_DAGSTER 
        WHERE ACTOR_COUNTRY_CODE IS NOT NULL 
        AND DATE >= {WEEK_AGO}
        ORDER BY DATE DESC, ABS(IMPACT_SCORE) DESC 
        LIMIT 10
    """
    df = safe_read_sql(engine, sql)
    if df.empty: return "Insufficient data.", None
    data = df.to_string(index=False)
    model = Gemini(model=GEMINI_MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
    brief = model.complete(f"Write a 3-bullet Executive Briefing based on this geopolitical data:\n{data}").text
    return brief, df

# --- 5. UI COMPONENTS ---

def render_sidebar(engine):
    with st.sidebar:
        st.title("⚙️ Control Panel")
        
        # Data diagnostics
        with st.expander("🔍 Data Status", expanded=False):
            data_info = check_data_availability(engine)
            if data_info:
                st.metric("Total Records", f"{data_info['total']:,}")
                st.metric("Recent (7 days)", f"{data_info['recent_count']:,}")
                if data_info['date_range'] is not None and not data_info['date_range'].empty:
                    st.caption(f"Date Range: {data_info['date_range'].iloc[0,0]} to {data_info['date_range'].iloc[0,1]}")
                if data_info['top_countries'] is not None:
                    st.caption("Top Countries:")
                    for _, row in data_info['top_countries'].iterrows():
                        st.caption(f"  • {row[0]}: {row[1]:,}")
        
        st.subheader("📋 Intelligence Report")
        if st.button("🔄 Generate Briefing", type="primary", use_container_width=True):
            with st.spinner("Synthesizing..."):
                report, source_df = generate_briefing(engine)
                st.session_state['generated_report'] = report
                st.session_state['report_sources'] = source_df
                st.success("Report Ready!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Data Throughput")
        try:
            count_df = safe_read_sql(engine, "SELECT COUNT(*) as C FROM EVENTS_DAGSTER")
            count = count_df.iloc[0,0] if not count_df.empty else 0
            st.metric("Total Events", f"{count:,}")
        except: st.metric("Total Events", "Connecting...")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Architecture")
        st.success("🦆 MotherDuck (Cloud)")
        st.success("🧠 Google Gemini 2.5")
        
        if st.button("Reset Session", use_container_width=True):
            st.session_state.clear(); st.rerun()

def render_hud(engine):
    sql_vol = "SELECT COUNT(*) FROM EVENTS_DAGSTER"
    sql_hotspot = "SELECT ACTOR_COUNTRY_CODE FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1"
    sql_crit = "SELECT COUNT(*) FROM EVENTS_DAGSTER WHERE ABS(IMPACT_SCORE) > 6"

    vol, hotspot, crit = 0, "Scanning...", 0
    try:
        df_vol = safe_read_sql(engine, sql_vol)
        if not df_vol.empty: vol = df_vol.iloc[0,0]

        df_hot = safe_read_sql(engine, sql_hotspot)
        if not df_hot.empty: 
            code = df_hot.iloc[0,0]
            try:
                c = pycountry.countries.get(alpha_2=code)
                hotspot = c.name if c else code
            except: hotspot = code

        df_crit = safe_read_sql(engine, sql_crit)
        if not df_crit.empty: crit = df_crit.iloc[0,0]
    except Exception: hotspot = "Offline"

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("📡 Signal Volume", f"{vol:,}", help="Total events.")
    with c2: st.metric("🔥 Active Hotspot", f"{hotspot}", delta="High Activity", help="Most active country.")
    with c3: st.metric("🚨 Critical Alerts", f"{crit}", delta="Extreme Impact", delta_color="inverse", help="Events > 6 Impact.")

def render_ticker(engine):
    df = safe_read_sql(engine, f"SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM EVENTS_DAGSTER WHERE IMPACT_SCORE < -2 AND ACTOR_COUNTRY_CODE IS NOT NULL AND DATE >= {WEEK_AGO} ORDER BY DATE DESC LIMIT 7")
    text_content = "⚠️ SYSTEM INITIALIZING... SCANNING GLOBAL FEEDS..."
    if not df.empty:
        df.columns = [c.upper() for c in df.columns]
        items = [f"⚠️ {r['MAIN_ACTOR']} ({r['ACTOR_COUNTRY_CODE']}) IMPACT: {r['IMPACT_SCORE']}" for _, r in df.iterrows()]
        text_content = " &nbsp; | &nbsp; ".join(items)
    html = f"""<!DOCTYPE html><html><head><style>.ticker-wrap {{ width: 100%; overflow: hidden; background-color: #7f1d1d; border-left: 5px solid #ef4444; padding: 10px 0; margin-bottom: 10px; }} .ticker {{ display: inline-block; white-space: nowrap; animation: marquee 35s linear infinite; font-family: monospace; font-weight: bold; font-size: 16px; color: #ffffff; }} @keyframes marquee {{ 0% {{ transform: translateX(100%); }} 100% {{ transform: translateX(-100%); }} }}</style></head><body style="margin:0;"><div class="ticker-wrap"><div class="ticker">{text_content}</div></div></body></html>"""
    components.html(html, height=55)

def render_visuals(engine):
    t_map, t_trending, t_feed = st.tabs(["🌍 3D MAP", "🔥 TRENDING NEWS", "📋 FEED"])
    with t_map:
        df = safe_read_sql(engine, "SELECT ACTOR_COUNTRY_CODE as \"Country\", COUNT(*) as \"Events\", AVG(IMPACT_SCORE) as \"Impact\" FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1")
        if not df.empty:
            fig = px.choropleth(df, locations="Country", locationmode='ISO-3', color="Events", hover_name="Country", hover_data=["Impact"], color_continuous_scale="Viridis", template="plotly_dark")
            fig.update_geos(projection_type="orthographic", showcoastlines=True, showland=True, landcolor="#0f172a", showocean=True, oceancolor="#1e293b")
            fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No Map Data")

    with t_trending:
        sql = f"""
            SELECT NEWS_LINK, ACTOR_COUNTRY_CODE, ARTICLE_COUNT, MAIN_ACTOR
            FROM EVENTS_DAGSTER 
            WHERE NEWS_LINK IS NOT NULL 
            AND DATE >= {WEEK_AGO}
            ORDER BY ARTICLE_COUNT DESC 
            LIMIT 50
        """
        df = safe_read_sql(engine, sql)
        if not df.empty:
            df.columns = [c.upper() for c in df.columns]
            df['Headline'] = df.apply(lambda x: format_headline(x['NEWS_LINK'], x['MAIN_ACTOR']), axis=1)
            df = df.drop_duplicates(subset=['Headline']).head(20)
            st.dataframe(
                df[['Headline', 'ACTOR_COUNTRY_CODE', 'ARTICLE_COUNT', 'NEWS_LINK']],
                hide_index=True,
                column_config={
                    "Headline": st.column_config.TextColumn("Trending Topic", width="large"),
                    "ACTOR_COUNTRY_CODE": st.column_config.TextColumn("Country", width="small"),
                    "ARTICLE_COUNT": st.column_config.NumberColumn("Reports", format="%d 📉"),
                    "NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read")
                }
            )
        else: st.info("No trending data available yet.")

    with t_feed:
        base_sql = f"""
            SELECT DATE, NEWS_LINK, MAX(MAIN_ACTOR) as MAIN_ACTOR, AVG(IMPACT_SCORE) as IMPACT_SCORE 
            FROM EVENTS_DAGSTER 
            WHERE NEWS_LINK IS NOT NULL 
            AND DATE >= {WEEK_AGO}
            GROUP BY 1, 2 
            ORDER BY 1 DESC 
            LIMIT 50
        """
        df = safe_read_sql(engine, base_sql)
        if not df.empty:
            df.columns = [c.upper() for c in df.columns] 
            df['Headline'] = df.apply(lambda x: format_headline(x['NEWS_LINK'], x['MAIN_ACTOR']), axis=1)
            try: df['Date'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b %Y')
            except: df['Date'] = df['DATE']
            df['Type'] = df['IMPACT_SCORE'].apply(lambda x: "🔥 Conflict" if x < -3 else ("🤝 Diplomacy" if x > 3 else "📢 General"))

            st.dataframe(
                df[['Date', 'Headline', 'Type', 'NEWS_LINK']], 
                hide_index=True, 
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Headline": st.column_config.TextColumn("Headline", width="large"),
                    "Type": st.column_config.TextColumn("Category", width="small"),
                    "NEWS_LINK": st.column_config.LinkColumn("Link", display_text="🔗 Read")
                }
            )
        else: st.info("No feed data.")

# --- 6. MAIN ---
def main():
    style_app()
    conn_ui = get_db_connection()
    engine_ai = get_sql_engine()
    
    if 'llm_locked' not in st.session_state: st.session_state['llm_locked'] = False
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant", "content":"Hello! I'm your geopolitical intelligence analyst. Ask me about recent global events."}]

    render_sidebar(conn_ui)
    st.title("Global Intelligence Command Center")
    st.markdown("**Real-Time Geopolitical Signal Processing**")
    
    if 'generated_report' in st.session_state:
        with st.container():
            st.markdown("<div class='report-box'>", unsafe_allow_html=True)
            st.subheader("📄 Executive Briefing")
            st.markdown(st.session_state['generated_report'])
            
            if 'report_sources' in st.session_state and st.session_state['report_sources'] is not None:
                try:
                    src_df = st.session_state['report_sources']
                    src_df.columns = [c.upper() for c in src_df.columns]
                    
                    if 'DATE' in src_df.columns:
                        try:
                            src_df['DATE'] = pd.to_datetime(src_df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b %Y')
                        except: pass
                    
                    if 'NEWS_LINK' in src_df.columns:
                        src_df['Headline'] = src_df.apply(lambda x: format_headline(x['NEWS_LINK']), axis=1)
                        
                        disp_cols = ['DATE', 'Headline', 'IMPACT_SCORE', 'NEWS_LINK']
                        disp_cols = [c for c in disp_cols if c in src_df.columns]
                        
                        st.caption("Intelligence Sources:")
                        st.dataframe(
                            src_df[disp_cols].rename(columns={'IMPACT_SCORE': 'Intensity'}),
                            column_config={
                                "NEWS_LINK": st.column_config.LinkColumn("Link", display_text="🔗 Read"),
                                "Headline": st.column_config.TextColumn("Incident / Headline", width="large"),
                                "DATE": st.column_config.TextColumn("Date", width="small")
                            }, 
                            hide_index=True
                        )
                except Exception as e: 
                    st.error(f"Error displaying sources: {e}")
                    
            if st.button("Close"): 
                del st.session_state['generated_report']
                if 'report_sources' in st.session_state: del st.session_state['report_sources']
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    render_hud(conn_ui)
    render_ticker(conn_ui)
    st.divider()
    
    c_chat, c_viz = st.columns([35, 65])
    with c_chat:
        st.subheader("💬 AI Analyst")
        
        st.markdown("""
        <div class="example-box">
            <strong>🎯 Try These Questions:</strong>
            <div class="example-item">• Show me crisis events from the last 48 hours</div>
            <div class="example-item">• What are recent conflicts in the Middle East?</div>
            <div class="example-item">• Compare activity between Russia and China this week</div>
            <div class="example-item">• Show me today's high-impact events</div>
            <div class="example-item">• What events happened in Ukraine recently?</div>
            <div class="example-item">• Show trending stories with high media coverage</div>
        </div>
        """, unsafe_allow_html=True)
        
        if prompt := st.chat_input("Ask about global events..."):
            if st.session_state['llm_locked']:
                st.warning("⚠️ Processing previous request...")
            else:
                st.session_state.messages.append({"role":"user", "content":prompt})
                st.chat_message("user").write(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Analyzing..."):
                        st.session_state['llm_locked'] = True
                        try:
                            qe = get_query_engine(engine_ai)
                            if qe:
                                result = execute_ai_query(qe, prompt, conn_ui)
                                
                                if result['success']:
                                    st.markdown(result['response'])
                                    
                                    if result['data'] is not None and not result['data'].empty:
                                        df_context = result['data']
                                        df_context.columns = [c.upper() for c in df_context.columns]
                                        
                                        if 'DATE' in df_context.columns:
                                            try:
                                                df_context['DATE'] = pd.to_datetime(
                                                    df_context['DATE'].astype(str), 
                                                    format='%Y%m%d'
                                                ).dt.strftime('%d %b %Y')
                                            except: pass
                                        
                                        if 'NEWS_LINK' in df_context.columns:
                                            df_context['Headline'] = df_context.apply(
                                                lambda x: format_headline(x.get('NEWS_LINK', '')), 
                                                axis=1
                                            )
                                            
                                            col_map = {
                                                'DATE': 'Date', 
                                                'IMPACT_SCORE': 'Impact', 
                                                'NEWS_LINK': 'Source',
                                                'ARTICLE_COUNT': 'Coverage'
                                            }
                                            df_context = df_context.rename(columns=col_map)
                                            
                                            priority_cols = ['Date', 'Headline', 'Impact', 'Coverage', 'Source']
                                            display_cols = [c for c in priority_cols if c in df_context.columns]
                                            
                                            st.caption("📊 Supporting Data:")
                                            st.dataframe(
                                                df_context[display_cols],
                                                column_config={
                                                    "Source": st.column_config.LinkColumn("🔗", display_text="Read"),
                                                    "Headline": st.column_config.TextColumn("Event", width="large"),
                                                    "Impact": st.column_config.NumberColumn("Impact", format="%.1f"),
                                                    "Coverage": st.column_config.NumberColumn("Media", format="%d 📰")
                                                },
                                                hide_index=True
                                            )
                                        else:
                                            st.dataframe(df_context, hide_index=True)
                                    
                                    if result['sql']:
                                        with st.expander("🔍 SQL Query"):
                                            st.code(result['sql'], language='sql')
                                    
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": result['response']
                                    })
                                else:
                                    st.error(f"❌ {result['error']}")
                                    st.info("💡 Try rephrasing your question or ask about 'recent events' or 'today's conflicts'")
                                    if result.get('sql'):
                                        with st.expander("🔍 Generated SQL (Debug)"):
                                            st.code(result['sql'], language='sql')
                            else:
                                st.error("AI Engine unavailable.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                            logger.exception("Query execution failed")
                        finally:
                            st.session_state['llm_locked'] = False

    with c_viz: 
        render_visuals(conn_ui)

if __name__ == "__main__":
    main()