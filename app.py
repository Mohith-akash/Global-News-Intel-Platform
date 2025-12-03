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
        .example-item { color: #94a3b8; font-size: 0.95em; margin-bottom: 8px; }
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

# [HEADLINE CLEANER - STRICT MODE]
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

# --- OPTIMIZED AI QUERY ENGINE ---

@st.cache_resource
def get_query_engine(_engine):
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # 1. Init Models with Better Settings
    llm = Gemini(
        model=GEMINI_MODEL, 
        api_key=api_key,
        temperature=0.1,  # Lower temperature for more precise SQL generation
    )
    embed_model = GeminiEmbedding(model_name=GEMINI_EMBED_MODEL, api_key=api_key)
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 2. Verify Table Exists
    try:
        inspector = inspect(_engine)
        combined_names = inspector.get_table_names() + inspector.get_view_names()
        target_table = next((t for t in combined_names if t.upper() == "EVENTS_DAGSTER"), None)
        
        if not target_table:
            st.error(f"❌ Table 'EVENTS_DAGSTER' not found. Found: {combined_names}")
            return None
        
        # 3. Get Schema Information
        with _engine.connect() as conn:
            schema_query = f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{target_table}'
            """
            try:
                schema_df = pd.read_sql(schema_query, conn)
                schema_info = schema_df.to_string(index=False)
            except:
                schema_info = "Schema unavailable"
        
        # 4. Build Engine
        sql_database = SQLDatabase(_engine, include_tables=[target_table])
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database, 
            llm=llm,
            synthesize_response=True  # Enable response synthesis
        )
        
        # 5. ENHANCED SYSTEM PROMPT with Schema Context
        enhanced_prompt = f"""You are an elite Geopolitical Intelligence Analyst with access to the GDELT database.

**DATABASE SCHEMA:**
Table: EVENTS_DAGSTER
{schema_info}

**KEY COLUMNS:**
- DATE: Format YYYYMMDD (e.g., 20250104)
- MAIN_ACTOR: Primary entity involved
- ACTOR_COUNTRY_CODE: ISO-2 country code
- EVENT_BASE_CODE: CAMEO event type
- IMPACT_SCORE: Goldstein scale (-10 to +10, negative=conflict, positive=cooperation)
- ARTICLE_COUNT: Media coverage intensity
- NEWS_LINK: Source URL
- AVG_TONE: Sentiment (-100 to +100)

**INTELLIGENCE ANALYSIS GUIDELINES:**

1. **TEMPORAL CONTEXT:**
   - Today's date is {datetime.datetime.now().strftime('%Y-%m-%d')} (format: {datetime.datetime.now().strftime('%Y%m%d')} in DATE column)
   - "Recent" = last 7 days
   - "Latest" = last 24-48 hours
   - Always filter by date when analyzing trends

2. **SQL GENERATION RULES:**
   - ALWAYS include: DATE, MAIN_ACTOR, NEWS_LINK, IMPACT_SCORE, ACTOR_COUNTRY_CODE
   - Filter NULL values: `WHERE IMPACT_SCORE IS NOT NULL AND NEWS_LINK IS NOT NULL`
   - For "recent" queries: `WHERE DATE >= {(datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y%m%d')}`
   - For "today": `WHERE DATE = {datetime.datetime.now().strftime('%Y%m%d')}`
   - Sort by relevance: `ORDER BY DATE DESC, ABS(IMPACT_SCORE) DESC`
   - Default limit: `LIMIT 10` (adjust based on query)
   - Use DuckDB syntax (e.g., `LIMIT`, not `TOP`)

3. **ANALYTICAL DEPTH:**
   - Identify patterns and anomalies
   - Compare IMPACT_SCORE with ARTICLE_COUNT (high coverage + high impact = critical)
   - Note sentiment trends (AVG_TONE)
   - Highlight escalations (multiple negative events for same actor)
   - Provide geopolitical context when relevant

4. **QUERY INTERPRETATION:**
   - "Crisis" → IMPACT_SCORE < -6
   - "Conflict" → IMPACT_SCORE < -3
   - "Cooperation" → IMPACT_SCORE > 5
   - "Trending" → High ARTICLE_COUNT
   - "Active" → Multiple recent events
   - Country names → Use ACTOR_COUNTRY_CODE (convert to ISO-2)

5. **RESPONSE QUALITY:**
   - Start with key findings (executive summary style)
   - Support claims with specific data points
   - Mention data limitations when relevant
   - Suggest follow-up queries for deeper analysis
   - Format numbers clearly (dates, scores, counts)

6. **ERROR HANDLING:**
   - If no results: explain why and suggest alternative queries
   - If ambiguous request: clarify interpretation before querying
   - If time-sensitive: prioritize recent data

**EXAMPLE QUERIES:**

"What's happening in Ukraine?"
→ SELECT DATE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK 
   FROM EVENTS_DAGSTER 
   WHERE ACTOR_COUNTRY_CODE = 'UA' 
   AND DATE >= {(datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y%m%d')}
   ORDER BY DATE DESC, ABS(IMPACT_SCORE) DESC 
   LIMIT 15

"Show me conflicts today"
→ SELECT DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, NEWS_LINK
   FROM EVENTS_DAGSTER
   WHERE DATE = {datetime.datetime.now().strftime('%Y%m%d')}
   AND IMPACT_SCORE < -3
   ORDER BY IMPACT_SCORE ASC
   LIMIT 20

"Which countries have the most activity?"
→ SELECT ACTOR_COUNTRY_CODE, 
          COUNT(*) as event_count,
          AVG(IMPACT_SCORE) as avg_impact,
          SUM(ARTICLE_COUNT) as media_coverage
   FROM EVENTS_DAGSTER
   WHERE DATE >= {(datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y%m%d')}
   AND ACTOR_COUNTRY_CODE IS NOT NULL
   GROUP BY ACTOR_COUNTRY_CODE
   ORDER BY event_count DESC
   LIMIT 10

Always return valid SQL in metadata and provide analytical insights in your response."""

        query_engine.update_prompts({"text_to_sql_prompt": enhanced_prompt})
        return query_engine

    except Exception as e:
        st.error(f"🔥 AI Engine Crash: {str(e)}")
        logger.exception("Query engine initialization failed")
        return None


# --- ENHANCED QUERY EXECUTION WITH RETRY LOGIC ---

def execute_ai_query(query_engine, prompt, conn_ui, max_retries=2):
    """Enhanced query execution with error handling and retry logic"""
    
    for attempt in range(max_retries):
        try:
            # Generate response
            resp = query_engine.query(prompt)
            
            # Validate SQL if present
            if hasattr(resp, 'metadata') and 'sql_query' in resp.metadata:
                sql = resp.metadata['sql_query']
                
                # Safety check
                if not is_safe_sql(sql):
                    return {
                        'success': False,
                        'error': 'Generated SQL contains forbidden operations',
                        'response': None
                    }
                
                # Execute SQL
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
                    if attempt < max_retries - 1:
                        # Retry with error feedback
                        error_msg = str(sql_error)
                        refined_prompt = f"{prompt}\n\nPrevious SQL failed with error: {error_msg}\nPlease generate corrected SQL."
                        continue
                    else:
                        return {
                            'success': False,
                            'error': f'SQL execution failed: {sql_error}',
                            'response': resp.response,
                            'sql': sql
                        }
            else:
                # No SQL generated (direct response)
                return {
                    'success': True,
                    'response': resp.response,
                    'sql': None,
                    'data': None
                }
                
        except Exception as e:
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
    sql = """
        SELECT DATE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, NEWS_LINK 
        FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL 
        ORDER BY DATE DESC, ABS(IMPACT_SCORE) DESC LIMIT 10
    """
    df = safe_read_sql(engine, sql)
    if df.empty: return "Insufficient data.", None
    data = df.to_string(index=False)
    model = Gemini(model=GEMINI_MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
    brief = model.complete(f"Write a 3-bullet Executive Briefing based on this data:\n{data}").text
    return brief, df

# --- 5. UI COMPONENTS ---

def render_sidebar(engine):
    with st.sidebar:
        st.title("⚙️ Control Panel")
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
    df = safe_read_sql(engine, "SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM EVENTS_DAGSTER WHERE IMPACT_SCORE < -2 AND ACTOR_COUNTRY_CODE IS NOT NULL ORDER BY DATE DESC LIMIT 7")
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
        sql = """
            SELECT NEWS_LINK, ACTOR_COUNTRY_CODE, ARTICLE_COUNT, MAIN_ACTOR
            FROM EVENTS_DAGSTER 
            WHERE NEWS_LINK IS NOT NULL 
            ORDER BY ARTICLE_COUNT DESC 
            LIMIT 70
        """
        df = safe_read_sql(engine, sql)
        if not df.empty:
            df.columns = [c.upper() for c in df.columns]
            df['Headline'] = df.apply(lambda x: format_headline(x['NEWS_LINK'], x['MAIN_ACTOR']), axis=1)
            df = df.drop_duplicates(subset=['Headline']).head(20)
            st.dataframe(
                df[['Headline', 'ACTOR_COUNTRY_CODE', 'ARTICLE_COUNT', 'NEWS_LINK']],
                use_container_width=True, hide_index=True,
                column_config={
                    "Headline": st.column_config.TextColumn("Trending Topic", width="large"),
                    "ACTOR_COUNTRY_CODE": st.column_config.TextColumn("Country", width="small"),
                    "ARTICLE_COUNT": st.column_config.NumberColumn("Reports", format="%d 📉"),
                    "NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read")
                }
            )
        else: st.info("No trending data available yet.")

    with t_feed:
        base_sql = """
            SELECT DATE, NEWS_LINK, MAX(MAIN_ACTOR) as MAIN_ACTOR, AVG(IMPACT_SCORE) as IMPACT_SCORE 
            FROM EVENTS_DAGSTER WHERE NEWS_LINK IS NOT NULL GROUP BY 1, 2 ORDER BY 1 DESC LIMIT 50
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
                use_container_width=True, hide_index=True, 
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
        st.session_state.messages = [{"role":"assistant", "content":"Hello! I am connected to MotherDuck GDELT stream. Ask me anything."}]

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
        
        st.markdown(f"""
        <div class="example-box">
            <strong>🎯 Smart Query Examples:</strong>
            <div class="example-item">1. What are the top 5 crisis events in the last 48 hours?</div>
            <div class="example-item">2. Compare military activity between Russia and China this week</div>
            <div class="example-item">3. Which countries had the most diplomatic events today?</div>
            <div class="example-item">4. Show me trending stories with high media coverage</div>
            <div class="example-item">5. Identify any escalating conflicts in the Middle East</div>
            <div class="example-item">6. What's the sentiment trend for United States events?</div>
        </div>
        """, unsafe_allow_html=True)
        
        if prompt := st.chat_input("Enter intelligence query..."):
            if st.session_state['llm_locked']:
                st.warning("⚠️ Processing previous request...")
            else:
                st.session_state.messages.append({"role":"user", "content":prompt})
                st.chat_message("user").write(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Analyzing intelligence data..."):
                        st.session_state['llm_locked'] = True
                        try:
                            qe = get_query_engine(engine_ai)
                            if qe:
                                # Execute with enhanced error handling
                                result = execute_ai_query(qe, prompt, conn_ui)
                                
                                if result['success']:
                                    # Display response
                                    st.markdown(result['response'])
                                    
                                    # Display data if available
                                    if result['data'] is not None and not result['data'].empty:
                                        df_context = result['data']
                                        df_context.columns = [c.upper() for c in df_context.columns]
                                        
                                        # Format dates
                                        if 'DATE' in df_context.columns:
                                            try:
                                                df_context['DATE'] = pd.to_datetime(
                                                    df_context['DATE'].astype(str), 
                                                    format='%Y%m%d'
                                                ).dt.strftime('%d %b %Y')
                                            except: pass
                                        
                                        # Create headlines if NEWS_LINK exists
                                        if 'NEWS_LINK' in df_context.columns:
                                            df_context['Headline'] = df_context.apply(
                                                lambda x: format_headline(x['NEWS_LINK']), 
                                                axis=1
                                            )
                                            
                                            col_map = {
                                                'DATE': 'Date', 
                                                'IMPACT_SCORE': 'Impact', 
                                                'NEWS_LINK': 'Source',
                                                'ARTICLE_COUNT': 'Coverage'
                                            }
                                            df_context = df_context.rename(columns=col_map)
                                            
                                            # Smart column selection
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
                                                hide_index=True,
                                                use_container_width=True
                                            )
                                        else:
                                            st.dataframe(df_context, hide_index=True)
                                    
                                    # SQL trace
                                    if result['sql']:
                                        with st.expander("🔍 SQL Trace"):
                                            st.code(result['sql'], language='sql')
                                    
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": result['response']
                                    })
                                else:
                                    st.error(f"❌ Query failed: {result['error']}")
                                    if result['sql']:
                                        with st.expander("Debug: Generated SQL"):
                                            st.code(result['sql'], language='sql')
                            else:
                                st.error("AI Engine unavailable.")
                        except Exception as e:
                            st.error(f"Query Failed: {e}")
                            logger.exception("Query execution failed")
                        finally:
                            st.session_state['llm_locked'] = False

    with c_viz: render_visuals(conn_ui)

if __name__ == "__main__":
    main()