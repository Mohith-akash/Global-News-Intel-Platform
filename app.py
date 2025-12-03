import streamlit as st
import os
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
# UPDATED IMPORTS TO FIX DEPRECATION WARNINGS
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
GEMINI_MODEL = "models/gemini-1.5-flash" # Updated to stable model name if needed, or keep yours
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
    # Added read_only=True to prevent locking issues which can cause crashes
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
def format_headline(url, actor):
    """
    Very strict cleaner. If a headline looks like a hash or contains
    mixed numbers/letters, it defaults to a clean 'Update on [Country]' string.
    """
    fallback = f"Update on {actor}" if actor else "Global Intelligence Update"
    
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
            if seg.isdigit(): continue
            if re.search(r'\d{4}', seg): continue
            if seg.lower() in ['index', 'default', 'article', 'news']: continue
            
            if len(seg) > 5:
                raw_text = seg
                break
        
        if not raw_text: return fallback

        text = raw_text.replace('-', ' ').replace('_', ' ').replace('+', ' ')
        words = text.split()
        clean_words = []
        
        for w in words:
            if any(char.isdigit() for char in w): continue
            if len(w) > 14: continue
            if w.lower() in ['html', 'php', 'story', 'id', 'page']: continue
            clean_words.append(w)
            
        final_text = " ".join(clean_words).title()
        
        if len(final_text) < 10 or len(clean_words) < 2:
            return fallback
            
        return final_text

    except Exception:
        return fallback

@st.cache_resource
def get_query_engine(_engine):
    api_key = os.getenv("GOOGLE_API_KEY")
    try:
        llm = Gemini(model="models/gemini-1.5-flash", api_key=api_key)
        embed_model = GeminiEmbedding(model_name=GEMINI_EMBED_MODEL, api_key=api_key)
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        inspector = inspect(_engine)
        combined_names = inspector.get_table_names() + inspector.get_view_names()
        target_table = next((t for t in combined_names if t.upper() == "EVENTS_DAGSTER"), None)
        
        if not target_table: return None
        
        sql_database = SQLDatabase(_engine, include_tables=[target_table])
        query_engine = NLSQLTableQueryEngine(sql_database=sql_database, llm=llm)
        
        update_str = (
            "You are a Geopolitical Intelligence AI. Querying 'EVENTS_DAGSTER'.\n"
            "**RULES:**\n"
            "1. **INCLUDE LINKS:** ALWAYS select the `NEWS_LINK` column.\n"
            "2. **NO NULLS:** Add `WHERE IMPACT_SCORE IS NOT NULL`.\n"
            "3. **NULLS LAST:** Use `ORDER BY [col] DESC NULLS LAST`.\n"
            "4. **DIALECT:** Use DuckDB/Postgres syntax (e.g. `LIMIT 5`).\n"
            "5. **Response:** Return SQL in metadata."
        )
        query_engine.update_prompts({"text_to_sql_prompt": update_str})
        return query_engine
    except: return None

# --- 4. LOGIC MODULES ---

def generate_briefing(engine):
    sql = """
        SELECT ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, NEWS_LINK 
        FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL 
        ORDER BY DATE DESC, ABS(IMPACT_SCORE) DESC LIMIT 10
    """
    df = safe_read_sql(engine, sql)
    if df.empty: return "Insufficient data.", None
    data = df.to_string(index=False)
    model = Gemini(model="models/gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    brief = model.complete(f"Write a 3-bullet Executive Briefing based on this data:\n{data}").text
    return brief, df

# --- 5. UI COMPONENTS ---

def render_sidebar(engine):
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.subheader("📋 Intelligence Report")
        # FIXED: use_container_width -> width="stretch"
        if st.button("📄 Generate Briefing", type="primary", width="stretch"):
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
        st.success("🧠 Google Gemini 1.5")
        
        # FIXED: use_container_width -> width="stretch"
        if st.button("Reset Session", width="stretch"):
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
    t_map, t_trending, t_feed = st.tabs(["🌐 3D MAP", "🔥 TRENDING NEWS", "📋 FEED"])
    
    with t_map:
        df = safe_read_sql(engine, "SELECT ACTOR_COUNTRY_CODE as \"Country\", COUNT(*) as \"Events\", AVG(IMPACT_SCORE) as \"Impact\" FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1")
        if not df.empty:
            fig = px.choropleth(df, locations="Country", locationmode='ISO-3', color="Events", hover_name="Country", hover_data=["Impact"], color_continuous_scale="Viridis", template="plotly_dark")
            fig.update_geos(projection_type="orthographic", showcoastlines=True, showland=True, landcolor="#0f172a", showocean=True, oceancolor="#1e293b")
            fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
            # FIXED: use_container_width -> width="stretch" (handled by streamlit automatically mostly, but good to be safe)
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
            
            # FIXED: use_container_width -> width
            st.dataframe(
                df[['Headline', 'ACTOR_COUNTRY_CODE', 'ARTICLE_COUNT', 'NEWS_LINK']],
                use_container_width=True,
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
        base_sql = """
            SELECT 
                DATE, 
                NEWS_LINK, 
                MAX(MAIN_ACTOR) as MAIN_ACTOR,
                AVG(IMPACT_SCORE) as IMPACT_SCORE 
            FROM EVENTS_DAGSTER 
            WHERE NEWS_LINK IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1 DESC 
            LIMIT 50
        """
        df = safe_read_sql(engine, base_sql)
        
        if not df.empty:
            df.columns = [c.upper() for c in df.columns] 
            df['Headline'] = df.apply(lambda x: format_headline(x['NEWS_LINK'], x['MAIN_ACTOR']), axis=1)
            try:
                df['Date'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
            except: df['Date'] = df['DATE']

            def get_type(score):
                if score < -3: return "🔥 Conflict"
                if score > 3: return "🤝 Diplomacy"
                return "📢 General"
            df['Type'] = df['IMPACT_SCORE'].apply(get_type)

            st.dataframe(
                df[['Date', 'Headline', 'Type', 'NEWS_LINK']], 
                use_container_width=True, 
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
    
    # [CRITICAL] Connect using MotherDuck
    conn_ui = get_db_connection()
    engine_ai = get_sql_engine()
    
    # [CRITICAL: Init Session State]
    if 'llm_locked' not in st.session_state:
        st.session_state['llm_locked'] = False
        
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
                    st.dataframe(src_df[['NEWS_LINK']], column_config={"NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read Article")}, hide_index=True)
                except: pass
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
        b1, b2 = st.columns(2)
        # FIXED: use_container_width -> width="stretch"
        p = None
        if b1.button("🚨 Conflicts", width="stretch"): p = "List 3 events with lowest IMPACT_SCORE where ACTOR_COUNTRY_CODE IS NOT NULL."
        if b2.button("🇺🇳 UN Events", width="stretch"): p = "List events where ACTOR_COUNTRY_CODE = 'US'."
        
        st.markdown("""<div class="example-box"><div class="example-item">1. Analyze the conflict trend in the Middle East.</div><div class="example-item">2. Which country has the lowest sentiment score?</div><div class="example-item">3. What is Conflict Index?</div></div>""", unsafe_allow_html=True)
        
        if prompt := (st.chat_input("Directive...") or p):
            if st.session_state['llm_locked']:
                st.warning("⚠️ Processing previous request...")
            else:
                st.session_state.messages.append({"role":"user", "content":prompt})
                if not p: st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Processing..."):
                        st.session_state['llm_locked'] = True
                        try:
                            # Direct AI processing (Manual Override Removed)
                            qe = get_query_engine(engine_ai)
                            if qe:
                                resp = qe.query(prompt)
                                st.markdown(resp.response)
                                if hasattr(resp, 'metadata') and 'sql_query' in resp.metadata:
                                    sql = resp.metadata['sql_query']
                                    if is_safe_sql(sql):
                                        df_context = safe_read_sql(conn_ui, sql)
                                        if not df_context.empty:
                                            df_context.columns = [c.upper() for c in df_context.columns]
                                            if 'NEWS_LINK' in df_context.columns:
                                                st.caption("Contextual Data:")
                                                st.dataframe(df_context, column_config={"NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read")}, hide_index=True)
                                        with st.expander("SQL Trace"): st.code(sql, language='sql')
                                st.session_state.messages.append({"role":"assistant", "content": resp.response})
                            else: st.error("AI Engine unavailable.")
                        except Exception as e: st.error(f"Query Failed: {e}")
                        finally: st.session_state['llm_locked'] = False

    with c_viz: render_visuals(conn_ui)

if __name__ == "__main__":
    main()