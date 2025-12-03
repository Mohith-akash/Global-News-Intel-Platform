import streamlit as st
import os
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, text, inspect
import logging
import re
from urllib.parse import urlparse
import pycountry

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
REQUIRED_ENVS = [
    "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_ACCOUNT", 
    "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA", 
    "GOOGLE_API_KEY"
]
missing = [k for k in REQUIRED_ENVS if not os.getenv(k)]
if missing:
    st.error(f"❌ CRITICAL ERROR: Missing env vars: {', '.join(missing)}")
    st.stop()

# --- SECURITY: Password Gate ---
# Set "APP_PASSWORD" in your Secrets for production. Default is 'admin123'.
REQUIRED_PASSWORD = os.getenv("APP_PASSWORD", "admin123") 

def check_password():
    """Simple password protection to prevent Snowflake bill shock."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    st.markdown("""
    <style>
        .stTextInput { width: 300px; margin: 0 auto; }
        .stButton { width: 300px; margin: 0 auto; display: block; }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔒 Classified Access")
        pwd = st.text_input("Enter Clearance Code", type="password")
        if st.button("Authenticate"):
            if pwd == REQUIRED_PASSWORD:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("Access Denied.")
    return False

if not check_password():
    st.stop()

# Constants
GEMINI_MODEL = "models/gemini-2.5-flash-preview-09-2025"
GEMINI_EMBED_MODEL = "models/embedding-001"

# Snowflake Config
SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "role": "ACCOUNTADMIN"
}

# --- 2. STYLING ---
def style_app():
    st.markdown("""
    <style>
        .stApp { background-color: #0b0f19; }
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        div[data-testid="stMetric"] { background-color: #111827; border: 1px solid #374151; border-radius: 8px; padding: 10px; }
        div[data-testid="stMetric"] label { color: #9ca3af; font-size: 0.8rem; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #f3f4f6; font-size: 1.5rem; }
        div[data-testid="stChatMessage"] { background-color: #1f2937; border: 1px solid #374151; border-radius: 12px; }
        div[data-testid="stChatMessageUser"] { background-color: #2563eb; color: white; }
        
        /* Report Box */
        .report-box { background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #475569; margin-bottom: 20px; }
        
        /* Example Questions */
        .example-box { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-top: 10px; }
        .example-item { color: #94a3b8; font-size: 0.9em; margin-bottom: 5px; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BACKEND ---
@st.cache_resource
def get_db_engine():
    url = (
        f"snowflake://{SNOWFLAKE_CONFIG['user']}:{SNOWFLAKE_CONFIG['password']}"
        f"@{SNOWFLAKE_CONFIG['account']}/{SNOWFLAKE_CONFIG['database']}/"
        f"{SNOWFLAKE_CONFIG['schema']}?warehouse={SNOWFLAKE_CONFIG['warehouse']}"
        f"&role={SNOWFLAKE_CONFIG['role']}"
    )
    return create_engine(url)

@st.cache_data(ttl=300)
def safe_read_sql(_engine, sql, params=None):
    try:
        with _engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return pd.DataFrame()

def is_safe_sql(sql: str) -> bool:
    if not sql: return False
    low = sql.lower()
    banned = ["delete ", "update ", "drop ", "alter ", "insert ", "grant ", "revoke ", "--"]
    return not any(b in low for b in banned)

def clean_key(text):
    return text.lower().replace("_", " ").replace("-", " ").strip()

# [HEAVILY IMPROVED HEADLINE CLEANER]
def format_headline(url, actor):
    if not url: return f"Update on {actor}"
    try:
        path = urlparse(url).path
        slug = path.rstrip('/').split('/')[-1]
        
        # If slug is bad (too short, digits, index.html), use previous segment
        if len(slug) < 5 or slug.replace('-','').isdigit() or 'index' in slug.lower():
            slug = path.rstrip('/').split('/')[-2]

        text = slug.replace('-', ' ').replace('_', ' ').replace('+', ' ')
        text = re.sub(r'\.html?$', '', text)
        
        # 1. REMOVE DATES (Aggressive)
        text = re.sub(r'\b20\d{2}[\s/-]?\d{1,2}[\s/-]?\d{1,2}\b', '', text) 
        text = re.sub(r'\b\d{8}\b', '', text) 
        
        # 2. REMOVE JUNK SUFFIXES (Long numbers at end)
        text = re.sub(r'\s[a-zA-Z0-9]{5,}$', '', text)
        text = re.sub(r'\s\d+$', '', text)
        
        # 3. GARBAGE DETECTOR (If mostly numbers/hex code)
        digit_count = sum(c.isdigit() for c in text)
        if len(text) > 0 and (digit_count / len(text) > 0.3):
            return f"Latest Intelligence: {actor}"

        # 4. Remove Prefixes
        text = re.sub(r'^(article|story|news|report|default)\s*', '', text, flags=re.IGNORECASE)

        headline = " ".join(text.split()).title()
        
        if len(headline) < 5: return f"Update on {actor}"
        
        return headline
    except:
        return f"Intelligence Brief: {actor}"

@st.cache_resource
def get_query_engine(_engine):
    api_key = os.getenv("GOOGLE_API_KEY")
    try:
        llm = Gemini(model=GEMINI_MODEL, api_key=api_key)
        embed_model = GeminiEmbedding(model_name=GEMINI_EMBED_MODEL, api_key=api_key)
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        inspector = inspect(_engine)
        combined_names = inspector.get_table_names() + inspector.get_view_names()
        target_table = "EVENTS_DAGSTER" 
        matched = next((t for t in combined_names if t.upper() == target_table), None)
        
        if not matched: return None
        
        sql_database = SQLDatabase(_engine, include_tables=[matched])
        query_engine = NLSQLTableQueryEngine(sql_database=sql_database, llm=llm)
        
        update_str = (
            "You are a Geopolitical Intelligence AI. Querying 'EVENTS_DAGSTER'.\n"
            "**RULES:**\n"
            "1. **INCLUDE LINKS:** ALWAYS select the `NEWS_LINK` column.\n"
            "2. **NO NULLS:** Add `WHERE IMPACT_SCORE IS NOT NULL`.\n"
            "3. **NULLS LAST:** Use `ORDER BY [col] DESC NULLS LAST`.\n"
            "4. **Response:** Return SQL in metadata."
        )
        query_engine.update_prompts({"text_to_sql_prompt": update_str})
        return query_engine
    except: return None

# --- 4. LOGIC MODULES ---

def run_manual_override(prompt, engine):
    p = prompt.lower()
    
    definitions = {
        "conflict index": "### 🛡️ Conflict Index Definition\nIntensity of negative events (Goldstein Scale).\n- **0-3:** Minor diplomatic comments.\n- **4-7:** Protests/Threats.\n- **8-10:** Military assault/War.",
        "stability": "### ⚖️ Stability Score\nOverall tone of coverage.\n- **< 40:** High Instability.\n- **> 60:** High Stability.",
        "impact score": "### 💥 Impact Score\nSignificance (-10 to 10).\n- **Negative:** Conflict.\n- **Positive:** Cooperation."
    }
    for key, explanation in definitions.items():
        if key in p:
            return True, None, explanation, "-- Knowledge Base Retrieval"

    if "compare" in p and "china" in p and ("usa" in p or "united states" in p):
        sql = """
            SELECT ACTOR_COUNTRY_CODE, COUNT(*) as ARTICLE_COUNT, AVG(SENTIMENT_SCORE) as AVG_SENTIMENT
            FROM EVENTS_DAGSTER 
            WHERE ACTOR_COUNTRY_CODE IN ('US', 'CH') 
            GROUP BY 1 ORDER BY 2 DESC
        """
        df = safe_read_sql(engine, sql)
        if not df.empty: 
            df.columns = [c.upper() for c in df.columns]
            summary = "### 🇨🇳 vs 🇺🇸 Superpower Analysis\n"
            for _, row in df.iterrows():
                summary += f"- **{row['ACTOR_COUNTRY_CODE']}**: {row['ARTICLE_COUNT']:,} events. (Sentiment: {row['AVG_SENTIMENT']:.2f})\n"
        else: summary = "No comparative data found."
        return True, df, summary, sql
        
    return False, None, None, None

def generate_briefing(engine):
    sql = """
        SELECT ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, NEWS_LINK 
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
        
        # Updated Hype Badge for 10M Scale
        st.info("🚀 **Monitoring 10M+ Incidents**\n90-Day Global Horizon (Parquet Optimized)")
        
        st.subheader("📋 Intelligence Report")
        if st.button("📄 Generate Briefing", type="primary", use_container_width=True):
            with st.spinner("Synthesizing..."):
                report, source_df = generate_briefing(engine)
                st.session_state['generated_report'] = report
                st.session_state['report_sources'] = source_df
                st.success("Report Ready!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Data Throughput")
        count_df = safe_read_sql(engine, "SELECT COUNT(*) as C FROM EVENTS_DAGSTER")
        count = count_df.iloc[0,0] if not count_df.empty else 0
        st.metric("Total Events", f"{count:,}")
        
        try:
            with engine.connect() as conn:
                res = conn.execute(text("SELECT MIN(DATE), MAX(DATE) FROM EVENTS_DAGSTER")).fetchone()
                if res and res[0]:
                    try:
                        # [FIXED DATE FORMATTING]
                        d_min = pd.to_datetime(str(res[0]), format='%Y%m%d').strftime('%d %b %Y')
                        d_max = pd.to_datetime(str(res[1]), format='%Y%m%d').strftime('%d %b %Y')
                        st.info(f"📅 **Window:**\n{d_min} to {d_max}")
                    except: st.info(f"📅 **Window:**\n{res[0]} to {res[1]}")
        except: pass
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Architecture")
        st.success("☁️ Snowflake Data Cloud")
        st.success("🧠 Google Gemini 2.5")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset Session", use_container_width=True):
            st.session_state.clear(); st.rerun()

# [NEW & IMPROVED METRICS]
def render_hud(engine):
    # 1. Volume
    sql_vol = "SELECT COUNT(*) FROM EVENTS_DAGSTER"
    
    # 2. Hotspot (Most active country code)
    sql_hotspot = "SELECT ACTOR_COUNTRY_CODE FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1"
    
    # 3. Critical Alerts (Events with Extreme Impact)
    sql_crit = "SELECT COUNT(*) FROM EVENTS_DAGSTER WHERE ABS(IMPACT_SCORE) > 6"

    # Defaults
    vol, hotspot, crit = 0, "Scanning...", 0

    try:
        # Get Volume
        df_vol = safe_read_sql(engine, sql_vol)
        if not df_vol.empty: vol = df_vol.iloc[0,0]

        # Get Hotspot (Convert Code to Name if possible)
        df_hot = safe_read_sql(engine, sql_hotspot)
        if not df_hot.empty: 
            code = df_hot.iloc[0,0]
            try:
                c = pycountry.countries.get(alpha_2=code)
                # Fallback to alpha_3 if 2 fails, or just use code
                hotspot = c.name if c else code
            except: hotspot = code

        # Get Critical Count
        df_crit = safe_read_sql(engine, sql_crit)
        if not df_crit.empty: crit = df_crit.iloc[0,0]

    except Exception as e:
        hotspot = "Offline"

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("📡 Signal Volume", f"{vol:,}", help="Total events ingested.")
    with c2: st.metric("🔥 Active Hotspot", f"{hotspot}", delta="High Activity", help="Country with highest event volume right now.")
    with c3: st.metric("🚨 Critical Alerts", f"{crit}", delta="Extreme Impact", delta_color="inverse", help="Number of high-intensity conflict/diplomacy events.")

def render_ticker(engine):
    df = safe_read_sql(engine, "SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM EVENTS_DAGSTER WHERE IMPACT_SCORE < -2 AND ACTOR_COUNTRY_CODE IS NOT NULL ORDER BY DATE DESC LIMIT 7")
    text_content = "⚠️ SYSTEM INITIALIZING... SCANNING GLOBAL FEEDS..."
    if not df.empty:
        df.columns = [c.upper() for c in df.columns]
        items = [f"⚠️ {r['MAIN_ACTOR']} ({r['ACTOR_COUNTRY_CODE']}) IMPACT: {r['IMPACT_SCORE']}" for _, r in df.iterrows()]
        text_content = " &nbsp; | &nbsp; ".join(items)
        
    html = f"""
    <!DOCTYPE html><html><head><style>
        .ticker-wrap {{ width: 100%; overflow: hidden; background-color: #7f1d1d; border-left: 5px solid #ef4444; padding: 10px 0; margin-bottom: 10px; }}
        .ticker {{ display: inline-block; white-space: nowrap; animation: marquee 35s linear infinite; font-family: monospace; font-weight: bold; font-size: 16px; color: #ffffff; }}
        @keyframes marquee {{ 0% {{ transform: translateX(100%); }} 100% {{ transform: translateX(-100%); }} }}
    </style></head><body style="margin:0;"><div class="ticker-wrap"><div class="ticker">{text_content}</div></div></body></html>
    """
    components.html(html, height=55)

def render_visuals(engine):
    t_map, t_trending, t_feed = st.tabs(["🌐 3D MAP", "🔥 TRENDING NEWS", "📋 FEED"])
    
    with t_map:
        df = safe_read_sql(engine, "SELECT ACTOR_COUNTRY_CODE as \"Country\", COUNT(*) as \"Events\", AVG(IMPACT_SCORE) as \"Impact\" FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1")
        if not df.empty:
            fig = px.choropleth(df, locations="Country", locationmode='ISO-3', color="Events", hover_name="Country", hover_data=["Impact"], color_continuous_scale="Viridis", template="plotly_dark")
            fig.update_geos(projection_type="orthographic", showcoastlines=True, showland=True, landcolor="#0f172a", showocean=True, oceancolor="#1e293b")
            fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No Map Data")

    # [TAB 2: VIRAL NEWS LEADERBOARD]
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
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Headline": st.column_config.TextColumn("Trending Topic", width="large"),
                    "ACTOR_COUNTRY_CODE": st.column_config.TextColumn("Country", width="small"),
                    "ARTICLE_COUNT": st.column_config.NumberColumn("Reports", format="%d 📉"),
                    "NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read")
                }
            )
        else:
            st.info("No trending data available yet.")

    # [TAB 3: GLOBAL FEED (CLEAN)]
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
            
            # 1. Clean Headlines
            df['Headline'] = df.apply(lambda x: format_headline(x['NEWS_LINK'], x['MAIN_ACTOR']), axis=1)
            
            # 2. Format Date (02 Dec)
            try:
                df['Date'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
            except:
                df['Date'] = df['DATE']

            # 3. Type (Impact Words)
            def get_type(score):
                if score < -3: return "🔥 Conflict"
                if score > 3: return "🤝 Diplomacy"
                return "📢 General"
            
            df['Type'] = df['IMPACT_SCORE'].apply(get_type)

            # 4. Display 4 Clean Columns
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
    engine = get_db_engine()
    
    # [CRITICAL: Init Session State]
    if 'llm_locked' not in st.session_state:
        st.session_state['llm_locked'] = False
        
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant", "content":"Hello! I am connected to the live GDELT stream. Ask me anything."}]

    render_sidebar(engine)
    st.title("Global Intelligence Command Center")
    st.markdown("**Real-Time Geopolitical Signal Processing**")
    
    if 'generated_report' in st.session_state:
        with st.container():
            st.markdown("<div class='report-box'>", unsafe_allow_html=True)
            st.subheader("📄 Executive Briefing")
            st.markdown(st.session_state['generated_report'])
            
            if 'report_sources' in st.session_state and st.session_state['report_sources'] is not None:
                st.caption("Sources:")
                try:
                    src_df = st.session_state['report_sources']
                    src_df.columns = [c.upper() for c in src_df.columns]
                    st.dataframe(
                        src_df[['NEWS_LINK']],
                        column_config={"NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read Article")},
                        hide_index=True
                    )
                except: pass

            if st.button("Close"): 
                del st.session_state['generated_report']
                if 'report_sources' in st.session_state: del st.session_state['report_sources']
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    render_hud(engine)
    render_ticker(engine)
    st.divider()
    
    c_chat, c_viz = st.columns([35, 65])
    with c_chat:
        st.subheader("💬 AI Analyst")
        
        b1, b2 = st.columns(2)
        p = None
        if b1.button("🚨 Conflicts", use_container_width=True): p = "List 3 events with lowest IMPACT_SCORE where ACTOR_COUNTRY_CODE IS NOT NULL."
        if b2.button("🇺🇳 UN Events", use_container_width=True): p = "List events where ACTOR_COUNTRY_CODE = 'US'."
        
        st.markdown("""
        <div class="example-box">
            <div class="example-item">1. Analyze the conflict trend in the Middle East.</div>
            <div class="example-item">2. Which country has the lowest sentiment score?</div>
            <div class="example-item">3. What is Conflict Index?</div>
            <div class="example-item">4. Compare media coverage of USA vs China.</div>
            <div class="example-item">5. Summarize activity involving Russia.</div>
        </div>
        """, unsafe_allow_html=True)
        
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
                            # 1. Manual Override
                            matched, m_df, m_txt, m_sql = run_manual_override(prompt, engine)
                            if matched:
                                st.markdown(m_txt)
                                if m_df is not None and not m_df.empty: 
                                    m_df.columns = [c.upper() for c in m_df.columns]
                                    if 'NEWS_LINK' in m_df.columns:
                                        st.caption("Top Trending Sources:")
                                        st.dataframe(
                                            m_df,
                                            column_config={"NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read Article")},
                                            hide_index=True
                                        )
                                    else:
                                        st.dataframe(m_df)
                                if m_sql and "-- Knowledge" not in m_sql:
                                    with st.expander("Override Trace"): st.code(m_sql, language='sql')
                                st.session_state.messages.append({"role":"assistant", "content": m_txt})
                            else:
                                # 2. AI Fallback
                                qe = get_query_engine(engine)
                                if qe:
                                    resp = qe.query(prompt)
                                    st.markdown(resp.response)
                                    
                                    if hasattr(resp, 'metadata') and 'sql_query' in resp.metadata:
                                        sql = resp.metadata['sql_query']
                                        if is_safe_sql(sql):
                                            df_context = safe_read_sql(engine, sql)
                                            if not df_context.empty:
                                                df_context.columns = [c.upper() for c in df_context.columns]
                                                if 'NEWS_LINK' in df_context.columns:
                                                    st.caption("Contextual Data:")
                                                    st.dataframe(
                                                        df_context, 
                                                        column_config={"NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read")},
                                                        hide_index=True
                                                    )
                                            with st.expander("SQL Trace"): st.code(sql, language='sql')
                                    
                                    st.session_state.messages.append({"role":"assistant", "content": resp.response})
                                else:
                                    st.error("AI Engine unavailable.")
                        except Exception as e:
                            st.error(f"Query Failed: {e}")
                        finally:
                            st.session_state['llm_locked'] = False

    with c_viz: render_visuals(engine)

if __name__ == "__main__":
    main()