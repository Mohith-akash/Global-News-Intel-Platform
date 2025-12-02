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
        
        /* HIDE PROFILE & FOOTER */
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        
        /* Spacing */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem; }
        
        /* Metrics */
        div[data-testid="stMetric"] { background-color: #111827; border: 1px solid #374151; border-radius: 8px; padding: 15px; }
        div[data-testid="stMetric"] label { color: #9ca3af; font-size: 0.9rem; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #f3f4f6; font-size: 1.8rem; }
        
        /* Chat */
        div[data-testid="stChatMessage"] { background-color: #1f2937; border: 1px solid #374151; border-radius: 12px; }
        div[data-testid="stChatMessageUser"] { background-color: #2563eb; color: white; }
        
        /* Report Box */
        .report-box { background-color: #1e293b; padding: 25px; border-radius: 10px; border: 1px solid #475569; margin-bottom: 25px; }
        
        /* Example Box */
        .example-box {
            background-color: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 20px;
        }
        .example-item { color: #94a3b8; font-size: 0.95em; margin-bottom: 8px; }
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
        
        # [SMART BOT BRAIN 3.0]
        # Includes specific fixes for sorting, links, and NULL values
        update_str = (
            "You are a Geopolitical Intelligence AI. Querying 'EVENTS_DAGSTER'.\n"
            "**DATA DICTIONARY:**\n"
            "- `ACTOR_COUNTRY_CODE`: Country Code (e.g., 'US'=USA, 'CH'=China).\n"
            "- `IMPACT_SCORE`: Scale -10 (Violent/Conflict) to 10 (Peace/Cooperation).\n"
            "- `NEWS_LINK`: URL to the source article.\n"
            "\n"
            "**CRITICAL RULES:**\n"
            "1. **INCLUDE LINKS:** ALWAYS select the `NEWS_LINK` column in your SQL so users can read the news.\n"
            "2. **DEFINING VIOLENCE:** 'Most violent' means the LOWEST Impact Score (e.g. -10). Sort ASCENDING.\n"
            "3. **NO NULLS:** When looking for extremes (highest/lowest), you MUST add `WHERE IMPACT_SCORE IS NOT NULL`.\n"
            "4. **NULLS LAST:** Always append `NULLS LAST` to ORDER BY.\n"
            "5. **Response:** Return SQL in metadata."
        )
        query_engine.update_prompts({"text_to_sql_prompt": update_str})
        return query_engine
    except: return None

# --- 4. LOGIC MODULES ---

def run_manual_override(prompt, engine):
    p = prompt.lower()
    
    # [KNOWLEDGE BASE]
    definitions = {
        "conflict index": "### 🛡️ Conflict Index Definition\nThis measures the **intensity** of negative events (Goldstein Scale).\n- **0-3:** Minor diplomatic comments.\n- **4-7:** Protests and threats.\n- **8-10:** Military assault and war.",
        "stability": "### ⚖️ Stability Score\nRepresents the overall **tone/sentiment** of news coverage.\n- **< 40:** High Instability (Crisis/War).\n- **> 60:** High Stability (Peace/Trade).",
        "impact score": "### 💥 Impact Score\nMeasures the significance of an event (-10 to 10).\n- **Negative:** Conflict (Riots, War).\n- **Positive:** Cooperation (Aid, Treaties)."
    }
    
    for key, explanation in definitions.items():
        if key in p:
            return True, None, explanation, "-- Knowledge Base Retrieval"

    # [SQL OVERRIDE] USA vs China
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
    # [FIX] Added NEWS_LINK selection
    sql = """
        SELECT ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, NEWS_LINK 
        FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL 
        ORDER BY DATE DESC, ABS(IMPACT_SCORE) DESC LIMIT 10
    """
    df = safe_read_sql(engine, sql)
    if df.empty: return "Insufficient data.", None
    data = df.to_string(index=False)
    model = Gemini(model=GEMINI_MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
    brief = model.complete(f"Write a 3-bullet Executive Intel Briefing based on this data. Use country names:\n{data}").text
    return brief, df

# --- 5. UI COMPONENTS ---

def render_sidebar(engine):
    with st.sidebar:
        st.title("⚙️ Control Panel")
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
        
        if count == 0:
            st.error("⚠️ No Data. Check Pipeline.")
        
        try:
            with engine.connect() as conn:
                res = conn.execute(text("SELECT MIN(DATE), MAX(DATE) FROM EVENTS_DAGSTER")).fetchone()
                if res and res[0]:
                    try:
                        d_min = pd.to_datetime(str(res[0]), format='%Y%m%d').strftime('%d %b %Y')
                        d_max = pd.to_datetime(str(res[1]), format='%Y%m%d').strftime('%d %b %Y')
                        st.info(f"📅 **Window:**\n{d_min} to {d_max}")
                    except:
                        st.info(f"📅 **Window:**\n{res[0]} to {res[1]}")
        except: pass
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Architecture")
        st.success("☁️ Snowflake Data Cloud")
        st.success("🧠 Google Gemini 2.5")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset Session", use_container_width=True):
            st.session_state.clear(); st.rerun()

def render_hud(engine):
    metrics = safe_read_sql(engine, "SELECT COUNT(*) as T, AVG(SENTIMENT_SCORE) as S, AVG(CASE WHEN IMPACT_SCORE<0 THEN IMPACT_SCORE END) as C FROM EVENTS_DAGSTER")
    if metrics.empty: return
    
    vol = metrics.iloc[0,0]
    sent = ((metrics.iloc[0,1] or 0) + 10) / 20 * 100
    conf = abs(metrics.iloc[0,2] or 0)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Signal Volume", f"{vol:,}", delta="Real-time")
    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=sent, 
            title={'text':"Stability Score"}, 
            gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#10b981" if sent>50 else "#ef4444"}}
        ))
        fig.update_layout(height=180, margin=dict(t=50,b=10,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)", font={'color':"white"})
        st.plotly_chart(fig, use_container_width=True)
    with c3: st.metric("Conflict Index", f"{conf:.2f} / 10", delta="Severity", delta_color="inverse")

def render_ticker(engine):
    df = safe_read_sql(engine, "SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM EVENTS_DAGSTER WHERE IMPACT_SCORE < -2 AND ACTOR_COUNTRY_CODE IS NOT NULL ORDER BY DATE DESC LIMIT 7")
    text_content = "⚠️ SYSTEM INITIALIZING... SCANNING GLOBAL FEEDS..."
    if not df.empty:
        df.columns = [c.upper() for c in df.columns]
        items = [f"⚠️ {r['MAIN_ACTOR']} ({r['ACTOR_COUNTRY_CODE']}) IMPACT: {r['IMPACT_SCORE']}" for _, r in df.iterrows()]
        text_content = " &nbsp; | &nbsp; ".join(items)
        
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        .ticker-wrap {{ width: 100%; overflow: hidden; background-color: #7f1d1d; border-left: 5px solid #ef4444; padding: 10px 0; margin-bottom: 10px; }}
        .ticker {{ display: inline-block; white-space: nowrap; animation: marquee 35s linear infinite; font-family: monospace; font-weight: bold; font-size: 16px; color: #ffffff; }}
        @keyframes marquee {{ 0% {{ transform: translateX(100%); }} 100% {{ transform: translateX(-100%); }} }}
    </style>
    </head>
    <body style="margin:0;"><div class="ticker-wrap"><div class="ticker">{text_content}</div></div></body></html>
    """
    components.html(html, height=55)

def render_visuals(engine):
    t_map, t_trends, t_feed = st.tabs(["🌐 3D MAP", "📈 TRENDS", "📋 FEED"])
    
    with t_map:
        df = safe_read_sql(engine, "SELECT ACTOR_COUNTRY_CODE as \"Country\", COUNT(*) as \"Events\", AVG(IMPACT_SCORE) as \"Impact\" FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1")
        if not df.empty:
            fig = px.choropleth(df, locations="Country", locationmode='ISO-3', color="Events", hover_name="Country", hover_data=["Impact"], color_continuous_scale="Viridis", template="plotly_dark")
            fig.update_geos(projection_type="orthographic", showcoastlines=True, showland=True, landcolor="#0f172a", showocean=True, oceancolor="#1e293b")
            fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No Map Data")

    with t_trends:
        sql_trend = """
            SELECT DATE, COUNT(*) as V 
            FROM EVENTS_DAGSTER 
            GROUP BY 1 ORDER BY 1 ASC
        """
        df = safe_read_sql(engine, sql_trend)
        if not df.empty:
            df.columns = ["Date", "Volume"]
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            except: pass
            
            st.altair_chart(alt.Chart(df).mark_area(line={'color':'#3b82f6'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='rgba(59, 130, 246, 0.5)', offset=0), alt.GradientStop(color='rgba(59, 130, 246, 0.0)', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x='Date', y='Volume').properties(height=350), use_container_width=True)
        else: st.info("No trend data available.")

    with t_feed:
        countries = safe_read_sql(engine, "SELECT DISTINCT ACTOR_COUNTRY_CODE FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL ORDER BY 1")
        opts = ["Global Stream"] + countries.iloc[:,0].tolist() if not countries.empty else ["Global Stream"]
        sel = st.selectbox("Target Selector:", opts)
        
        base_sql = """
            SELECT 
                DATE as "Date", ACTOR_COUNTRY_CODE as "Region", MAIN_ACTOR as "Actor",
                CAST(SENTIMENT_SCORE as FLOAT) as "Sentiment",
                CASE 
                    WHEN IMPACT_SCORE < -5 THEN '🔥 Conflict'
                    WHEN IMPACT_SCORE BETWEEN -5 AND 2 THEN '😐 Neutral'
                    WHEN IMPACT_SCORE > 2 THEN '🤝 Diplomacy'
                END as "Type",
                NEWS_LINK as "Source"
            FROM EVENTS_DAGSTER 
            WHERE ACTOR_COUNTRY_CODE IS NOT NULL
        """
        params = {}
        if sel != "Global Stream":
            base_sql += " AND ACTOR_COUNTRY_CODE = :c"
            params = {"c": sel}
        base_sql += " ORDER BY DATE DESC LIMIT 50"
        
        df = safe_read_sql(engine, base_sql, params)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True, column_config={
                "Date": st.column_config.TextColumn("Date"),
                "Sentiment": st.column_config.ProgressColumn("Sentiment", min_value=-10, max_value=10, format="%.1f"),
                "Source": st.column_config.LinkColumn("Source", display_text="🔗 Read")
            })
        else: st.info("No feed data.")

# --- 6. MAIN ---
def main():
    style_app()
    engine = get_db_engine()
    
    if 'llm_locked' not in st.session_state:
        st.session_state['llm_locked'] = False

    render_sidebar(engine)
    st.title("Global Intelligence Command Center")
    st.markdown("**Real-Time Geopolitical Signal Processing**")
    
    if 'generated_report' in st.session_state:
        with st.container():
            st.markdown("<div class='report-box'>", unsafe_allow_html=True)
            st.subheader("📄 Executive Briefing")
            st.markdown(st.session_state['generated_report'])
            
            # [NEW] Source Table for Briefing
            if 'report_sources' in st.session_state and st.session_state['report_sources'] is not None:
                st.caption("Sources used for this briefing:")
                st.dataframe(
                    st.session_state['report_sources'],
                    column_config={"NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Read Article")},
                    hide_index=True
                )

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
        
        b1, b2, b3 = st.columns(3)
        p = None
        if b1.button("🚨 Conflicts"): p = "List 3 events with lowest IMPACT_SCORE where ACTOR_COUNTRY_CODE IS NOT NULL."
        if b2.button("🇺🇳 UN"): p = "List events where ACTOR_COUNTRY_CODE = 'US'."
        if b3.button("📈 Trends"): p = "Analyze the top 3 countries by event volume and summarize their geopolitical significance."
        
        st.markdown("""
        <div class="example-box">
            <div class="example-item">1. Analyze the conflict trend in the Middle East.</div>
            <div class="example-item">2. Which country has the lowest sentiment score?</div>
            <div class="example-item">3. What is Conflict Index?</div>
            <div class="example-item">4. Compare media coverage of USA vs China.</div>
            <div class="example-item">5. Summarize activity involving Russia.</div>
        </div>
        """, unsafe_allow_html=True)
        
        if "messages" not in st.session_state: st.session_state.messages = [{"role":"assistant", "content":"Hello! I am connected to the live GDELT stream. Ask me anything."}]
        for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
        
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
                                if m_df is not None and not m_df.empty: st.dataframe(m_df)
                                if m_sql and "-- Knowledge" not in m_sql:
                                    with st.expander("Override Trace"): st.code(m_sql, language='sql')
                                st.session_state.messages.append({"role":"assistant", "content": m_txt})
                            else:
                                # 2. AI Fallback
                                qe = get_query_engine(engine)
                                if qe:
                                    resp = qe.query(prompt)
                                    st.markdown(resp.response)
                                    
                                    # [NEW] SMART LINK INJECTOR
                                    # If the AI generated SQL, we run it again to get the table with LINKS
                                    if hasattr(resp, 'metadata') and 'sql_query' in resp.metadata:
                                        sql = resp.metadata['sql_query']
                                        if is_safe_sql(sql):
                                            # We run the query to get the dataframe for display
                                            df_context = safe_read_sql(engine, sql)
                                            if not df_context.empty:
                                                # If NEWS_LINK is present, we show the table with buttons
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