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
        section[data-testid="stSidebar"] .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
        
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

def get_iso3(country_name):
    try: return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
    except: return None

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
        
        # [SMARTER AI BRAIN]
        # We give it a "Data Dictionary" so it understands the codes and metrics.
        update_str = (
            "You are a Senior Geopolitical Intelligence Analyst. Querying 'EVENTS_DAGSTER'.\n"
            "**DATA DICTIONARY:**\n"
            "- `ACTOR_COUNTRY_CODE`: FIPS 10-4 Country Codes (e.g., 'US'=USA, 'CH'=China, 'RS'=Russia, 'UK'=United Kingdom, 'IZ'=Israel, 'PL'=Palestine).\n"
            "- `IMPACT_SCORE`: Scale -10 to 10. (Negative=Conflict/Crisis, Positive=Diplomacy/Aid).\n"
            "- `SENTIMENT_SCORE`: Tone of news coverage.\n"
            "\n"
            "**RULES:**\n"
            "1. **Narrative Responses:** After running the SQL, do NOT just list numbers. Explain what they mean. If trends are flat, say so. If US events are high, mention it implies high diplomatic activity.\n"
            "2. **Nulls:** ALWAYS `WHERE ACTOR_COUNTRY_CODE IS NOT NULL`.\n"
            "3. **Dates:** Use COUNT(*) grouped by `DATE` for trends.\n"
            "4. **Response:** Return SQL in metadata."
        )
        query_engine.update_prompts({"text_to_sql_prompt": update_str})
        return query_engine
    except: return None

# --- 4. LOGIC MODULES (The "Knowledge Base") ---

def run_manual_override(prompt, engine):
    p = prompt.lower()
    
    # [NEW] KNOWLEDGE BASE INTERCEPTOR
    # This answers definitions immediately without asking the DB
    definitions = {
        "conflict index": "### 🛡️ Conflict Index Definition\nThe **Conflict Index** is a measure of the intensity of negative geopolitical events.\n\nIt is calculated using the **Goldstein Scale**, which rates events from **-10 (Military Attack)** to **+10 (Military Assistance)**. \n- A score of **0 to 5** indicates minor diplomatic tension.\n- A score **above 5** indicates serious conflict or instability.",
        "stability": "### ⚖️ Stability Score\n**Stability** represents the overall tone of media coverage for a region.\n\n- **0-40 (Red):** Highly Unstable / Negative Coverage (War, Crisis).\n- **40-60 (Yellow):** Neutral / Mixed Coverage.\n- **60-100 (Green):** Stable / Positive Coverage (Diplomacy, Trade deals).",
        "impact score": "### 💥 Impact Score\nThe **Impact Score** quantifies the significance of an event on a scale of **-10 to 10**.\n\nNegative values denote **conflict** (e.g., riots, war), while positive values denote **cooperation** (e.g., treaties, aid). Zero indicates a neutral event or statement.",
    }
    
    for key, explanation in definitions.items():
        if key in p:
            return True, None, explanation, "-- Knowledge Base Retrieval"

    # [EXISTING] SQL OVERRIDE (USA vs China)
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
            # Map codes to names for the user
            country_map = {'US': '🇺🇸 United States', 'CH': '🇨🇳 China'}
            df['ACTOR_COUNTRY_CODE'] = df['ACTOR_COUNTRY_CODE'].map(country_map).fillna(df['ACTOR_COUNTRY_CODE'])
        
        summary = "### 🇨🇳 vs 🇺🇸 Superpower Standoff\n"
        if not df.empty:
            for _, row in df.iterrows():
                summary += f"- **{row['ACTOR_COUNTRY_CODE']}**: {row['ARTICLE_COUNT']:,} tracked events. (Avg Sentiment: {row['AVG_SENTIMENT']:.2f})\n"
            summary += "\n*Data indicates relative media volume and tone between the two nations.*"
        else: summary += "No comparative data found in current window."
        return True, df, summary, sql
        
    return False, None, None, None

def generate_briefing(engine):
    sql = """
        SELECT ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE 
        FROM EVENTS_DAGSTER WHERE ACTOR_COUNTRY_CODE IS NOT NULL 
        ORDER BY DATE DESC, ABS(IMPACT_SCORE) DESC LIMIT 20
    """
    df = safe_read_sql(engine, sql)
    if df.empty: return "Insufficient data."
    data = df.to_string(index=False)
    model = Gemini(model=GEMINI_MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
    return model.complete(f"Write a 3-bullet Executive Intel Briefing based on this data. Use country names, not codes (e.g. US=USA, RS=Russia):\n{data}").text

# --- 5. UI COMPONENTS ---

def render_sidebar(engine):
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.subheader("📋 Intelligence Report")
        if st.button("📄 Generate Briefing", type="primary", use_container_width=True):
            with st.spinner("Synthesizing..."):
                st.session_state['generated_report'] = generate_briefing(engine)
                st.success("Report Generated!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Data Throughput")
        count_df = safe_read_sql(engine, "SELECT COUNT(*) as C FROM EVENTS_DAGSTER")
        count = count_df.iloc[0,0] if not count_df.empty else 0
        st.metric("Total Events", f"{count:,}", help="Total number of geopolitical events ingested.")
        
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
    with c1: 
        st.metric("Signal Volume", f"{vol:,}", delta="Real-time", help="Live count of ingested news events.")
    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=sent, 
            title={'text':"Stability Score"}, 
            gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#10b981" if sent>50 else "#ef4444"}}
        ))
        fig.update_layout(
            height=170, 
            margin=dict(t=40,b=10,l=20,r=20), 
            paper_bgcolor="rgba(0,0,0,0)", 
            font={'color':"white"}
        )
        st.plotly_chart(fig, use_container_width=True)
    with c3: 
        st.metric(
            "Conflict Index", 
            f"{conf:.2f} / 10", 
            delta="Severity", 
            delta_color="inverse", 
            help="Average intensity of negative events (0=Peace, 10=War)."
        )

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
        .ticker-wrap {{
            width: 100%;
            overflow: hidden;
            background-color: #7f1d1d;
            border-left: 5px solid #ef4444;
            padding: 10px 0;
            margin-bottom: 10px;
            box-sizing: border-box;
        }}
        .ticker {{
            display: inline-block;
            white-space: nowrap;
            animation: marquee 35s linear infinite;
            font-family: monospace;
            font-weight: bold;
            font-size: 16px;
            color: #ffffff;
        }}
        @keyframes marquee {{
            0% {{ transform: translateX(100%); }}
            100% {{ transform: translateX(-100%); }}
        }}
    </style>
    </head>
    <body style="margin:0;">
        <div class="ticker-wrap">
            <div class="ticker">{text_content}</div>
        </div>
    </body>
    </html>
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
            GROUP BY 1 ORDER BY 1 DESC LIMIT 30
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
                "Source": st.column_config.LinkColumn("Source", display_text="Read Report")
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
            if st.button("Close"): del st.session_state['generated_report']; st.rerun()
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
        if b3.button("📈 Trends"): p = "Which country (no nulls) has highest event count?"
        
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
                            # 1. Manual Override (Knowledge Base + Hardcoded SQL)
                            matched, m_df, m_txt, m_sql = run_manual_override(prompt, engine)
                            if matched:
                                st.markdown(m_txt)
                                if m_df is not None and not m_df.empty: st.dataframe(m_df)
                                if m_sql and "-- Knowledge" not in m_sql:
                                    with st.expander("Override Trace"): st.code(m_sql, language='sql')
                                st.session_state.messages.append({"role":"assistant", "content": m_txt})
                            else:
                                # 2. AI Fallback (NL-to-SQL)
                                qe = get_query_engine(engine)
                                if qe:
                                    resp = qe.query(prompt)
                                    st.write(resp.response)
                                    if hasattr(resp, 'metadata') and resp.metadata:
                                        sql = resp.metadata.get('sql_query', '')
                                        if is_safe_sql(sql):
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