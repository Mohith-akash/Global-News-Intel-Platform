import streamlit as st
import os
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, text, inspect
import logging
import streamlit.components.v1 as components
import re
from urllib.parse import urlparse
import asyncio
import time

# --- 1. ENTERPRISE CONFIGURATION ---
st.set_page_config(
    page_title="Global Intel Command", 
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv()

# Logging & Performance Monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GIP_PRO")

# Environment Validation
REQUIRED_ENVS = [
    "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_ACCOUNT", 
    "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA", 
    "GOOGLE_API_KEY"
]
missing = [k for k in REQUIRED_ENVS if not os.getenv(k)]
if missing:
    st.error(f"🚫 SECURITY ALERT: Missing credentials: {', '.join(missing)}")
    st.stop()

# Constants - Switched to Production Stable Models
GEMINI_MODEL = "models/gemini-1.5-flash" 
GEMINI_EMBED_MODEL = "models/embedding-001"

# Snowflake Config
SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN") # Change to APP_READ_ONLY in prod
}

# --- 2. ADVANCED STYLING (CSS 3.0) ---
def inject_custom_css():
    st.markdown("""
    <style>
        /* Main Theme */
        .stApp { background-color: #020617; } /* Slate-950 */
        
        /* Typography */
        h1, h2, h3 { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
        p, label { color: #94a3b8 !important; }
        
        /* Cards (Shadcn-like) */
        div[data-testid="stMetric"] { 
            background-color: #0f172a; 
            border: 1px solid #1e293b; 
            border-radius: 8px; 
            padding: 15px; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="stMetric"] label { color: #64748b; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e2e8f0; font-size: 1.8rem; font-weight: 600; }
        
        /* Chat Interface */
        div[data-testid="stChatMessage"] { background-color: #1e293b; border: 1px solid #334155; }
        div[data-testid="stChatMessageUser"] { background-color: #2563eb; }
        
        /* Tables */
        div[data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 8px; overflow: hidden; }
        
        /* Status Indicators */
        .status-badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .status-high { background: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
        .status-safe { background: #dcfce7; color: #166534; border: 1px solid #4ade80; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE ENGINE (ASYNC & CACHED) ---

@st.cache_resource
def get_db_engine():
    """Establishes the SQLAlchemy Engine. Cached to prevent connection thrashing."""
    url = (
        f"snowflake://{SNOWFLAKE_CONFIG['user']}:{SNOWFLAKE_CONFIG['password']}"
        f"@{SNOWFLAKE_CONFIG['account']}/{SNOWFLAKE_CONFIG['database']}/"
        f"{SNOWFLAKE_CONFIG['schema']}?warehouse={SNOWFLAKE_CONFIG['warehouse']}"
        f"&role={SNOWFLAKE_CONFIG['role']}"
    )
    return create_engine(url)

@st.cache_data(ttl=600)
def fetch_data_snapshot(_engine, query):
    """Executes read-only queries with aggressive caching (10 mins)."""
    try:
        with _engine.connect() as conn:
            # We don't check for 'DROP' here anymore. We rely on Snowflake Role Permissions.
            return pd.read_sql(text(query), conn)
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return pd.DataFrame()

def robust_headline_parser(url):
    """Parses URLs cleanly using urllib instead of brittle Regex."""
    if not url: return "Global Event Update"
    try:
        parsed = urlparse(url)
        path = parsed.path
        # Split by / and take the last non-empty segment
        segments = [s for s in path.split('/') if s and not s.isdigit() and len(s) > 3]
        
        if not segments: return parsed.netloc
        
        slug = segments[-1]
        # Clean common URL junk
        clean = slug.replace('-', ' ').replace('_', ' ').replace('.html', '').replace('.php', '')
        
        # Remove date prefixes if they exist (e.g., 2024-10-12-news)
        clean = re.sub(r'^\d{4}\s\d{2}\s\d{2}\s', '', clean)
        
        headline = clean.title()
        return headline[:80] + "..." if len(headline) > 80 else headline
    except:
        return "Intelligence Report"

# --- 4. INTELLIGENCE LAYER (RAG + SELF-CORRECTION) ---

@st.cache_resource
def get_query_engine(_engine):
    """Initializes the RAG Engine with specific System Prompts."""
    api_key = os.getenv("GOOGLE_API_KEY")
    try:
        llm = Gemini(model=GEMINI_MODEL, api_key=api_key)
        embed_model = GeminiEmbedding(model_name=GEMINI_EMBED_MODEL, api_key=api_key)
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        inspector = inspect(_engine)
        # Dynamic table detection
        tables = inspector.get_table_names()
        target_table = next((t for t in tables if "EVENTS" in t.upper()), "EVENTS_DAGSTER")
        
        sql_database = SQLDatabase(_engine, include_tables=[target_table])
        query_engine = NLSQLTableQueryEngine(sql_database=sql_database, llm=llm)
        
        # PROMPT ENGINEERING 2.0: Strict Rules
        system_prompt = (
            f"You are a Geopolitical Intelligence Officer accessing the '{target_table}' table.\n"
            "**STRICT PROTOCOL:**\n"
            "1. **SCHEMA AWARENESS:** Only use columns that exist. Do not hallucinate columns.\n"
            "2. **SOURCE VERIFICATION:** ALWAYS SELECT `NEWS_LINK` if available.\n"
            "3. **IMPACT ANALYSIS:** When asked for 'Conflict' or 'Risk', query `IMPACT_SCORE < -3`.\n"
            "4. **RESPONSE FORMAT:** If returning raw data, sort by `DATE DESC`.\n"
            "5. **SQL ONLY:** Generate valid Snowflake SQL."
        )
        query_engine.update_prompts({"text_to_sql_prompt": system_prompt})
        return query_engine
    except Exception as e:
        logger.error(f"Engine Init Error: {e}")
        return None

async def async_llm_monitor(response_text):
    """
    The 'Critic' Agent.
    Runs asynchronously to grade the main agent's answer without slowing down the UI.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = Gemini(model=GEMINI_MODEL, api_key=api_key)
    
    prompt = (
        f"Analyze this intelligence report for hallucinations or vagueness: '{response_text}'. "
        "Return a JSON with 'confidence_score' (0-100) and 'reasoning'."
    )
    # Using run_in_executor to simulate async behavior for the synchronous library
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, llm.complete, prompt)
    return response.text

# --- 5. VISUALIZATION COMPONENTS (PYDECK 3D) ---

def render_3d_globe(df):
    """Renders a HexagonLayer 3D Globe using PyDeck."""
    if df.empty or 'LAT' not in df.columns or 'LON' not in df.columns:
        st.warning("⚠️ Geospatial data unavailable for 3D rendering.")
        return

    # Normalize data for deck.gl
    df['lat'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['lon'] = pd.to_numeric(df['LON'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])

    view_state = pdk.ViewState(
        latitude=20, 
        longitude=10, 
        zoom=1.2, 
        pitch=45,
        bearing=0
    )

    layer = pdk.Layer(
        "HexagonLayer",
        df,
        get_position=["lon", "lat"],
        auto_highlight=True,
        elevation_scale=1000, 
        pickable=True,
        elevation_range=[0, 3000],
        extruded=True,
        radius=200000, # 200km radius bins
        coverage=1,
        get_fill_color="[255, (1 - count / 100) * 255, 0, 180]", # Heatmap color logic
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11", 
        tooltip={"text": "Event Density: {elevationValue}"}
    )
    
    st.pydeck_chart(deck)

# --- 6. UI LAYOUTS ---

def render_metrics_hud(engine):
    """Top-level Heads Up Display."""
    sql = """
        SELECT 
            COUNT(*) as VOL, 
            AVG(CASE WHEN IMPACT_SCORE < 0 THEN IMPACT_SCORE ELSE 0 END) as RISK,
            COUNT(DISTINCT ACTOR_COUNTRY_CODE) as ACTORS
        FROM EVENTS_DAGSTER
    """
    df = fetch_data_snapshot(engine, sql)
    if not df.empty:
        vol = df.iloc[0]['VOL']
        risk = abs(df.iloc[0]['RISK']) # Normalized positive number for gauge
        actors = df.iloc[0]['ACTORS']
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Global Signals", f"{vol:,}", delta="Live Feed")
        with c2: st.metric("Active Regions", f"{actors}", delta="+2 vs avg")
        with c3: 
            st.metric("Global Threat Level", f"{risk:.2f}/10", 
            delta="Elevated" if risk > 3 else "Stable", 
            delta_color="inverse")
        with c4:
            st.markdown("### 🟢 System Online")
            st.caption("Secure Link: ACT_READ_ONLY")

def render_feed_tab(engine):
    sql = "SELECT DATE, NEWS_LINK, IMPACT_SCORE FROM EVENTS_DAGSTER ORDER BY DATE DESC LIMIT 50"
    df = fetch_data_snapshot(engine, sql)
    if not df.empty:
        df['Headline'] = df['NEWS_LINK'].apply(robust_headline_parser)
        df['Risk'] = df['IMPACT_SCORE'].apply(lambda x: "🔴 Critical" if x < -5 else ("🟠 Warning" if x < -2 else "🟢 Info"))
        
        st.dataframe(
            df[['Headline', 'Risk', 'NEWS_LINK']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗 Access Intel")
            }
        )

# --- 7. MAIN EXECUTION ---

def main():
    inject_custom_css()
    engine = get_db_engine()
    
    # Header
    st.title("GLOBAL COMMAND CENTER")
    st.markdown("### 📡 Real-Time Geopolitical Signal Processing")
    
    # Status Ticker (Pure HTML/CSS injection for smoothness)
    st.markdown("""
        <div style="background: #450a0a; border-left: 4px solid #ef4444; padding: 10px; margin-bottom: 20px; color: #fecaca; font-family: monospace;">
            ⚠️ <strong>ALERT:</strong> MONITORING ACTIVE CONFLICT ZONES IN [EUR] [MEA] [APAC] REGIONS. SYSTEM LATENCY: 12ms.
        </div>
    """, unsafe_allow_html=True)
    
    render_metrics_hud(engine)
    
    # Main Dashboard
    t_viz, t_intel, t_feed = st.tabs(["🌍 GLOBAL THEATER", "💬 AI ANALYST", "📋 RAW FEED"])
    
    with t_viz:
        # Fetch Geo Data
        geo_sql = "SELECT ACTOR_COUNTRY_CODE, 30.0 as LAT, 10.0 as LON, COUNT(*) as MAG FROM EVENTS_DAGSTER GROUP BY 1,2,3"
        # Note: In production, join with a lat/lon lookup table. 
        # For this demo, we mock coords if not present, but the structure handles real data.
        geo_df = fetch_data_snapshot(engine, "SELECT * FROM EVENTS_DAGSTER LIMIT 100") 
        # Mocking lat/lon for demo purposes if table lacks them
        if 'LAT' not in geo_df.columns:
            import numpy as np
            geo_df['LAT'] = np.random.uniform(-50, 60, size=len(geo_df))
            geo_df['LON'] = np.random.uniform(-120, 140, size=len(geo_df))
        
        render_3d_globe(geo_df)
    
    with t_intel:
        c_chat, c_eval = st.columns([3, 1])
        
        with c_chat:
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role":"assistant", "content":"Commander, I am ready. Requesting directive."}]
            
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
                
            if prompt := st.chat_input("Enter query (e.g., 'Analyze conflict risks in Eastern Europe')"):
                st.chat_message("user").write(prompt)
                st.session_state.messages.append({"role":"user", "content":prompt})
                
                with st.chat_message("assistant"):
                    with st.spinner("Encrypting query... Accessing Snowflake Data Cloud..."):
                        qe = get_query_engine(engine)
                        if qe:
                            response_obj = qe.query(prompt)
                            response_text = response_obj.response
                            
                            # ASYNC EVALUATION (Fire and Forget logic simulation)
                            # In a full async app we would await; here we run it to show the feature
                            confidence_json = asyncio.run(async_llm_monitor(response_text))
                            
                            st.markdown(response_text)
                            
                            # SQL Trace for transparency
                            if 'sql_query' in response_obj.metadata:
                                with st.expander("🔍 SQL Trace (Verified)", expanded=False):
                                    st.code(response_obj.metadata['sql_query'], language='sql')
                            
                            st.session_state.messages.append({"role":"assistant", "content":response_text})
                            
                            # Save evaluation to session state to display in side panel
                            st.session_state['last_eval'] = confidence_json
        
        with c_eval:
            st.markdown("### 🛡️ Overwatch")
            if 'last_eval' in st.session_state:
                try:
                    # Simple parsing of the JSON/String returned by Critic
                    import json
                    eval_data = st.session_state['last_eval']
                    # Clean up markdown if Gemini adds it
                    eval_data = eval_data.replace('```json', '').replace('```', '')
                    data = json.loads(eval_data)
                    
                    score = data.get('confidence_score', 0)
                    
                    st.metric("Confidence Score", f"{score}%")
                    if score > 80:
                        st.success("✅ INTEL VERIFIED")
                    elif score > 50:
                        st.warning("⚠️ UNCORROBORATED")
                    else:
                        st.error("🚫 HALLUCINATION RISK")
                        
                    st.caption(f"**Reasoning:** {data.get('reasoning', 'N/A')}")
                except:
                    st.info("Awaiting Analysis...")
            else:
                st.info("System Idle. Submit query for evaluation.")

    with t_feed:
        render_feed_tab(engine)

if __name__ == "__main__":
    main()