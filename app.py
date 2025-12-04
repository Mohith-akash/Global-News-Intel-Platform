"""
🦅 GLOBAL INTELLIGENCE COMMAND CENTER
A Real-Time Geopolitical Analytics Platform
Built by: [Your Name] | Portfolio Project for AI/ML Engineering Roles

Architecture: GDELT → GitHub Actions → MotherDuck → Gemini AI → Streamlit
"""

import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, inspect
import datetime
import pycountry
import logging
import streamlit.components.v1 as components
import re
from urllib.parse import urlparse, unquote
import duckdb
import time
import json

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & SETUP
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SIGINT Command Center | Global Intelligence Platform",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/Mohith-akash/global-news-intel-platform',
        'Report a bug': 'https://github.com/Mohith-akash/global-news-intel-platform/issues',
        'About': "Real-time geopolitical intelligence powered by GDELT, MotherDuck & Gemini AI"
    }
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sigint")

# Validate environment
REQUIRED_ENVS = ["MOTHERDUCK_TOKEN", "GOOGLE_API_KEY"]
missing = [k for k in REQUIRED_ENVS if not os.getenv(k)]
if missing:
    st.error(f"❌ SYSTEM CRITICAL: Missing environment variables: {', '.join(missing)}")
    st.info("Please configure your secrets in the Streamlit Cloud dashboard.")
    st.stop()

# Constants
GEMINI_MODEL = "models/gemini-2.5-flash-preview-09-2025"
GEMINI_EMBED_MODEL = "models/embedding-001"

# Date calculations
NOW = datetime.datetime.now()
TODAY = f"'{NOW.strftime('%Y%m%d')}'"
YESTERDAY = f"'{(NOW - datetime.timedelta(days=1)).strftime('%Y%m%d')}'"
TWO_DAYS_AGO = f"'{(NOW - datetime.timedelta(days=2)).strftime('%Y%m%d')}'"
WEEK_AGO = f"'{(NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')}'"
MONTH_AGO = f"'{(NOW - datetime.timedelta(days=30)).strftime('%Y%m%d')}'"

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PROFESSIONAL STYLING - Intelligence Command Center Aesthetic
# ═══════════════════════════════════════════════════════════════════════════════

def inject_custom_css():
    st.markdown("""
    <style>
        /* ═══════════════ IMPORT FONTS ═══════════════ */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* ═══════════════ ROOT VARIABLES ═══════════════ */
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #0d1320;
            --bg-card: #111827;
            --bg-elevated: #1a2332;
            --border-color: #1e3a5f;
            --border-glow: #0ea5e9;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-blue: #0ea5e9;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --accent-purple: #8b5cf6;
            --gradient-blue: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
            --gradient-danger: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            --gradient-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }
        
        /* ═══════════════ GLOBAL STYLES ═══════════════ */
        .stApp {
            background: var(--bg-primary);
            background-image: 
                radial-gradient(ellipse at top, rgba(14, 165, 233, 0.03) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(6, 182, 212, 0.02) 0%, transparent 50%);
        }
        
        /* Hide Streamlit defaults */
        header[data-testid="stHeader"] { display: none; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stDeployButton { display: none; }
        div[data-testid="stToolbar"] { display: none; }
        div[data-testid="stDecoration"] { display: none; }
        
        /* ═══════════════ TYPOGRAPHY ═══════════════ */
        html, body, .stApp, .stMarkdown, p, span, div {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-primary);
        }
        
        h1, h2, h3, .header-title {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        
        code, pre, .mono {
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* ═══════════════ CONTAINER LAYOUT ═══════════════ */
        .block-container {
            padding: 1.5rem 2rem 3rem 2rem;
            max-width: 100%;
        }
        
        /* ═══════════════ COMMAND HEADER ═══════════════ */
        .command-header {
            background: linear-gradient(180deg, var(--bg-secondary) 0%, transparent 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 0 1.5rem 0;
            margin-bottom: 1.5rem;
        }
        
        .header-grid {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 1.5rem;
        }
        
        .logo-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .logo-icon {
            font-size: 2.5rem;
            filter: drop-shadow(0 0 10px rgba(14, 165, 233, 0.5));
        }
        
        .logo-text {
            display: flex;
            flex-direction: column;
        }
        
        .logo-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin: 0;
            line-height: 1.2;
        }
        
        .logo-subtitle {
            font-size: 0.7rem;
            color: var(--accent-cyan);
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-emerald);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            50% { opacity: 0.8; box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        }
        
        /* ═══════════════ METRIC CARDS ═══════════════ */
        div[data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        div[data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-blue);
        }
        
        div[data-testid="stMetric"]:hover {
            border-color: var(--border-glow);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(14, 165, 233, 0.15);
        }
        
        div[data-testid="stMetric"] label {
            color: var(--text-secondary);
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: var(--text-primary);
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* ═══════════════ CUSTOM CARDS ═══════════════ */
        .intel-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .intel-card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 4px 20px rgba(14, 165, 233, 0.1);
        }
        
        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .card-icon {
            font-size: 1.25rem;
        }
        
        .card-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0;
        }
        
        /* ═══════════════ TABS ═══════════════ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 4px;
            border: 1px solid var(--border-color);
        }
        
        .stTabs [data-baseweb="tab"] {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-secondary);
            background: transparent;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            letter-spacing: 0.02em;
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--bg-elevated);
            color: var(--accent-cyan);
            border: 1px solid var(--border-color);
        }
        
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }
        
        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }
        
        /* ═══════════════ CHAT INTERFACE ═══════════════ */
        div[data-testid="stChatMessage"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
        }
        
        div[data-testid="stChatMessageContent"] p {
            color: var(--text-primary);
        }
        
        .stChatInput > div {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }
        
        .stChatInput input {
            background: transparent;
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }
        
        .stChatInput input::placeholder {
            color: var(--text-muted);
        }
        
        /* ═══════════════ DATAFRAMES ═══════════════ */
        div[data-testid="stDataFrame"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }
        
        div[data-testid="stDataFrame"] table {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
        }
        
        div[data-testid="stDataFrame"] th {
            background: var(--bg-elevated) !important;
            color: var(--text-secondary) !important;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* ═══════════════ BUTTONS ═══════════════ */
        .stButton > button {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            background: var(--bg-elevated);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
        }
        
        .stButton > button[kind="primary"] {
            background: var(--gradient-blue);
            border: none;
            color: white;
        }
        
        /* ═══════════════ EXPANDERS ═══════════════ */
        .streamlit-expanderHeader {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        .streamlit-expanderContent {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-top: none;
            border-radius: 0 0 8px 8px;
        }
        
        /* ═══════════════ ALERT TICKER ═══════════════ */
        .alert-ticker {
            background: linear-gradient(90deg, 
                rgba(239, 68, 68, 0.15) 0%, 
                rgba(239, 68, 68, 0.05) 50%,
                rgba(239, 68, 68, 0.15) 100%);
            border-left: 4px solid var(--accent-red);
            border-radius: 0 8px 8px 0;
            padding: 0.75rem 1rem;
            margin: 1rem 0;
            overflow: hidden;
        }
        
        .ticker-content {
            display: flex;
            animation: scroll 40s linear infinite;
            white-space: nowrap;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--accent-red);
        }
        
        @keyframes scroll {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        /* ═══════════════ ARCHITECTURE DIAGRAM ═══════════════ */
        .arch-container {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            margin: 1rem 0;
        }
        
        .arch-node {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            transition: all 0.3s ease;
        }
        
        .arch-node:hover {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
        }
        
        .arch-arrow {
            color: var(--accent-cyan);
            font-size: 1.5rem;
            margin: 0 0.5rem;
        }
        
        /* ═══════════════ SELECTBOX & INPUTS ═══════════════ */
        .stSelectbox > div > div {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        .stSelectbox [data-baseweb="select"] {
            background: var(--bg-secondary);
        }
        
        /* ═══════════════ PLOTLY CHARTS ═══════════════ */
        .js-plotly-plot {
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* ═══════════════ SCROLLBAR ═══════════════ */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-blue);
        }
        
        /* ═══════════════ SIDEBAR ═══════════════ */
        section[data-testid="stSidebar"] {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding: 1rem;
        }
        
        /* ═══════════════ DIVIDERS ═══════════════ */
        hr {
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 1.5rem 0;
        }
        
        /* ═══════════════ SPECIAL ELEMENTS ═══════════════ */
        .stat-highlight {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--gradient-blue);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .tech-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 0.4rem 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin: 0.25rem;
        }
        
        .tech-badge:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATABASE CONNECTION & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_db_connection():
    """Get direct DuckDB connection to MotherDuck"""
    token = os.getenv("MOTHERDUCK_TOKEN")
    return duckdb.connect(f'md:gdelt_db?motherduck_token={token}', read_only=True)

@st.cache_resource
def get_sql_engine():
    """Get SQLAlchemy engine for LlamaIndex"""
    token = os.getenv("MOTHERDUCK_TOKEN")
    return create_engine(f'duckdb:///md:gdelt_db?motherduck_token={token}')

def safe_query(conn, query):
    """Execute query safely with error handling"""
    try:
        return conn.execute(query).df()
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return pd.DataFrame()

def is_safe_sql(sql: str) -> bool:
    """Validate SQL for safety"""
    if not sql:
        return False
    forbidden = ["delete ", "update ", "drop ", "alter ", "insert ", "grant ", "revoke ", "--"]
    return not any(f in sql.lower() for f in forbidden)

def get_country_name(code):
    """Convert country code to full name"""
    try:
        if not code or pd.isna(code):
            return "Unknown"
        country = pycountry.countries.get(alpha_2=code)
        return country.name if country else code
    except:
        return code

def format_headline(url, actor=None):
    """Extract readable headline from URL"""
    fallback = "Global Event Report"
    if not url:
        return fallback
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        segments = [s for s in path.split('/') if s]
        if not segments:
            return fallback

        for seg in reversed(segments[-3:]):
            seg = re.sub(r'\.(html|htm|php|asp|aspx)$', '', seg, flags=re.IGNORECASE)
            if seg.isdigit() or re.search(r'\d{4}', seg):
                continue
            if seg.lower() in ['index', 'default', 'article', 'news', 'story']:
                continue
            if len(seg) > 5:
                text = seg.replace('-', ' ').replace('_', ' ')
                words = [w for w in text.split() if len(w) < 15 and not any(c.isdigit() for c in w)]
                headline = " ".join(words).title()
                if len(headline) >= 10:
                    return headline[:80] + "..." if len(headline) > 80 else headline
        return fallback
    except:
        return fallback

# ═══════════════════════════════════════════════════════════════════════════════
# 4. AI QUERY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_query_engine(_engine):
    """Initialize the AI-powered SQL query engine"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    llm = Gemini(
        model=GEMINI_MODEL,
        api_key=api_key,
        temperature=0.0,
    )
    embed_model = GeminiEmbedding(model_name=GEMINI_EMBED_MODEL, api_key=api_key)
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    try:
        inspector = inspect(_engine)
        tables = inspector.get_table_names() + inspector.get_view_names()
        target = next((t for t in tables if t.upper() == "EVENTS_DAGSTER"), None)
        
        if not target:
            st.error("❌ EVENTS_DAGSTER table not found in database")
            return None
        
        sql_database = SQLDatabase(_engine, include_tables=[target])
        
        enhanced_prompt = f"""You are an expert SQL analyst for geopolitical intelligence data.

TABLE: EVENTS_DAGSTER

COLUMNS:
- DATE (VARCHAR, format: 'YYYYMMDD')
- MAIN_ACTOR (text) - Actor involved in event
- ACTOR_COUNTRY_CODE (text) - ISO-2 country code
- IMPACT_SCORE (float) - Event intensity (-10 to +10, negative=conflict)
- ARTICLE_COUNT (integer) - Media coverage count
- NEWS_LINK (text) - Source URL
- SENTIMENT_SCORE (float) - Media sentiment

DATE REFERENCES:
- Today: {TODAY}
- Yesterday: {YESTERDAY}  
- 2 days ago: {TWO_DAYS_AGO}
- Week ago: {WEEK_AGO}

SQL RULES:
1. DATE is VARCHAR - use string comparison: DATE >= '20241127'
2. Never use date functions
3. Always include: DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, NEWS_LINK
4. Add: WHERE IMPACT_SCORE IS NOT NULL AND NEWS_LINK IS NOT NULL
5. Default ORDER BY: DATE DESC, ABS(IMPACT_SCORE) DESC
6. Default LIMIT: 15

QUERY INTERPRETATIONS:
- "crisis" → IMPACT_SCORE < -5
- "conflict" → IMPACT_SCORE < -3
- "recent" → DATE >= {WEEK_AGO}
- "today" → DATE = {TODAY}
- "48 hours" → DATE >= {TWO_DAYS_AGO}
- "trending" → ORDER BY ARTICLE_COUNT DESC

COUNTRY CODES:
US, RU, CN, UA, IL, PS, IR, SY, IQ, SA, IN, GB, FR, DE, JP

Return ONLY valid SQL."""

        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            llm=llm,
            synthesize_response=True
        )
        query_engine.update_prompts({"text_to_sql_prompt": enhanced_prompt})
        return query_engine
    except Exception as e:
        logger.exception("Query engine initialization failed")
        return None

def execute_ai_query(query_engine, prompt, conn):
    """Execute natural language query through AI"""
    try:
        resp = query_engine.query(prompt)
        
        if hasattr(resp, 'metadata') and 'sql_query' in resp.metadata:
            sql = resp.metadata['sql_query'].strip()
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            if not is_safe_sql(sql):
                return {'success': False, 'error': 'Unsafe SQL detected'}
            
            df = safe_query(conn, sql)
            return {
                'success': True,
                'response': resp.response,
                'sql': sql,
                'data': df
            }
        return {
            'success': True,
            'response': resp.response,
            'sql': None,
            'data': None
        }
    except Exception as e:
        logger.exception("Query execution failed")
        return {'success': False, 'error': str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATA FETCHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_dashboard_metrics(_conn):
    """Fetch key metrics for dashboard"""
    metrics = {}
    
    # Total events
    df = safe_query(_conn, "SELECT COUNT(*) as c FROM EVENTS_DAGSTER")
    metrics['total'] = df.iloc[0, 0] if not df.empty else 0
    
    # Recent events (7 days)
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    df = safe_query(_conn, f"SELECT COUNT(*) as c FROM EVENTS_DAGSTER WHERE DATE >= '{week_ago}'")
    metrics['recent'] = df.iloc[0, 0] if not df.empty else 0
    
    # Critical alerts
    df = safe_query(_conn, "SELECT COUNT(*) as c FROM EVENTS_DAGSTER WHERE ABS(IMPACT_SCORE) > 6")
    metrics['critical'] = df.iloc[0, 0] if not df.empty else 0
    
    # Top hotspot
    df = safe_query(_conn, """
        SELECT ACTOR_COUNTRY_CODE, COUNT(*) as c 
        FROM EVENTS_DAGSTER 
        WHERE ACTOR_COUNTRY_CODE IS NOT NULL 
        GROUP BY 1 ORDER BY 2 DESC LIMIT 1
    """)
    if not df.empty:
        metrics['hotspot_code'] = df.iloc[0, 0]
        metrics['hotspot_name'] = get_country_name(df.iloc[0, 0])
        metrics['hotspot_count'] = df.iloc[0, 1]
    else:
        metrics['hotspot_code'] = 'N/A'
        metrics['hotspot_name'] = 'Scanning...'
        metrics['hotspot_count'] = 0
    
    # Data freshness
    df = safe_query(_conn, "SELECT MAX(DATE) as d FROM EVENTS_DAGSTER")
    if not df.empty:
        try:
            latest = str(df.iloc[0, 0])
            metrics['latest_date'] = datetime.datetime.strptime(latest, '%Y%m%d').strftime('%d %b %Y')
        except:
            metrics['latest_date'] = 'Unknown'
    else:
        metrics['latest_date'] = 'Unknown'
    
    return metrics

@st.cache_data(ttl=300)
def get_alert_events(_conn):
    """Fetch recent high-impact events for ticker"""
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    return safe_query(_conn, f"""
        SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, DATE 
        FROM EVENTS_DAGSTER 
        WHERE IMPACT_SCORE < -3 
        AND ACTOR_COUNTRY_CODE IS NOT NULL 
        AND DATE >= '{week_ago}'
        ORDER BY DATE DESC, IMPACT_SCORE ASC 
        LIMIT 10
    """)

@st.cache_data(ttl=300)
def get_country_data(_conn):
    """Fetch country-level aggregations"""
    return safe_query(_conn, """
        SELECT 
            ACTOR_COUNTRY_CODE as country,
            COUNT(*) as events,
            AVG(IMPACT_SCORE) as avg_impact,
            SUM(ARTICLE_COUNT) as total_coverage
        FROM EVENTS_DAGSTER 
        WHERE ACTOR_COUNTRY_CODE IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
    """)

@st.cache_data(ttl=300)
def get_time_series(_conn):
    """Fetch daily event counts"""
    month_ago = (NOW - datetime.timedelta(days=30)).strftime('%Y%m%d')
    return safe_query(_conn, f"""
        SELECT 
            DATE,
            COUNT(*) as events,
            AVG(IMPACT_SCORE) as avg_impact,
            SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as conflicts
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{month_ago}'
        GROUP BY 1
        ORDER BY 1
    """)

@st.cache_data(ttl=300)
def get_trending_news(_conn):
    """Fetch trending stories by media coverage"""
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    return safe_query(_conn, f"""
        SELECT 
            NEWS_LINK,
            ACTOR_COUNTRY_CODE,
            MAIN_ACTOR,
            MAX(ARTICLE_COUNT) as coverage,
            AVG(IMPACT_SCORE) as impact
        FROM EVENTS_DAGSTER 
        WHERE NEWS_LINK IS NOT NULL 
        AND DATE >= '{week_ago}'
        GROUP BY 1, 2, 3
        ORDER BY coverage DESC 
        LIMIT 30
    """)

@st.cache_data(ttl=300)
def get_recent_feed(_conn):
    """Fetch recent events feed"""
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    return safe_query(_conn, f"""
        SELECT 
            DATE,
            NEWS_LINK,
            MAIN_ACTOR,
            ACTOR_COUNTRY_CODE,
            IMPACT_SCORE,
            ARTICLE_COUNT
        FROM EVENTS_DAGSTER 
        WHERE NEWS_LINK IS NOT NULL 
        AND DATE >= '{week_ago}'
        ORDER BY DATE DESC, ABS(IMPACT_SCORE) DESC
        LIMIT 50
    """)

@st.cache_data(ttl=300)  
def get_actor_network(_conn):
    """Fetch actor co-occurrence data for network viz"""
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    return safe_query(_conn, f"""
        SELECT 
            ACTOR_COUNTRY_CODE as source,
            COUNT(*) as weight,
            AVG(IMPACT_SCORE) as sentiment
        FROM EVENTS_DAGSTER
        WHERE ACTOR_COUNTRY_CODE IS NOT NULL
        AND DATE >= '{week_ago}'
        GROUP BY 1
        HAVING COUNT(*) > 10
        ORDER BY 2 DESC
        LIMIT 15
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_header():
    """Render the command center header"""
    st.markdown("""
    <div class="command-header">
        <div class="header-grid">
            <div class="logo-container">
                <span class="logo-icon">🦅</span>
                <div class="logo-text">
                    <span class="logo-title">SIGINT Command</span>
                    <span class="logo-subtitle">Global Intelligence Platform</span>
                </div>
            </div>
            <div></div>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>SYSTEM ONLINE</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics(conn):
    """Render key performance indicators"""
    metrics = get_dashboard_metrics(conn)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📡 SIGNAL VOLUME",
            value=f"{metrics['total']:,}",
            delta="Total Events",
            help="Total events in database"
        )
    
    with col2:
        st.metric(
            label="⚡ RECENT (7D)",
            value=f"{metrics['recent']:,}",
            delta="This Week",
            help="Events from last 7 days"
        )
    
    with col3:
        st.metric(
            label="🚨 CRITICAL",
            value=f"{metrics['critical']:,}",
            delta="High Impact",
            delta_color="inverse",
            help="Events with impact > 6"
        )
    
    with col4:
        st.metric(
            label="🔥 HOTSPOT",
            value=metrics['hotspot_name'][:12],
            delta=f"{metrics['hotspot_count']:,} events",
            help="Most active region"
        )
    
    with col5:
        st.metric(
            label="📅 DATA UPDATED",
            value=metrics['latest_date'],
            delta="Latest Record",
            help="Most recent data point"
        )

def render_alert_ticker(conn):
    """Render scrolling alert ticker"""
    df = get_alert_events(conn)
    
    if df.empty:
        ticker_text = "⚠️ INITIALIZING THREAT DETECTION... SCANNING GLOBAL FEEDS..."
    else:
        items = []
        for _, row in df.iterrows():
            actor = row['MAIN_ACTOR'][:30] if row['MAIN_ACTOR'] else 'Unknown'
            country = row['ACTOR_COUNTRY_CODE']
            impact = row['IMPACT_SCORE']
            items.append(f"⚠️ {actor} ({country}) • Impact: {impact:.1f}")
        ticker_text = " &nbsp;&nbsp;│&nbsp;&nbsp; ".join(items)
    
    components.html(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; background: transparent; }}
            .ticker-wrap {{
                width: 100%;
                overflow: hidden;
                background: linear-gradient(90deg, 
                    rgba(239, 68, 68, 0.12) 0%, 
                    rgba(239, 68, 68, 0.05) 50%,
                    rgba(239, 68, 68, 0.12) 100%);
                border-left: 4px solid #ef4444;
                border-radius: 0 8px 8px 0;
                padding: 12px 0;
            }}
            .ticker {{
                display: inline-block;
                white-space: nowrap;
                animation: scroll 45s linear infinite;
                font-family: 'JetBrains Mono', 'SF Mono', monospace;
                font-size: 13px;
                font-weight: 500;
                color: #fca5a5;
                letter-spacing: 0.02em;
            }}
            @keyframes scroll {{
                0% {{ transform: translateX(100%); }}
                100% {{ transform: translateX(-100%); }}
            }}
        </style>
    </head>
    <body>
        <div class="ticker-wrap">
            <div class="ticker">{ticker_text}</div>
        </div>
    </body>
    </html>
    """, height=50)

def render_globe_map(conn):
    """Render 3D globe visualization"""
    df = get_country_data(conn)
    
    if df.empty:
        st.info("🌍 Loading geospatial data...")
        return
    
    # Convert ISO-2 to ISO-3 for plotly
    df['iso3'] = df['country'].apply(lambda x: pycountry.countries.get(alpha_2=x).alpha_3 if pycountry.countries.get(alpha_2=x) else None)
    df = df.dropna(subset=['iso3'])
    df['country_name'] = df['country'].apply(get_country_name)
    
    fig = px.choropleth(
        df,
        locations="iso3",
        color="events",
        hover_name="country_name",
        hover_data={
            "events": ":,",
            "avg_impact": ":.2f",
            "iso3": False
        },
        color_continuous_scale=[
            [0, "#0d1320"],
            [0.2, "#164e63"],
            [0.4, "#0891b2"],
            [0.7, "#06b6d4"],
            [1, "#22d3ee"]
        ],
        labels={
            "events": "Event Count",
            "avg_impact": "Avg Impact"
        }
    )
    
    fig.update_geos(
        projection_type="orthographic",
        showcoastlines=True,
        coastlinecolor="#1e3a5f",
        showland=True,
        landcolor="#0d1320",
        showocean=True,
        oceancolor="#0a0e17",
        showlakes=False,
        showcountries=True,
        countrycolor="#1e3a5f",
        bgcolor="rgba(0,0,0,0)"
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title=dict(text="Events", font=dict(color="#94a3b8", size=11)),
            tickfont=dict(color="#94a3b8", size=10),
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor="#1e3a5f",
            borderwidth=1
        ),
        geo=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_time_series_chart(conn):
    """Render time series analysis"""
    df = get_time_series(conn)
    
    if df.empty:
        st.info("📈 Loading temporal data...")
        return
    
    # Parse dates
    df['date_parsed'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Event Volume", "Conflict Intensity")
    )
    
    # Event volume (area chart)
    fig.add_trace(
        go.Scatter(
            x=df['date_parsed'],
            y=df['events'],
            fill='tozeroy',
            fillcolor='rgba(6, 182, 212, 0.2)',
            line=dict(color='#06b6d4', width=2),
            name='Events',
            hovertemplate='%{x|%b %d}<br>Events: %{y:,}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Conflict events (bar chart)
    fig.add_trace(
        go.Bar(
            x=df['date_parsed'],
            y=df['conflicts'],
            marker=dict(
                color=df['conflicts'],
                colorscale=[[0, '#164e63'], [0.5, '#f59e0b'], [1, '#ef4444']],
                line=dict(width=0)
            ),
            name='Conflicts',
            hovertemplate='%{x|%b %d}<br>Conflicts: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=40, b=20),
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
    )
    
    fig.update_xaxes(
        showgrid=False,
        linecolor="#1e3a5f",
        tickfont=dict(size=10)
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(30, 58, 95, 0.3)",
        linecolor="#1e3a5f",
        tickfont=dict(size=10)
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12, color='#94a3b8', family='JetBrains Mono')
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_country_bar_chart(conn):
    """Render top countries bar chart"""
    df = get_country_data(conn)
    
    if df.empty:
        st.info("📊 Loading country data...")
        return
    
    df = df.head(10)
    df['country_name'] = df['country'].apply(get_country_name)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['country_name'],
        x=df['events'],
        orientation='h',
        marker=dict(
            color=df['events'],
            colorscale=[[0, '#0d4754'], [0.5, '#0891b2'], [1, '#22d3ee']],
            line=dict(width=0)
        ),
        text=df['events'].apply(lambda x: f'{x:,}'),
        textposition='outside',
        textfont=dict(size=11, color='#94a3b8', family='JetBrains Mono'),
        hovertemplate='%{y}<br>Events: %{x:,}<extra></extra>'
    ))
    
    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=100, r=60, t=20, b=20),
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(30, 58, 95, 0.3)",
            linecolor="#1e3a5f",
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=False,
            linecolor="#1e3a5f",
            tickfont=dict(size=11),
            autorange="reversed"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_impact_distribution(conn):
    """Render impact score distribution"""
    df = safe_query(conn, """
        SELECT 
            CASE 
                WHEN IMPACT_SCORE < -5 THEN 'Critical Conflict'
                WHEN IMPACT_SCORE < -2 THEN 'Moderate Tension'
                WHEN IMPACT_SCORE < 2 THEN 'Neutral'
                WHEN IMPACT_SCORE < 5 THEN 'Positive Development'
                ELSE 'Major Cooperation'
            END as category,
            COUNT(*) as count
        FROM EVENTS_DAGSTER
        WHERE IMPACT_SCORE IS NOT NULL
        GROUP BY 1
    """)
    
    if df.empty:
        st.info("Loading distribution data...")
        return
    
    colors = {
        'Critical Conflict': '#ef4444',
        'Moderate Tension': '#f59e0b',
        'Neutral': '#6b7280',
        'Positive Development': '#10b981',
        'Major Cooperation': '#22d3ee'
    }
    
    df['color'] = df['category'].map(colors)
    
    fig = go.Figure(data=[go.Pie(
        labels=df['category'],
        values=df['count'],
        hole=0.6,
        marker=dict(colors=df['color'], line=dict(color='#0d1320', width=2)),
        textinfo='percent',
        textfont=dict(size=11, color='white', family='JetBrains Mono'),
        hovertemplate='%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(
            font=dict(size=9, color='#94a3b8'),
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        annotations=[dict(
            text='<b>IMPACT</b>',
            x=0.5, y=0.5,
            font=dict(size=12, color='#94a3b8', family='JetBrains Mono'),
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_conflict_gauge(conn):
    """Render a conflict vs cooperation gauge meter"""
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    df = safe_query(conn, f"""
        SELECT 
            AVG(IMPACT_SCORE) as avg_impact,
            SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as conflicts,
            SUM(CASE WHEN IMPACT_SCORE > 3 THEN 1 ELSE 0 END) as cooperations,
            COUNT(*) as total
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{week_ago}'
        AND IMPACT_SCORE IS NOT NULL
    """)
    
    if df.empty:
        st.info("Loading gauge data...")
        return
    
    avg_impact = df.iloc[0]['avg_impact'] or 0
    conflicts = df.iloc[0]['conflicts'] or 0
    cooperations = df.iloc[0]['cooperations'] or 0
    total = df.iloc[0]['total'] or 1
    
    # Normalize to 0-100 scale (impact is -10 to +10)
    gauge_value = ((avg_impact + 10) / 20) * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_value,
        number={'suffix': '%', 'font': {'size': 40, 'color': '#e2e8f0', 'family': 'JetBrains Mono'}},
        delta={'reference': 50, 'increasing': {'color': '#10b981'}, 'decreasing': {'color': '#ef4444'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#1e3a5f', 'tickfont': {'color': '#64748b', 'size': 10}},
            'bar': {'color': '#06b6d4', 'thickness': 0.7},
            'bgcolor': '#1e293b',
            'borderwidth': 2,
            'bordercolor': '#1e3a5f',
            'steps': [
                {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [30, 50], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [50, 70], 'color': 'rgba(107, 114, 128, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#f59e0b', 'width': 3},
                'thickness': 0.8,
                'value': 50
            }
        },
        title={'text': "Global Stability Index", 'font': {'size': 14, 'color': '#94a3b8', 'family': 'JetBrains Mono'}}
    ))
    
    fig.update_layout(
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=60, b=20),
        font=dict(family='JetBrains Mono')
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Stats below gauge
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">
            <div style="font-size: 1.5rem; font-weight: bold; color: #ef4444; font-family: 'JetBrains Mono';">{int(conflicts):,}</div>
            <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;">Conflicts</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background: rgba(107, 114, 128, 0.1); border-radius: 8px; border: 1px solid rgba(107, 114, 128, 0.3);">
            <div style="font-size: 1.5rem; font-weight: bold; color: #9ca3af; font-family: 'JetBrains Mono';">{int(total):,}</div>
            <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;">Total Events</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);">
            <div style="font-size: 1.5rem; font-weight: bold; color: #10b981; font-family: 'JetBrains Mono';">{int(cooperations):,}</div>
            <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;">Cooperations</div>
        </div>
        """, unsafe_allow_html=True)


def render_sparkline_trend(conn):
    """Render a compact 7-day activity sparkline"""
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    df = safe_query(conn, f"""
        SELECT 
            DATE,
            COUNT(*) as events,
            SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as conflicts
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{week_ago}'
        GROUP BY 1
        ORDER BY 1
    """)
    
    if df.empty or len(df) < 2:
        st.info("Loading trend data...")
        return
    
    df['date_parsed'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    
    fig = go.Figure()
    
    # Events area
    fig.add_trace(go.Scatter(
        x=df['date_parsed'],
        y=df['events'],
        fill='tozeroy',
        fillcolor='rgba(6, 182, 212, 0.2)',
        line=dict(color='#06b6d4', width=2),
        name='Events',
        hovertemplate='%{x|%b %d}: %{y:,} events<extra></extra>'
    ))
    
    # Conflicts line
    fig.add_trace(go.Scatter(
        x=df['date_parsed'],
        y=df['conflicts'],
        line=dict(color='#ef4444', width=2, dash='dot'),
        name='Conflicts',
        hovertemplate='%{x|%b %d}: %{y:,} conflicts<extra></extra>'
    ))
    
    fig.update_layout(
        height=150,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1,
            xanchor='right',
            x=1,
            font=dict(size=10, color='#64748b'),
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=True,
            tickfont=dict(size=9, color='#64748b'),
            tickformat='%d %b'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(30, 58, 95, 0.2)',
            showticklabels=False
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_top_actors(conn):
    """Render top actors with activity breakdown"""
    week_ago = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
    
    df = safe_query(conn, f"""
        SELECT 
            MAIN_ACTOR,
            ACTOR_COUNTRY_CODE,
            COUNT(*) as events,
            AVG(IMPACT_SCORE) as avg_impact,
            SUM(ARTICLE_COUNT) as media_coverage
        FROM EVENTS_DAGSTER
        WHERE DATE >= '{week_ago}'
        AND MAIN_ACTOR IS NOT NULL
        AND LENGTH(MAIN_ACTOR) > 2
        GROUP BY 1, 2
        ORDER BY events DESC
        LIMIT 12
    """)
    
    if df.empty:
        st.info("Loading actor data...")
        return
    
    # Create a horizontal bar chart with diverging colors based on impact
    df['color'] = df['avg_impact'].apply(
        lambda x: '#ef4444' if x < -3 else ('#f59e0b' if x < 0 else ('#10b981' if x > 3 else '#06b6d4'))
    )
    df['actor_label'] = df.apply(
        lambda x: f"{x['MAIN_ACTOR'][:20]}... ({x['ACTOR_COUNTRY_CODE']})" if len(str(x['MAIN_ACTOR'])) > 20 
        else f"{x['MAIN_ACTOR']} ({x['ACTOR_COUNTRY_CODE']})", axis=1
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['actor_label'],
        x=df['events'],
        orientation='h',
        marker=dict(
            color=df['color'],
            line=dict(width=0)
        ),
        text=df['events'].apply(lambda x: f'{x:,}'),
        textposition='outside',
        textfont=dict(size=10, color='#94a3b8', family='JetBrains Mono'),
        hovertemplate='<b>%{y}</b><br>Events: %{x:,}<br>Avg Impact: %{customdata:.2f}<extra></extra>',
        customdata=df['avg_impact']
    ))
    
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=50, t=10, b=10),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(30, 58, 95, 0.2)',
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=10, color='#94a3b8'),
            autorange='reversed'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Legend
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 0.5rem; flex-wrap: wrap;">
        <span style="font-size: 0.7rem; color: #ef4444;">● Conflict</span>
        <span style="font-size: 0.7rem; color: #f59e0b;">● Tension</span>
        <span style="font-size: 0.7rem; color: #06b6d4;">● Neutral</span>
        <span style="font-size: 0.7rem; color: #10b981;">● Positive</span>
    </div>
    """, unsafe_allow_html=True)

def render_trending_table(conn):
    """Render trending news table"""
    df = get_trending_news(conn)
    
    if df.empty:
        st.info("📰 Loading trending stories...")
        return
    
    # Process data
    df.columns = [c.upper() for c in df.columns]
    df['HEADLINE'] = df.apply(lambda x: format_headline(x.get('NEWS_LINK', ''), x.get('MAIN_ACTOR', '')), axis=1)
    df['COUNTRY'] = df['ACTOR_COUNTRY_CODE'].apply(get_country_name)
    df = df.drop_duplicates(subset=['HEADLINE']).head(15)
    
    # Categorize impact
    df['TYPE'] = df['IMPACT'].apply(
        lambda x: "🔴 Crisis" if x < -4 else ("🟡 Tension" if x < -1 else ("🟢 Positive" if x > 2 else "⚪ Neutral"))
    )
    
    st.dataframe(
        df[['HEADLINE', 'COUNTRY', 'COVERAGE', 'TYPE', 'NEWS_LINK']],
        hide_index=True,
        column_config={
            "HEADLINE": st.column_config.TextColumn("Story", width="large"),
            "COUNTRY": st.column_config.TextColumn("Region", width="small"),
            "COVERAGE": st.column_config.NumberColumn("📊 Coverage", format="%d"),
            "TYPE": st.column_config.TextColumn("Status", width="small"),
            "NEWS_LINK": st.column_config.LinkColumn("Source", display_text="🔗")
        },
        use_container_width=True
    )

def render_feed_table(conn):
    """Render recent events feed"""
    df = get_recent_feed(conn)
    
    if df.empty:
        st.info("📋 Loading event feed...")
        return
    
    df.columns = [c.upper() for c in df.columns]
    df['HEADLINE'] = df.apply(lambda x: format_headline(x.get('NEWS_LINK', ''), x.get('MAIN_ACTOR', '')), axis=1)
    df['COUNTRY'] = df['ACTOR_COUNTRY_CODE'].apply(get_country_name)
    
    try:
        df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
    except:
        df['DATE_FMT'] = df['DATE']
    
    df['IMPACT_FMT'] = df['IMPACT_SCORE'].apply(
        lambda x: f"🔴 {x:.1f}" if x < -4 else (f"🟡 {x:.1f}" if x < -1 else f"🟢 {x:.1f}")
    )
    
    st.dataframe(
        df[['DATE_FMT', 'HEADLINE', 'COUNTRY', 'IMPACT_FMT', 'NEWS_LINK']].head(30),
        hide_index=True,
        column_config={
            "DATE_FMT": st.column_config.TextColumn("Date", width="small"),
            "HEADLINE": st.column_config.TextColumn("Event", width="large"),
            "COUNTRY": st.column_config.TextColumn("Region", width="small"),
            "IMPACT_FMT": st.column_config.TextColumn("Impact", width="small"),
            "NEWS_LINK": st.column_config.LinkColumn("Link", display_text="🔗")
        },
        use_container_width=True
    )

def render_ai_chat(conn, engine):
    """Render AI analyst chat interface"""
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "🦅 **SIGINT Analyst Online**\n\nI'm your AI intelligence analyst with access to real-time GDELT data. Ask me about:\n- Recent conflicts and crises\n- Regional activity patterns\n- Country comparisons\n- Trending news stories"
        }]
    
    # Example queries
    st.markdown("""
    <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
        <p style="color: #94a3b8; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.05em;">💡 Example Queries</p>
        <p style="color: #64748b; font-size: 0.85rem; margin: 0.25rem 0;">• Show crisis events from the last 48 hours</p>
        <p style="color: #64748b; font-size: 0.85rem; margin: 0.25rem 0;">• What's happening in the Middle East?</p>
        <p style="color: #64748b; font-size: 0.85rem; margin: 0.25rem 0;">• Compare Russia and China activity this week</p>
        <p style="color: #64748b; font-size: 0.85rem; margin: 0.25rem 0;">• Show trending high-coverage stories</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container(height=400)
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about global events..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Analyzing intelligence data..."):
                    qe = get_query_engine(engine)
                    if qe:
                        result = execute_ai_query(qe, prompt, conn)
                        
                        if result['success']:
                            response = result['response']
                            st.markdown(response)
                            
                            if result['data'] is not None and not result['data'].empty:
                                df = result['data']
                                df.columns = [c.upper() for c in df.columns]
                                
                                if 'DATE' in df.columns:
                                    try:
                                        df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b %Y')
                                    except:
                                        pass
                                
                                if 'NEWS_LINK' in df.columns:
                                    df['HEADLINE'] = df.apply(lambda x: format_headline(x.get('NEWS_LINK', '')), axis=1)
                                    cols = [c for c in ['DATE', 'HEADLINE', 'ACTOR_COUNTRY_CODE', 'IMPACT_SCORE', 'NEWS_LINK'] if c in df.columns]
                                    st.dataframe(
                                        df[cols],
                                        hide_index=True,
                                        column_config={
                                            "NEWS_LINK": st.column_config.LinkColumn("🔗"),
                                            "HEADLINE": st.column_config.TextColumn("Event", width="large")
                                        }
                                    )
                                else:
                                    st.dataframe(df, hide_index=True)
                            
                            if result['sql']:
                                with st.expander("🔍 View SQL Query"):
                                    st.code(result['sql'], language='sql')
                            
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            st.error(f"❌ {result.get('error', 'Query failed')}")
                            st.info("💡 Try: 'Show recent high-impact events'")
                    else:
                        st.error("AI Engine unavailable")

def render_architecture():
    """Render architecture documentation for portfolio"""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="font-family: 'JetBrains Mono', monospace; color: #e2e8f0; margin-bottom: 0.5rem;">
            🏗️ System Architecture
        </h2>
        <p style="color: #64748b; font-size: 0.9rem;">
            End-to-end data engineering pipeline with AI-powered analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture flow
    st.markdown("""
    <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 2rem; margin: 1rem 0; text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 0.5rem;">
            <span style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 0.6rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;">
                🌐 GDELT
            </span>
            <span style="color: #06b6d4; font-size: 1.2rem;">→</span>
            <span style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 0.6rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;">
                ⚙️ GitHub Actions
            </span>
            <span style="color: #06b6d4; font-size: 1.2rem;">→</span>
            <span style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 0.6rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;">
                🦆 MotherDuck
            </span>
            <span style="color: #06b6d4; font-size: 1.2rem;">→</span>
            <span style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 0.6rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;">
                🧠 Gemini AI
            </span>
            <span style="color: #06b6d4; font-size: 1.2rem;">→</span>
            <span style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 0.6rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;">
                📊 Streamlit
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; height: 100%;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #06b6d4; font-size: 0.9rem; margin-bottom: 1rem;">
                📥 DATA INGESTION
            </h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>Source:</strong> GDELT Project (Global Database of Events)</li>
                <li><strong>Pipeline:</strong> Dagster orchestration</li>
                <li><strong>Schedule:</strong> GitHub Actions (30-min intervals)</li>
                <li><strong>Format:</strong> CSV → Parquet optimization</li>
                <li><strong>Volume:</strong> ~10M+ events processed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; margin-top: 1rem;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #10b981; font-size: 0.9rem; margin-bottom: 1rem;">
                🗄️ DATA STORAGE
            </h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>Current:</strong> MotherDuck (Cloud DuckDB)</li>
                <li><strong>Originally:</strong> Built on Snowflake</li>
                <li><strong>Migration:</strong> Moved to MotherDuck for cost optimization</li>
                <li><strong>Benefits:</strong> Serverless, free tier, same SQL</li>
                <li><strong>Query Engine:</strong> SQLAlchemy + DuckDB</li>
            </ul>
            <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(16, 185, 129, 0.1); border-radius: 6px; border-left: 3px solid #10b981;">
                <span style="color: #10b981; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">💡 COST SAVINGS: Migrated from Snowflake → MotherDuck free tier</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; height: 100%;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #8b5cf6; font-size: 0.9rem; margin-bottom: 1rem;">
                🤖 AI LAYER
            </h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>LLM:</strong> Google Gemini 2.5 Flash</li>
                <li><strong>Tier:</strong> Free API (cost-optimized)</li>
                <li><strong>Framework:</strong> LlamaIndex</li>
                <li><strong>Feature:</strong> Natural Language to SQL</li>
                <li><strong>Embeddings:</strong> Gemini Embedding-001</li>
            </ul>
            <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(139, 92, 246, 0.1); border-radius: 6px; border-left: 3px solid #8b5cf6;">
                <span style="color: #8b5cf6; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">🆓 Using Gemini Free Tier API</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; margin-top: 1rem;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #f59e0b; font-size: 0.9rem; margin-bottom: 1rem;">
                📊 VISUALIZATION
            </h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>Framework:</strong> Streamlit</li>
                <li><strong>Charts:</strong> Plotly (Interactive)</li>
                <li><strong>Maps:</strong> Choropleth (Orthographic)</li>
                <li><strong>Deployment:</strong> Streamlit Cloud</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Tech stack badges
    st.markdown("""
    <div style="margin-top: 2rem; text-align: center;">
        <p style="color: #64748b; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.1em;">
            Technology Stack
        </p>
        <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem;">
            <span class="tech-badge">🐍 Python</span>
            <span class="tech-badge">❄️ Snowflake</span>
            <span class="tech-badge">🦆 DuckDB</span>
            <span class="tech-badge">☁️ MotherDuck</span>
            <span class="tech-badge">⚙️ Dagster</span>
            <span class="tech-badge">🤖 Gemini AI</span>
            <span class="tech-badge">🦙 LlamaIndex</span>
            <span class="tech-badge">📊 Plotly</span>
            <span class="tech-badge">🎨 Streamlit</span>
            <span class="tech-badge">🔄 GitHub Actions</span>
            <span class="tech-badge">🐼 Pandas</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key features
    st.markdown("---")
    st.markdown("""
    <h3 style="font-family: 'JetBrains Mono', monospace; color: #e2e8f0; text-align: center; margin: 2rem 0 1.5rem 0;">
        ✨ Key Engineering Highlights
    </h3>
    """, unsafe_allow_html=True)
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <span style="font-size: 2rem;">🔄</span>
            <h4 style="color: #e2e8f0; font-size: 1rem; margin: 0.75rem 0 0.5rem 0;">Automated Pipeline</h4>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">
                30-minute automated data refresh using GitHub Actions + Dagster orchestration
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <span style="font-size: 2rem;">🧠</span>
            <h4 style="color: #e2e8f0; font-size: 1rem; margin: 0.75rem 0 0.5rem 0;">AI-Powered Queries</h4>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">
                Natural language to SQL conversion using Gemini + LlamaIndex
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <span style="font-size: 2rem;">📈</span>
            <h4 style="color: #e2e8f0; font-size: 1rem; margin: 0.75rem 0 0.5rem 0;">Scale & Performance</h4>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">
                10M+ events with sub-second query response via MotherDuck
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_about():
    """Render about/contact section"""
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h2 style="font-family: 'JetBrains Mono', monospace; color: #e2e8f0; margin-bottom: 1rem;">
            👋 About This Project
        </h2>
        <p style="color: #94a3b8; font-size: 1rem; max-width: 700px; margin: 0 auto 2rem auto; line-height: 1.7;">
            This real-time geopolitical intelligence platform demonstrates end-to-end 
            data engineering and AI capabilities. Originally built on <strong>Snowflake</strong>, 
            then strategically migrated to <strong>MotherDuck</strong> for cost optimization while 
            maintaining performance. Uses <strong>Gemini AI free tier</strong> for intelligent queries — 
            showcasing how to build production-grade systems on a budget.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #06b6d4; font-size: 0.9rem; margin-bottom: 1rem;">
                🎯 PROJECT GOALS
            </h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li>Demonstrate data pipeline engineering</li>
                <li>Showcase cloud-native architecture</li>
                <li>Implement AI/ML integration</li>
                <li>Build production-grade UI/UX</li>
                <li>Optimize costs (Snowflake → MotherDuck)</li>
                <li>Leverage free tiers effectively</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem;">
            <h4 style="font-family: 'JetBrains Mono', monospace; color: #10b981; font-size: 0.9rem; margin-bottom: 1rem;">
                🛠️ SKILLS DEMONSTRATED
            </h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
                <li>Python, SQL, Data Engineering</li>
                <li>ETL/ELT Pipeline Development</li>
                <li>Cloud Platforms (Snowflake, DuckDB, MotherDuck)</li>
                <li>LLM Integration (Gemini, LlamaIndex)</li>
                <li>Data Visualization & Dashboards</li>
                <li>Cost Optimization & Migration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 2rem; background: linear-gradient(180deg, #111827 0%, transparent 100%); border-radius: 12px;">
        <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 1rem;">
            Interested in discussing this project or opportunities?
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <a href="https://github.com/Mohith-akash" target="_blank" style="
                display: inline-flex; align-items: center; gap: 0.5rem;
                background: #1e293b; border: 1px solid #334155; border-radius: 8px;
                padding: 0.6rem 1.2rem; color: #94a3b8; text-decoration: none;
                font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
                transition: all 0.3s ease;
            ">
                ⭐ GitHub
            </a>
            <a href="https://www.linkedin.com/in/mohith-akash/" target="_blank" style="
                display: inline-flex; align-items: center; gap: 0.5rem;
                background: #1e293b; border: 1px solid #334155; border-radius: 8px;
                padding: 0.6rem 1.2rem; color: #94a3b8; text-decoration: none;
                font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
                transition: all 0.3s ease;
            ">
                💼 LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    
    # Inject CSS
    inject_custom_css()
    
    # Get connections
    conn = get_db_connection()
    engine = get_sql_engine()
    
    # Render header
    render_header()
    
    # Main navigation tabs
    tab_dashboard, tab_analytics, tab_ai, tab_arch, tab_about = st.tabs([
        "📊 DASHBOARD",
        "📈 ANALYTICS", 
        "🤖 AI ANALYST",
        "🏗️ ARCHITECTURE",
        "👤 ABOUT"
    ])
    
    # ═══════════════ DASHBOARD TAB ═══════════════
    with tab_dashboard:
        # Metrics row
        render_metrics(conn)
        
        # Alert ticker
        render_alert_ticker(conn)
        
        st.markdown("---")
        
        # Row 1: Main visualizations
        col_left, col_right = st.columns([5, 5])
        
        with col_left:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">⚡</span>
                <span class="card-title">Real-Time Conflict Monitor</span>
            </div>
            """, unsafe_allow_html=True)
            render_conflict_gauge(conn)
            
            st.markdown("""
            <div class="card-header" style="margin-top: 1.5rem;">
                <span class="card-icon">📈</span>
                <span class="card-title">7-Day Activity Trend</span>
            </div>
            """, unsafe_allow_html=True)
            render_sparkline_trend(conn)
        
        with col_right:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">🎯</span>
                <span class="card-title">Top Active Actors</span>
            </div>
            """, unsafe_allow_html=True)
            render_top_actors(conn)
        
        st.markdown("---")
        
        # Row 2: Map and Distribution
        col_map, col_dist = st.columns([6, 4])
        
        with col_map:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">🌍</span>
                <span class="card-title">Global Activity Heatmap</span>
            </div>
            """, unsafe_allow_html=True)
            render_globe_map(conn)
        
        with col_dist:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">📊</span>
                <span class="card-title">Event Classification</span>
            </div>
            """, unsafe_allow_html=True)
            render_impact_distribution(conn)
            
            st.markdown("""
            <div class="card-header" style="margin-top: 1rem;">
                <span class="card-icon">🏆</span>
                <span class="card-title">Regional Leaderboard</span>
            </div>
            """, unsafe_allow_html=True)
            render_country_bar_chart(conn)
    
    # ═══════════════ ANALYTICS TAB ═══════════════
    with tab_analytics:
        st.markdown("""
        <div class="card-header">
            <span class="card-icon">📈</span>
            <span class="card-title">Temporal Analysis (30 Days)</span>
        </div>
        """, unsafe_allow_html=True)
        render_time_series_chart(conn)
        
        st.markdown("---")
        
        col_trend, col_feed = st.columns(2)
        
        with col_trend:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">🔥</span>
                <span class="card-title">Trending Stories</span>
            </div>
            """, unsafe_allow_html=True)
            render_trending_table(conn)
        
        with col_feed:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">📋</span>
                <span class="card-title">Recent Events Feed</span>
            </div>
            """, unsafe_allow_html=True)
            render_feed_table(conn)
    
    # ═══════════════ AI ANALYST TAB ═══════════════
    with tab_ai:
        col_chat, col_info = st.columns([7, 3])
        
        with col_chat:
            st.markdown("""
            <div class="card-header">
                <span class="card-icon">🤖</span>
                <span class="card-title">Intelligence Query Interface</span>
            </div>
            """, unsafe_allow_html=True)
            render_ai_chat(conn, engine)
        
        with col_info:
            st.markdown("""
            <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem;">
                <h4 style="font-family: 'JetBrains Mono', monospace; color: #06b6d4; font-size: 0.85rem; margin-bottom: 1rem;">
                    ℹ️ HOW IT WORKS
                </h4>
                <p style="color: #94a3b8; font-size: 0.8rem; line-height: 1.7;">
                    The AI analyst uses <strong>Google Gemini 2.5</strong> with <strong>LlamaIndex</strong> 
                    to convert your natural language questions into SQL queries against the GDELT database.
                </p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #1e3a5f;">
                    <p style="color: #64748b; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; margin-bottom: 0.5rem;">
                        DATA SCHEMA
                    </p>
                    <code style="font-size: 0.7rem; color: #94a3b8; display: block; line-height: 1.6;">
                        • DATE (YYYYMMDD)<br>
                        • MAIN_ACTOR<br>
                        • ACTOR_COUNTRY_CODE<br>
                        • IMPACT_SCORE (-10 to +10)<br>
                        • ARTICLE_COUNT<br>
                        • NEWS_LINK
                    </code>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.5rem; margin-top: 1rem;">
                <h4 style="font-family: 'JetBrains Mono', monospace; color: #f59e0b; font-size: 0.85rem; margin-bottom: 1rem;">
                    ⚡ QUERY TIPS
                </h4>
                <ul style="color: #94a3b8; font-size: 0.8rem; line-height: 1.8; padding-left: 1rem;">
                    <li>Use country names (Russia, China, US)</li>
                    <li>Specify timeframes (today, this week)</li>
                    <li>Ask about "crisis" or "conflict" events</li>
                    <li>Request "trending" for high-coverage</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # ═══════════════ ARCHITECTURE TAB ═══════════════
    with tab_arch:
        render_architecture()
    
    # ═══════════════ ABOUT TAB ═══════════════
    with tab_about:
        render_about()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; border-top: 1px solid #1e3a5f; margin-top: 2rem;">
        <p style="color: #475569; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
            SIGINT COMMAND CENTER v2.0 | Built by <a href="https://www.linkedin.com/in/mohith-akash/" target="_blank" style="color: #06b6d4; text-decoration: none;">Mohith Akash</a>
        </p>
        <p style="color: #374151; font-size: 0.65rem; font-family: 'JetBrains Mono', monospace; margin-top: 0.25rem;">
            Snowflake → MotherDuck | Gemini AI Free Tier | Zero Infrastructure Cost
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
