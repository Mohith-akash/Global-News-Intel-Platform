"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🌐 GLOBAL NEWS INTELLIGENCE PLATFORM                       ║
║  Real-Time Analytics Dashboard for GDELT                                     ║
║  Author: Mohith Akash | Portfolio Project for AI/Data Engineering Roles     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

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
from sqlalchemy import create_engine, text
import datetime
import pycountry
import logging
import re
from urllib.parse import urlparse, unquote
import duckdb

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: APP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Global News Intelligence | GDELT Analytics",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/Mohith-akash/global-news-intel-platform',
        'Report a bug': 'https://github.com/Mohith-akash/global-news-intel-platform/issues',
        'About': "Real-time global news analytics powered by GDELT, MotherDuck & Gemini AI"
    }
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gdelt_platform")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ENVIRONMENT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_ENVS = ["MOTHERDUCK_TOKEN", "GOOGLE_API_KEY"]
missing = [k for k in REQUIRED_ENVS if not os.getenv(k)]

if missing:
    st.error(f"❌ Missing environment variables: {', '.join(missing)}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GEMINI_MODEL = "models/gemini-2.5-flash-preview-05-20"
GEMINI_EMBED_MODEL = "models/embedding-001"

# Date calculations - GDELT uses YYYYMMDD format
NOW = datetime.datetime.now()
TODAY = NOW.strftime('%Y%m%d')
WEEK_AGO = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
MONTH_AGO = (NOW - datetime.timedelta(days=30)).strftime('%Y%m%d')

# Table name - will be detected automatically
TABLE_NAME = None

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
        
        :root {
            --bg-primary: #0a0e17;
            --bg-card: #111827;
            --bg-elevated: #1a2332;
            --border-color: #1e3a5f;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-red: #ef4444;
        }
        
        .stApp { background: var(--bg-primary); }
        header[data-testid="stHeader"], #MainMenu, footer, .stDeployButton { display: none !important; }
        
        html, body, .stApp, p, span, div { font-family: 'Inter', sans-serif; color: var(--text-primary); }
        h1, h2, h3, code { font-family: 'JetBrains Mono', monospace; }
        
        .block-container { padding: 1.5rem 2rem; max-width: 100%; }
        
        /* Header */
        .command-header { border-bottom: 1px solid var(--border-color); padding: 1rem 0 1.5rem 0; margin-bottom: 1.5rem; }
        .header-grid { display: flex; justify-content: space-between; align-items: center; }
        .logo-container { display: flex; align-items: center; gap: 0.75rem; }
        .logo-icon { font-size: 2.5rem; }
        .logo-title { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; text-transform: uppercase; }
        .logo-subtitle { font-size: 0.7rem; color: var(--accent-cyan); }
        .status-badge { display: flex; align-items: center; gap: 0.5rem; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.75rem; }
        .status-dot { width: 8px; height: 8px; background: var(--accent-emerald); border-radius: 50%; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        /* Metrics */
        div[data-testid="stMetric"] { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; }
        div[data-testid="stMetric"] label { color: var(--text-secondary); font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        
        /* Cards */
        .card-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border-color); }
        .card-title { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 0; background: #0d1320; border-radius: 8px; padding: 4px; border: 1px solid var(--border-color); overflow-x: auto; }
        .stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-secondary); padding: 0.5rem 0.9rem; white-space: nowrap; }
        .stTabs [aria-selected="true"] { background: var(--bg-elevated); color: var(--accent-cyan); border-radius: 6px; }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
        
        /* DataFrames */
        div[data-testid="stDataFrame"] { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; }
        div[data-testid="stDataFrame"] th { background: var(--bg-elevated) !important; color: var(--text-secondary) !important; font-size: 0.75rem; text-transform: uppercase; }
        
        /* Ticker */
        .ticker-container { background: linear-gradient(90deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05), rgba(239,68,68,0.15)); border-left: 4px solid #ef4444; border-radius: 0 8px 8px 0; padding: 0.6rem 0; overflow: hidden; position: relative; margin: 0.5rem 0; }
        .ticker-label { position: absolute; left: 0; top: 0; bottom: 0; background: linear-gradient(90deg, rgba(127,29,29,0.98), transparent); padding: 0.6rem 1.25rem 0.6rem 0.75rem; font-size: 0.7rem; font-weight: 600; color: #ef4444; display: flex; align-items: center; gap: 0.5rem; z-index: 2; }
        .ticker-dot { width: 7px; height: 7px; background: #ef4444; border-radius: 50%; animation: ticker-pulse 1s infinite; }
        @keyframes ticker-pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.7); } }
        .ticker-content { display: inline-block; white-space: nowrap; padding-left: 95px; animation: ticker-scroll 40s linear infinite; font-size: 0.8rem; color: #fca5a5; }
        @keyframes ticker-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        
        /* Tech badges */
        .tech-badge { display: inline-flex; background: var(--bg-elevated); border: 1px solid var(--border-color); border-radius: 20px; padding: 0.4rem 0.8rem; font-size: 0.75rem; color: var(--text-secondary); margin: 0.25rem; }
        
        /* Mobile */
        @media (max-width: 768px) {
            .block-container { padding: 1rem 0.75rem; }
            .logo-title { font-size: 1rem !important; }
            .logo-subtitle { display: none; }
            div[data-testid="stMetric"] { padding: 0.6rem !important; }
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
        }
        
        .swipe-hint { text-align: center; padding: 0.4rem; color: var(--accent-cyan); font-size: 0.7rem; display: none; }
        @media (max-width: 768px) { .swipe-hint { display: block; } }
        
        hr { border: none; border-top: 1px solid var(--border-color); margin: 1.5rem 0; }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_db_connection():
    """Get DuckDB connection to MotherDuck"""
    token = os.getenv("MOTHERDUCK_TOKEN")
    return duckdb.connect(f'md:gdelt_db?motherduck_token={token}', read_only=True)

@st.cache_resource
def get_sql_engine():
    """Get SQLAlchemy engine for LlamaIndex"""
    token = os.getenv("MOTHERDUCK_TOKEN")
    return create_engine(f"duckdb:///md:gdelt_db?motherduck_token={token}")

def get_table_name(conn):
    """Detect the actual table name (handles case sensitivity)"""
    global TABLE_NAME
    if TABLE_NAME:
        return TABLE_NAME
    
    try:
        # Try to find the events table
        tables = conn.execute("SHOW TABLES").df()
        if not tables.empty:
            table_names = tables.iloc[:, 0].tolist()
            # Look for events_dagster in any case
            for t in table_names:
                if 'events' in t.lower() and 'dagster' in t.lower():
                    TABLE_NAME = t
                    return TABLE_NAME
            # Fallback to first table if no match
            if table_names:
                TABLE_NAME = table_names[0]
                return TABLE_NAME
    except Exception as e:
        logger.error(f"Error detecting table: {e}")
    
    # Default fallback
    TABLE_NAME = "events_dagster"
    return TABLE_NAME

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: AI SETUP
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_engine(_sql_engine, table_name):
    """Initialize LlamaIndex with Gemini"""
    try:
        llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"), model=GEMINI_MODEL, temperature=0.1)
        embed = GeminiEmbedding(api_key=os.getenv("GOOGLE_API_KEY"), model_name=GEMINI_EMBED_MODEL)
        Settings.llm = llm
        Settings.embed_model = embed
        return SQLDatabase(_sql_engine, include_tables=[table_name])
    except Exception as e:
        logger.error(f"AI engine init failed: {e}")
        return None

@st.cache_resource
def get_query_engine(_engine, table_name):
    """Create NL to SQL query engine"""
    if _engine is None:
        return None
    try:
        return NLSQLTableQueryEngine(sql_database=_engine, tables=[table_name])
    except Exception as e:
        logger.error(f"Query engine failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def safe_query(conn, sql):
    """Execute SQL safely, return empty DataFrame on error"""
    try:
        return conn.execute(sql).df()
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return pd.DataFrame()

def get_country_name(code):
    """Convert country code to name, with None handling"""
    if not code or not isinstance(code, str) or len(code) != 2:
        return "Unknown"
    try:
        country = pycountry.countries.get(alpha_2=code.upper())
        return country.name if country else code
    except:
        return code if code else "Unknown"

def format_headline(url, actor=None):
    """Extract headline from URL"""
    if not url:
        return actor if actor else "News Event"
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        segments = [s for s in path.split('/') if s and len(s) > 3]
        if segments:
            headline = segments[-1]
            headline = re.sub(r'\.(html?|php|aspx?)$', '', headline)
            headline = re.sub(r'[-_]', ' ', headline).title()[:80]
            if len(headline) > 10:
                return headline
        return parsed.netloc.replace('www.', '').split('.')[0].title()
    except:
        return actor if actor else "News Event"

def format_number(num):
    """Format large numbers with K/M suffix"""
    if num is None:
        return "0"
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    if num >= 1000:
        return f"{num/1000:.1f}K"
    return str(int(num))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def get_dashboard_metrics(_conn, table):
    """Fetch main KPIs"""
    df = safe_query(_conn, f"""
        SELECT COUNT(*) as total,
            SUM(CASE WHEN DATE >= '{WEEK_AGO}' THEN 1 ELSE 0 END) as recent,
            SUM(CASE WHEN ABS(IMPACT_SCORE) > 6 AND DATE >= '{WEEK_AGO}' THEN 1 ELSE 0 END) as critical
        FROM {table}
    """)
    
    hotspot_df = safe_query(_conn, f"""
        SELECT ACTOR_COUNTRY_CODE, COUNT(*) as cnt FROM {table}
        WHERE DATE >= '{WEEK_AGO}' AND ACTOR_COUNTRY_CODE IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC LIMIT 1
    """)
    
    return {
        'total': df.iloc[0]['total'] if not df.empty else 0,
        'recent': df.iloc[0]['recent'] if not df.empty else 0,
        'critical': df.iloc[0]['critical'] if not df.empty else 0,
        'hotspot': hotspot_df.iloc[0]['ACTOR_COUNTRY_CODE'] if not hotspot_df.empty else 'N/A'
    }

@st.cache_data(ttl=600)
def get_alert_events(_conn, table):
    """Fetch high-impact events for ticker"""
    three_days = (NOW - datetime.timedelta(days=3)).strftime('%Y%m%d')
    return safe_query(_conn, f"""
        SELECT DATE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, NEWS_LINK
        FROM {table}
        WHERE DATE >= '{three_days}' AND IMPACT_SCORE < -4 AND MAIN_ACTOR IS NOT NULL
        ORDER BY IMPACT_SCORE ASC LIMIT 15
    """)

@st.cache_data(ttl=600)
def get_country_data(_conn, table):
    """Get country aggregates"""
    return safe_query(_conn, f"""
        SELECT ACTOR_COUNTRY_CODE as country, COUNT(*) as events, AVG(IMPACT_SCORE) as avg_impact
        FROM {table} WHERE DATE >= '{MONTH_AGO}' AND ACTOR_COUNTRY_CODE IS NOT NULL
        GROUP BY 1 ORDER BY events DESC
    """)

@st.cache_data(ttl=600)
def get_time_series(_conn, table):
    """Get daily event counts"""
    return safe_query(_conn, f"""
        SELECT DATE, COUNT(*) as events,
            SUM(CASE WHEN IMPACT_SCORE < -2 THEN 1 ELSE 0 END) as negative,
            SUM(CASE WHEN IMPACT_SCORE > 2 THEN 1 ELSE 0 END) as positive
        FROM {table} WHERE DATE >= '{MONTH_AGO}' GROUP BY 1 ORDER BY 1
    """)

@st.cache_data(ttl=600)
def get_trending_news(_conn, table):
    """Get trending stories by coverage"""
    return safe_query(_conn, f"""
        SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT
        FROM {table} WHERE DATE >= '{WEEK_AGO}' AND ARTICLE_COUNT > 3 AND NEWS_LINK IS NOT NULL
        ORDER BY ARTICLE_COUNT DESC, DATE DESC LIMIT 50
    """)

@st.cache_data(ttl=600)
def get_recent_feed(_conn, table):
    """Get recent events"""
    return safe_query(_conn, f"""
        SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT
        FROM {table} WHERE DATE >= '{WEEK_AGO}' AND NEWS_LINK IS NOT NULL
        ORDER BY DATE DESC LIMIT 50
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div class="command-header">
        <div class="header-grid">
            <div class="logo-container">
                <span class="logo-icon">🌐</span>
                <div>
                    <div class="logo-title">Global News Intelligence</div>
                    <div class="logo-subtitle">Powered by GDELT • Real-Time Analytics</div>
                </div>
            </div>
            <div class="status-badge"><span class="status-dot"></span> LIVE DATA</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics(conn, table):
    metrics = get_dashboard_metrics(conn, table)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("📡 TOTAL", format_number(metrics['total']), "All time")
    with c2: st.metric("⚡ 7 DAYS", format_number(metrics['recent']), "Recent")
    with c3: st.metric("🚨 CRITICAL", format_number(metrics['critical']), "High impact")
    with c4: 
        hs = metrics['hotspot']
        st.metric("🔥 HOTSPOT", get_country_name(hs)[:12] if hs else "N/A", hs if hs else "")
    with c5: st.metric("📅 UPDATED", NOW.strftime("%H:%M"), NOW.strftime("%d %b"))

def render_alert_ticker(conn, table):
    df = get_alert_events(conn, table)
    
    if df.empty:
        ticker_text = "⚡ Monitoring global news feeds for significant events │ Platform powered by GDELT │ "
    else:
        items = []
        for _, row in df.iterrows():
            actor = str(row['MAIN_ACTOR'])[:25] if row['MAIN_ACTOR'] else 'Unknown'
            country = get_country_name(row.get('ACTOR_COUNTRY_CODE', ''))
            country = country[:15] if country else 'Unknown'
            impact = row['IMPACT_SCORE'] if row['IMPACT_SCORE'] else 0
            items.append(f"⚠️ {actor} ({country}) • {impact:.1f}")
        ticker_text = " │ ".join(items) + " │ "
    
    double_text = ticker_text + ticker_text
    st.markdown(f"""
    <div class="ticker-container">
        <div class="ticker-label"><span class="ticker-dot"></span> LIVE</div>
        <div class="ticker-content">{double_text}</div>
    </div>
    """, unsafe_allow_html=True)

def render_quick_briefing(conn, table):
    df = safe_query(conn, f"""
        SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT
        FROM {table} WHERE NEWS_LINK IS NOT NULL AND ARTICLE_COUNT > 5 AND DATE >= '{WEEK_AGO}'
        ORDER BY DATE DESC, ARTICLE_COUNT DESC LIMIT 25
    """)
    
    if df.empty:
        st.info("📰 Loading headlines...")
        return
    
    df.columns = [c.upper() for c in df.columns]
    df['HEADLINE'] = df.apply(lambda x: format_headline(x.get('NEWS_LINK', ''), x.get('MAIN_ACTOR', '')), axis=1)
    df['COUNTRY'] = df['ACTOR_COUNTRY_CODE'].apply(lambda x: get_country_name(x) if x else 'Unknown')
    df = df.drop_duplicates(subset=['HEADLINE']).head(12)
    
    try:
        df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
    except:
        df['DATE_FMT'] = df['DATE']
    
    df['TONE'] = df['IMPACT_SCORE'].apply(lambda x: "🔴" if x and x < -4 else ("🟡" if x and x < -1 else ("🟢" if x and x > 2 else "⚪")))
    
    st.dataframe(df[['TONE', 'DATE_FMT', 'HEADLINE', 'COUNTRY', 'NEWS_LINK']], hide_index=True, height=350,
        column_config={
            "TONE": st.column_config.TextColumn("", width="small"),
            "DATE_FMT": st.column_config.TextColumn("Date", width="small"),
            "HEADLINE": st.column_config.TextColumn("Headline", width="large"),
            "COUNTRY": st.column_config.TextColumn("Region", width="small"),
            "NEWS_LINK": st.column_config.LinkColumn("🔗")
        }, use_container_width=True)

def render_conflict_gauge(conn, table):
    df = safe_query(conn, f"""
        SELECT AVG(IMPACT_SCORE) as avg_impact,
            SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as conflicts,
            SUM(CASE WHEN IMPACT_SCORE > 3 THEN 1 ELSE 0 END) as positive,
            COUNT(*) as total
        FROM {table} WHERE DATE >= '{WEEK_AGO}' AND IMPACT_SCORE IS NOT NULL
    """)
    
    if df.empty:
        st.info("Loading...")
        return
    
    avg = df.iloc[0]['avg_impact'] or 0
    conflicts = int(df.iloc[0]['conflicts'] or 0)
    positive = int(df.iloc[0]['positive'] or 0)
    total = int(df.iloc[0]['total'] or 1)
    
    if avg < -2: status, color = "⚠️ ELEVATED", "#ef4444"
    elif avg < 0: status, color = "🟡 MODERATE", "#f59e0b"
    elif avg < 2: status, color = "🟢 STABLE", "#10b981"
    else: status, color = "✨ POSITIVE", "#06b6d4"
    
    st.markdown(f"""
    <div style="text-align:center; padding:1.25rem; background:linear-gradient(135deg,rgba(14,165,233,0.1),rgba(6,182,212,0.05)); border-radius:12px; border:1px solid #1e3a5f; margin-bottom:1rem;">
        <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase;">Weekly Sentiment</div>
        <div style="font-size:1.75rem; font-weight:700; color:{color}; font-family:'JetBrains Mono';">{status}</div>
        <div style="font-size:0.8rem; color:#94a3b8;">Avg: <span style="color:{color}">{avg:.2f}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div style="text-align:center; padding:0.75rem; background:rgba(239,68,68,0.1); border-radius:8px;"><div style="font-size:1.25rem; font-weight:700; color:#ef4444;">{conflicts:,}</div><div style="font-size:0.6rem; color:#94a3b8;">NEGATIVE</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div style="text-align:center; padding:0.75rem; background:rgba(107,114,128,0.1); border-radius:8px;"><div style="font-size:1.25rem; font-weight:700; color:#9ca3af;">{total:,}</div><div style="font-size:0.6rem; color:#94a3b8;">TOTAL</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div style="text-align:center; padding:0.75rem; background:rgba(16,185,129,0.1); border-radius:8px;"><div style="font-size:1.25rem; font-weight:700; color:#10b981;">{positive:,}</div><div style="font-size:0.6rem; color:#94a3b8;">POSITIVE</div></div>', unsafe_allow_html=True)

def render_time_series_chart(conn, table):
    df = get_time_series(conn, table)
    if df.empty:
        st.info("📈 Loading...")
        return
    
    df['date_parsed'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['date_parsed'], y=df['events'], fill='tozeroy', fillcolor='rgba(6,182,212,0.15)', line=dict(color='#06b6d4', width=2), name='Total'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['date_parsed'], y=df['negative'], line=dict(color='#ef4444', width=2), name='Negative'), secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date_parsed'], y=df['positive'], line=dict(color='#10b981', width=2), name='Positive'), secondary_y=True)
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=30,b=0), showlegend=True, legend=dict(orientation='h', y=1.02, font=dict(size=11, color='#94a3b8')), xaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')), yaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_top_actors(conn, table):
    df = safe_query(conn, f"""
        SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, COUNT(*) as events, AVG(IMPACT_SCORE) as avg_impact
        FROM {table} WHERE DATE >= '{WEEK_AGO}' AND MAIN_ACTOR IS NOT NULL AND LENGTH(MAIN_ACTOR) > 2
        GROUP BY 1, 2 ORDER BY events DESC LIMIT 10
    """)
    
    if df.empty:
        st.info("🎯 Loading...")
        return
    
    df['label'] = df.apply(lambda x: f"{str(x['MAIN_ACTOR'])[:18]} ({x['ACTOR_COUNTRY_CODE']})" if x['ACTOR_COUNTRY_CODE'] else str(x['MAIN_ACTOR'])[:20], axis=1)
    df['color'] = df['avg_impact'].apply(lambda x: '#ef4444' if x and x < -3 else ('#f59e0b' if x and x < 0 else ('#10b981' if x and x > 3 else '#06b6d4')))
    
    fig = go.Figure(go.Bar(x=df['events'], y=df['label'], orientation='h', marker_color=df['color'], text=df['events'].apply(lambda x: f'{x:,}'), textposition='outside', textfont=dict(color='#94a3b8', size=10)))
    fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=50,t=10,b=0), xaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')), yaxis=dict(showgrid=False, tickfont=dict(color='#e2e8f0', size=11), autorange='reversed'), bargap=0.3)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_impact_distribution(conn, table):
    df = safe_query(conn, f"""
        SELECT CASE WHEN IMPACT_SCORE < -5 THEN 'Crisis' WHEN IMPACT_SCORE < -2 THEN 'Negative' WHEN IMPACT_SCORE < 2 THEN 'Neutral' WHEN IMPACT_SCORE < 5 THEN 'Positive' ELSE 'Very Positive' END as category, COUNT(*) as count
        FROM {table} WHERE DATE >= '{WEEK_AGO}' AND IMPACT_SCORE IS NOT NULL GROUP BY 1
    """)
    
    if df.empty:
        st.info("📊 Loading...")
        return
    
    colors = {'Crisis': '#ef4444', 'Negative': '#f59e0b', 'Neutral': '#64748b', 'Positive': '#10b981', 'Very Positive': '#06b6d4'}
    fig = go.Figure(data=[go.Pie(labels=df['category'], values=df['count'], hole=0.6, marker_colors=[colors.get(c, '#64748b') for c in df['category']], textinfo='percent', textfont=dict(size=11, color='#e2e8f0'))])
    fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=10), showlegend=True, legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center', font=dict(size=10, color='#94a3b8')))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_country_bar_chart(conn, table):
    df = get_country_data(conn, table)
    if df.empty:
        st.info("🏆 Loading...")
        return
    
    df = df.head(8)
    df['country_name'] = df['country'].apply(lambda x: get_country_name(x) if x else 'Unknown')
    fig = go.Figure(go.Bar(x=df['country_name'], y=df['events'], marker_color='#06b6d4', text=df['events'].apply(format_number), textposition='outside', textfont=dict(color='#94a3b8', size=10)))
    fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=9), tickangle=-45), yaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', showticklabels=False), bargap=0.4)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_trending_table(conn, table):
    df = get_trending_news(conn, table)
    if df.empty:
        st.info("🔥 Loading...")
        return
    
    df.columns = [c.upper() for c in df.columns]
    df['HEADLINE'] = df.apply(lambda x: format_headline(x.get('NEWS_LINK', ''), x.get('MAIN_ACTOR', '')), axis=1)
    df['COUNTRY'] = df['ACTOR_COUNTRY_CODE'].apply(lambda x: get_country_name(x) if x else 'Unknown')
    df = df.drop_duplicates(subset=['HEADLINE']).head(15)
    try: df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
    except: df['DATE_FMT'] = df['DATE']
    
    st.dataframe(df[['DATE_FMT', 'HEADLINE', 'COUNTRY', 'ARTICLE_COUNT', 'NEWS_LINK']], hide_index=True, height=400,
        column_config={"DATE_FMT": "Date", "HEADLINE": st.column_config.TextColumn("Story", width="large"), "COUNTRY": "Region", "ARTICLE_COUNT": "📰", "NEWS_LINK": st.column_config.LinkColumn("🔗")}, use_container_width=True)

def render_feed_table(conn, table):
    df = get_recent_feed(conn, table)
    if df.empty:
        st.info("📋 Loading...")
        return
    
    df.columns = [c.upper() for c in df.columns]
    df['HEADLINE'] = df.apply(lambda x: format_headline(x.get('NEWS_LINK', ''), x.get('MAIN_ACTOR', '')), axis=1)
    df['COUNTRY'] = df['ACTOR_COUNTRY_CODE'].apply(lambda x: get_country_name(x) if x else 'Unknown')
    try: df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
    except: df['DATE_FMT'] = df['DATE']
    df['TONE'] = df['IMPACT_SCORE'].apply(lambda x: "🔴" if x and x < -3 else ("🟡" if x and x < 0 else ("🟢" if x and x > 2 else "⚪")))
    df = df.drop_duplicates(subset=['HEADLINE']).head(15)
    
    st.dataframe(df[['TONE', 'DATE_FMT', 'HEADLINE', 'COUNTRY', 'NEWS_LINK']], hide_index=True, height=400,
        column_config={"TONE": "", "DATE_FMT": "Date", "HEADLINE": st.column_config.TextColumn("Event", width="large"), "COUNTRY": "Region", "NEWS_LINK": st.column_config.LinkColumn("🔗")}, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: AI CHAT
# ═══════════════════════════════════════════════════════════════════════════════

def execute_ai_query(query_engine, question, conn):
    try:
        response = query_engine.query(question)
        sql = response.metadata.get('sql_query', None)
        data = safe_query(conn, sql) if sql else None
        return {'success': True, 'response': str(response), 'sql': sql, 'data': data}
    except Exception as e:
        return {'success': False, 'error': str(e), 'response': None, 'sql': None, 'data': None}

def render_ai_chat(conn, engine, table):
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "🌐 **GDELT Analyst Ready**\n\nAsk me questions about global news!"}]
    
    st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;margin-bottom:1rem;"><span style="color:#64748b;font-size:0.7rem;">💡 TRY:</span> <span style="color:#94a3b8;font-size:0.75rem;">"Show crisis events" • "What\'s happening in Russia?" • "Top countries"</span></div>', unsafe_allow_html=True)
    
    prompt = st.chat_input("Ask about global news...")
    
    for msg in st.session_state.messages[-10:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Querying..."):
                qe = get_query_engine(engine, table)
                if qe:
                    result = execute_ai_query(qe, prompt, conn)
                    if result['success']:
                        st.markdown(result['response'])
                        if result['data'] is not None and not result['data'].empty:
                            st.dataframe(result['data'], hide_index=True)
                        if result['sql']:
                            with st.expander("🔍 SQL"): st.code(result['sql'], language='sql')
                        st.session_state.messages.append({"role": "assistant", "content": result['response']})
                    else:
                        st.error(f"❌ {result.get('error', 'Failed')}")
                else:
                    st.error("AI unavailable")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: ARCHITECTURE & ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

def render_architecture():
    st.markdown("""
    <div style="text-align:center;margin-bottom:2rem;">
        <h2 style="font-family:'JetBrains Mono';color:#e2e8f0;">🏗️ System Architecture</h2>
        <p style="color:#64748b;">End-to-end pipeline: GDELT → MotherDuck → Gemini AI → Streamlit</p>
    </div>
    <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:2rem;text-align:center;margin-bottom:2rem;">
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1rem;display:inline-block;margin:0.25rem;">📰 GDELT</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1rem;display:inline-block;margin:0.25rem;">⚡ GitHub Actions</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1rem;display:inline-block;margin:0.25rem;">🦆 MotherDuck</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1rem;display:inline-block;margin:0.25rem;">🤖 Gemini AI</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1rem;display:inline-block;margin:0.25rem;">🎨 Streamlit</span>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
            <h4 style="color:#06b6d4;font-size:0.9rem;">📥 DATA SOURCE: GDELT</h4>
            <p style="color:#94a3b8;font-size:0.8rem;">Global Database of Events, Language & Tone - monitors news worldwide in 100+ languages.</p>
            <ul style="color:#94a3b8;font-size:0.85rem;padding-left:1.2rem;">
                <li><b>Updates:</b> Every 15 minutes</li><li><b>Pipeline:</b> Dagster + dbt</li><li><b>Automation:</b> GitHub Actions</li>
            </ul>
        </div>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;">
            <h4 style="color:#8b5cf6;font-size:0.9rem;">🤖 GENERATIVE AI</h4>
            <ul style="color:#94a3b8;font-size:0.85rem;padding-left:1.2rem;">
                <li><b>LLM:</b> Google Gemini 2.5</li><li><b>Framework:</b> LlamaIndex</li><li><b>Feature:</b> Text-to-SQL</li><li><b>Cost:</b> Free Tier</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
            <h4 style="color:#10b981;font-size:0.9rem;">🗄️ DATA STORAGE</h4>
            <p style="color:#94a3b8;font-size:0.8rem;"><b>Originally:</b> Snowflake → <b>Migrated:</b> MotherDuck</p>
            <ul style="color:#94a3b8;font-size:0.85rem;padding-left:1.2rem;">
                <li><b>Database:</b> DuckDB (columnar)</li><li><b>Cloud:</b> MotherDuck (serverless)</li><li><b>Performance:</b> Sub-second queries</li>
            </ul>
            <div style="margin-top:0.75rem;padding:0.5rem;background:rgba(16,185,129,0.1);border-radius:6px;border-left:3px solid #10b981;">
                <span style="color:#10b981;font-size:0.75rem;">💡 COST: Snowflake → Free Tier</span>
            </div>
        </div>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;">
            <h4 style="color:#f59e0b;font-size:0.9rem;">📊 VISUALIZATION</h4>
            <ul style="color:#94a3b8;font-size:0.85rem;padding-left:1.2rem;">
                <li><b>Framework:</b> Streamlit</li><li><b>Charts:</b> Plotly</li><li><b>Hosting:</b> Streamlit Cloud</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<h3 style="text-align:center;color:#e2e8f0;">🛠️ Tech Stack</h3>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;padding:1rem;"><span class="tech-badge">🐍 Python</span><span class="tech-badge">❄️ Snowflake</span><span class="tech-badge">🦆 DuckDB</span><span class="tech-badge">☁️ MotherDuck</span><span class="tech-badge">⚙️ Dagster</span><span class="tech-badge">🔧 dbt</span><span class="tech-badge">🤖 Gen AI</span><span class="tech-badge">🦙 LlamaIndex</span><span class="tech-badge">✨ Gemini</span><span class="tech-badge">📊 Plotly</span><span class="tech-badge">🎨 Streamlit</span><span class="tech-badge">🔄 CI/CD</span><span class="tech-badge">⚡ GitHub Actions</span><span class="tech-badge">🐼 Pandas</span><span class="tech-badge">🗃️ SQL</span></div>', unsafe_allow_html=True)

def render_about():
    st.markdown("""
    <div style="text-align:center;padding:2rem 0;">
        <h2 style="font-family:'JetBrains Mono';color:#e2e8f0;">👋 About This Project</h2>
        <p style="color:#94a3b8;max-width:750px;margin:0 auto 1.5rem;line-height:1.7;">
            This platform analyzes data from <b>GDELT</b> — the world's largest open database of human society, monitoring news from every country in 100+ languages.
        </p>
        <p style="color:#64748b;max-width:700px;margin:0 auto 2rem;line-height:1.7;">
            Originally built on <b>Snowflake</b>, then migrated to <b>MotherDuck</b> for cost optimization. Uses <b>Gemini AI free tier</b> for natural language queries.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;"><h4 style="color:#06b6d4;font-size:0.9rem;">🎯 PROJECT GOALS</h4><ul style="color:#94a3b8;font-size:0.85rem;line-height:1.8;padding-left:1.2rem;"><li>End-to-end data engineering</li><li>AI/LLM integration</li><li>Production-grade dashboards</li><li>Cost optimization</li></ul></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;"><h4 style="color:#10b981;font-size:0.9rem;">🛠️ SKILLS</h4><ul style="color:#94a3b8;font-size:0.85rem;line-height:1.8;padding-left:1.2rem;"><li>Python, SQL, Data Engineering</li><li>ETL/ELT (Dagster, dbt)</li><li>Cloud (Snowflake, MotherDuck)</li><li>Gen AI / LLM Integration</li><li>CI/CD, Visualization</li></ul></div>', unsafe_allow_html=True)
    
    st.markdown('---<div style="text-align:center;"><h4 style="color:#e2e8f0;">📬 GET IN TOUCH</h4><div style="display:flex;justify-content:center;gap:1rem;"><a href="https://github.com/Mohith-akash" target="_blank" style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1.25rem;color:#e2e8f0;text-decoration:none;">⭐ GitHub</a><a href="https://www.linkedin.com/in/mohith-akash/" target="_blank" style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1.25rem;color:#e2e8f0;text-decoration:none;">💼 LinkedIn</a></div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    inject_custom_css()
    
    # Initialize connections
    conn = get_db_connection()
    table = get_table_name(conn)
    engine = get_ai_engine(get_sql_engine(), table)
    
    render_header()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 HOME", "📈 TRENDS", "🤖 AI", "🏗️ TECH", "👤 ABOUT"])
    st.markdown('<div class="swipe-hint">👆 Swipe for more tabs →</div>', unsafe_allow_html=True)
    
    with tab1:
        render_metrics(conn, table)
        render_alert_ticker(conn, table)
        st.markdown("---")
        c1, c2 = st.columns([6, 4])
        with c1:
            st.markdown('<div class="card-header"><span>📰</span><span class="card-title">Latest Headlines</span></div>', unsafe_allow_html=True)
            render_quick_briefing(conn, table)
        with c2:
            st.markdown('<div class="card-header"><span>⚡</span><span class="card-title">Weekly Sentiment</span></div>', unsafe_allow_html=True)
            render_conflict_gauge(conn, table)
        st.markdown("---")
        c1, c2 = st.columns([6, 4])
        with c1:
            st.markdown('<div class="card-header"><span>🎯</span><span class="card-title">Most Mentioned</span></div>', unsafe_allow_html=True)
            render_top_actors(conn, table)
        with c2:
            st.markdown('<div class="card-header"><span>📊</span><span class="card-title">Tone Breakdown</span></div>', unsafe_allow_html=True)
            render_impact_distribution(conn, table)
            st.markdown('<div class="card-header" style="margin-top:1rem;"><span>🏆</span><span class="card-title">Top Countries</span></div>', unsafe_allow_html=True)
            render_country_bar_chart(conn, table)
    
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card-header"><span>🔥</span><span class="card-title">Trending Stories</span></div>', unsafe_allow_html=True)
            render_trending_table(conn, table)
        with c2:
            st.markdown('<div class="card-header"><span>📋</span><span class="card-title">Recent Events</span></div>', unsafe_allow_html=True)
            render_feed_table(conn, table)
        st.markdown("---")
        st.markdown('<div class="card-header"><span>📈</span><span class="card-title">30-Day Trend</span></div>', unsafe_allow_html=True)
        render_time_series_chart(conn, table)
    
    with tab3:
        c1, c2 = st.columns([7, 3])
        with c1:
            st.markdown('<div class="card-header"><span>🤖</span><span class="card-title">Ask in Plain English</span></div>', unsafe_allow_html=True)
            render_ai_chat(conn, engine, table)
        with c2:
            st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.25rem;"><h4 style="color:#06b6d4;font-size:0.85rem;">ℹ️ HOW IT WORKS</h4><p style="color:#94a3b8;font-size:0.8rem;">Your question → Gemini AI → SQL → Results</p><p style="font-size:0.75rem;color:#94a3b8;margin-top:1rem;">📅 Dates • 👤 Actors • 📊 Scores • 🔗 Links</p></div>', unsafe_allow_html=True)
    
    with tab4:
        render_architecture()
    
    with tab5:
        render_about()
    
    st.markdown('<div style="text-align:center;padding:2rem 0 1rem;border-top:1px solid #1e3a5f;margin-top:2rem;"><p style="color:#64748b;font-size:0.8rem;"><b>GDELT</b> monitors worldwide news in real-time.</p><p style="color:#475569;font-size:0.75rem;">Built by <a href="https://www.linkedin.com/in/mohith-akash/" style="color:#06b6d4;">Mohith Akash</a> | Snowflake → MotherDuck | Gemini AI</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
