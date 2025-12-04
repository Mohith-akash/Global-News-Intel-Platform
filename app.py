"""
🌐 GLOBAL NEWS INTELLIGENCE PLATFORM
Real-Time Analytics Dashboard for GDELT
Author: Mohith Akash | Portfolio Project
"""

import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine
import datetime
import pycountry
import logging
import re
from urllib.parse import urlparse, unquote
import duckdb

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Global News Intelligence", page_icon="🌐", layout="wide", initial_sidebar_state="collapsed")
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gdelt")

REQUIRED_ENVS = ["MOTHERDUCK_TOKEN", "GOOGLE_API_KEY"]
missing = [k for k in REQUIRED_ENVS if not os.getenv(k)]
if missing:
    st.error(f"❌ Missing: {', '.join(missing)}")
    st.stop()

GEMINI_MODEL = "models/gemini-2.5-flash-preview-05-20"
NOW = datetime.datetime.now()
WEEK_AGO = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
MONTH_AGO = (NOW - datetime.timedelta(days=30)).strftime('%Y%m%d')

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
        :root { --bg:#0a0e17; --card:#111827; --border:#1e3a5f; --text:#e2e8f0; --muted:#94a3b8; --cyan:#06b6d4; --green:#10b981; --red:#ef4444; --amber:#f59e0b; }
        .stApp { background: var(--bg); }
        header[data-testid="stHeader"], #MainMenu, footer, .stDeployButton { display: none !important; }
        html, body, p, span, div { font-family: 'Inter', sans-serif; color: var(--text); }
        h1, h2, h3, code { font-family: 'JetBrains Mono', monospace; }
        .block-container { padding: 1.5rem 2rem; max-width: 100%; }
        .header { border-bottom: 1px solid var(--border); padding: 1rem 0 1.5rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; }
        .logo { display: flex; align-items: center; gap: 0.75rem; }
        .logo-icon { font-size: 2.5rem; }
        .logo-title { font-family: 'JetBrains Mono'; font-size: 1.4rem; font-weight: 700; text-transform: uppercase; }
        .logo-sub { font-size: 0.7rem; color: var(--cyan); }
        .live-badge { display: flex; align-items: center; gap: 0.5rem; background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.75rem; }
        .live-dot { width: 8px; height: 8px; background: var(--green); border-radius: 50%; animation: pulse 2s infinite; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
        div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; }
        div[data-testid="stMetric"] label { color: var(--muted); font-size: 0.7rem; font-family: 'JetBrains Mono'; text-transform: uppercase; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono'; }
        .card-hdr { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border); }
        .card-title { font-family: 'JetBrains Mono'; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
        .stTabs [data-baseweb="tab-list"] { gap: 0; background: #0d1320; border-radius: 8px; padding: 4px; border: 1px solid var(--border); overflow-x: auto; }
        .stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono'; font-size: 0.75rem; color: var(--muted); padding: 0.5rem 0.9rem; white-space: nowrap; }
        .stTabs [aria-selected="true"] { background: #1a2332; color: var(--cyan); border-radius: 6px; }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
        div[data-testid="stDataFrame"] { background: var(--card); border: 1px solid var(--border); border-radius: 12px; }
        div[data-testid="stDataFrame"] th { background: #1a2332 !important; color: var(--muted) !important; font-size: 0.75rem; text-transform: uppercase; }
        .ticker { background: linear-gradient(90deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05)); border-left: 4px solid var(--red); border-radius: 0 8px 8px 0; padding: 0.6rem 0; overflow: hidden; position: relative; margin: 0.5rem 0; }
        .ticker-label { position: absolute; left: 0; top: 0; bottom: 0; background: linear-gradient(90deg, rgba(127,29,29,0.98), transparent); padding: 0.6rem 1.25rem 0.6rem 0.75rem; font-size: 0.7rem; font-weight: 600; color: var(--red); display: flex; align-items: center; gap: 0.5rem; z-index: 2; }
        .ticker-dot { width: 7px; height: 7px; background: var(--red); border-radius: 50%; animation: blink 1s infinite; }
        @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
        .ticker-text { display: inline-block; white-space: nowrap; padding-left: 95px; animation: scroll 40s linear infinite; font-size: 0.8rem; color: #fca5a5; }
        @keyframes scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        .tech-badge { display: inline-flex; background: #1a2332; border: 1px solid var(--border); border-radius: 20px; padding: 0.4rem 0.8rem; font-size: 0.75rem; color: var(--muted); margin: 0.25rem; }
        @media (max-width: 768px) { .block-container { padding: 1rem 0.75rem; } .logo-title { font-size: 1rem !important; } .logo-sub { display: none; } }
        hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_db():
    return duckdb.connect(f'md:gdelt_db?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}', read_only=True)

@st.cache_resource
def get_engine():
    return create_engine(f"duckdb:///md:gdelt_db?motherduck_token={os.getenv('MOTHERDUCK_TOKEN')}")

@st.cache_data(ttl=3600)
def detect_table(_conn):
    try:
        result = _conn.execute("SHOW TABLES").df()
        if not result.empty:
            for t in result.iloc[:, 0].tolist():
                if 'event' in t.lower():
                    return t
            return result.iloc[0, 0]
    except: pass
    return 'events_dagster'

def safe_query(conn, sql):
    try: return conn.execute(sql).df()
    except Exception as e:
        logger.error(f"Query error: {e}")
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════════
# AI ENGINE - Fixed to auto-discover tables
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_engine(_engine):
    try:
        llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"), model=GEMINI_MODEL, temperature=0.1)
        embed = GeminiEmbedding(api_key=os.getenv("GOOGLE_API_KEY"), model_name="models/embedding-001")
        Settings.llm = llm
        Settings.embed_model = embed
        # Let SQLDatabase auto-discover tables - no include_tables!
        sql_db = SQLDatabase(_engine)
        logger.info(f"AI Engine OK. Tables: {sql_db.get_usable_table_names()}")
        return sql_db
    except Exception as e:
        logger.error(f"AI init: {e}")
        return None

@st.cache_resource
def get_query_engine(_sql_db):
    if not _sql_db: return None
    try:
        tables = list(_sql_db.get_usable_table_names())
        target = next((t for t in tables if 'event' in t.lower()), tables[0] if tables else None)
        if target:
            return NLSQLTableQueryEngine(sql_database=_sql_db, tables=[target])
        return NLSQLTableQueryEngine(sql_database=_sql_db)
    except Exception as e:
        logger.error(f"QE error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# HEADLINE CLEANING - COMPREHENSIVE FIX
# ═══════════════════════════════════════════════════════════════════════════════

def get_country(code):
    if not code or not isinstance(code, str) or len(code) != 2: return None
    try:
        c = pycountry.countries.get(alpha_2=code.upper())
        return c.name if c else None
    except: return None

def clean_headline(text):
    """Clean and validate headline text"""
    if not text: return None
    text = str(text).strip()
    
    # 1. Strip leading numbers/codes like "25672040." or "C90000"
    text = re.sub(r'^[\dA-Fa-f]{5,}[\.\s\-_]*', '', text)
    text = re.sub(r'^\d+[\.\s\-_]+', '', text)
    
    # 2. Strip dates at start like "2025 12 02" or "2025-12-02"
    text = re.sub(r'^20\d{2}[\s\-_/]?\d{2}[\s\-_/]?\d{2}[\s\-_]*', '', text)
    
    # 3. Remove file extensions
    text = re.sub(r'\.(html?|php|aspx?|jsp|shtml|htm)$', '', text, flags=re.I)
    
    # 4. Replace separators with spaces
    text = re.sub(r'[-_]+', ' ', text)
    
    # 5. Clean up whitespace
    text = ' '.join(text.split())
    
    # 6. Title case if needed
    if text.isupper() or text.islower():
        text = text.title()
    
    # 7. Validate - reject garbage
    if len(text) < 8: return None
    
    # Too many numbers = garbage
    nums = sum(c.isdigit() for c in text.replace(' ', ''))
    if nums > len(text) * 0.3: return None
    
    # Hex hash = garbage
    if re.match(r'^[A-Fa-f0-9]{10,}$', text.replace(' ', '')): return None
    
    # Known garbage patterns
    garbage = ['govno', 'news=', 'wxii', 'trend', 'bbs', 'posnews']
    if text.lower() in garbage or any(g in text.lower() for g in garbage): return None
    
    # Single word (likely domain) = garbage unless long
    if ' ' not in text and len(text) < 15: return None
    
    return text[:120] if len(text) >= 8 else None

def extract_headline(url, actor=None):
    """Extract clean headline from URL"""
    if not url:
        return clean_headline(actor)
    
    try:
        parsed = urlparse(str(url))
        path = unquote(parsed.path)
        
        # Try path segments from end
        for seg in reversed([s for s in path.split('/') if s and len(s) > 5]):
            headline = clean_headline(seg)
            if headline and len(headline) > 20:
                return headline
        
        # Try actor name + domain
        if actor:
            actor_clean = clean_headline(actor)
            if actor_clean:
                domain = parsed.netloc.replace('www.', '').split('.')[0].title()
                if domain and len(domain) > 2 and not domain.isdigit():
                    return f"{actor_clean} - {domain}"
                return actor_clean
        
        # Last resort - domain context
        domain = parsed.netloc.replace('www.', '').split('.')[0].title()
        if domain and len(domain) > 3 and not domain.isdigit():
            return f"News from {domain}"
        
        return None
    except:
        return clean_headline(actor)

def process_df(df):
    """Process dataframe with clean headlines"""
    if df.empty: return df
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    
    # Create clean headlines
    headlines = []
    for _, row in df.iterrows():
        h = extract_headline(row.get('NEWS_LINK', ''), row.get('MAIN_ACTOR', ''))
        headlines.append(h if h else "Breaking News Event")
    df['HEADLINE'] = headlines
    
    # Country names
    df['REGION'] = df['ACTOR_COUNTRY_CODE'].apply(lambda x: get_country(x) or x if x else 'Global')
    
    # Format dates
    try: df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
    except: df['DATE_FMT'] = df['DATE']
    
    # Tone emoji
    df['TONE'] = df['IMPACT_SCORE'].apply(lambda x: "🔴" if x and x < -4 else ("🟡" if x and x < -1 else ("🟢" if x and x > 2 else "⚪")))
    
    # Remove garbage headlines and duplicates
    df = df[df['HEADLINE'] != "Breaking News Event"]
    df = df.drop_duplicates(subset=['HEADLINE'])
    
    # If we filtered too much, add some back
    if len(df) < 5:
        df = df.copy()
    
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def get_metrics(_c, t):
    df = safe_query(_c, f"SELECT COUNT(*) as total, SUM(CASE WHEN DATE >= '{WEEK_AGO}' THEN 1 ELSE 0 END) as recent, SUM(CASE WHEN ABS(IMPACT_SCORE) > 6 AND DATE >= '{WEEK_AGO}' THEN 1 ELSE 0 END) as critical FROM {t}")
    hs = safe_query(_c, f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as c FROM {t} WHERE DATE >= '{WEEK_AGO}' AND ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 1")
    return {'total': df.iloc[0]['total'] if not df.empty else 0, 'recent': df.iloc[0]['recent'] if not df.empty else 0, 'critical': df.iloc[0]['critical'] if not df.empty else 0, 'hotspot': hs.iloc[0]['ACTOR_COUNTRY_CODE'] if not hs.empty else None}

@st.cache_data(ttl=600)
def get_alerts(_c, t):
    d = (NOW - datetime.timedelta(days=3)).strftime('%Y%m%d')
    return safe_query(_c, f"SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t} WHERE DATE >= '{d}' AND IMPACT_SCORE < -4 AND MAIN_ACTOR IS NOT NULL ORDER BY IMPACT_SCORE LIMIT 15")

@st.cache_data(ttl=600)
def get_headlines(_c, t):
    return safe_query(_c, f"SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t} WHERE NEWS_LINK IS NOT NULL AND ARTICLE_COUNT > 5 AND DATE >= '{WEEK_AGO}' ORDER BY DATE DESC, ARTICLE_COUNT DESC LIMIT 60")

@st.cache_data(ttl=600)
def get_trending(_c, t):
    return safe_query(_c, f"SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT FROM {t} WHERE DATE >= '{WEEK_AGO}' AND ARTICLE_COUNT > 3 AND NEWS_LINK IS NOT NULL ORDER BY ARTICLE_COUNT DESC LIMIT 60")

@st.cache_data(ttl=600)
def get_feed(_c, t):
    return safe_query(_c, f"SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t} WHERE DATE >= '{WEEK_AGO}' AND NEWS_LINK IS NOT NULL ORDER BY DATE DESC LIMIT 60")

@st.cache_data(ttl=600)
def get_countries(_c, t):
    return safe_query(_c, f"SELECT ACTOR_COUNTRY_CODE as country, COUNT(*) as events FROM {t} WHERE DATE >= '{MONTH_AGO}' AND ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")

@st.cache_data(ttl=600)
def get_timeseries(_c, t):
    return safe_query(_c, f"SELECT DATE, COUNT(*) as events, SUM(CASE WHEN IMPACT_SCORE < -2 THEN 1 ELSE 0 END) as negative, SUM(CASE WHEN IMPACT_SCORE > 2 THEN 1 ELSE 0 END) as positive FROM {t} WHERE DATE >= '{MONTH_AGO}' GROUP BY 1 ORDER BY 1")

@st.cache_data(ttl=600)
def get_sentiment(_c, t):
    return safe_query(_c, f"SELECT AVG(IMPACT_SCORE) as avg, SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as neg, SUM(CASE WHEN IMPACT_SCORE > 3 THEN 1 ELSE 0 END) as pos, COUNT(*) as total FROM {t} WHERE DATE >= '{WEEK_AGO}' AND IMPACT_SCORE IS NOT NULL")

@st.cache_data(ttl=600)
def get_actors(_c, t):
    return safe_query(_c, f"SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, COUNT(*) as events, AVG(IMPACT_SCORE) as avg_impact FROM {t} WHERE DATE >= '{WEEK_AGO}' AND MAIN_ACTOR IS NOT NULL AND LENGTH(MAIN_ACTOR) > 3 GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10")

@st.cache_data(ttl=600)
def get_distribution(_c, t):
    return safe_query(_c, f"SELECT CASE WHEN IMPACT_SCORE < -5 THEN 'Crisis' WHEN IMPACT_SCORE < -2 THEN 'Negative' WHEN IMPACT_SCORE < 2 THEN 'Neutral' WHEN IMPACT_SCORE < 5 THEN 'Positive' ELSE 'Very Positive' END as cat, COUNT(*) as cnt FROM {t} WHERE DATE >= '{WEEK_AGO}' AND IMPACT_SCORE IS NOT NULL GROUP BY 1")

# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown('<div class="header"><div class="logo"><span class="logo-icon">🌐</span><div><div class="logo-title">Global News Intelligence</div><div class="logo-sub">Powered by GDELT • Real-Time Analytics</div></div></div><div class="live-badge"><span class="live-dot"></span> LIVE DATA</div></div>', unsafe_allow_html=True)

def render_metrics(c, t):
    m = get_metrics(c, t)
    c1, c2, c3, c4, c5 = st.columns(5)
    def fmt(n): return f"{n/1000000:.1f}M" if n and n >= 1000000 else (f"{n/1000:.1f}K" if n and n >= 1000 else str(int(n or 0)))
    with c1: st.metric("📡 TOTAL", fmt(m['total']), "All time")
    with c2: st.metric("⚡ 7 DAYS", fmt(m['recent']), "Recent")
    with c3: st.metric("🚨 CRITICAL", fmt(m['critical']), "High impact")
    hs = m['hotspot']
    with c4: st.metric("🔥 HOTSPOT", (get_country(hs) or hs or "N/A")[:12], hs or "")
    with c5: st.metric("📅 UPDATED", NOW.strftime("%H:%M"), NOW.strftime("%d %b"))

def render_ticker(c, t):
    df = get_alerts(c, t)
    if df.empty:
        txt = "⚡ Monitoring global news for significant events │ Platform powered by GDELT │ "
    else:
        items = []
        for _, r in df.iterrows():
            actor = clean_headline(r.get('MAIN_ACTOR', '')) or "Event"
            # Ensure we always have a valid string for country
            country_code = r.get('ACTOR_COUNTRY_CODE', '')
            country = get_country(country_code) or country_code or 'Global'
            # Additional safety check to ensure they're strings
            actor = str(actor) if actor else "Event"
            country = str(country) if country else "Global"
            items.append(f"⚠️ {actor[:25]} ({country[:12]}) • {r.get('IMPACT_SCORE', 0):.1f}")
        txt = " │ ".join(items) + " │ "
    st.markdown(f'<div class="ticker"><div class="ticker-label"><span class="ticker-dot"></span> LIVE</div><div class="ticker-text">{txt + txt}</div></div>', unsafe_allow_html=True)

def render_headlines(c, t):
    df = get_headlines(c, t)
    if df.empty: st.info("📰 Loading..."); return
    df = process_df(df).head(12)
    if df.empty: st.info("📰 No headlines available"); return
    st.dataframe(df[['TONE', 'DATE_FMT', 'HEADLINE', 'REGION', 'NEWS_LINK']], hide_index=True, height=350,
        column_config={"TONE": st.column_config.TextColumn("", width="small"), "DATE_FMT": st.column_config.TextColumn("Date", width="small"), "HEADLINE": st.column_config.TextColumn("Headline", width="large"), "REGION": st.column_config.TextColumn("Region", width="small"), "NEWS_LINK": st.column_config.LinkColumn("🔗", width="small")}, use_container_width=True)

def render_sentiment(c, t):
    df = get_sentiment(c, t)
    if df.empty: st.info("Loading..."); return
    avg, neg, pos, total = df.iloc[0]['avg'] or 0, int(df.iloc[0]['neg'] or 0), int(df.iloc[0]['pos'] or 0), int(df.iloc[0]['total'] or 1)
    status, color = ("⚠️ ELEVATED", "#ef4444") if avg < -2 else (("🟡 MODERATE", "#f59e0b") if avg < 0 else (("🟢 STABLE", "#10b981") if avg < 2 else ("✨ POSITIVE", "#06b6d4")))
    st.markdown(f'<div style="text-align:center;padding:1.25rem;background:linear-gradient(135deg,rgba(14,165,233,0.1),rgba(6,182,212,0.05));border-radius:12px;border:1px solid #1e3a5f;margin-bottom:1rem;"><div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;">Weekly Sentiment</div><div style="font-size:1.75rem;font-weight:700;color:{color};">{status}</div><div style="font-size:0.8rem;color:#94a3b8;">Avg: <span style="color:{color}">{avg:.2f}</span></div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div style="text-align:center;padding:0.75rem;background:rgba(239,68,68,0.1);border-radius:8px;"><div style="font-size:1.25rem;font-weight:700;color:#ef4444;">{neg:,}</div><div style="font-size:0.6rem;color:#94a3b8;">NEGATIVE</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div style="text-align:center;padding:0.75rem;background:rgba(107,114,128,0.1);border-radius:8px;"><div style="font-size:1.25rem;font-weight:700;color:#9ca3af;">{total:,}</div><div style="font-size:0.6rem;color:#94a3b8;">TOTAL</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div style="text-align:center;padding:0.75rem;background:rgba(16,185,129,0.1);border-radius:8px;"><div style="font-size:1.25rem;font-weight:700;color:#10b981;">{pos:,}</div><div style="font-size:0.6rem;color:#94a3b8;">POSITIVE</div></div>', unsafe_allow_html=True)

def render_actors(c, t):
    df = get_actors(c, t)
    if df.empty: st.info("🎯 Loading..."); return
    labels = [f"{(clean_headline(r['MAIN_ACTOR']) or 'Unknown')[:18]} ({r.get('ACTOR_COUNTRY_CODE', '')})" if r.get('ACTOR_COUNTRY_CODE') else (clean_headline(r['MAIN_ACTOR']) or 'Unknown')[:20] for _, r in df.iterrows()]
    colors = ['#ef4444' if x and x < -3 else ('#f59e0b' if x and x < 0 else ('#10b981' if x and x > 3 else '#06b6d4')) for x in df['avg_impact']]
    fig = go.Figure(go.Bar(x=df['events'], y=labels, orientation='h', marker_color=colors, text=df['events'].apply(lambda x: f'{x:,}'), textposition='outside', textfont=dict(color='#94a3b8', size=10)))
    fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=50,t=10,b=0), xaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')), yaxis=dict(showgrid=False, tickfont=dict(color='#e2e8f0', size=11), autorange='reversed'), bargap=0.3)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_distribution(c, t):
    df = get_distribution(c, t)
    if df.empty: st.info("📊 Loading..."); return
    colors = {'Crisis': '#ef4444', 'Negative': '#f59e0b', 'Neutral': '#64748b', 'Positive': '#10b981', 'Very Positive': '#06b6d4'}
    fig = go.Figure(data=[go.Pie(labels=df['cat'], values=df['cnt'], hole=0.6, marker_colors=[colors.get(c, '#64748b') for c in df['cat']], textinfo='percent', textfont=dict(size=11, color='#e2e8f0'))])
    fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=10), showlegend=True, legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center', font=dict(size=10, color='#94a3b8')))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_countries(c, t):
    df = get_countries(c, t)
    if df.empty: st.info("🏆 Loading..."); return
    df = df.head(8)
    df['name'] = df['country'].apply(lambda x: get_country(x) or x or 'Unknown')
    fmt = lambda n: f"{n/1000:.1f}K" if n >= 1000 else str(int(n))
    fig = go.Figure(go.Bar(x=df['name'], y=df['events'], marker_color='#06b6d4', text=df['events'].apply(fmt), textposition='outside', textfont=dict(color='#94a3b8', size=10)))
    fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=9), tickangle=-45), yaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', showticklabels=False), bargap=0.4)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_trending(c, t):
    df = get_trending(c, t)
    if df.empty: st.info("🔥 Loading..."); return
    df = process_df(df).head(15)
    if df.empty: st.info("🔥 No stories"); return
    st.dataframe(df[['DATE_FMT', 'HEADLINE', 'REGION', 'ARTICLE_COUNT', 'NEWS_LINK']], hide_index=True, height=400,
        column_config={"DATE_FMT": st.column_config.TextColumn("Date", width="small"), "HEADLINE": st.column_config.TextColumn("Story", width="large"), "REGION": st.column_config.TextColumn("Region", width="small"), "ARTICLE_COUNT": st.column_config.NumberColumn("📰", width="small"), "NEWS_LINK": st.column_config.LinkColumn("🔗", width="small")}, use_container_width=True)

def render_feed(c, t):
    df = get_feed(c, t)
    if df.empty: st.info("📋 Loading..."); return
    df = process_df(df).head(15)
    if df.empty: st.info("📋 No events"); return
    st.dataframe(df[['TONE', 'DATE_FMT', 'HEADLINE', 'REGION', 'NEWS_LINK']], hide_index=True, height=400,
        column_config={"TONE": st.column_config.TextColumn("", width="small"), "DATE_FMT": st.column_config.TextColumn("Date", width="small"), "HEADLINE": st.column_config.TextColumn("Event", width="large"), "REGION": st.column_config.TextColumn("Region", width="small"), "NEWS_LINK": st.column_config.LinkColumn("🔗", width="small")}, use_container_width=True)

def render_timeseries(c, t):
    df = get_timeseries(c, t)
    if df.empty: st.info("📈 Loading..."); return
    df['date'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['date'], y=df['events'], fill='tozeroy', fillcolor='rgba(6,182,212,0.15)', line=dict(color='#06b6d4', width=2), name='Total'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['date'], y=df['negative'], line=dict(color='#ef4444', width=2), name='Negative'), secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df['positive'], line=dict(color='#10b981', width=2), name='Positive'), secondary_y=True)
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=30,b=0), showlegend=True, legend=dict(orientation='h', y=1.02, font=dict(size=11, color='#94a3b8')), xaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')), yaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ═══════════════════════════════════════════════════════════════════════════════
# AI CHAT
# ═══════════════════════════════════════════════════════════════════════════════

def render_ai_chat(c, sql_db):
    if "msgs" not in st.session_state:
        st.session_state.msgs = [{"role": "assistant", "content": "🌐 **GDELT Analyst Ready**\n\nAsk me about global news events!"}]
    
    st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;margin-bottom:1rem;"><span style="color:#64748b;font-size:0.7rem;">💡 TRY:</span> <span style="color:#94a3b8;font-size:0.75rem;">"Show crisis events" • "What\'s happening in Russia?" • "Top 5 countries"</span></div>', unsafe_allow_html=True)
    
    prompt = st.chat_input("Ask about global news...")
    for msg in st.session_state.msgs[-8:]:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    if prompt:
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Querying..."):
                qe = get_query_engine(sql_db)
                if qe:
                    try:
                        response = qe.query(prompt)
                        answer = str(response)
                        st.markdown(answer)
                        sql = response.metadata.get('sql_query')
                        if sql:
                            data = safe_query(c, sql)
                            if not data.empty: st.dataframe(data.head(20), hide_index=True, use_container_width=True)
                            with st.expander("🔍 SQL"): st.code(sql, language='sql')
                        st.session_state.msgs.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {str(e)[:100]}")
                        st.info("💡 Try: 'Show recent events'")
                else:
                    st.error("AI engine not ready. Check API keys.")

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE & ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

def render_arch():
    st.markdown('<div style="text-align:center;margin-bottom:2rem;"><h2 style="font-family:JetBrains Mono;color:#e2e8f0;">🏗️ System Architecture</h2><p style="color:#64748b;">GDELT → GitHub Actions → MotherDuck → Gemini AI → Streamlit</p></div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:2rem;text-align:center;margin-bottom:2rem;"><span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.25rem;">📰 GDELT</span><span style="color:#06b6d4;margin:0 0.5rem;">→</span><span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.25rem;">⚡ GitHub Actions</span><span style="color:#06b6d4;margin:0 0.5rem;">→</span><span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.25rem;">🦆 MotherDuck</span><span style="color:#06b6d4;margin:0 0.5rem;">→</span><span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.25rem;">🤖 Gemini AI</span><span style="color:#06b6d4;margin:0 0.5rem;">→</span><span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.25rem;">🎨 Streamlit</span></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;margin-bottom:1rem;"><h4 style="color:#06b6d4;font-size:0.9rem;">📥 DATA: GDELT</h4><p style="color:#94a3b8;font-size:0.8rem;">Global Database of Events - monitors news worldwide in 100+ languages.</p><ul style="color:#94a3b8;font-size:0.85rem;"><li>Updates every 15 min</li><li>Dagster + dbt pipeline</li><li>GitHub Actions</li></ul></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;"><h4 style="color:#8b5cf6;font-size:0.9rem;">🤖 GEN AI</h4><ul style="color:#94a3b8;font-size:0.85rem;"><li>Google Gemini 2.5</li><li>LlamaIndex</li><li>Text-to-SQL</li><li>Free Tier</li></ul></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;margin-bottom:1rem;"><h4 style="color:#10b981;font-size:0.9rem;">🗄️ STORAGE</h4><p style="color:#94a3b8;font-size:0.8rem;">Snowflake → MotherDuck migration</p><ul style="color:#94a3b8;font-size:0.85rem;"><li>DuckDB (columnar)</li><li>Serverless</li><li>Sub-second queries</li></ul><div style="margin-top:0.5rem;padding:0.5rem;background:rgba(16,185,129,0.1);border-radius:6px;border-left:3px solid #10b981;"><span style="color:#10b981;font-size:0.75rem;">💡 COST: $0/month</span></div></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;"><h4 style="color:#f59e0b;font-size:0.9rem;">📊 VISUALIZATION</h4><ul style="color:#94a3b8;font-size:0.85rem;"><li>Streamlit</li><li>Plotly</li><li>Cloud hosting</li></ul></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<h3 style="text-align:center;color:#e2e8f0;">🛠️ Tech Stack</h3><div style="text-align:center;padding:1rem;"><span class="tech-badge">🐍 Python</span><span class="tech-badge">❄️ Snowflake</span><span class="tech-badge">🦆 DuckDB</span><span class="tech-badge">☁️ MotherDuck</span><span class="tech-badge">⚙️ Dagster</span><span class="tech-badge">🔧 dbt</span><span class="tech-badge">🤖 Gen AI</span><span class="tech-badge">🦙 LlamaIndex</span><span class="tech-badge">✨ Gemini</span><span class="tech-badge">📊 Plotly</span><span class="tech-badge">🎨 Streamlit</span><span class="tech-badge">🔄 CI/CD</span></div>', unsafe_allow_html=True)

def render_about():
    st.markdown('<div style="text-align:center;padding:2rem 0;"><h2 style="font-family:JetBrains Mono;color:#e2e8f0;">👋 About This Project</h2><p style="color:#94a3b8;max-width:750px;margin:0 auto 1.5rem;">Real-time analytics for <b>GDELT</b> — world\'s largest open database monitoring global news in 100+ languages.</p><p style="color:#64748b;max-width:700px;margin:0 auto 2rem;">Built on Snowflake, migrated to MotherDuck. Uses Gemini AI free tier.</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;"><h4 style="color:#06b6d4;font-size:0.9rem;">🎯 GOALS</h4><ul style="color:#94a3b8;font-size:0.85rem;line-height:1.8;"><li>End-to-end data engineering</li><li>AI/LLM integration</li><li>Production dashboards</li><li>Cost optimization</li></ul></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;"><h4 style="color:#10b981;font-size:0.9rem;">🛠️ SKILLS</h4><ul style="color:#94a3b8;font-size:0.85rem;line-height:1.8;"><li>Python, SQL, Data Engineering</li><li>ETL/ELT (Dagster, dbt)</li><li>Cloud (Snowflake, MotherDuck)</li><li>Gen AI, CI/CD, Visualization</li></ul></div>', unsafe_allow_html=True)
    st.markdown('---<div style="text-align:center;"><h4 style="color:#e2e8f0;">📬 CONTACT</h4><div style="display:flex;justify-content:center;gap:1rem;"><a href="https://github.com/Mohith-akash" target="_blank" style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1.25rem;color:#e2e8f0;text-decoration:none;">⭐ GitHub</a><a href="https://www.linkedin.com/in/mohith-akash/" target="_blank" style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1.25rem;color:#e2e8f0;text-decoration:none;">💼 LinkedIn</a></div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()
    conn = get_db()
    tbl = detect_table(conn)
    sql_db = get_ai_engine(get_engine())
    
    render_header()
    tabs = st.tabs(["📊 HOME", "📈 TRENDS", "🤖 AI", "🏗️ TECH", "👤 ABOUT"])
    
    with tabs[0]:
        render_metrics(conn, tbl)
        render_ticker(conn, tbl)
        st.markdown("---")
        c1, c2 = st.columns([6, 4])
        with c1:
            st.markdown('<div class="card-hdr"><span>📰</span><span class="card-title">Latest Headlines</span></div>', unsafe_allow_html=True)
            render_headlines(conn, tbl)
        with c2:
            st.markdown('<div class="card-hdr"><span>⚡</span><span class="card-title">Weekly Sentiment</span></div>', unsafe_allow_html=True)
            render_sentiment(conn, tbl)
        st.markdown("---")
        c1, c2 = st.columns([6, 4])
        with c1:
            st.markdown('<div class="card-hdr"><span>🎯</span><span class="card-title">Most Mentioned</span></div>', unsafe_allow_html=True)
            render_actors(conn, tbl)
        with c2:
            st.markdown('<div class="card-hdr"><span>📊</span><span class="card-title">Tone Breakdown</span></div>', unsafe_allow_html=True)
            render_distribution(conn, tbl)
            st.markdown('<div class="card-hdr" style="margin-top:1rem;"><span>🏆</span><span class="card-title">Top Countries</span></div>', unsafe_allow_html=True)
            render_countries(conn, tbl)
    
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card-hdr"><span>🔥</span><span class="card-title">Trending Stories</span></div>', unsafe_allow_html=True)
            render_trending(conn, tbl)
        with c2:
            st.markdown('<div class="card-hdr"><span>📋</span><span class="card-title">Recent Events</span></div>', unsafe_allow_html=True)
            render_feed(conn, tbl)
        st.markdown("---")
        st.markdown('<div class="card-hdr"><span>📈</span><span class="card-title">30-Day Trend</span></div>', unsafe_allow_html=True)
        render_timeseries(conn, tbl)
    
    with tabs[2]:
        c1, c2 = st.columns([7, 3])
        with c1:
            st.markdown('<div class="card-hdr"><span>🤖</span><span class="card-title">Ask in Plain English</span></div>', unsafe_allow_html=True)
            render_ai_chat(conn, sql_db)
        with c2:
            st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.25rem;"><h4 style="color:#06b6d4;font-size:0.85rem;">ℹ️ HOW IT WORKS</h4><p style="color:#94a3b8;font-size:0.8rem;">Your question → Gemini AI → SQL → Results</p><p style="font-size:0.75rem;color:#94a3b8;margin-top:1rem;">📅 Dates • 👤 Actors • 📊 Scores • 🔗 Links</p></div>', unsafe_allow_html=True)
    
    with tabs[3]: render_arch()
    with tabs[4]: render_about()
    
    st.markdown('<div style="text-align:center;padding:2rem 0 1rem;border-top:1px solid #1e3a5f;margin-top:2rem;"><p style="color:#64748b;font-size:0.8rem;"><b>GDELT</b> monitors worldwide news in real-time.</p><p style="color:#475569;font-size:0.75rem;">Built by <a href="https://www.linkedin.com/in/mohith-akash/" style="color:#06b6d4;">Mohith Akash</a></p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()