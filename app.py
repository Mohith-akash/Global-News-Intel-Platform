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
from llama_index.llms.google_genai import GoogleGenAI as Gemini
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
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

def get_secret(key):
    val = os.getenv(key)
    if val: return val
    try: return st.secrets.get(key)
    except: return None

REQUIRED_ENVS = ["MOTHERDUCK_TOKEN", "GOOGLE_API_KEY"]
missing = [k for k in REQUIRED_ENVS if not get_secret(k)]
if missing:
    st.error(f"❌ Missing: {', '.join(missing)}")
    st.stop()

for key in REQUIRED_ENVS:
    val = get_secret(key)
    if val: os.environ[key] = val

GEMINI_MODEL = "gemini-2.5-flash-lite"

NOW = datetime.datetime.now()
WEEK_AGO = (NOW - datetime.timedelta(days=7)).strftime('%Y%m%d')
MONTH_AGO = (NOW - datetime.timedelta(days=30)).strftime('%Y%m%d')

# Initialize session state for tab persistence
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

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
# AI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_engine(_engine):
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found!")
            return None
        
        llm = Gemini(api_key=api_key, model=GEMINI_MODEL, temperature=0.1)
        embed = GoogleGenAIEmbedding(api_key=api_key, model_name="text-embedding-004")
        Settings.llm = llm
        Settings.embed_model = embed
        
        conn = get_db()
        main_table = detect_table(conn)
        sql_db = SQLDatabase(_engine, include_tables=[main_table])
        logger.info(f"AI Engine initialized with table: {main_table}")
        return sql_db
    except Exception as e:
        logger.error(f"AI init failed: {e}")
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
# HEADLINE CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def get_country(code):
    if not code or not isinstance(code, str) or len(code) < 2: return None
    try:
        c = pycountry.countries.get(alpha_2=code[:2].upper())
        return c.name if c else None
    except: return None

def clean_headline(text):
    if not text: return None
    text = str(text).strip()
    
    # Strip leading numbers/codes
    text = re.sub(r'^[\dA-Fa-f]{5,}[\.\s\-_]*', '', text)
    text = re.sub(r'^\d+[\.\s\-_]+', '', text)
    
    # Strip ALL date patterns
    text = re.sub(r'^20\d{2}[\s\-_/]?\d{0,2}[\s\-_/]?\d{0,2}[\s\-_]*', '', text)
    text = re.sub(r'\b\d{1,2}\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b', '', text, flags=re.I)
    text = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{1,2}\b', '', text, flags=re.I)
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text)
    text = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', text)
    
    # Remove file extensions
    text = re.sub(r'\.(html?|php|aspx?|jsp|shtml|htm)$', '', text, flags=re.I)
    
    # Replace separators with spaces
    text = re.sub(r'[-_]+', ' ', text)
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    # Title case if needed
    if text and (text.isupper() or text.islower()):
        text = text.title()
    
    # Validate
    if not text or len(text) < 10: return None
    
    nums = sum(c.isdigit() for c in text.replace(' ', ''))
    if nums > len(text) * 0.3: return None
    if re.match(r'^[A-Fa-f0-9]{10,}$', text.replace(' ', '')): return None
    
    garbage = ['govno', 'news=', 'wxii', 'trend', 'bbs', 'posnews', 'tradearabia']
    if text.lower() in garbage or any(g in text.lower() for g in garbage): return None
    if ' ' not in text and len(text) < 15: return None
    
    return text[:100]

def extract_headline(url, actor=None):
    if not url:
        return clean_headline(actor)
    try:
        parsed = urlparse(str(url))
        path = unquote(parsed.path)
        
        for seg in reversed([s for s in path.split('/') if s and len(s) > 5]):
            headline = clean_headline(seg)
            if headline and len(headline) > 20:
                return headline
        
        if actor:
            actor_clean = clean_headline(actor)
            if actor_clean:
                domain = parsed.netloc.replace('www.', '').split('.')[0].title()
                if domain and len(domain) > 2 and not domain.isdigit():
                    return f"{actor_clean} - {domain}"
                return actor_clean
        
        domain = parsed.netloc.replace('www.', '').split('.')[0].title()
        if domain and len(domain) > 3 and not domain.isdigit():
            return f"News from {domain}"
        return None
    except:
        return clean_headline(actor)

def process_df(df):
    if df.empty: return df
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    
    headlines = []
    for _, row in df.iterrows():
        h = extract_headline(row.get('NEWS_LINK', ''), row.get('MAIN_ACTOR', ''))
        if h and len(h) > 10:
            headlines.append(h)
        else:
            actor = row.get('MAIN_ACTOR', '')
            if actor and len(str(actor)) > 5:
                cleaned = clean_headline(str(actor))
                if cleaned:
                    headlines.append(cleaned[:60])
                else:
                    headlines.append(None)
            else:
                headlines.append(None)
    df['HEADLINE'] = headlines
    
    df['REGION'] = df['ACTOR_COUNTRY_CODE'].apply(lambda x: get_country(x) or x if x else 'Global')
    
    try: df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d %b')
    except: df['DATE_FMT'] = df['DATE']
    
    df['TONE'] = df['IMPACT_SCORE'].apply(lambda x: "🔴" if x and x < -4 else ("🟡" if x and x < -1 else ("🟢" if x and x > 2 else "⚪")))
    
    df = df[df['HEADLINE'].notna()]
    df = df.drop_duplicates(subset=['HEADLINE'])
    
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
    return safe_query(_c, f"SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t} WHERE DATE >= '{d}' AND IMPACT_SCORE < -4 AND MAIN_ACTOR IS NOT NULL AND LENGTH(MAIN_ACTOR) > 5 AND ACTOR_COUNTRY_CODE IS NOT NULL ORDER BY IMPACT_SCORE ASC LIMIT 15")

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
    if df.empty or len(df) < 3:
        txt = "⚡ Monitoring global news for critical events │ Real-time GDELT analysis │ AI-powered insights │ "
    else:
        items = []
        for _, r in df.iterrows():
            actor = r.get('MAIN_ACTOR', '')
            if not actor or len(str(actor)) <= 3: continue
            actor = str(actor)[:30]
            country_code = r.get('ACTOR_COUNTRY_CODE', '')
            country = get_country(country_code) if country_code else None
            country = country[:15] if country else (country_code if country_code else 'Global')
            impact = r.get('IMPACT_SCORE', 0) or 0
            items.append(f"⚠️ {actor} ({country}) • {impact:.1f}")
        txt = " │ ".join(items[:10]) + " │ " if items else "⚡ Monitoring global news │ "
    st.markdown(f'<div class="ticker"><div class="ticker-label"><span class="ticker-dot"></span> LIVE</div><div class="ticker-text">{txt + txt}</div></div>', unsafe_allow_html=True)

def render_headlines(c, t):
    df = get_headlines(c, t)
    if df.empty: st.info("📰 Loading..."); return
    df = process_df(df).head(12)
    if df.empty: st.info("📰 No headlines available"); return
    st.dataframe(df[['TONE', 'DATE_FMT', 'HEADLINE', 'REGION', 'NEWS_LINK']], hide_index=True, height=350,
        column_config={"TONE": st.column_config.TextColumn("", width="small"), "DATE_FMT": st.column_config.TextColumn("Date", width="small"), "HEADLINE": st.column_config.TextColumn("Headline", width="large"), "REGION": st.column_config.TextColumn("Region", width="small"), "NEWS_LINK": st.column_config.LinkColumn("🔗", display_text="Open", width="small")}, use_container_width=True)

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
    labels = []
    for _, r in df.iterrows():
        actor = clean_headline(r['MAIN_ACTOR']) or 'Unknown'
        cc = r.get('ACTOR_COUNTRY_CODE', '')
        labels.append(f"{actor[:18]} ({cc})" if cc else actor[:20])
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
        column_config={"DATE_FMT": st.column_config.TextColumn("Date", width="small"), "HEADLINE": st.column_config.TextColumn("Story", width="large"), "REGION": st.column_config.TextColumn("Region", width="small"), "ARTICLE_COUNT": st.column_config.NumberColumn("📰", width="small"), "NEWS_LINK": st.column_config.LinkColumn("🔗", display_text="Open", width="small")}, use_container_width=True)

def render_feed(c, t):
    df = get_feed(c, t)
    if df.empty: st.info("📋 Loading..."); return
    df = process_df(df).head(15)
    if df.empty: st.info("📋 No events"); return
    st.dataframe(df[['TONE', 'DATE_FMT', 'HEADLINE', 'REGION', 'NEWS_LINK']], hide_index=True, height=400,
        column_config={"TONE": st.column_config.TextColumn("", width="small"), "DATE_FMT": st.column_config.TextColumn("Date", width="small"), "HEADLINE": st.column_config.TextColumn("Event", width="large"), "REGION": st.column_config.TextColumn("Region", width="small"), "NEWS_LINK": st.column_config.LinkColumn("🔗", display_text="Open", width="small")}, use_container_width=True)

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
        st.session_state.msgs = [{"role": "assistant", "content": "🌐 **GDELT Analyst Ready** - Ask me about global news events!"}]
    
    st.markdown('''
        <div style="
            background:#111827;
            border:1px solid #1e3a5f;
            border-radius:8px;
            padding:0.5rem 0.75rem;
            margin-bottom:0.75rem;
        ">

            <span style="display:block;color:#94a3b8;font-size:0.75rem;">
                💡 Try the following questions:<br>
            </span>
            
            <span style="display:block;color:#94a3b8;font-size:0.75rem;margin-top:0.4rem;">
                "Top 10 countries with negative news"
            </span>
            
            <span style="display:block;color:#94a3b8;font-size:0.75rem;margin-top:0.4rem;">
                "Analyze the conflict trend in the Middle East."
            </span>
            
            <span style="display:block;color:#94a3b8;font-size:0.75rem;margin-top:0.4rem;">
                "Compare media coverage of USA vs China"
            </span>
            
            <span style="display:block;color:#94a3b8;font-size:0.75rem;margin-top:0.4rem;">
                "Which country has the lowest sentiment score?"
            </span>

        </div>
        ''', unsafe_allow_html=True)
    for msg in st.session_state.msgs[-6:]:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    prompt = st.chat_input("Ask about global news...")
    
    if prompt:
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing..."):
                qe = get_query_engine(sql_db)
                if qe:
                    try:
                        enhanced_prompt = f"""You are a GDELT news analyst. User asked: "{prompt}"

Table columns:
- DATE (YYYYMMDD integer)
- MAIN_ACTOR (person/org)
- ACTOR_COUNTRY_CODE (2-letter: US, RU, CN)
- IMPACT_SCORE (negative=conflict, positive=cooperation)
- ARTICLE_COUNT (media coverage count)
- NEWS_LINK (URL)

IMPORTANT RULES:
1. ALWAYS filter NULL values: WHERE column IS NOT NULL
2. For country queries: WHERE ACTOR_COUNTRY_CODE IS NOT NULL
3. Use LIMIT 15 max
4. Recent data: DATE >= '{WEEK_AGO}'
5. Negative events: IMPACT_SCORE < -3
6. Positive events: IMPACT_SCORE > 3

Generate SQL to answer the question."""
                        
                        response = qe.query(enhanced_prompt)
                        answer = str(response)
                        st.markdown(answer)
                        sql = response.metadata.get('sql_query')
                        if sql:
                            data = safe_query(c, sql)
                            if not data.empty:
                                st.dataframe(data.head(15), hide_index=True, use_container_width=True)
                            with st.expander("SQL"): st.code(sql, language='sql')
                        st.session_state.msgs.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {str(e)[:100]}")
                else:
                    st.warning("⚠️ AI initializing... Refresh page.")

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
    
    tab_names = ["📊 HOME", "📈 TRENDS", "🤖 AI", "🏗️ TECH", "👤 ABOUT"]
    tabs = st.tabs(tab_names)
    
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
            st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem;"><h4 style="color:#06b6d4;font-size:0.8rem;">ℹ️ HOW IT WORKS</h4><p style="color:#94a3b8;font-size:0.75rem;">Question → Gemini AI → SQL → Results</p></div>', unsafe_allow_html=True)
    
    with tabs[3]: render_arch()
    with tabs[4]: render_about()
    
    st.markdown('<div style="text-align:center;padding:2rem 0 1rem;border-top:1px solid #1e3a5f;margin-top:2rem;"><p style="color:#475569;font-size:0.75rem;">Built by <a href="https://www.linkedin.com/in/mohith-akash/" style="color:#06b6d4;">Mohith Akash</a> • GDELT Real-Time Analytics</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()