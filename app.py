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
        hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
    </style>
    """, unsafe_allow_html=True)

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
                if 'event' in t.lower(): return t
            return result.iloc[0, 0]
    except: pass
    return 'events_dagster'

def safe_query(conn, sql):
    try: return conn.execute(sql).df()
    except Exception as e:
        logger.error(f"Query error: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_ai_engine(_engine):
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: return None
        llm = Gemini(api_key=api_key, model=GEMINI_MODEL, temperature=0.1)
        embed = GoogleGenAIEmbedding(api_key=api_key, model_name="text-embedding-004")
        Settings.llm = llm
        Settings.embed_model = embed
        conn = get_db()
        main_table = detect_table(conn)
        sql_db = SQLDatabase(_engine, include_tables=[main_table])
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
        if target: return NLSQLTableQueryEngine(sql_database=_sql_db, tables=[target])
        return NLSQLTableQueryEngine(sql_database=_sql_db)
    except: return None

def get_country(code):
    if not code or not isinstance(code, str): return None
    code = code.strip().upper()
    if len(code) < 2: return None
    try:
        if len(code) == 2: 
            c = pycountry.countries.get(alpha_2=code)
            if c: return c.name
        if len(code) == 3: 
            c = pycountry.countries.get(alpha_3=code)
            if c: return c.name
        return None
    except: return None

def get_impact_label(score):
    if score is None: return "Neutral"
    score = float(score)
    if score <= -8: return "🔴 Severe Crisis"
    if score <= -5: return "🔴 Major Conflict"
    if score <= -3: return "🟠 Rising Tensions"
    if score <= -1: return "🟡 Minor Dispute"
    if score < 1: return "⚪ Neutral"
    if score < 3: return "🟢 Cooperation"
    if score < 5: return "🟢 Partnership"
    return "✨ Major Agreement"

def clean_headline(text):
    """AGGRESSIVE cleaning - remove ALL garbage patterns from headlines"""
    if not text: return None
    text = str(text).strip()
    
    reject_patterns = [
        r'^[a-f0-9]{8}[-\s][a-f0-9]{4}',
        r'^[a-f0-9\s\-]{20,}$',
        r'^(article|post|item|id)[\s\-_]*[a-f0-9]{8}',
    ]
    for pattern in reject_patterns:
        if re.match(pattern, text.lower()): return None
    
    for _ in range(5):
        text = re.sub(r'^\d{4}\s+\d{1,2}\s+\d{1,2}\s+', '', text)
        text = re.sub(r'^\d{1,2}\s+\d{1,2}\s+', '', text)
        text = re.sub(r'^\d{1,2}[/\-\.]\d{1,2}\s+', '', text)
        text = re.sub(r'^\d{4}\s+', '', text)
        text = re.sub(r'^\d{8}\s*', '', text)
        text = re.sub(r'^\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\s*', '', text)
    
    text = re.sub(r'\s+\d{1,2}\.\d{5,}', ' ', text)
    text = re.sub(r'\s+\d{5,}', ' ', text)
    text = re.sub(r'\s+[a-z]{3,5}\d[a-z\d]{4,}', ' ', text, flags=re.I)
    text = re.sub(r'\s+[a-z0-9]{12,}(?=\s|$)', ' ', text, flags=re.I)
    text = re.sub(r'[\s,]+\d{1,3}$', '', text)
    text = re.sub(r'\.(html?|php|aspx?|jsp|shtml)$', '', text, flags=re.I)
    text = re.sub(r'[-_]+', ' ', text)
    text = ' '.join(text.split())
    
    if len(text) < 10: return None
    text_no_spaces = text.replace(' ', '')
    if text_no_spaces:
        num_count = sum(c.isdigit() for c in text_no_spaces)
        if num_count > len(text_no_spaces) * 0.2: return None
    hex_count = sum(c in '0123456789abcdefABCDEF' for c in text_no_spaces)
    if hex_count > len(text_no_spaces) * 0.35: return None
    if ' ' not in text: return None
    words = text.split()
    if len(words) < 3: return None
    
    return text[:100]

def enhance_headline(text, impact_score=None, actor=None):
    """Make headlines more engaging"""
    if not text: return None
    words = text.split()
    capitalized = []
    important_words = {'president', 'minister', 'government', 'military', 'congress', 'senate', 
                      'crisis', 'attack', 'strike', 'protest', 'emergency', 'war', 'peace',
                      'agreement', 'deal', 'summit', 'meeting', 'vote', 'election', 'law',
                      'court', 'judge', 'police', 'fire', 'flood', 'earthquake', 'storm'}
    
    for i, word in enumerate(words):
        word_lower = word.lower()
        if i == 0:
            capitalized.append(word.capitalize())
        elif word_lower in important_words:
            capitalized.append(word.capitalize())
        elif word.isupper() and len(word) > 2:
            capitalized.append(word.title())
        else:
            capitalized.append(word)
    
    return ' '.join(capitalized)

def extract_headline(url, actor=None, impact_score=None):
    if not url and actor: 
        cleaned = clean_headline(actor)
        return enhance_headline(cleaned, impact_score, actor) if cleaned else None
    if not url: return None
    try:
        parsed = urlparse(str(url))
        path = unquote(parsed.path)
        segments = [s for s in path.split('/') if s and len(s) > 8]
        for seg in reversed(segments):
            cleaned = clean_headline(seg)
            if cleaned and len(cleaned) > 20:
                return enhance_headline(cleaned, impact_score, actor)
        if actor:
            cleaned = clean_headline(actor)
            return enhance_headline(cleaned, impact_score, actor) if cleaned else None
        return None
    except:
        if actor:
            cleaned = clean_headline(actor)
            return enhance_headline(cleaned, impact_score, actor) if cleaned else None
        return None

def process_df(df):
    if df.empty: return df
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    headlines = []
    for _, row in df.iterrows():
        h = extract_headline(row.get('NEWS_LINK', ''), row.get('MAIN_ACTOR', ''), row.get('IMPACT_SCORE', None))
        headlines.append(h if h else None)
    df['HEADLINE'] = headlines
    df = df[df['HEADLINE'].notna()]
    df['REGION'] = df['ACTOR_COUNTRY_CODE'].apply(lambda x: get_country(x) or x if x else 'Global')
    try: 
        df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d/%m')
    except: 
        df['DATE_FMT'] = df['DATE']
    df['TONE'] = df['IMPACT_SCORE'].apply(lambda x: "🔴" if x and x < -4 else ("🟡" if x and x < -1 else ("🟢" if x and x > 2 else "⚪")))
    df = df.drop_duplicates(subset=['HEADLINE'])
    return df

@st.cache_data(ttl=300)
def get_metrics(_c, t):
    df = safe_query(_c, f"SELECT COUNT(*) as total, SUM(CASE WHEN DATE >= '{WEEK_AGO}' THEN 1 ELSE 0 END) as recent, SUM(CASE WHEN ABS(IMPACT_SCORE) > 6 AND DATE >= '{WEEK_AGO}' THEN 1 ELSE 0 END) as critical FROM {t}")
    hs = safe_query(_c, f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as c FROM {t} WHERE DATE >= '{WEEK_AGO}' AND ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 1")
    return {'total': df.iloc[0]['total'] if not df.empty else 0, 'recent': df.iloc[0]['recent'] if not df.empty else 0, 'critical': df.iloc[0]['critical'] if not df.empty else 0, 'hotspot': hs.iloc[0]['ACTOR_COUNTRY_CODE'] if not hs.empty else None}

@st.cache_data(ttl=300)
def get_alerts(_c, t):
    d = (NOW - datetime.timedelta(days=3)).strftime('%Y%m%d')
    return safe_query(_c, f"SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t} WHERE DATE >= '{d}' AND IMPACT_SCORE < -4 AND MAIN_ACTOR IS NOT NULL ORDER BY IMPACT_SCORE LIMIT 15")

@st.cache_data(ttl=300)
def get_headlines(_c, t):
    return safe_query(_c, f"SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t} WHERE NEWS_LINK IS NOT NULL AND ARTICLE_COUNT > 5 AND DATE >= '{WEEK_AGO}' ORDER BY DATE DESC, ARTICLE_COUNT DESC LIMIT 60")

@st.cache_data(ttl=300)
def get_trending(_c, t):
    return safe_query(_c, f"SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT FROM {t} WHERE DATE >= '{WEEK_AGO}' AND ARTICLE_COUNT > 3 AND NEWS_LINK IS NOT NULL ORDER BY ARTICLE_COUNT DESC LIMIT 60")

@st.cache_data(ttl=300)
def get_feed(_c, t):
    return safe_query(_c, f"SELECT DATE, NEWS_LINK, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t} WHERE DATE >= '{WEEK_AGO}' AND NEWS_LINK IS NOT NULL ORDER BY DATE DESC LIMIT 60")

@st.cache_data(ttl=300)
def get_countries(_c, t):
    return safe_query(_c, f"SELECT ACTOR_COUNTRY_CODE as country, COUNT(*) as events FROM {t} WHERE DATE >= '{MONTH_AGO}' AND ACTOR_COUNTRY_CODE IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")

@st.cache_data(ttl=300)
def get_timeseries(_c, t):
    return safe_query(_c, f"SELECT DATE, COUNT(*) as events, SUM(CASE WHEN IMPACT_SCORE < -2 THEN 1 ELSE 0 END) as negative, SUM(CASE WHEN IMPACT_SCORE > 2 THEN 1 ELSE 0 END) as positive FROM {t} WHERE DATE >= '{MONTH_AGO}' GROUP BY 1 ORDER BY 1")

@st.cache_data(ttl=300)
def get_sentiment(_c, t):
    return safe_query(_c, f"SELECT AVG(IMPACT_SCORE) as avg, SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as neg, SUM(CASE WHEN IMPACT_SCORE > 3 THEN 1 ELSE 0 END) as pos, COUNT(*) as total FROM {t} WHERE DATE >= '{WEEK_AGO}' AND IMPACT_SCORE IS NOT NULL")

@st.cache_data(ttl=300)
def get_actors(_c, t):
    return safe_query(_c, f"SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, COUNT(*) as events, AVG(IMPACT_SCORE) as avg_impact FROM {t} WHERE DATE >= '{WEEK_AGO}' AND MAIN_ACTOR IS NOT NULL AND LENGTH(MAIN_ACTOR) > 3 GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10")

@st.cache_data(ttl=300)
def get_distribution(_c, t):
    return safe_query(_c, f"SELECT CASE WHEN IMPACT_SCORE < -5 THEN 'Crisis' WHEN IMPACT_SCORE < -2 THEN 'Negative' WHEN IMPACT_SCORE < 2 THEN 'Neutral' WHEN IMPACT_SCORE < 5 THEN 'Positive' ELSE 'Very Positive' END as cat, COUNT(*) as cnt FROM {t} WHERE DATE >= '{WEEK_AGO}' AND IMPACT_SCORE IS NOT NULL GROUP BY 1")

def render_header():
    st.markdown('<div class="header"><div class="logo"><span class="logo-icon">🌐</span><div><div class="logo-title">Global News Intelligence</div><div class="logo-sub">Powered by GDELT • Real-Time Analytics</div></div></div><div class="live-badge"><span class="live-dot"></span> LIVE DATA</div></div>', unsafe_allow_html=True)

def render_metrics(c, t):
    m = get_metrics(c, t)
    c1, c2, c3, c4, c5 = st.columns(5)
    def fmt(n): return f"{int(n or 0):,}"
    with c1: st.metric("📡 TOTAL", fmt(m['total']), "All time")
    with c2: st.metric("⚡ 7 DAYS", fmt(m['recent']), "Recent")
    with c3: st.metric("🚨 CRITICAL", fmt(m['critical']), "High impact")
    hs = m['hotspot']
    with c4: st.metric("🔥 HOTSPOT", (get_country(hs) or hs or "N/A")[:12], hs or "")
    with c5: st.metric("📅 UPDATED", NOW.strftime("%H:%M"), NOW.strftime("%d %b"))

def render_ticker(c, t):
    df = get_alerts(c, t)
    if df.empty:
        txt = "⚡ Monitoring global news │ "
    else:
        items = []
        for _, r in df.iterrows():
            actor = r.get('MAIN_ACTOR', '')[:30] or "Event"
            country = get_country(r.get('ACTOR_COUNTRY_CODE', '')) or 'Global'
            items.append(f"⚠️ {actor} ({country}) • {r.get('IMPACT_SCORE', 0):.1f}")
        txt = " │ ".join(items) + " │ "
    st.markdown(f'<div class="ticker"><div class="ticker-label"><span class="ticker-dot"></span> LIVE</div><div class="ticker-text">{txt + txt}</div></div>', unsafe_allow_html=True)

def render_headlines(c, t):
    df = get_headlines(c, t)
    if df.empty: st.info("📰 Loading..."); return
    df = process_df(df).head(12)
    if df.empty: st.info("📰 No headlines"); return
    st.dataframe(df[['TONE', 'DATE_FMT', 'HEADLINE', 'REGION', 'NEWS_LINK']], hide_index=True, height=350,
        column_config={"TONE": st.column_config.TextColumn("", width="small"), "DATE_FMT": st.column_config.TextColumn("Date", width="small"), "HEADLINE": st.column_config.TextColumn("Headline", width="large"), "REGION": st.column_config.TextColumn("Region", width="small"), "NEWS_LINK": st.column_config.LinkColumn("🔗", width="small")}, width='stretch')

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
        actor = r['MAIN_ACTOR'][:25]
        country = get_country(r.get('ACTOR_COUNTRY_CODE', ''))
        if country:
            labels.append(f"{actor} ({country[:10]})")
        else:
            labels.append(actor)
    colors = ['#ef4444' if x and x < -3 else ('#f59e0b' if x and x < 0 else ('#10b981' if x and x > 3 else '#06b6d4')) for x in df['avg_impact']]
    fig = go.Figure(go.Bar(x=df['events'], y=labels, orientation='h', marker_color=colors, text=df['events'].apply(lambda x: f'{x:,}'), textposition='outside', textfont=dict(color='#94a3b8', size=10)))
    fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=50,t=10,b=0), xaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')), yaxis=dict(showgrid=False, tickfont=dict(color='#e2e8f0', size=11), autorange='reversed'), bargap=0.3)
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch')

def render_distribution(c, t):
    df = get_distribution(c, t)
    if df.empty: st.info("📊 Loading..."); return
    colors = {'Crisis': '#ef4444', 'Negative': '#f59e0b', 'Neutral': '#64748b', 'Positive': '#10b981', 'Very Positive': '#06b6d4'}
    fig = go.Figure(data=[go.Pie(labels=df['cat'], values=df['cnt'], hole=0.6, marker_colors=[colors.get(c, '#64748b') for c in df['cat']], textinfo='percent', textfont=dict(size=11, color='#e2e8f0'))])
    fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=10), showlegend=True, legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center', font=dict(size=10, color='#94a3b8')))
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch')

def render_countries(c, t):
    df = get_countries(c, t)
    if df.empty: st.info("🏆 Loading..."); return
    df = df.head(8)
    df['name'] = df['country'].apply(lambda x: get_country(x) or x or 'Unknown')
    fmt = lambda n: f"{n/1000:.1f}K" if n >= 1000 else str(int(n))
    fig = go.Figure(go.Bar(x=df['name'], y=df['events'], marker_color='#06b6d4', text=df['events'].apply(fmt), textposition='outside', textfont=dict(color='#94a3b8', size=10)))
    fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=9), tickangle=-45), yaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', showticklabels=False), bargap=0.4)
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch')

def render_trending(c, t):
    df = get_trending(c, t)
    if df.empty: st.info("🔥 Loading..."); return
    df = process_df(df).head(15)
    if df.empty: st.info("🔥 No stories"); return
    st.dataframe(df[['DATE_FMT', 'HEADLINE', 'REGION', 'ARTICLE_COUNT', 'NEWS_LINK']], hide_index=True, height=400,
        column_config={"DATE_FMT": st.column_config.TextColumn("Date", width="small"), "HEADLINE": st.column_config.TextColumn("Story", width="large"), "REGION": st.column_config.TextColumn("Region", width="small"), "ARTICLE_COUNT": st.column_config.NumberColumn("📰", width="small"), "NEWS_LINK": st.column_config.LinkColumn("🔗", width="small")}, width='stretch')

def render_feed(c, t):
    df = get_feed(c, t)
    if df.empty: st.info("📋 Loading..."); return
    df = process_df(df).head(15)
    if df.empty: st.info("📋 No events"); return
    st.dataframe(df[['TONE', 'DATE_FMT', 'HEADLINE', 'REGION', 'NEWS_LINK']], hide_index=True, height=400,
        column_config={"TONE": st.column_config.TextColumn("", width="small"), "DATE_FMT": st.column_config.TextColumn("Date", width="small"), "HEADLINE": st.column_config.TextColumn("Event", width="large"), "REGION": st.column_config.TextColumn("Region", width="small"), "NEWS_LINK": st.column_config.LinkColumn("🔗", width="small")}, width='stretch')

def render_timeseries(c, t):
    df = get_timeseries(c, t)
    if df.empty: st.info("📈 Loading..."); return
    df['date'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['date'], y=df['events'], fill='tozeroy', fillcolor='rgba(6,182,212,0.15)', line=dict(color='#06b6d4', width=2), name='Total'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['date'], y=df['negative'], line=dict(color='#ef4444', width=2), name='Negative'), secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df['positive'], line=dict(color='#10b981', width=2), name='Positive'), secondary_y=True)
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=30,b=0), showlegend=True, legend=dict(orientation='h', y=1.02, font=dict(size=11, color='#94a3b8')), xaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')), yaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')), hovermode='x unified')
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch')

def render_ai_chat(c, sql_db):
    if "msgs" not in st.session_state:
        st.session_state.msgs = [{"role": "assistant", "content": "🌐 Ask me about global news!"}]
    
    st.markdown('<div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;margin-bottom:1rem;"><span style="color:#64748b;font-size:0.7rem;">💡 EXAMPLE QUESTIONS:</span> <span style="color:#94a3b8;font-size:0.75rem;">"What major events happened this week?" • "Top 5 countries by event count" • "Show crisis-level events" • "What are the most severe events?"</span></div>', unsafe_allow_html=True)
    
    for msg in st.session_state.msgs[-8:]:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])
    
    prompt = st.chat_input("Ask about global events...", key="chat")
    
    if prompt:
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            qe = get_query_engine(sql_db)
            if not qe:
                st.error("❌ AI not available")
                return
            
            try:
                # ENHANCED PROMPT with explicit examples
                short_prompt = f"""Query: "{prompt}"

Table: events_dagster
Columns: DATE (VARCHAR in YYYYMMDD format like '20241206'), MAIN_ACTOR (VARCHAR), ACTOR_COUNTRY_CODE (VARCHAR 3-letter), IMPACT_SCORE (FLOAT), ARTICLE_COUNT (SMALLINT), NEWS_LINK (VARCHAR)

CRITICAL: DATE is stored as VARCHAR like '20241206', NOT as date type!

MANDATORY FILTERS:
WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND ACTOR_COUNTRY_CODE != ''

For recent events (last 7 days), use: DATE >= '{WEEK_AGO}'

EXAMPLES:
1. "top 5 countries by event count":
SELECT ACTOR_COUNTRY_CODE, COUNT(*) as count FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND DATE >= '{WEEK_AGO}' GROUP BY ACTOR_COUNTRY_CODE ORDER BY count DESC LIMIT 5

2. "show crisis events":
SELECT DATE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND IMPACT_SCORE < -3 AND DATE >= '{WEEK_AGO}' ORDER BY IMPACT_SCORE ASC LIMIT 10

3. "what happened this week":
SELECT DATE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND DATE >= '{WEEK_AGO}' ORDER BY DATE DESC LIMIT 10

DO NOT use date(), strftime(), or any date functions. Just compare DATE >= '{WEEK_AGO}'.

CRISIS = IMPACT_SCORE < -3

ALWAYS include NEWS_LINK. Write complete SQL only."""
                
                with st.spinner("🔍 Querying..."):
                    response = qe.query(short_prompt)
                    answer = str(response)
                    st.markdown(answer)
                    
                    sql = response.metadata.get('sql_query')
                    logger.info(f"Generated SQL: {sql}")  # LOG THE SQL
                    
                    # FALLBACK: If no SQL and user asked for crisis, generate it manually
                    if not sql and ('crisis' in prompt.lower() or 'severe' in prompt.lower()):
                        sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND IMPACT_SCORE < -3 AND DATE >= '{WEEK_AGO}' ORDER BY IMPACT_SCORE ASC LIMIT 10"
                        logger.info(f"Using fallback crisis SQL: {sql}")
                        st.info("🔧 Using built-in crisis query")
                    
                    # FALLBACK: If no SQL and user asked for top countries
                    if not sql and ('top' in prompt.lower() and 'countr' in prompt.lower()):
                        limit = 5  # default
                        import re
                        match = re.search(r'top\s+(\d+)', prompt.lower())
                        if match:
                            limit = int(match.group(1))
                        sql = f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as count FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND DATE >= '{WEEK_AGO}' GROUP BY ACTOR_COUNTRY_CODE ORDER BY count DESC LIMIT {limit}"
                        logger.info(f"Using fallback top countries SQL: {sql}")
                        st.info("🔧 Using built-in top countries query")
                    
                    # FALLBACK: If no SQL and user asked "what happened"
                    if not sql and ('what' in prompt.lower() and ('happen' in prompt.lower() or 'event' in prompt.lower())):
                        sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND DATE >= '{WEEK_AGO}' ORDER BY DATE DESC, ARTICLE_COUNT DESC LIMIT 10"
                        logger.info(f"Using fallback what happened SQL: {sql}")
                        st.info("🔧 Using built-in recent events query")
                    
                    if sql:
                        data = safe_query(c, sql)
                        if not data.empty:
                            data_display = data.copy()
                            data_display.columns = [c.upper() for c in data_display.columns]
                            
                            # Remove EVENT_ID
                            if 'EVENT_ID' in data_display.columns:
                                data_display = data_display.drop(columns=['EVENT_ID'])
                            
                            # Convert country codes - FILTER OUT UNKNOWNS
                            if 'ACTOR_COUNTRY_CODE' in data_display.columns:
                                data_display['COUNTRY'] = data_display['ACTOR_COUNTRY_CODE'].apply(
                                    lambda x: get_country(x) if x and isinstance(x, str) and len(x.strip()) > 0 else None
                                )
                                # Remove rows where country conversion failed
                                data_display = data_display[data_display['COUNTRY'].notna()]
                                data_display = data_display.drop(columns=['ACTOR_COUNTRY_CODE'])
                            
                            # For aggregated queries (like "top countries"), rename count column
                            if 'COUNT(*)' in data_display.columns:
                                data_display = data_display.rename(columns={'COUNT(*)': 'EVENTS'})
                            elif 'COUNT' in data_display.columns:
                                data_display = data_display.rename(columns={'COUNT': 'EVENTS'})
                            
                            # Add severity labels
                            if 'IMPACT_SCORE' in data_display.columns:
                                data_display['SEVERITY'] = data_display['IMPACT_SCORE'].apply(get_impact_label)
                            
                            # Format dates
                            if 'DATE' in data_display.columns:
                                try:
                                    data_display['DATE'] = pd.to_datetime(
                                        data_display['DATE'].astype(str), format='%Y%m%d'
                                    ).dt.strftime('%d/%m')
                                except: pass
                            
                            # AGGRESSIVE NULL/UNKNOWN FILTERING
                            for col in data_display.columns:
                                if data_display[col].dtype == 'object':
                                    data_display = data_display[
                                        (data_display[col].notna()) & 
                                        (data_display[col].astype(str) != 'None') & 
                                        (data_display[col].astype(str) != '') &
                                        (data_display[col].astype(str) != 'Unknown')
                                    ]
                            
                            # Rename links
                            if 'NEWS_LINK' in data_display.columns:
                                data_display = data_display.rename(columns={'NEWS_LINK': '🔗'})
                            
                            # Smart column ordering based on query type
                            if 'EVENTS' in data_display.columns:
                                # Aggregated query (like "top countries")
                                preferred_order = ['COUNTRY', 'EVENTS', 'MAIN_ACTOR']
                            else:
                                # Regular query
                                preferred_order = ['DATE', 'COUNTRY', 'MAIN_ACTOR', 'SEVERITY', 'IMPACT_SCORE', 'ARTICLE_COUNT']
                            
                            link_cols = [c for c in data_display.columns if '🔗' in c]
                            other_cols = [c for c in data_display.columns if c not in preferred_order and c not in link_cols]
                            final_order = [c for c in preferred_order if c in data_display.columns] + other_cols + link_cols
                            data_display = data_display[final_order]
                            
                            # Column config
                            col_config = {
                                "DATE": st.column_config.TextColumn("DATE", width="small"),
                                "COUNTRY": st.column_config.TextColumn("COUNTRY", width="medium"),
                                "EVENTS": st.column_config.NumberColumn("EVENTS", format="%d", width="small"),
                                "MAIN_ACTOR": st.column_config.TextColumn("ACTOR", width="medium"),
                                "SEVERITY": st.column_config.TextColumn("SEVERITY", width="medium"),
                                "IMPACT_SCORE": st.column_config.NumberColumn("SCORE", format="%.1f", width="small"),
                                "ARTICLE_COUNT": st.column_config.NumberColumn("ARTICLES", width="small"),
                            }
                            for col in link_cols:
                                col_config[col] = st.column_config.LinkColumn(col, width="small")
                            
                            # Show results only if we have valid data
                            if not data_display.empty:
                                st.dataframe(data_display.head(10), hide_index=True, width='stretch', column_config=col_config)
                            else:
                                st.info("📭 No valid results after filtering")
                        else:
                            st.warning("📭 No results found")
                        
                        with st.expander("🔍 SQL Query"): 
                            st.code(sql, language='sql')
                    else:
                        st.warning("⚠️ Could not generate SQL query. Try rephrasing your question or use one of the examples above.")
                    
                    st.session_state.msgs.append({"role": "assistant", "content": answer})
                    
            except Exception as e:
                error_msg = str(e)
                if "MAX_TOKENS" in error_msg:
                    st.error("⚠️ Response too long. Try: 'Top 5 countries by event count'")
                else:
                    st.error(f"❌ Error: {error_msg[:100]}")
                logger.error(f"AI error: {e}")

def render_arch():
    st.markdown("""
    <div style="text-align:center;margin-bottom:2rem;">
        <h2 style="font-family:JetBrains Mono;color:#e2e8f0;">🏗️ System Architecture</h2>
        <p style="color:#64748b;">End-to-end data pipeline processing 2M+ daily events</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:2rem;text-align:center;margin-bottom:2rem;">
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">📰 GDELT API</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">⚡ Dagster Orchestration</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">🔧 dbt Transformations</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">🦆 MotherDuck DWH</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">🤖 Gemini AI</span>
        <span style="color:#06b6d4;margin:0 0.5rem;">→</span>
        <span style="background:#1a2332;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem;display:inline-block;margin:0.5rem;">🎨 Streamlit</span>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
            <h4 style="color:#06b6d4;font-size:0.9rem;">📥 DATA INGESTION</h4>
            <p style="color:#94a3b8;font-size:0.85rem;">GDELT Project monitors 100+ languages, 2M+ daily events</p>
            <ul style="color:#94a3b8;font-size:0.85rem;">
                <li>15-minute update intervals</li>
                <li>GitHub Actions scheduler</li>
                <li>Dagster orchestration</li>
                <li>Incremental loads</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;">
            <h4 style="color:#10b981;font-size:0.9rem;">🔧 TRANSFORMATION</h4>
            <ul style="color:#94a3b8;font-size:0.85rem;">
                <li>dbt models for data quality</li>
                <li>Aggregations & metrics</li>
                <li>Country code mapping</li>
                <li>Impact scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
            <h4 style="color:#f59e0b;font-size:0.9rem;">🗄️ DATA WAREHOUSE</h4>
            <p style="color:#94a3b8;font-size:0.85rem;">Migrated from Snowflake → MotherDuck</p>
            <ul style="color:#94a3b8;font-size:0.85rem;">
                <li>DuckDB columnar format</li>
                <li>Sub-second queries</li>
                <li>Serverless architecture</li>
            </ul>
            <div style="margin-top:0.5rem;padding:0.5rem;background:rgba(16,185,129,0.1);border-radius:6px;border-left:3px solid #10b981;">
                <span style="color:#10b981;font-size:0.75rem;">💰 COST: $0/month</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;">
            <h4 style="color:#8b5cf6;font-size:0.9rem;">🤖 AI LAYER</h4>
            <ul style="color:#94a3b8;font-size:0.85rem;">
                <li>Google Gemini 2.5 Flash</li>
                <li>LlamaIndex text-to-SQL</li>
                <li>Natural language queries</li>
                <li>Free tier usage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <h3 style="text-align:center;color:#e2e8f0;margin-bottom:1rem;">🛠️ Tech Stack</h3>
    <div style="text-align:center;padding:1rem;">
        <span class="tech-badge">🐍 Python</span>
        <span class="tech-badge">❄️ Snowflake</span>
        <span class="tech-badge">🦆 DuckDB</span>
        <span class="tech-badge">☁️ MotherDuck</span>
        <span class="tech-badge">⚙️ Dagster</span>
        <span class="tech-badge">🔧 dbt</span>
        <span class="tech-badge">🤖 Gen AI</span>
        <span class="tech-badge">🦙 LlamaIndex</span>
        <span class="tech-badge">✨ Gemini</span>
        <span class="tech-badge">📊 Plotly</span>
        <span class="tech-badge">🎨 Streamlit</span>
        <span class="tech-badge">🔄 GitHub Actions</span>
    </div>
    """, unsafe_allow_html=True)

def render_about():
    st.markdown("""
    <div style="text-align:center;padding:2rem 0;">
        <h2 style="font-family:JetBrains Mono;color:#e2e8f0;">👋 About This Project</h2>
        <p style="color:#94a3b8;max-width:750px;margin:0 auto 1.5rem;font-size:1.1rem;">
            Real-time analytics for <b>GDELT</b> — the world's largest open database monitoring global news in 100+ languages
        </p>
        <p style="color:#64748b;max-width:700px;margin:0 auto 2rem;">
            This project showcases end-to-end data engineering, from ingestion to visualization, with AI-powered natural language querying
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;">
            <h4 style="color:#06b6d4;font-size:0.9rem;">🎯 PROJECT GOALS</h4>
            <ul style="color:#94a3b8;font-size:0.85rem;line-height:1.8;">
                <li>Demonstrate production-ready data pipelines</li>
                <li>Showcase modern data stack (Dagster, dbt, DuckDB)</li>
                <li>Integrate AI/LLM capabilities (Gemini, LlamaIndex)</li>
                <li>Build scalable, cost-effective architecture</li>
                <li>Create intuitive data visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;">
            <h4 style="color:#10b981;font-size:0.9rem;">🛠️ TECHNICAL SKILLS</h4>
            <ul style="color:#94a3b8;font-size:0.85rem;line-height:1.8;">
                <li><b>Languages:</b> Python, SQL</li>
                <li><b>Data Engineering:</b> ETL/ELT, Data Modeling</li>
                <li><b>Orchestration:</b> Dagster, dbt, GitHub Actions</li>
                <li><b>Cloud:</b> Snowflake, MotherDuck (DuckDB)</li>
                <li><b>AI/ML:</b> LLMs, RAG, Text-to-SQL</li>
                <li><b>Visualization:</b> Streamlit, Plotly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:2rem;margin:2rem 0;">
        <h4 style="color:#e2e8f0;text-align:center;margin-bottom:1rem;">📈 PROJECT HIGHLIGHTS</h4>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;">
            <div style="text-align:center;padding:1rem;background:rgba(6,182,212,0.1);border-radius:8px;">
                <div style="font-size:2rem;font-weight:700;color:#06b6d4;">2M+</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Daily Events Processed</div>
            </div>
            <div style="text-align:center;padding:1rem;background:rgba(16,185,129,0.1);border-radius:8px;">
                <div style="font-size:2rem;font-weight:700;color:#10b981;">$0</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Monthly Operating Cost</div>
            </div>
            <div style="text-align:center;padding:1rem;background:rgba(245,158,11,0.1);border-radius:8px;">
                <div style="font-size:2rem;font-weight:700;color:#f59e0b;">&lt;1s</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Average Query Time</div>
            </div>
            <div style="text-align:center;padding:1rem;background:rgba(139,92,246,0.1);border-radius:8px;">
                <div style="font-size:2rem;font-weight:700;color:#8b5cf6;">100+</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Languages Monitored</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align:center;">
        <h4 style="color:#e2e8f0;">📬 CONTACT</h4>
        <p style="color:#94a3b8;margin-bottom:1rem;">Interested in data engineering roles or collaborations</p>
        <div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;">
            <a href="https://github.com/Mohith-akash" target="_blank" style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1.25rem;color:#e2e8f0;text-decoration:none;display:inline-block;">
                ⭐ GitHub
            </a>
            <a href="https://www.linkedin.com/in/mohith-akash/" target="_blank" style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.75rem 1.25rem;color:#e2e8f0;text-decoration:none;display:inline-block;">
                💼 LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            st.markdown("""
            <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.25rem;">
                <h4 style="color:#06b6d4;font-size:0.85rem;">ℹ️ HOW IT WORKS</h4>
                <p style="color:#94a3b8;font-size:0.8rem;">Your question → Gemini AI → SQL query → Results with links</p>
                <hr style="border-color:#1e3a5f;margin:1rem 0;">
                <p style="color:#94a3b8;font-size:0.75rem;">📅 Dates: YYYYMMDD<br>👤 Actors: People/Orgs<br>📊 Impact: -10 to +10<br>🔗 Links: News sources</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[3]:
        render_arch()
    
    with tabs[4]:
        render_about()
    
    st.markdown('<div style="text-align:center;padding:2rem 0 1rem;border-top:1px solid #1e3a5f;margin-top:2rem;"><p style="color:#64748b;font-size:0.8rem;"><b>GDELT</b> monitors worldwide news in real-time • 2M+ daily events</p><p style="color:#475569;font-size:0.75rem;">Built by <a href="https://www.linkedin.com/in/mohith-akash/" style="color:#06b6d4;">Mohith Akash</a> • Portfolio Project</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()