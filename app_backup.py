"""
GDELT News Intelligence Dashboard
Real-time global news analytics powered by AI.
"""

import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from llama_index.llms.cerebras import Cerebras
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine
import datetime
import logging
import re
import duckdb
from urllib.parse import urlparse, unquote

from config import CEREBRAS_MODEL, COUNTRY_ALIASES, REQUIRED_ENVS
from utils import get_country_code, get_dates, get_country, get_impact_label, get_intensity_label, detect_query_type
from styles import inject_css

# Page config
st.set_page_config(
    page_title="Global News Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gdelt")

# =============================================================================
# SECURITY & SECRETS
# =============================================================================

def get_secret(key):
    """Get API keys from environment or Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key)
    except:
        return None

missing = [k for k in REQUIRED_ENVS if not get_secret(k)]
if missing:
    st.error(f"❌ Missing required API keys: {', '.join(missing)}")
    st.stop()

for key in REQUIRED_ENVS:
    val = get_secret(key)
    if val:
        os.environ[key] = val

# =============================================================================
# DATABASE
# =============================================================================

@st.cache_resource
def get_db():
    """Connect to MotherDuck."""
    return duckdb.connect(
        f'md:gdelt_db?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}',
        read_only=True
    )

@st.cache_resource
def get_engine():
    """Create SQLAlchemy engine."""
    return create_engine(
        f"duckdb:///md:gdelt_db?motherduck_token={os.getenv('MOTHERDUCK_TOKEN')}"
    )

@st.cache_data(ttl=3600)
def detect_table(_conn):
    """Find the main events table."""
    try:
        result = _conn.execute("SHOW TABLES").df()
        if not result.empty:
            for name in result.iloc[:, 0].tolist():
                if 'event' in name.lower():
                    return name
            return result.iloc[0, 0]
    except Exception:
        pass
    return 'events_dagster'

def safe_query(conn, sql):
    """Execute SQL safely."""
    try:
        return conn.execute(sql).df()
    except Exception as e:
        logger.error(f"Query error: {e}")
        return pd.DataFrame()

# =============================================================================
# AI SETUP
# =============================================================================

@st.cache_resource
def get_ai_engine(_engine):
    """Set up AI query engine."""
    try:
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            return None
        
        llm = Cerebras(api_key=api_key, model=CEREBRAS_MODEL, temperature=0.1)
        Settings.llm = llm
        
        conn = get_db()
        main_table = detect_table(conn)
        sql_db = SQLDatabase(_engine, include_tables=[main_table], sample_rows_in_table_info=0)
        
        return sql_db
    except Exception as e:
        logger.error(f"AI initialization failed: {e}")
        return None

@st.cache_resource
def get_query_engine(_sql_db):
    """Create AI query engine."""
    if not _sql_db:
        return None
    try:
        tables = list(_sql_db.get_usable_table_names())
        target = next((t for t in tables if 'event' in t.lower()), tables[0] if tables else None)
        if target:
            return NLSQLTableQueryEngine(sql_database=_sql_db, tables=[target])
        return NLSQLTableQueryEngine(sql_database=_sql_db)
    except Exception:
        return None

@st.cache_resource
def get_cerebras_llm():
    """Initialize Cerebras LLM."""
    try:
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            return None
        return Cerebras(api_key=api_key, model=CEREBRAS_MODEL, temperature=0.1)
    except:
        return None

# =============================================================================
# DATA PROCESSING
# =============================================================================

def clean_headline(text):
    """Remove garbage patterns and dates from headlines."""
    if not text: 
        return None
    
    text = str(text).strip()
    
    # Remove leading/trailing punctuation
    text = re.sub(r'^[.,;:\'"!?\-_\s]+', '', text)
    text = re.sub(r'[.,;:\'"!?\-_\s]+$', '', text)
    
    # Reject if too short or single word
    if len(text) < 20 or ' ' not in text:
        return None
    
    words = text.split()
    if len(words) < 4:
        return None
    
    # Reject if it's just a country/city/entity name (all caps single concept)
    if text.isupper() and len(words) <= 3:
        return None
    
    # Reject common garbage patterns
    reject_patterns = [
        r'^[a-f0-9]{8}[-\s][a-f0-9]{4}',
        r'^[a-f0-9\s\-]{20,}$',
        r'^(article|post|item|id)[\s\-_]*[a-f0-9]{8}',
        r'^\d+$',
        r'^[A-Z]{2,5}\s*\d{5,}',
    ]
    
    for pattern in reject_patterns:
        if re.match(pattern, text.lower()): 
            return None
    
    # Remove date patterns at start
    for _ in range(5):
        text = re.sub(r'^\d{4}\s*\d{1,2}\s*\d{1,2}\s+', '', text)
        text = re.sub(r'^\d{8}\s*', '', text)
        text = re.sub(r'^\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\s*', '', text)
        text = re.sub(r'^\d{4}\s+', '', text)
        text = re.sub(r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\s*', '', text)
    
    # Remove garbage patterns anywhere
    text = re.sub(r'\s+\d{1,2}\.\d{5,}', ' ', text)
    text = re.sub(r'\s+\d{4,}', ' ', text)  # Remove 4+ digit numbers anywhere
    text = re.sub(r'\s+[a-z]{1,3}\d[a-z\d]{3,}', ' ', text, flags=re.I)
    text = re.sub(r'\s+[a-z0-9]{10,}(?=\s|$)', ' ', text, flags=re.I)
    text = re.sub(r'\.(html?|php|aspx?|jsp|shtml)$', '', text, flags=re.I)
    text = re.sub(r'[-_]+', ' ', text)
    
    # Remove trailing junk
    text = re.sub(r'\s+\d{1,8}$', '', text)
    text = re.sub(r'\s+[A-Za-z]\d[A-Za-z0-9]{1,5}$', '', text)
    text = re.sub(r'\s+[A-Z]{1,3}\d+$', '', text)
    text = re.sub(r'[\s,]+\d{1,6}$', '', text)
    text = re.sub(r'^[A-Za-z]{1,2}\d+\s+', '', text)
    
    text = ' '.join(text.split())
    
    # Quality checks - must have at least 4 words and 20 chars
    words = text.split()
    if len(text) < 20 or len(words) < 4:
        return None
    
    text_no_spaces = text.replace(' ', '')
    if text_no_spaces:
        num_count = sum(c.isdigit() for c in text_no_spaces)
        if num_count > len(text_no_spaces) * 0.15:
            return None
    
    hex_count = sum(c in '0123456789abcdefABCDEF' for c in text_no_spaces)
    if hex_count > len(text_no_spaces) * 0.3:
        return None
    
    # Reject if last word looks like a code
    last_word = words[-1]
    if re.match(r'^[A-Za-z]{0,2}\d+[A-Za-z]*$', last_word) and len(last_word) < 8:
        words = words[:-1]
        text = ' '.join(words)
        if len(words) < 4:
            return None
    
    # Truncate to 100 chars but don't cut mid-word
    if len(text) > 100:
        text = text[:100].rsplit(' ', 1)[0]
    
    # Remove trailing prepositions and fragments
    trailing_junk = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                     'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be', 'if',
                     'with', 'from', 'into', 'that', 'this', 'than', 'when', 'where',
                     'n', 'b', 'na', 'th', 'wh', 's', 't'}
    
    words = text.split()
    while words and (words[-1].lower() in trailing_junk or len(words[-1]) <= 2):
        words.pop()
        if len(words) < 4:
            return None
    
    text = ' '.join(words)
    
    if len(text) < 20 or len(text.split()) < 4:
        return None
    
    return text

def enhance_headline(text, impact_score=None, actor=None):
    """Proper title case - capitalize first letter of each significant word."""
    if not text: 
        return None
    
    # Remove leading/trailing punctuation first
    text = re.sub(r'^[.,;:\'"!?\-_\s]+', '', text)
    text = re.sub(r'[.,;:\'"!?\-_\s]+$', '', text)
    
    # Remove trailing numbers (like "1396")
    text = re.sub(r'\s+\d{3,}$', '', text)
    
    if not text or len(text) < 20:
        return None
    
    # Must have at least 4 words
    words = text.split()
    if len(words) < 4:
        return None
    
    # Remove trailing prepositions and fragments
    trailing_junk = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                     'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be', 'if',
                     'with', 'from', 'into', 'that', 'this', 'than', 'when', 'where',
                     'n', 'b', 'na', 'th', 'wh', 's', 't'}
    
    while words and (words[-1].lower() in trailing_junk or len(words[-1]) <= 2):
        words.pop()
        if len(words) < 4:
            return None
    
    if len(words) < 4:
        return None
    
    # Convert to title case
    result = []
    small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                   'to', 'by', 'in', 'of', 'up', 'as', 'is', 'it', 'so', 'be'}
    
    for i, word in enumerate(words):
        word_lower = word.lower()
        if i == 0:
            result.append(word_lower.capitalize())
        elif word_lower in small_words:
            result.append(word_lower)
        else:
            result.append(word_lower.capitalize())
    
    return ' '.join(result)

def extract_headline(url, actor=None, impact_score=None):
    """Extract a readable headline from a news article URL."""
    if not url: 
        return None
    
    try:
        parsed = urlparse(str(url))
        path = unquote(parsed.path)
        
        segments = [s for s in path.split('/') if s and len(s) > 15]
        
        for seg in reversed(segments):
            cleaned = clean_headline(seg)
            if cleaned and len(cleaned) > 20 and len(cleaned.split()) >= 4:
                return enhance_headline(cleaned, impact_score, actor)
        
        return None
    except:
        return None

def process_df(df):
    """Process raw database results into display-ready format."""
    if df.empty: 
        return df
    
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    
    # Use database HEADLINE if available, otherwise extract from URL
    headlines = []
    for _, row in df.iterrows():
        headline = None
        
        # First try database headline - must be at least 25 chars and 4 words
        db_headline = row.get('HEADLINE')
        if db_headline and isinstance(db_headline, str) and len(db_headline.strip()) > 25:
            # Skip if it's just a country/city name (all caps, short)
            if not (db_headline.isupper() and len(db_headline.split()) <= 3):
                cleaned = clean_headline(db_headline)
                if cleaned and len(cleaned.split()) >= 4:
                    headline = enhance_headline(cleaned)
        
        # Fall back to URL extraction only
        if not headline:
            headline = extract_headline(
                row.get('NEWS_LINK', ''), 
                None,  # Don't use actor
                row.get('IMPACT_SCORE', None)
            )
            # Validate extracted headline - must be 4+ words
            if headline and len(headline.split()) < 4:
                headline = None
        
        headlines.append(headline)
    
    df['HEADLINE'] = headlines
    df = df[df['HEADLINE'].notna()]
    
    # Convert country codes to full names
    df['REGION'] = df['ACTOR_COUNTRY_CODE'].apply(
        lambda x: get_country(x) or x if x else 'Global'
    )
    
    # Format dates
    try:
        df['DATE_FMT'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d').dt.strftime('%d/%m')
    except:
        df['DATE_FMT'] = df['DATE']
    
    # Add tone indicators
    df['TONE'] = df['IMPACT_SCORE'].apply(
        lambda x: "🔴" if x and x < -4 else ("🟡" if x and x < -1 else ("🟢" if x and x > 2 else "⚪"))
    )
    
    # Add intensity labels
    df['INTENSITY'] = df['IMPACT_SCORE'].apply(get_intensity_label)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['HEADLINE'])
    return df

# =============================================================================
# DATA QUERIES
# =============================================================================

@st.cache_data(ttl=300)
def get_metrics(_c, t):
    dates = get_dates()
    df = safe_query(_c, f"""
        SELECT COUNT(*) as total,
            SUM(CASE WHEN DATE >= '{dates['week_ago']}' THEN 1 ELSE 0 END) as recent,
            SUM(CASE WHEN ABS(IMPACT_SCORE) > 6 AND DATE >= '{dates['week_ago']}' THEN 1 ELSE 0 END) as critical
        FROM {t}
    """)
    hs = safe_query(_c, f"""
        SELECT ACTOR_COUNTRY_CODE, COUNT(*) as c FROM {t} 
        WHERE DATE >= '{dates['week_ago']}' AND ACTOR_COUNTRY_CODE IS NOT NULL 
        GROUP BY 1 ORDER BY 2 DESC LIMIT 1
    """)
    return {
        'total': df.iloc[0]['total'] if not df.empty else 0,
        'recent': df.iloc[0]['recent'] if not df.empty else 0,
        'critical': df.iloc[0]['critical'] if not df.empty else 0,
        'hotspot': hs.iloc[0]['ACTOR_COUNTRY_CODE'] if not hs.empty else None
    }

@st.cache_data(ttl=300)
def get_alerts(_c, t):
    three_days = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime('%Y%m%d')
    return safe_query(_c, f"""
        SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE FROM {t} 
        WHERE DATE >= '{three_days}' AND IMPACT_SCORE < -4 AND MAIN_ACTOR IS NOT NULL 
        ORDER BY IMPACT_SCORE LIMIT 15
    """)

@st.cache_data(ttl=300)
def get_headlines(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT DATE, NEWS_LINK, HEADLINE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE 
        FROM {t} WHERE NEWS_LINK IS NOT NULL AND ARTICLE_COUNT > 5 AND DATE >= '{dates['week_ago']}' AND ACTOR_COUNTRY_CODE IS NOT NULL
        ORDER BY DATE DESC, ARTICLE_COUNT DESC LIMIT 500
    """)

@st.cache_data(ttl=300)
def get_trending(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT DATE, NEWS_LINK, HEADLINE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE, ARTICLE_COUNT 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND ARTICLE_COUNT > 3 AND NEWS_LINK IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL
        ORDER BY ARTICLE_COUNT DESC LIMIT 500
    """)

@st.cache_data(ttl=300)
def get_feed(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT DATE, NEWS_LINK, HEADLINE, MAIN_ACTOR, ACTOR_COUNTRY_CODE, IMPACT_SCORE 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND NEWS_LINK IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL
        ORDER BY DATE DESC LIMIT 500
    """)

@st.cache_data(ttl=300)
def get_countries(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT ACTOR_COUNTRY_CODE as country, COUNT(*) as events FROM {t} 
        WHERE DATE >= '{dates['month_ago']}' AND ACTOR_COUNTRY_CODE IS NOT NULL 
        GROUP BY 1 ORDER BY 2 DESC
    """)

@st.cache_data(ttl=300)
def get_timeseries(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT DATE, COUNT(*) as events, 
            SUM(CASE WHEN IMPACT_SCORE < -2 THEN 1 ELSE 0 END) as negative, 
            SUM(CASE WHEN IMPACT_SCORE > 2 THEN 1 ELSE 0 END) as positive 
        FROM {t} WHERE DATE >= '{dates['month_ago']}' GROUP BY 1 ORDER BY 1
    """)

@st.cache_data(ttl=300)
def get_sentiment(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT AVG(IMPACT_SCORE) as avg, 
            SUM(CASE WHEN IMPACT_SCORE < -3 THEN 1 ELSE 0 END) as neg, 
            SUM(CASE WHEN IMPACT_SCORE > 3 THEN 1 ELSE 0 END) as pos, 
            COUNT(*) as total 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND IMPACT_SCORE IS NOT NULL
    """)

@st.cache_data(ttl=300)
def get_actors(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT MAIN_ACTOR, ACTOR_COUNTRY_CODE, COUNT(*) as events, AVG(IMPACT_SCORE) as avg_impact 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND MAIN_ACTOR IS NOT NULL AND LENGTH(MAIN_ACTOR) > 3 
        GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10
    """)

@st.cache_data(ttl=300)
def get_distribution(_c, t):
    dates = get_dates()
    return safe_query(_c, f"""
        SELECT CASE 
            WHEN IMPACT_SCORE < -5 THEN 'Crisis' 
            WHEN IMPACT_SCORE < -2 THEN 'Negative' 
            WHEN IMPACT_SCORE < 2 THEN 'Neutral' 
            WHEN IMPACT_SCORE < 5 THEN 'Positive' 
            ELSE 'Very Positive' END as cat, COUNT(*) as cnt 
        FROM {t} WHERE DATE >= '{dates['week_ago']}' AND IMPACT_SCORE IS NOT NULL GROUP BY 1
    """)

# =============================================================================
# RENDER FUNCTIONS - Using CSS classes from styles.py
# =============================================================================

def render_header():
    st.markdown('''
        <div class="header">
            <div class="logo">
                <span class="logo-icon">🌐</span>
                <div>
                    <div class="logo-title">Global News Intelligence</div>
                    <div class="logo-sub">Powered by GDELT • Real-Time Analytics</div>
                </div>
            </div>
            <div class="live-badge"><span class="live-dot"></span> LIVE DATA</div>
        </div>
    ''', unsafe_allow_html=True)

def render_metrics(c, t):
    m = get_metrics(c, t)
    c1, c2, c3, c4, c5 = st.columns(5)
    fmt = lambda n: f"{int(n or 0):,}"
    
    with c1:
        st.metric("📡 TOTAL", fmt(m['total']), "All time")
        st.markdown('''
            <div style="text-align:center;margin-top:-0.5rem;">
                <span style="font-size:0.7rem;color:#64748b;">
                    💡 Total global events tracked in database
                </span>
            </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.metric("⚡ 7 DAYS", fmt(m['recent']), "Recent")
        st.markdown('''
            <div style="text-align:center;margin-top:-0.5rem;">
                <span style="font-size:0.7rem;color:#64748b;">
                    💡 Events from the past week
                </span>
            </div>
        ''', unsafe_allow_html=True)
    with c3:
        st.metric("🚨 CRITICAL", fmt(m['critical']), "High impact")
        st.markdown('''
            <div style="text-align:center;margin-top:-0.5rem;">
                <span style="font-size:0.7rem;color:#64748b;">
                    💡 Severe events (impact score > 6) this week
                </span>
            </div>
        ''', unsafe_allow_html=True)
    with c4:
        hs = m['hotspot']
        name = get_country(hs) or hs or "N/A"
        display_name = name if len(name) <= 15 else name[:15] + "..."
        st.metric("🔥 HOTSPOT", display_name, hs or "")
        st.markdown('''
            <div style="text-align:center;margin-top:-0.5rem;">
                <span style="font-size:0.7rem;color:#64748b;">
                    💡 Country with most events this week
                </span>
            </div>
        ''', unsafe_allow_html=True)
    with c5:
        now = datetime.datetime.now()
        st.metric("📅 UPDATED", now.strftime("%H:%M"), now.strftime("%d %b"))
        st.markdown('''
            <div style="text-align:center;margin-top:-0.5rem;">
                <span style="font-size:0.7rem;color:#64748b;">
                    💡 UTC timezone • Refreshes every 5 min
                </span>
            </div>
        ''', unsafe_allow_html=True)

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
    
    st.markdown('''
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;
                    padding:0.5rem;margin-bottom:0.5rem;text-align:center;">
            <span style="font-size:0.7rem;color:#64748b;">
                💡 <b>LIVE TICKER:</b> Shows high-impact events (score < -4) from 
                the past 3 days. Numbers indicate severity level (-10 to +10 scale, 
                where negative = conflict/crisis)
            </span>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown(f'<div class="ticker"><div class="ticker-label"><span class="ticker-dot"></span> LIVE</div><div class="ticker-text">{txt + txt}</div></div>', unsafe_allow_html=True)

def render_sentiment(c, t):
    df = get_sentiment(c, t)
    if df.empty:
        st.info("Loading...")
        return
    
    avg = df.iloc[0]['avg'] or 0
    neg = int(df.iloc[0]['neg'] or 0)
    pos = int(df.iloc[0]['pos'] or 0)
    total = int(df.iloc[0]['total'] or 1)
    
    if avg < -2:
        status, color = ("⚠️ ELEVATED", "#ef4444")
    elif avg < 0:
        status, color = ("🟡 MODERATE", "#f59e0b")
    elif avg < 2:
        status, color = ("🟢 STABLE", "#10b981")
    else:
        status, color = ("✨ POSITIVE", "#06b6d4")
    
    st.markdown(f'''
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:0.75rem;text-align:center;margin-bottom:0.5rem;">
            <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;margin-bottom:0.25rem;">Weekly Sentiment</div>
            <div style="font-size:1.25rem;font-weight:700;color:{color};">{status}</div>
            <div style="font-size:0.7rem;color:#94a3b8;">Avg: <span style="color:{color}">{avg:.2f}</span></div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;">
            <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);border-radius:8px;padding:0.5rem;text-align:center;">
                <div style="font-size:1rem;font-weight:700;color:#ef4444;">{neg:,}</div>
                <div style="font-size:0.6rem;color:#64748b;">Negative</div>
            </div>
            <div style="background:rgba(107,114,128,0.1);border:1px solid rgba(107,114,128,0.2);border-radius:8px;padding:0.5rem;text-align:center;">
                <div style="font-size:1rem;font-weight:700;color:#9ca3af;">{total:,}</div>
                <div style="font-size:0.6rem;color:#64748b;">Total</div>
            </div>
            <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.2);border-radius:8px;padding:0.5rem;text-align:center;">
                <div style="font-size:1rem;font-weight:700;color:#10b981;">{pos:,}</div>
                <div style="font-size:0.6rem;color:#64748b;">Positive</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

def render_actors(c, t):
    df = get_actors(c, t)
    if df.empty:
        st.info("🎯 Loading...")
        return
    
    labels = []
    for _, r in df.iterrows():
        actor = r['MAIN_ACTOR'][:25]
        country = get_country(r.get('ACTOR_COUNTRY_CODE', ''))
        labels.append(f"{actor} ({country[:10]})" if country else actor)
    
    colors = ['#ef4444' if x and x < -3 else ('#f59e0b' if x and x < 0 else ('#10b981' if x and x > 3 else '#06b6d4')) for x in df['avg_impact']]
    
    fig = go.Figure(go.Bar(
        x=df['events'], y=labels, orientation='h', marker_color=colors,
        text=df['events'].apply(lambda x: f'{x:,}'), textposition='outside',
        textfont=dict(color='#94a3b8', size=10)
    ))
    fig.update_layout(
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=50, t=10, b=0),
        xaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')),
        yaxis=dict(showgrid=False, tickfont=dict(color='#e2e8f0', size=11), autorange='reversed'),
        bargap=0.3
    )
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch', key='actors_chart')

def render_distribution(c, t, chart_key='distribution'):
    df = get_distribution(c, t)
    if df.empty:
        st.info("📊 Loading...")
        return
    
    colors = {'Crisis': '#ef4444', 'Negative': '#f59e0b', 'Neutral': '#64748b', 'Positive': '#10b981', 'Very Positive': '#06b6d4'}
    fig = go.Figure(data=[go.Pie(
        labels=df['cat'], values=df['cnt'], hole=0.6,
        marker_colors=[colors.get(c, '#64748b') for c in df['cat']],
        textinfo='percent', textfont=dict(size=11, color='#e2e8f0')
    )])
    fig.update_layout(
        height=200, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True, legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center', font=dict(size=10, color='#94a3b8'))
    )
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch', key=chart_key)

def render_countries(c, t):
    df = get_countries(c, t)
    if df.empty:
        st.info("🏆 Loading...")
        return
    
    df = df.head(8)
    df['name'] = df['country'].apply(lambda x: get_country(x) or x or 'Unknown')
    fmt = lambda n: f"{n/1000:.1f}K" if n >= 1000 else str(int(n))
    
    fig = go.Figure(go.Bar(
        x=df['name'], y=df['events'], marker_color='#06b6d4',
        text=df['events'].apply(fmt), textposition='outside', textfont=dict(color='#94a3b8', size=10)
    ))
    fig.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=10), tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', showticklabels=False),
        bargap=0.35
    )
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch', key='countries_chart')

def render_trending(c, t):
    df = get_trending(c, t)
    if df.empty:
        st.info("🔥 Loading...")
        return
    df = process_df(df).head(20)
    if df.empty:
        st.info("🔥 No stories")
        return
    
    st.dataframe(
        df[['DATE_FMT', 'INTENSITY', 'HEADLINE', 'REGION', 'ARTICLE_COUNT', 'NEWS_LINK']],
        hide_index=True, height=400, width='stretch',
        column_config={
            "DATE_FMT": st.column_config.TextColumn("Date", width=60),
            "INTENSITY": st.column_config.TextColumn("Intensity", width=140),
            "HEADLINE": st.column_config.TextColumn("Story", width=None),
            "REGION": st.column_config.TextColumn("Region", width=100),
            "ARTICLE_COUNT": st.column_config.NumberColumn("📰", width=50),
            "NEWS_LINK": st.column_config.LinkColumn("🔗", width=40)
        }
    )

def render_feed(c, t):
    df = get_feed(c, t)
    if df.empty:
        st.info("📋 Loading...")
        return
    df = process_df(df).head(50)
    if df.empty:
        st.info("📋 No events")
        return
    
    st.dataframe(
        df[['DATE_FMT', 'INTENSITY', 'HEADLINE', 'REGION', 'NEWS_LINK']],
        hide_index=True, height=600, width='stretch',
        column_config={
            "DATE_FMT": st.column_config.TextColumn("Date", width=60),
            "INTENSITY": st.column_config.TextColumn("Intensity Level", width=140),
            "HEADLINE": st.column_config.TextColumn("Event", width=None),
            "REGION": st.column_config.TextColumn("Region", width=100),
            "NEWS_LINK": st.column_config.LinkColumn("🔗", width=40)
        }
    )

def render_timeseries(c, t):
    df = get_timeseries(c, t)
    if df.empty:
        st.info("📈 Loading...")
        return
    
    df['date'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['events'], fill='tozeroy', fillcolor='rgba(6,182,212,0.15)', line=dict(color='#06b6d4', width=2), name='Total'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['date'], y=df['negative'], line=dict(color='#ef4444', width=2), name='Negative'), secondary_y=True)
    fig.add_trace(go.Scatter(x=df['date'], y=df['positive'], line=dict(color='#10b981', width=2), name='Positive'), secondary_y=True)
    
    fig.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0), showlegend=True,
        legend=dict(orientation='h', y=1.02, font=dict(size=11, color='#94a3b8')),
        xaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')),
        yaxis=dict(showgrid=True, gridcolor='rgba(30,58,95,0.3)', tickfont=dict(color='#64748b')),
        hovermode='x unified'
    )
    st.plotly_chart(fig, config={'displayModeBar': False}, width='stretch', key='timeseries_chart')

def render_ai_chat(c, sql_db):
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    if st.session_state.qa_history:
        past = st.session_state.qa_history[-5:]
        with st.expander("🕒 Previous Conversations", expanded=False):
            idx = st.selectbox("Select", range(len(past)), format_func=lambda i: (past[i]["question"][:70] + "…") if len(past[i]["question"]) > 70 else past[i]["question"], key="prev_select")
            sel = past[idx]
            st.markdown(f'''<div class="prev-convo-card">
                <div class="prev-convo-label">💬 Previous Conversation</div>
                <div class="prev-convo-q"><b>Q:</b></div><div class="prev-convo-text">{sel['question']}</div>
                <div class="prev-convo-q" style="margin-top:0.5rem;"><b>A:</b></div><div class="prev-convo-text">{sel['answer']}</div>
            </div>''', unsafe_allow_html=True)
            if sel.get("sql"):
                with st.expander("🔍 SQL Query"):
                    st.code(sel["sql"], language="sql")

    st.markdown('''<div class="ai-info-card">
        <div class="ai-example-label">💡 EXAMPLE QUESTIONS:</div>
        <div class="ai-examples">• "What major events happened this week?"<br>• "Top 5 countries by event count"<br>• "Show crisis-level events"</div>
    </div>''', unsafe_allow_html=True)

    prompt = st.chat_input("Ask about global events...", key="chat")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            qe = get_query_engine(sql_db) if sql_db else None
            llm = get_cerebras_llm()
            if not llm:
                st.error("❌ Cerebras AI not available")
                return
            try:
                dates = get_dates()
                qi = detect_query_type(prompt)
                
                if qi['is_specific_date'] and qi['specific_date']:
                    date_filter = f"DATE = '{qi['specific_date']}'"
                elif qi['time_period'] == 'all' or qi['is_aggregate']:
                    date_filter = f"DATE >= '{dates['three_months_ago']}'"
                elif qi['time_period'] == 'month':
                    date_filter = f"DATE >= '{dates['month_ago']}'"
                elif qi['time_period'] == 'day':
                    date_filter = f"DATE = '{dates['today']}'"
                else:
                    date_filter = f"DATE >= '{dates['week_ago']}'"

                sql = None
                answer = ""
                is_country_aggregate = False
                is_count_aggregate = False
                country_filter_name = None
                with st.spinner("🔍 Querying..."):
                    # Determine display limit from query (default 5, max 10)
                    limit = 5
                    m = re.search(r'(\d+)\s*(events?|results?|items?)', prompt.lower())
                    if m: 
                        limit = min(int(m.group(1)), 10)
                    m2 = re.search(r'top\s+(\d+)', prompt.lower())
                    if m2:
                        limit = min(int(m2.group(1)), 10)
                    
                    # Fetch more rows to account for filtering and deduplication
                    fetch_limit = limit * 20  # Fetch 20x more since many headlines will be filtered or duplicated
                    
                    # Helper: detect country codes in prompt
                    def get_country_codes_from_prompt(text):
                        codes = []
                        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
                        
                        # Check for multi-word phrases first
                        multi_word_regions = [
                            'middle east', 'united states', 'united kingdom', 'great britain',
                            'south korea', 'north korea', 'saudi arabia', 'south africa',
                            'new zealand'
                        ]
                        for phrase in multi_word_regions:
                            if phrase in clean_text:
                                code = get_country_code(phrase)
                                if code and code not in codes:
                                    codes.append(code)
                        
                        # Then check individual words
                        for w in clean_text.split():
                            if len(w) >= 2:
                                code = get_country_code(w)
                                if code and code not in codes: 
                                    codes.append(code)
                        return codes
                    
                    prompt_lower = prompt.lower()
                    has_crisis = 'crisis' in prompt_lower or 'severe' in prompt_lower
                    has_country_word = 'countr' in prompt_lower
                    has_major = 'major' in prompt_lower or 'important' in prompt_lower or 'significant' in prompt_lower or 'biggest' in prompt_lower or 'trending' in prompt_lower
                    
                    # Check for specific query types (ORDER MATTERS!)
                    
                    # 1. COUNTRIES WITH CRISIS - must come before plain crisis
                    if has_crisis and has_country_word:
                        is_country_aggregate = True
                        sql = f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as EVENT_COUNT FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND IMPACT_SCORE < -3 AND {date_filter} GROUP BY ACTOR_COUNTRY_CODE ORDER BY EVENT_COUNT DESC LIMIT {limit}"
                    
                    # 2. Plain crisis events
                    elif has_crisis:
                        sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND IMPACT_SCORE < -3 AND {date_filter} ORDER BY IMPACT_SCORE ASC LIMIT {fetch_limit}"
                    
                    # 3. MAJOR/IMPORTANT events - high article count (trending stories)
                    elif has_major:
                        sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND ARTICLE_COUNT > 50 AND {date_filter} ORDER BY ARTICLE_COUNT DESC LIMIT {fetch_limit}"
                    
                    # 4. TOP COUNTRIES - check this BEFORE is_aggregate
                    elif 'top' in prompt_lower and has_country_word:
                        is_country_aggregate = True
                        sql = f"SELECT ACTOR_COUNTRY_CODE, COUNT(*) as EVENT_COUNT FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {date_filter} GROUP BY ACTOR_COUNTRY_CODE ORDER BY EVENT_COUNT DESC LIMIT {limit}"
                    
                    # 5. Aggregate queries (how many, count, total) - now with country support
                    elif qi['is_aggregate']:
                        is_count_aggregate = True
                        codes = get_country_codes_from_prompt(prompt)
                        if codes:
                            cf = f"ACTOR_COUNTRY_CODE = '{codes[0]}'"
                            country_filter_name = get_country(codes[0]) or codes[0]
                            sql = f"SELECT COUNT(*) as TOTAL_EVENTS FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {cf} AND {date_filter}"
                        else:
                            sql = f"SELECT COUNT(*) as TOTAL_EVENTS FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {date_filter}"
                    
                    # 6. Default: specific events query - prioritize high article count
                    else:
                        if qe:
                            try:
                                resp = qe.query(prompt)
                                sql = resp.metadata.get('sql_query')
                            except Exception: pass
                        if not sql:
                            codes = get_country_codes_from_prompt(prompt)
                            if codes:
                                if len(codes) == 1:
                                    cf = f"ACTOR_COUNTRY_CODE = '{codes[0]}'"
                                else:
                                    codes_str = "', '".join(codes)
                                    cf = f"ACTOR_COUNTRY_CODE IN ('{codes_str}')"
                                sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND {cf} AND {date_filter} ORDER BY ARTICLE_COUNT DESC, DATE DESC LIMIT {fetch_limit}"
                            else:
                                # Default: get high article count events (most covered stories)
                                sql = f"SELECT DATE, ACTOR_COUNTRY_CODE, HEADLINE, MAIN_ACTOR, IMPACT_SCORE, ARTICLE_COUNT, NEWS_LINK FROM events_dagster WHERE MAIN_ACTOR IS NOT NULL AND ACTOR_COUNTRY_CODE IS NOT NULL AND ARTICLE_COUNT > 20 AND {date_filter} ORDER BY ARTICLE_COUNT DESC LIMIT {fetch_limit}"
                    
                    # Enforce LIMIT on aggregate queries only (event queries need more rows for filtering)
                    if sql and (is_count_aggregate or is_country_aggregate):
                        if 'LIMIT' not in sql.upper():
                            sql = sql.rstrip(';') + f' LIMIT {limit}'
                    
                    if sql:
                        data = safe_query(c, sql)
                        if not data.empty:
                            dd = data.copy()
                            dd.columns = [col.upper() for col in dd.columns]
                            
                            # Handle COUNT(*) aggregate queries
                            if is_count_aggregate:
                                total = dd.iloc[0]['TOTAL_EVENTS']
                                location = f"in {country_filter_name}" if country_filter_name else "globally"
                                
                                ai_prompt = f"""Database query result: {total:,} events recorded {location} during {qi['period_label']}.

Question: {prompt}

Provide a brief, factual answer using ONLY this data. State the count clearly."""

                                answer = str(llm.complete(ai_prompt))
                                st.markdown(answer)
                                
                                st.markdown(f"""
                                <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1.5rem;text-align:center;margin:1rem 0;">
                                    <div style="font-size:0.8rem;color:#64748b;text-transform:uppercase;">Total Events {f"in {country_filter_name}" if country_filter_name else ""}</div>
                                    <div style="font-size:2.5rem;font-weight:700;color:#06b6d4;">{total:,}</div>
                                    <div style="font-size:0.75rem;color:#94a3b8;">{qi['period_label']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                with st.expander("🔍 SQL"):
                                    st.code(sql, language='sql')
                            
                            # Handle country aggregate query differently
                            elif is_country_aggregate:
                                # Convert country codes to names
                                dd['COUNTRY'] = dd['ACTOR_COUNTRY_CODE'].apply(lambda x: get_country(x) or x)
                                
                                # Build summary for AI
                                country_list = []
                                for _, row in dd.iterrows():
                                    country_list.append(f"- {row['COUNTRY']}: {row['EVENT_COUNT']:,} events")
                                summary_text = "\n".join(country_list)
                                
                                ai_prompt = f"""Top {len(dd)} countries by events ({qi['period_label']}):

{summary_text}

Question: {prompt}

Briefly explain why these countries lead and any notable patterns. Keep response concise."""

                                answer = str(llm.complete(ai_prompt))
                                st.markdown(answer)
                                
                                st.dataframe(
                                    dd[['COUNTRY', 'EVENT_COUNT']],
                                    hide_index=True,
                                    width='stretch',
                                    column_config={
                                        "COUNTRY": st.column_config.TextColumn("Country", width=200),
                                        "EVENT_COUNT": st.column_config.NumberColumn("Event Count", width=120)
                                    }
                                )
                                
                                with st.expander("🔍 SQL"):
                                    st.code(sql, language='sql')
                            else:
                                # Regular event query - extract headlines, show details
                                # Convert country code to full name
                                if 'ACTOR_COUNTRY_CODE' in dd.columns:
                                    dd['COUNTRY'] = dd['ACTOR_COUNTRY_CODE'].apply(lambda x: get_country(x) or x)
                                
                                if 'NEWS_LINK' in dd.columns:
                                    headlines = []
                                    for _, row in dd.iterrows():
                                        headline = None
                                        
                                        # First try database HEADLINE
                                        db_headline = row.get('HEADLINE')
                                        if db_headline and isinstance(db_headline, str) and len(db_headline.strip()) > 25:
                                            if not (db_headline.isupper() and len(db_headline.split()) <= 3):
                                                cleaned = clean_headline(db_headline)
                                                if cleaned and len(cleaned.split()) >= 4:
                                                    headline = enhance_headline(cleaned)
                                        
                                        # Fall back to URL extraction
                                        if not headline:
                                            headline = extract_headline(
                                                row.get('NEWS_LINK', ''),
                                                None,
                                                row.get('IMPACT_SCORE', None)
                                            )
                                            if headline and len(headline.split()) < 4:
                                                headline = None
                                        
                                        headlines.append(headline)
                                    dd['HEADLINE'] = headlines
                                    
                                    # Filter out rows with no valid headline
                                    dd = dd[dd['HEADLINE'].notna()]
                                    
                                    # Deduplicate by headline to avoid showing same story multiple times
                                    dd = dd.drop_duplicates(subset=['HEADLINE'])
                                
                                # Add severity label
                                if 'IMPACT_SCORE' in dd.columns:
                                    dd['SEVERITY'] = dd['IMPACT_SCORE'].apply(get_impact_label)
                                
                                # Format date
                                if 'DATE' in dd.columns:
                                    try: 
                                        dd['DATE'] = pd.to_datetime(dd['DATE'].astype(str), format='%Y%m%d').dt.strftime('%b %d')
                                    except Exception: pass
                                
                                # Check if we have any valid data after filtering
                                if dd.empty:
                                    st.warning("📭 No events with proper headlines found for this query")
                                    answer = "No events with valid headlines were found."
                                else:
                                    # Prepare data for AI summary (include headlines)
                                    summary_data = []
                                    for _, row in dd.head(limit).iterrows():
                                        headline = row.get('HEADLINE', 'Event')
                                        country = row.get('COUNTRY', row.get('ACTOR_COUNTRY_CODE', 'Global'))
                                        date = row.get('DATE', '')
                                        severity = row.get('SEVERITY', '')
                                        score = row.get('IMPACT_SCORE', 0)
                                        summary_data.append(f"- {headline} | {country} | {date} | Severity: {severity} ({score})")
                                    
                                    summary_text = "\n".join(summary_data)
                                    
                                    ai_prompt = f"""Events from {qi['period_label']}:

{summary_text}

Question: {prompt}

Give 2-3 sentences about each event - what happened, who's involved, why it matters."""

                                    answer = str(llm.complete(ai_prompt))
                                    st.markdown(answer)
                                    
                                    display_cols = ['DATE', 'HEADLINE', 'COUNTRY', 'SEVERITY']
                                    if 'NEWS_LINK' in dd.columns:
                                        display_cols.append('NEWS_LINK')
                                    
                                    display_cols = [col for col in display_cols if col in dd.columns]
                                    
                                    st.dataframe(
                                        dd[display_cols].head(limit),
                                        hide_index=True,
                                        width='stretch',
                                        column_config={
                                            "DATE": st.column_config.TextColumn("Date", width=70),
                                            "HEADLINE": st.column_config.TextColumn("Event", width=None),
                                            "COUNTRY": st.column_config.TextColumn("Country", width=100),
                                            "SEVERITY": st.column_config.TextColumn("Severity", width=120),
                                            "NEWS_LINK": st.column_config.LinkColumn("🔗", width=40)
                                        }
                                    )
                                
                                with st.expander("🔍 SQL"):
                                    st.code(sql, language='sql')
                        else:
                            st.warning("📭 No results found for this query")
                            answer = "No results found."
                    else:
                        st.warning("⚠️ Could not generate query")
                        answer = "Could not process query."
                
                st.session_state.qa_history.append({"question": prompt, "answer": answer, "sql": sql})
                # Limit history to 50 messages to prevent unbounded memory growth
                if len(st.session_state.qa_history) > 50:
                    st.session_state.qa_history = st.session_state.qa_history[-50:]
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:100]}")

def render_about():
    """About page with architecture, tool comparison, and evolution."""
    
    # TITLE
    st.markdown("""
    <div style="text-align:center;padding:0.75rem 0;">
        <h2 style="font-family:JetBrains Mono;color:#e2e8f0;font-size:1.5rem;margin:0;">🏗️ About This Project</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # ARCHITECTURE - Full width edge to edge
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:1rem 0;margin-bottom:0.5rem;">
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin-right:0.5rem;">
            <div style="font-size:1.75rem;">📰</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">GDELT API</div>
            <div style="color:#64748b;font-size:0.7rem;">100K+ events/day</div>
        </div>
        <span style="color:#06b6d4;font-size:1.75rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin:0 0.5rem;">
            <div style="font-size:1.75rem;">⚡</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">Dagster</div>
            <div style="color:#64748b;font-size:0.7rem;">Orchestration</div>
        </div>
        <span style="color:#06b6d4;font-size:1.75rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin:0 0.5rem;">
            <div style="font-size:1.75rem;">🦆</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">MotherDuck</div>
            <div style="color:#64748b;font-size:0.7rem;">Cloud DuckDB</div>
        </div>
        <span style="color:#06b6d4;font-size:1.75rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin:0 0.5rem;">
            <div style="font-size:1.75rem;">🦙</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">LlamaIndex</div>
            <div style="color:#64748b;font-size:0.7rem;">Text-to-SQL</div>
        </div>
        <span style="color:#06b6d4;font-size:1.75rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin:0 0.5rem;">
            <div style="font-size:1.75rem;">🧠</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">Cerebras</div>
            <div style="color:#64748b;font-size:0.7rem;">Llama 3.1 8B</div>
        </div>
        <span style="color:#06b6d4;font-size:1.75rem;font-weight:bold;">→</span>
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.5rem;text-align:center;flex:1;margin-left:0.5rem;">
            <div style="font-size:1.75rem;">🎨</div>
            <div style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">Streamlit</div>
            <div style="color:#64748b;font-size:0.7rem;">Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ENTERPRISE vs MY STACK - Full width, bigger font
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:1rem;">
        <h4 style="color:#e2e8f0;text-align:center;margin-bottom:0.75rem;font-size:1.1rem;">💰 ENTERPRISE TOOLS vs MY STACK</h4>
        <table style="width:100%;border-collapse:collapse;font-size:0.95rem;">
            <tr style="border-bottom:1px solid #1e3a5f;">
                <th style="text-align:left;padding:0.5rem;color:#f59e0b;width:28%;">Enterprise Tool</th>
                <th style="text-align:left;padding:0.5rem;color:#10b981;width:18%;">My Stack</th>
                <th style="text-align:left;padding:0.5rem;color:#64748b;">How I Replaced It</th>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>Spark/PySpark</b> <span style="color:#ef4444;font-size:0.75rem;">~$500/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>DuckDB</b></td>
                <td style="padding:0.5rem;color:#64748b;">Columnar OLAP for 100K+ events — no cluster needed, runs in-process</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>Snowflake/Hadoop</b> <span style="color:#ef4444;font-size:0.75rem;">~$300/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>MotherDuck</b></td>
                <td style="padding:0.5rem;color:#64748b;">Serverless cloud DWH, same SQL syntax, free tier handles my scale</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>Managed Airflow</b> <span style="color:#ef4444;font-size:0.75rem;">~$300/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>Dagster</b></td>
                <td style="padding:0.5rem;color:#64748b;">Asset-based DAGs with lineage tracking — modern orchestration UI</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>dbt Cloud</b> <span style="color:#ef4444;font-size:0.75rem;">~$100/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>SQL in Python</b></td>
                <td style="padding:0.5rem;color:#64748b;">Data transformations via raw SQL in pipeline.py — same result, no cost</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>AWS Lambda/CI</b> <span style="color:#ef4444;font-size:0.75rem;">~$100/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>GitHub Actions</b></td>
                <td style="padding:0.5rem;color:#64748b;">Scheduled ETL runs every 30 min — free CI/CD with cron triggers</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f22;">
                <td style="padding:0.5rem;color:#94a3b8;"><b>OpenAI GPT-4</b> <span style="color:#ef4444;font-size:0.75rem;">~$50/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>Cerebras</b></td>
                <td style="padding:0.5rem;color:#64748b;">Llama 3.1 8B via free tier — fastest LLM inference for Text-to-SQL</td>
            </tr>
            <tr>
                <td style="padding:0.5rem;color:#94a3b8;"><b>Tableau/Power BI</b> <span style="color:#ef4444;font-size:0.75rem;">~$70/mo</span></td>
                <td style="padding:0.5rem;color:#e2e8f0;"><b>Streamlit</b></td>
                <td style="padding:0.5rem;color:#64748b;">Python-native dashboards with Plotly — free Streamlit Cloud hosting</td>
            </tr>
        </table>
        <div style="display:flex;justify-content:space-around;margin-top:1rem;padding-top:1rem;border-top:1px solid #1e3a5f;">
            <div style="text-align:center;">
                <div style="color:#ef4444;font-size:1.5rem;font-weight:700;">$1,420+</div>
                <div style="color:#64748b;font-size:0.8rem;">Enterprise monthly</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#10b981;font-size:1.75rem;font-weight:700;">$0</div>
                <div style="color:#64748b;font-size:0.8rem;">My monthly cost</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # TWO COLUMNS - Evolution (left) + Tech Stack with Metrics (right)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # EVOLUTION - Half width
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:1rem;margin-top:0.75rem;">
            <h4 style="color:#e2e8f0;text-align:center;margin-bottom:0.75rem;font-size:0.95rem;">🔄 TECHNOLOGY EVOLUTION</h4>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;margin-bottom:0.5rem;">
                <div><span style="color:#06b6d4;font-size:0.7rem;">DATA WAREHOUSE</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">❄️ Snowflake → 🦆 <b>MotherDuck</b></span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">Migrated for $0 cost, same SQL syntax</div>
            </div>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;margin-bottom:0.5rem;">
                <div><span style="color:#8b5cf6;font-size:0.7rem;">AI / LLM (RAG)</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">✨ Gemini → ⚡ Groq → 🧠 <b>Cerebras</b></span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">LlamaIndex RAG + Text-to-SQL pipeline</div>
            </div>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;margin-bottom:0.5rem;">
                <div><span style="color:#f59e0b;font-size:0.7rem;">MODELS</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">Llama 70B → <b>Llama 3.1 8B</b></span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">Smaller model, faster, good enough for task</div>
            </div>
            <div style="background:#1a2332;border-radius:6px;padding:0.6rem;">
                <div><span style="color:#10b981;font-size:0.7rem;">ETL PIPELINE (CI/CD)</span> <span style="color:#e2e8f0;font-size:0.9rem;margin-left:0.5rem;">Manual → <b>GitHub Actions 30min</b></span></div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem;">Dagster orchestration, fully automated</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # TECH STACK + KEY METRICS combined
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e3a5f;border-radius:10px;padding:1rem;margin-top:0.75rem;">
            <h4 style="color:#e2e8f0;text-align:center;margin-bottom:0.75rem;font-size:0.95rem;">🛠️ TECH STACK</h4>
            <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.5rem;">
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🐍 Python</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">📝 SQL</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🐼 Pandas</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🦆 DuckDB</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">☁️ MotherDuck</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">⚙️ Dagster</span>
            </div>
            <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.75rem;">
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🦙 LlamaIndex</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">⚡ Cerebras</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">📊 Plotly</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🎨 Streamlit</span>
                <span style="background:#1e3a5f;border-radius:6px;padding:0.5rem 0.75rem;color:#e2e8f0;font-size:0.85rem;">🔄 GitHub Actions</span>
            </div>
            <div style="display:flex;justify-content:space-around;padding-top:0.75rem;border-top:1px solid #1e3a5f;">
                <div style="text-align:center;">
                    <div style="font-size:1.25rem;font-weight:700;color:#06b6d4;">100K+</div>
                    <div style="font-size:0.65rem;color:#64748b;">Events/day</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.25rem;font-weight:700;color:#10b981;">$0</div>
                    <div style="font-size:0.65rem;color:#64748b;">Cost</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.25rem;font-weight:700;color:#f59e0b;">&lt;1s</div>
                    <div style="font-size:0.65rem;color:#64748b;">Query</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.25rem;font-weight:700;color:#8b5cf6;">100+</div>
                    <div style="font-size:0.65rem;color:#64748b;">Languages</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # CONTACT
    st.markdown("""
    <div style="text-align:center;margin-top:1.25rem;padding-top:1rem;border-top:1px solid #1e3a5f;">
        <span style="color:#94a3b8;font-size:0.9rem;">📬 Open to opportunities</span>
        <a href="https://github.com/Mohith-akash" target="_blank" style="margin-left:1rem;background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.5rem 1rem;color:#e2e8f0;text-decoration:none;">⭐ GitHub</a>
        <a href="https://www.linkedin.com/in/mohith-akash/" target="_blank" style="margin-left:0.5rem;background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:0.5rem 1rem;color:#e2e8f0;text-decoration:none;">💼 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)



# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    inject_css()
    conn = get_db()
    tbl = detect_table(conn)
    
    try:
        sql_db = get_ai_engine(get_engine())
    except:
        sql_db = None
    
    render_header()
    tabs = st.tabs(["📊 HOME", "📈 TRENDS", "🤖 AI", "👤 ABOUT"])
    
    with tabs[0]:
        render_metrics(conn, tbl)
        render_ticker(conn, tbl)
        st.markdown("---")
        c1, c2 = st.columns([6, 4])
        with c1:
            st.markdown('<div class="card-hdr"><span>🔥</span><span class="card-title">Trending News</span><span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem;">(This Week)</span></div>', unsafe_allow_html=True)
            render_trending(conn, tbl)
        with c2:
            st.markdown('<div class="card-hdr"><span>⚡</span><span class="card-title">Weekly Sentiment</span><span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem;">(7 Days)</span></div>', unsafe_allow_html=True)
            render_sentiment(conn, tbl)
            st.markdown('<div class="card-hdr" style="margin-top:1rem;"><span>📊</span><span class="card-title">Tone Breakdown</span><span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem;">(This Week)</span></div>', unsafe_allow_html=True)
            render_distribution(conn, tbl, 'home_dist')
        st.markdown("---")
        c1, c2 = st.columns([6, 4])
        with c1:
            st.markdown('<div class="card-hdr"><span>🎯</span><span class="card-title">Most Mentioned</span><span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem;">(This Week)</span></div>', unsafe_allow_html=True)
            render_actors(conn, tbl)
        with c2:
            st.markdown('<div class="card-hdr"><span>🏆</span><span class="card-title">Top Countries</span><span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem;">(30 Days)</span></div>', unsafe_allow_html=True)
            render_countries(conn, tbl)
    
    with tabs[1]:
        c1, c2 = st.columns([7, 3])
        with c1:
            st.markdown('<div class="card-hdr"><span>📋</span><span class="card-title">Recent Events Feed</span><span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem;">(This Week)</span></div>', unsafe_allow_html=True)
            render_feed(conn, tbl)
        with c2:
            st.markdown('<div class="card-hdr"><span>📊</span><span class="card-title">Intensity Guide</span></div>', unsafe_allow_html=True)
            st.markdown('''<div class="card">
                <h4 class="title-cyan">🎯 EVENT INTENSITY LEVELS</h4>
                <div class="intensity-item intensity-red"><div class="intensity-title">⚔️ Armed Conflict</div><div class="intensity-score">Score: -10 to -8</div></div>
                <div class="intensity-item intensity-red"><div class="intensity-title">🔴 Major Crisis</div><div class="intensity-score">Score: -7 to -6</div></div>
                <div class="intensity-item intensity-amber"><div class="intensity-title">🟠 Serious Tension</div><div class="intensity-score">Score: -5 to -4</div></div>
                <div class="intensity-item intensity-yellow"><div class="intensity-title">🟡 Verbal Dispute</div><div class="intensity-score">Score: -3 to -2</div></div>
                <div class="intensity-item intensity-gray"><div class="intensity-title">⚪ Neutral Event</div><div class="intensity-score">Score: -2 to 2</div></div>
                <div class="intensity-item intensity-green"><div class="intensity-title">🟢 Diplomatic Talk</div><div class="intensity-score">Score: 2 to 4</div></div>
                <div class="intensity-item intensity-green"><div class="intensity-title">🤝 Partnership</div><div class="intensity-score">Score: 4 to 6</div></div>
                <div class="intensity-item intensity-cyan"><div class="intensity-title">✨ Peace Agreement</div><div class="intensity-score">Score: 6+</div></div>
            </div>''', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="card-hdr"><span>📈</span><span class="card-title">30-Day Trend Analysis</span></div>', unsafe_allow_html=True)
        render_timeseries(conn, tbl)
    
    with tabs[2]:
        c1, c2 = st.columns([7, 3])
        with c1:
            st.markdown('<div class="card-hdr"><span>🤖</span><span class="card-title">Ask in Plain English</span></div>', unsafe_allow_html=True)
            render_ai_chat(conn, sql_db)
        with c2:
            st.markdown('''<div class="card">
                <h4 class="title-cyan">ℹ️ HOW IT WORKS</h4>
                <p class="text-muted">Your question → Cerebras AI → SQL query → Results</p>
                <hr style="border-color:#1e3a5f;margin:1rem 0;">
                <p class="text-xs text-muted">📅 Dates: YYYYMMDD<br>👤 Actors: People/Orgs<br>📊 Impact: -10 to +10<br>🔗 Links: News sources</p>
            </div>''', unsafe_allow_html=True)
    
    with tabs[3]:
        render_about()
    
    st.markdown('''<div class="footer">
        <p class="footer-main"><b>GDELT</b> monitors worldwide news in real-time • 100K+ daily events</p>
        <p class="footer-sub">Built by <a href="https://www.linkedin.com/in/mohith-akash/" class="footer-link">Mohith Akash</a></p>
    </div>''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()