"""
GDELT News Intelligence Dashboard
Real-time global news analytics powered by AI.
"""

import streamlit as st
import os
from dotenv import load_dotenv
import logging

from config import REQUIRED_ENVS
from styles import inject_css
from database import get_db, get_engine, detect_table
from ai_engine import get_ai_engine
from components import (
    render_header,
    render_metrics,
    render_ticker,
    render_sentiment,
    render_actors,
    render_distribution,
    render_countries,
    render_trending,
    render_feed,
    render_timeseries,
    render_ai_chat,
    render_about
)

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