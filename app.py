"""
Global News Intelligence: a Streamlit dashboard over GDELT news events,
with an AI chat (SQL templates and RAG). Data is ingested hourly.
"""

import faulthandler

# On a native crash (segfault), print the python line every thread was
# executing to stderr - it lands in the cloud log and names the culprit.
faulthandler.enable()

import logging  # noqa: E402
import os  # noqa: E402

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from components import (render_about, render_actors, render_ai_chat, render_countries,  # noqa: E402
                        render_distribution, render_emotions_tab, render_feed, render_header,
                        render_metrics, render_sentiment, render_ticker, render_timeseries,
                        render_trending)
from components.ui import card_header  # noqa: E402
from src.config import REQUIRED_ENVS, get_secret  # noqa: E402
from src.database import WarehouseUnavailable, detect_table  # noqa: E402
from src.styles import inject_css  # noqa: E402
from src.utils import INTENSITY_SCALE  # noqa: E402

st.set_page_config(page_title="Global News Intelligence", page_icon="🌐", layout="wide",
                   initial_sidebar_state="collapsed")
load_dotenv()
logging.basicConfig(level=logging.INFO)

missing = [k for k in REQUIRED_ENVS if not get_secret(k)]
if missing:
    st.error(f"Missing required API keys: {', '.join(missing)}")
    st.stop()
# Streamlit secrets aren't visible to the query worker subprocess; env vars are.
for key in REQUIRED_ENVS:
    os.environ[key] = get_secret(key)


def render_intensity_guide():
    rows = "".join(
        f'<div class="intensity-item"><div class="intensity-title">{label}</div>'
        f'<div class="intensity-score">Score {low} to {high}</div></div>'
        for label, low, high, _ in INTENSITY_SCALE)
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)


HOW_IT_WORKS = """<div class="card"><h4 class="title-cyan">How it works</h4>
    <p class="text-muted"><b>SQL mode:</b> the question picks a tested query template, the
    warehouse runs it, and the LLM writes the answer from the returned rows.</p>
    <p class="text-muted"><b>RAG mode:</b> the question is embedded with Voyage AI, similar
    headlines are found by vector search, and the LLM answers from those headlines.</p>
    <p class="text-xs text-muted">Impact scores run from -10 (conflict) to +10 (cooperation).
    Answers are cached for an hour.</p></div>"""


def main():
    inject_css()
    tbl = detect_table()
    render_header()
    home, feed, emotions, chat, about = st.tabs(["Overview", "Feed", "Emotions", "Ask AI", "About"])

    with home:
        render_metrics(tbl)
        render_ticker(tbl)
        st.markdown("---")
        c1, c2 = st.columns([6, 4])
        with c1:
            card_header("Trending news", "this week")
            render_trending(tbl)
        with c2:
            card_header("Weekly sentiment", "7 days")
            render_sentiment(tbl)
            card_header("Tone breakdown", "this week")
            render_distribution(tbl, 'home_dist')
        st.markdown("---")
        c1, c2 = st.columns([6, 4])
        with c1:
            card_header("Most mentioned", "this week")
            render_actors(tbl)
        with c2:
            card_header("Top countries", "30 days")
            render_countries(tbl)

    with feed:
        c1, c2 = st.columns([7, 3])
        with c1:
            card_header("Recent events", "this week")
            render_feed(tbl)
        with c2:
            card_header("Intensity guide")
            render_intensity_guide()
        st.markdown("---")
        card_header("30-day trend")
        render_timeseries(tbl)

    with emotions:
        render_emotions_tab()

    with chat:
        c1, c2 = st.columns([7, 3])
        with c1:
            card_header("Ask in plain English")
            render_ai_chat(tbl)
        with c2:
            st.markdown(HOW_IT_WORKS, unsafe_allow_html=True)

    with about:
        render_about()

    st.markdown('''<div class="footer">
        <p class="footer-main">Data: <b>GDELT</b>, which publishes new events every 15 minutes (about 100K a day). This app ingests them hourly.</p>
        <p class="footer-sub">Built by <a href="https://www.linkedin.com/in/mohith-akash/" class="footer-link">Mohith Akash</a></p>
    </div>''', unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except WarehouseUnavailable:
        st.warning(
            "The data warehouse is temporarily unreachable. "
            "Dashboards come back on their own once it responds again - "
            "ingestion keeps running in the background."
        )
        st.stop()