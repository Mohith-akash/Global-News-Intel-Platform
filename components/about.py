"""About tab: how the platform is built and why."""

import streamlit as st

FLOW = [
    ("GDELT", "events + GKG, a new batch every 15 min"),
    ("Polars + Dagster", "hourly ingest, Great Expectations checks"),
    ("MotherDuck", "warehouse + vector search"),
    ("Streamlit", "this app"),
]
SIDE = [
    ("dbt Core", "staging and marts, 30+ tests, built weekly"),
    ("Voyage AI + Cerebras", "embeddings every 12 h, answers in the AI tab"),
]
CHOICES = [
    ("MotherDuck", "One free-tier warehouse for SQL and vectors: <code>array_cosine_similarity</code> "
                   "replaced a separate vector database."),
    ("GitHub Actions", "Cron runs the Dagster jobs, so there is no scheduler server to keep alive."),
    ("Polars", "The slowest stage went from 91 s in pandas to 6 s on 116K rows."),
    ("Query templates", "The SQL chat first let the LLM write SQL. It got dates and country codes wrong, "
                        "so questions now map to tested templates and the LLM only writes the answer."),
    ("Direct LLM calls", "LlamaIndex sent no output cap, and the reasoning model billed its hidden "
                         "thinking. Calling the API directly made the cost predictable."),
    ("Cerebras credits", "The only running cost, a few dollars a month. Everything else runs on free tiers."),
]
STATS = [("30M+", "events stored"), ("~100K", "new events a day"), ("Hourly", "ingestion"), ("~$3", "per month")]


def render_about():
    flow = '<span class="flow-arrow">→</span>'.join(
        f'<div class="flow-box"><b>{name}</b><span>{note}</span></div>' for name, note in FLOW)
    side = "".join(f'<div class="flow-box flow-side"><b>{name}</b><span>{note}</span></div>' for name, note in SIDE)
    choices = "".join(f'<li><b>{name}:</b> {text}</li>' for name, text in CHOICES)
    stats = "".join(f'<div class="stat"><b>{value}</b><span>{label}</span></div>' for value, label in STATS)
    st.markdown(f'''
        <h3 class="about-title">About this project</h3>
        <div class="about-flow">{flow}</div>
        <div class="about-flow about-side">{side}</div>
        <div class="about-card"><h4>Choices along the way</h4><ul class="about-list">{choices}</ul></div>
        <div class="stat-row">{stats}</div>
        <div class="about-contact">
            <a href="https://github.com/Mohith-akash/Global-News-Intel-Platform" target="_blank">Source on GitHub</a>
            <a href="https://www.linkedin.com/in/mohith-akash/" target="_blank">LinkedIn</a>
        </div>''', unsafe_allow_html=True)
