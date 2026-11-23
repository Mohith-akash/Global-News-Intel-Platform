"""Home and Feed tab widgets."""

import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.ui import GRID, MUTED, TEXT, metric_row, show_chart
from src.data_processing import process_df
from src.queries import (get_actors, get_alerts, get_countries, get_distribution, get_feed,
                         get_metrics, get_sentiment, get_timeseries, get_trending)
from src.utils import get_country


def render_header():
    st.markdown('''<div class="header"><div class="logo"><div>
            <div class="logo-title">Global News Intelligence</div>
            <div class="logo-sub">News events from GDELT, ingested hourly</div>
        </div></div></div>''', unsafe_allow_html=True)


def render_metrics(t):
    m = get_metrics(t)
    fmt = lambda n: f"{int(n):,}" if n else "—"
    hot = get_country(m['hotspot']) or m['hotspot'] or "—"
    fetched = datetime.datetime.fromisoformat(m['fetched_at'])
    metric_row([
        ("Total events", fmt(m['total']), "Every event in the warehouse"),
        ("Last 7 days", fmt(m['recent']), "Events from the past week"),
        ("High impact", fmt(m['critical']), "Impact beyond ±6 this week"),
        ("Top country", hot if len(hot) <= 15 else hot[:15] + "…", "Most events this week"),
        ("Data as of", fetched.strftime("%H:%M UTC"), fetched.strftime("%d %b") + ", cached up to 4 h"),
    ])


def render_ticker(t):
    df = get_alerts(t)
    if df.empty:
        text = "No high-impact events in the last 3 days │ "
    else:
        text = " │ ".join(f"{r['MAIN_ACTOR'][:30]} ({get_country(r['ACTOR_COUNTRY_CODE']) or 'Global'}) "
                          f"{r['IMPACT_SCORE']:.1f}" for _, r in df.iterrows()) + " │ "
    st.markdown('<div class="ticker-hint">High-impact events from the last 3 days (impact below −4, '
                'on a scale from −10 conflict to +10 cooperation)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ticker"><div class="ticker-label">ALERTS</div>'
                f'<div class="ticker-text">{text + text}</div></div>', unsafe_allow_html=True)


def render_sentiment(t):
    df = get_sentiment(t)
    if df.empty:
        st.info("No sentiment data for this week.")
        return
    row = df.iloc[0]
    avg = row['avg'] or 0
    status, color = next((s, c) for limit, s, c in ((-2, "Elevated", "#ef4444"), (0, "Moderate", "#f59e0b"),
                                                     (2, "Stable", "#10b981"), (float('inf'), "Positive", "#06b6d4"))
                         if avg < limit)
    counts = [("Negative", int(row['neg'] or 0), "neg"), ("Total", int(row['total'] or 1), "total"),
              ("Positive", int(row['pos'] or 0), "pos")]
    boxes = "".join(f'<div class="sent-box sent-{cls}"><div class="sent-num">{n:,}</div>'
                    f'<div class="sent-label">{label}</div></div>' for label, n, cls in counts)
    st.markdown(f'''<div class="sent-card">
        <div class="sent-status" style="color:{color};">{status}</div>
        <div class="sent-avg">Average impact <span style="color:{color}">{avg:.2f}</span></div></div>
        <div class="sent-grid">{boxes}</div>''', unsafe_allow_html=True)


def render_actors(t):
    df = get_actors(t)
    if df.empty:
        st.info("No actor data for this week.")
        return
    labels = [f"{a[:25]} ({c[:10]})" if (c := get_country(code)) else a[:25]
              for a, code in zip(df['MAIN_ACTOR'], df['ACTOR_COUNTRY_CODE'])]
    colors = ['#ef4444' if x < -3 else '#f59e0b' if x < 0 else '#10b981' if x > 3 else '#06b6d4'
              for x in df['avg_impact'].fillna(0)]
    fig = go.Figure(go.Bar(x=df['events'], y=labels, orientation='h', marker_color=colors,
                           text=df['events'].map('{:,}'.format), textposition='outside',
                           textfont=dict(color=TEXT, size=10)))
    show_chart(fig, 'actors_chart', 350, margin=dict(l=0, r=70, t=10, b=0), bargap=0.3,
               xaxis=dict(showgrid=True, gridcolor=GRID, tickfont=dict(color=MUTED)),
               yaxis=dict(showgrid=False, tickfont=dict(color='#e2e8f0', size=11), autorange='reversed'))


def render_distribution(t, chart_key='distribution'):
    df = get_distribution(t)
    if df.empty:
        st.info("No tone data for this week.")
        return
    colors = {'Crisis': '#ef4444', 'Negative': '#f59e0b', 'Neutral': MUTED, 'Positive': '#10b981', 'Very Positive': '#06b6d4'}
    fig = go.Figure(go.Pie(labels=df['cat'], values=df['cnt'], hole=0.6, textinfo='percent',
                           marker_colors=[colors.get(c, MUTED) for c in df['cat']],
                           textfont=dict(size=11, color='#e2e8f0')))
    show_chart(fig, chart_key, 200, margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
               legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center', font=dict(size=10, color=TEXT)))


def render_countries(t):
    df = get_countries(t)
    if df.empty:
        st.info("No country data for the last 30 days.")
        return
    df = df.head(8)
    names = [get_country(c) or c or 'Unknown' for c in df['country']]
    fig = go.Figure(go.Bar(x=names, y=df['events'], marker_color='#06b6d4', textposition='outside',
                           text=[f"{n / 1000:.1f}K" if n >= 1000 else str(int(n)) for n in df['events']],
                           textfont=dict(color=TEXT, size=10)))
    show_chart(fig, 'countries_chart', 300, margin=dict(l=0, r=0, t=10, b=0), bargap=0.35,
               xaxis=dict(showgrid=False, tickfont=dict(color=TEXT, size=10), tickangle=-45),
               yaxis=dict(showgrid=True, gridcolor=GRID, showticklabels=False))


def _stories(df, rows, empty, height, extra_cols=(), headline_label="Story"):
    df = process_df(df).head(rows) if not df.empty else df
    if df.empty:
        st.info(empty)
        return
    cols = ['DATE_FMT', 'INTENSITY', 'HEADLINE', 'REGION', *extra_cols, 'NEWS_LINK']
    st.dataframe(df[cols], hide_index=True, height=height, width='stretch', column_config={
        "DATE_FMT": st.column_config.TextColumn("Date", width=60),
        "INTENSITY": st.column_config.TextColumn("Intensity", width=140),
        "HEADLINE": st.column_config.TextColumn(headline_label, width=None),
        "REGION": st.column_config.TextColumn("Region", width=100),
        "ARTICLE_COUNT": st.column_config.NumberColumn("Articles", width=60),
        "NEWS_LINK": st.column_config.LinkColumn("Link", width=50, display_text="open"),
    })


def render_trending(t):
    _stories(get_trending(t), 20, "No stories this week.", 400, extra_cols=('ARTICLE_COUNT',))


def render_feed(t):
    _stories(get_feed(t), 50, "No events this week.", 600, headline_label="Event")


def render_timeseries(t):
    df = get_timeseries(t)
    if df.empty:
        st.info("No data for the last 30 days.")
        return
    dates = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=dates, y=df['events'], fill='tozeroy', fillcolor='rgba(6,182,212,0.15)',
                             line=dict(color='#06b6d4', width=2), name='Total'), secondary_y=False)
    for col, color, name in (('negative', '#ef4444', 'Negative'), ('positive', '#10b981', 'Positive')):
        fig.add_trace(go.Scatter(x=dates, y=df[col], line=dict(color=color, width=2), name=name), secondary_y=True)
    show_chart(fig, 'timeseries_chart', 300, margin=dict(l=0, r=0, t=30, b=0), showlegend=True,
               legend=dict(orientation='h', y=1.02, font=dict(size=11, color=TEXT)), hovermode='x unified',
               xaxis=dict(showgrid=True, gridcolor=GRID, tickfont=dict(color=MUTED)),
               yaxis=dict(showgrid=True, gridcolor=GRID, tickfont=dict(color=MUTED)))
