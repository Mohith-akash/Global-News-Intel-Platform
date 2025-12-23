"""
Render functions for dashboard UI components.
"""

import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.queries import (
    get_metrics, get_alerts, get_trending, get_feed, get_countries,
    get_timeseries, get_sentiment, get_actors, get_distribution
)
from src.data_processing import process_df
from src.utils import get_country


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
