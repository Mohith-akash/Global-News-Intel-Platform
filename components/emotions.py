"""
Emotions tab: mood, emotion mix and trending themes from GDELT's GKG feed
(the gkg_emotions table), over the last 24 hours. Emotion values are words
per 1,000 from the WordNet Affect dictionary in GCAM.
"""

import datetime

import plotly.graph_objects as go
import streamlit as st

from components.ui import card_header, metric_row, show_chart
from src.database import WarehouseUnavailable, retry_cache_race, safe_query_batch

EMOTIONS = ['fear', 'anger', 'sadness', 'joy', 'confidence', 'anxiety', 'anticipation']
EMOTION_COLORS = ['#ef4444', '#f97316', '#3b82f6', '#22c55e', '#06b6d4', '#eab308', '#8b5cf6']
MOOD_BANDS = [(-3, "Very negative", "#ef4444"), (-1, "Negative", "#f97316"), (1, "Neutral", "#eab308"),
              (3, "Positive", "#84cc16"), (float('inf'), "Very positive", "#22c55e")]

# common GDELT theme codes -> readable names
THEME_NAMES = {
    'TAX_FNCACT': 'Finance', 'EPU_POLICY': 'Policy', 'TAX_ETHNICITY': 'Ethnicity',
    'TAX_WORLDLANGUAGES': 'Languages', 'CRISISLEX_CRISISLEXREC': 'Crisis',
    'UNGP_FORESTS_RIVERS': 'Environment', 'USPEC_POLITICS_GENERAL1': 'Politics',
    'TAX_ECON_PRICE': 'Economy', 'GENERAL_GOVERNMENT': 'Government',
    'MANMADE_DISASTER_IMPLIED': 'Disaster', 'EPU_ECONOMY_HISTORIC': 'History',
    'EDUCATION': 'Education', 'SOC_POINTSOFINTEREST': 'Society', 'GENERAL_HEALTH': 'Health',
    'LEADER': 'Leadership', 'TERROR': 'Security', 'PROTEST': 'Protests', 'MILITARY': 'Military',
    'ARREST': 'Law', 'KILL': 'Conflict',
}
THEME_PREFIXES = ['Tax ', 'Epu ', 'Soc ', 'Wb ', 'Ungp ', 'Uspec ', 'Crisislex ']


def _cutoff_24h():
    """24h ago as GKG's YYYYMMDDHHMMSS number, rounded to the hour so the cache key holds."""
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
    return int(cutoff.strftime('%Y%m%d%H0000'))


@retry_cache_race
@st.cache_data(ttl=14400, show_spinner=False)
def _emotions_bundle(cutoff):
    """Both emotion queries in one worker call, cached 4h."""
    averages = ", ".join(f"AVG(EMOTION_{e.upper()}) AS {e}" for e in EMOTIONS)
    return safe_query_batch({
        'summary': (f"""
            SELECT COUNT(*) AS articles, AVG(AVG_TONE) AS tone,
                   AVG(POSITIVE_SCORE) AS positive, AVG(NEGATIVE_SCORE) AS negative, {averages}
            FROM gkg_emotions WHERE DATE >= {cutoff}
        """, None),
        # counted over every article in the window
        'themes': (f"""
            SELECT TRIM(theme) AS theme, COUNT(*) AS mentions
            FROM (SELECT unnest(string_split(TOP_THEMES, ',')) AS theme
                  FROM gkg_emotions WHERE DATE >= {cutoff} AND TOP_THEMES IS NOT NULL)
            WHERE LENGTH(TRIM(theme)) > 2
            GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 30
        """, None),
    })


def humanize_theme(theme):
    """GDELT theme code -> readable name, or None if it stays cryptic."""
    if not theme:
        return None
    code = theme.upper().strip()
    for prefix, name in THEME_NAMES.items():
        if code.startswith(prefix):
            return name
    cleaned = theme.replace('_', ' ').title()
    for prefix in THEME_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned if 2 < len(cleaned) < 20 else None


def _num(row, col):
    v = row[col]
    return 0 if v is None or v != v else v


def render_emotion_stats(row):
    metric_row([
        ("Articles", f"{int(row['articles']):,}", "GKG articles, last 24 h"),
        ("Positive", f"{_num(row, 'positive'):.1f}%", "Share of positive words"),
        ("Negative", f"{_num(row, 'negative'):.1f}%", "Share of negative words"),
        ("Fear", f"{_num(row, 'fear'):.2f}", "Fear words per 1,000"),
        ("Joy", f"{_num(row, 'joy'):.2f}", "Joy words per 1,000"),
    ])


def render_emotion_insight(row):
    tone, fear, joy = _num(row, 'tone'), _num(row, 'fear'), _num(row, 'joy')
    if fear > joy * 1.5:
        text = "Coverage is dominated by <b style='color:#ef4444'>fear and anxiety</b>."
    elif joy > fear * 1.5:
        text = "Coverage leans <b style='color:#22c55e'>positive</b>, optimism outweighs concern."
    elif tone < -2:
        text = "Media tone is <b style='color:#f97316'>clearly negative</b>."
    elif tone > 2:
        text = "Coverage reflects <b style='color:#84cc16'>positive developments</b>."
    else:
        text = "Sentiment is <b style='color:#eab308'>mixed</b>, no emotion dominates."
    st.markdown(f'<div class="insight"><span class="insight-label">Status:</span> {text}</div>',
                unsafe_allow_html=True)


def render_trending_themes(themes):
    top, seen = [], set()
    for theme, mentions in zip(themes['theme'], themes['mentions']):
        name = humanize_theme(theme)
        if name and name not in seen:
            seen.add(name)
            top.append((name, int(mentions)))
        if len(top) == 8:
            break
    if not top:
        st.info("No readable themes in the last 24 hours.")
        return
    names, values = zip(*top)
    colors = ['#00d4ff', '#06b6d4', '#0ea5e9', '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef']
    fig = go.Figure(go.Bar(x=values, y=names, orientation='h', marker=dict(color=colors[:len(names)]),
                           text=[f'{v:,}' for v in values], textposition='inside',
                           textfont=dict(color='white', size=11),
                           hovertemplate='<b>%{y}</b><br>Mentions: %{x:,}<extra></extra>'))
    show_chart(fig, 'themes_chart', 280, margin=dict(l=0, r=20, t=10, b=10), bargap=0.3,
               font=dict(color='#e2e8f0'), xaxis=dict(gridcolor='#1e3a5f', zeroline=False, showticklabels=False),
               yaxis=dict(showgrid=False, tickfont=dict(size=12), autorange='reversed'))


def render_emotions_pulse(row):
    mood = _num(row, 'tone')
    label, color = next((lab, col) for limit, lab, col in MOOD_BANDS if mood < limit)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=mood,
        number={'font': {'size': 34, 'color': color}, 'valueformat': '.2f'},
        gauge={
            'axis': {'range': [-10, 10], 'tickcolor': '#64748b', 'tickfont': {'color': '#64748b', 'size': 11},
                     'tickwidth': 1, 'dtick': 5},
            'bar': {'color': color, 'thickness': 0.3}, 'bgcolor': '#0f2744', 'borderwidth': 0,
            'steps': [{'range': r, 'color': c} for r, c in (
                ([-10, -3], 'rgba(239,68,68,0.15)'), ([-3, -1], 'rgba(249,115,22,0.15)'),
                ([-1, 1], 'rgba(234,179,8,0.15)'), ([1, 3], 'rgba(132,204,22,0.15)'), ([3, 10], 'rgba(34,197,94,0.15)'))],
            'threshold': {'line': {'color': '#ffffff', 'width': 3}, 'thickness': 0.8, 'value': mood},
        }))
    show_chart(fig, 'mood_gauge', 230, margin=dict(l=30, r=30, t=40, b=10), font=dict(color='#e2e8f0'))
    st.markdown(f'''<div class="mood"><span class="mood-badge" style="color:{color};border-color:{color};background:{color}22;">{label}</span>
        <div class="mood-note">Based on <b>{int(row['articles']):,}</b> articles</div></div>''', unsafe_allow_html=True)


def render_emotion_breakdown(row):
    values = [_num(row, e) for e in EMOTIONS]
    names = [e.title() for e in EMOTIONS]
    fig = go.Figure([
        go.Scatterpolar(r=values + values[:1], theta=names + names[:1], fill='toself',
                        fillcolor='rgba(0, 212, 255, 0.15)', line=dict(color='#00d4ff', width=2),
                        hovertemplate='%{theta}: %{r:.2f} per 1,000 words<extra></extra>'),
        go.Scatterpolar(r=values, theta=names, mode='markers',
                        marker=dict(size=10, color=EMOTION_COLORS, line=dict(color='#0a192f', width=2)),
                        hovertemplate='%{theta}: %{r:.2f} per 1,000 words<extra></extra>'),
    ])
    axis = dict(gridcolor='#1e3a5f', linecolor='#1e3a5f')
    show_chart(fig, 'emotion_radar', 300, showlegend=False, margin=dict(l=50, r=50, t=20, b=20),
               polar=dict(bgcolor='rgba(0,0,0,0)',
                          radialaxis=dict(visible=True, range=[0, max(values) * 1.3 or 1],
                                          tickfont=dict(color='#64748b', size=9), **axis),
                          angularaxis=dict(tickfont=dict(color='#e2e8f0', size=11), **axis)))
    top = max(range(len(values)), key=values.__getitem__)
    st.markdown(f'<div class="dominant">Dominant: <b>{names[top]}</b> ({values[top]:.2f} per 1,000 words)</div>',
                unsafe_allow_html=True)


def render_emotions_tab():
    try:
        data = _emotions_bundle(_cutoff_24h())
    except WarehouseUnavailable:
        st.warning("The data warehouse is not reachable right now. Try again in a few minutes.")
        return
    summary = data['summary']
    if summary.empty:  # table missing or query failed
        st.info("Emotion data isn't available yet. The GKG pipeline runs hourly.")
        return
    row = summary.iloc[0]
    if not row['articles']:
        st.info("No GKG articles in the last 24 hours yet.")
        return

    render_emotion_stats(row)
    render_emotion_insight(row)
    card_header("Trending topics", "last 24 h")
    render_trending_themes(data['themes'])
    col1, col2 = st.columns(2)
    with col1:
        card_header("Global mood index")
        render_emotions_pulse(row)
    with col2:
        card_header("Emotion radar")
        render_emotion_breakdown(row)
