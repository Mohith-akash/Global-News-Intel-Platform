"""
Emotions & Themes Dashboard Component
Visualizes GDELT GKG emotion data and trending themes.
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime


def check_gkg_table_exists(conn):
    """Check if gkg_emotions table exists and has data."""
    try:
        result = conn.execute("""
            SELECT COUNT(*) as cnt FROM gkg_emotions LIMIT 1
        """).df()
        return result['cnt'].iloc[0] > 0
    except:
        return False


def render_emotions_pulse(conn):
    """Render the global emotion pulse meter with a beautiful gauge."""
    try:
        df = conn.execute("""
            SELECT 
                AVG(AVG_TONE) as avg_mood,
                AVG(EMOTION_FEAR) as avg_fear,
                AVG(EMOTION_JOY) as avg_joy,
                AVG(EMOTION_ANGER) as avg_anger,
                AVG(EMOTION_TRUST) as avg_trust,
                COUNT(*) as article_count
            FROM gkg_emotions
        """).df()
        
        if df.empty:
            st.info("📊 Emotion data is being collected...")
            return
        
        row = df.iloc[0]
        mood = row['avg_mood'] if row['avg_mood'] else 0
        articles = int(row['article_count'])
        
        # Create a gauge chart for global mood
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mood,
            number={'suffix': '', 'font': {'size': 40, 'color': '#e2e8f0'}},
            title={'text': "Global Mood Index", 'font': {'size': 18, 'color': '#94a3b8'}},
            gauge={
                'axis': {'range': [-10, 10], 'tickcolor': '#64748b', 
                         'tickfont': {'color': '#64748b'}},
                'bar': {'color': '#00d4ff'},
                'bgcolor': '#0f2744',
                'borderwidth': 2,
                'bordercolor': '#1e3a5f',
                'steps': [
                    {'range': [-10, -5], 'color': '#ef4444'},
                    {'range': [-5, -2], 'color': '#f97316'},
                    {'range': [-2, 2], 'color': '#eab308'},
                    {'range': [2, 5], 'color': '#84cc16'},
                    {'range': [5, 10], 'color': '#22c55e'},
                ],
                'threshold': {
                    'line': {'color': '#00d4ff', 'width': 4},
                    'thickness': 0.75,
                    'value': mood
                }
            }
        ))
        
        fig.update_layout(
            height=220,
            margin=dict(l=30, r=30, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.plotly_chart(fig, width='stretch')
            st.markdown(f"""
                <div style="text-align: center; margin-top: -20px;">
                    <span style="color: #64748b; font-size: 0.85rem;">
                        Based on <b style="color: #00d4ff;">{articles:,}</b> articles analyzed
                    </span>
                </div>
            """, unsafe_allow_html=True)
                    
    except Exception as e:
        st.info("📊 Emotion data loading...")


def render_emotion_breakdown(conn):
    """Render emotion breakdown as a beautiful radar chart."""
    try:
        df = conn.execute("""
            SELECT 
                AVG(EMOTION_FEAR) as fear,
                AVG(EMOTION_ANGER) as anger,
                AVG(EMOTION_SADNESS) as sadness,
                AVG(EMOTION_JOY) as joy,
                AVG(EMOTION_TRUST) as trust,
                AVG(EMOTION_ANXIETY) as anxiety,
                AVG(EMOTION_ANTICIPATION) as anticipation
            FROM gkg_emotions
        """).df()
        
        if df.empty:
            st.info("📊 Collecting emotion data...")
            return
        
        row = df.iloc[0]
        
        # Prepare data for radar chart
        emotions = ['Fear', 'Anger', 'Sadness', 'Joy', 'Trust', 'Anxiety', 'Anticipation']
        values = [
            row['fear'] if row['fear'] else 0,
            row['anger'] if row['anger'] else 0,
            row['sadness'] if row['sadness'] else 0,
            row['joy'] if row['joy'] else 0,
            row['trust'] if row['trust'] else 0,
            row['anxiety'] if row['anxiety'] else 0,
            row['anticipation'] if row['anticipation'] else 0,
        ]
        
        # Close the radar chart
        emotions_closed = emotions + [emotions[0]]
        values_closed = values + [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=emotions_closed,
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.2)',
            line=dict(color='#00d4ff', width=2),
            name='Current'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2 if max(values) > 0 else 10],
                    gridcolor='#1e3a5f',
                    tickfont=dict(color='#64748b', size=10),
                ),
                angularaxis=dict(
                    gridcolor='#1e3a5f',
                    tickfont=dict(color='#e2e8f0', size=12),
                ),
                bgcolor='rgba(0,0,0,0)',
            ),
            showlegend=False,
            height=320,
            margin=dict(l=60, r=60, t=30, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Show top 3 emotions below
        emotion_values = list(zip(emotions, values))
        top_3 = sorted(emotion_values, key=lambda x: x[1], reverse=True)[:3]
        
        cols = st.columns(3)
        colors = ['#00d4ff', '#8b5cf6', '#06b6d4']
        for i, (emo, val) in enumerate(top_3):
            with cols[i]:
                st.markdown(f"""
                    <div style="text-align: center; padding: 0.5rem; background: #0f2744; border-radius: 8px; border-left: 3px solid {colors[i]};">
                        <div style="color: #94a3b8; font-size: 0.75rem;">#{i+1}</div>
                        <div style="color: #e2e8f0; font-weight: bold;">{emo}</div>
                        <div style="color: {colors[i]}; font-size: 1.1rem;">{val:.1f}</div>
                    </div>
                """, unsafe_allow_html=True)
        
    except Exception as e:
        st.info("📊 Emotion breakdown loading...")


# Map common GDELT theme codes to human-readable names
THEME_TRANSLATIONS = {
    'TAX_FNCACT': 'Financial Activity',
    'EPU_POLICY': 'Economic Policy',
    'TAX_ETHNICITY': 'Ethnic Issues',
    'TAX_WORLDLANGUAGES': 'Languages',
    'CRISISLEX_CRISISLEXREC': 'Crisis Events',
    'UNGP_FORESTS_RIVERS': 'Environment',
    'USPEC_POLITICS_GENERAL1': 'Politics',
    'TAX_ECON_PRICE': 'Pricing & Economy',
    'GENERAL_GOVERNMENT': 'Government',
    'MANMADE_DISASTER_IMPLIED': 'Disasters',
    'EPU_ECONOMY_HISTORIC': 'Economic History',
    'EDUCATION': 'Education',
    'SOC_POINTSOFINTEREST': 'Social Issues',
    'GENERAL_HEALTH': 'Healthcare',
    'LEADER': 'Leadership',
    'TERROR': 'Terrorism',
    'PROTEST': 'Protests',
    'MILITARY': 'Military',
    'ARREST': 'Law Enforcement',
    'KILL': 'Violence',
}


def humanize_theme(theme):
    """Convert GDELT theme code to human-readable name."""
    if not theme:
        return None
    theme_upper = theme.upper()
    if theme_upper in THEME_TRANSLATIONS:
        return THEME_TRANSLATIONS[theme_upper]
    for prefix, name in THEME_TRANSLATIONS.items():
        if theme_upper.startswith(prefix):
            return name
    # Clean up for display
    cleaned = theme.replace('_', ' ').title()
    # Remove common prefixes
    for prefix in ['Tax ', 'Epu ', 'Soc ', 'Wb ', 'Ungp ']:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned if len(cleaned) > 2 else None


def render_trending_themes(conn):
    """Render trending themes with beautiful styling."""
    try:
        df = conn.execute("""
            SELECT 
                TRIM(theme.value) as theme,
                COUNT(*) as mention_count
            FROM gkg_emotions,
            LATERAL UNNEST(STRING_SPLIT(TOP_THEMES, ',')) AS theme(value)
            WHERE TOP_THEMES IS NOT NULL AND TOP_THEMES != ''
            GROUP BY TRIM(theme.value)
            HAVING COUNT(*) >= 3
            ORDER BY mention_count DESC
            LIMIT 12
        """).df()
        
        if df.empty:
            st.info("📊 Theme data is being collected...")
            return
        
        # Get max for relative sizing
        max_count = df['mention_count'].max()
        
        for idx, row in df.iterrows():
            theme = row['theme']
            count = row['mention_count']
            
            display_theme = humanize_theme(theme)
            if not display_theme or len(display_theme) < 3:
                continue
            if len(display_theme) > 20:
                display_theme = display_theme[:18] + "..."
            
            # Calculate bar width percentage
            bar_pct = (count / max_count) * 100 if max_count > 0 else 0
            
            st.markdown(f"""
                <div style="position: relative; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem; background: #0f2744; border-radius: 6px; overflow: hidden;">
                    <div style="position: absolute; left: 0; top: 0; bottom: 0; width: {bar_pct}%; background: linear-gradient(90deg, rgba(0,212,255,0.3) 0%, rgba(0,212,255,0.05) 100%);"></div>
                    <div style="position: relative; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #e2e8f0; font-weight: 500; font-size: 0.9rem;">#{display_theme}</span>
                        <span style="color: #00d4ff; font-size: 0.8rem; font-weight: bold;">{count:,}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.info("📊 Theme data loading...")


def render_emotion_stats(conn):
    """Render emotion statistics cards."""
    try:
        df = conn.execute("""
            SELECT 
                COUNT(*) as total_articles,
                AVG(POSITIVE_SCORE) as avg_positive,
                AVG(NEGATIVE_SCORE) as avg_negative,
                AVG(AVG_TONE) as avg_tone
            FROM gkg_emotions
        """).df()
        
        if df.empty:
            return
        
        row = df.iloc[0]
        
        cols = st.columns(4)
        
        metrics = [
            ("📰", "Articles", f"{int(row['total_articles']):,}", "#00d4ff"),
            ("👍", "Positive", f"{row['avg_positive']:.1f}%", "#22c55e"),
            ("👎", "Negative", f"{row['avg_negative']:.1f}%", "#ef4444"),
            ("📊", "Avg Tone", f"{row['avg_tone']:.2f}", "#8b5cf6"),
        ]
        
        for i, (icon, label, value, color) in enumerate(metrics):
            with cols[i]:
                st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e3a5f 0%, #0a192f 100%); border-radius: 10px; border: 1px solid #1e3a5f;">
                        <div style="font-size: 1.5rem;">{icon}</div>
                        <div style="color: {color}; font-size: 1.3rem; font-weight: bold;">{value}</div>
                        <div style="color: #64748b; font-size: 0.75rem;">{label}</div>
                    </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        pass


def render_emotions_tab(conn):
    """Main render function for Emotions & Themes tab."""
    
    if not check_gkg_table_exists(conn):
        st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #1e3a5f 0%, #0a192f 100%); border-radius: 12px; border: 1px solid #1e3a5f; margin: 2rem 0;">
                <div style="font-size: 4rem;">🧠</div>
                <h2 style="color: #00d4ff; margin: 1rem 0;">Emotions & Themes Coming Soon!</h2>
                <p style="color: #94a3b8; max-width: 500px; margin: 0 auto;">
                    This feature analyzes 2,200+ emotional dimensions from global news.
                    <br><br>
                    <b>Status:</b> Waiting for GKG data collection.<br>
                    The pipeline runs every 15 minutes. Check back soon!
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Stats row at top
    render_emotion_stats(conn)
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([5, 5])
    
    with col1:
        st.markdown('<div class="card-hdr"><span>🎯</span><span class="card-title">Global Mood Index</span></div>', unsafe_allow_html=True)
        render_emotions_pulse(conn)
    
    with col2:
        st.markdown('<div class="card-hdr"><span>📊</span><span class="card-title">Emotion Radar</span></div>', unsafe_allow_html=True)
        render_emotion_breakdown(conn)
    
    st.markdown("---")
    
    st.markdown('<div class="card-hdr"><span>🔥</span><span class="card-title">Trending Topics</span><span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem;">(from news themes)</span></div>', unsafe_allow_html=True)
    render_trending_themes(conn)
