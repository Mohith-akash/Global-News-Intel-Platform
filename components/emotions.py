"""
Emotions & Themes Dashboard Component
Visualizes GDELT GKG emotion data and trending themes.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


def get_emoji_for_mood(mood_score):
    """Return emoji based on mood score."""
    if mood_score < -5:
        return "😡"
    elif mood_score < -2:
        return "😨"
    elif mood_score < 0:
        return "😟"
    elif mood_score < 2:
        return "😐"
    elif mood_score < 5:
        return "🙂"
    else:
        return "😊"


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
    """Render the global emotion pulse meter."""
    try:
        # Query raw gkg_emotions table directly (not dbt)
        df = conn.execute("""
            SELECT 
                SUBSTR(DATE, 1, 8) as date_key,
                AVG(AVG_TONE) as avg_mood,
                AVG(EMOTION_FEAR) as avg_fear,
                AVG(EMOTION_JOY) as avg_joy,
                AVG(EMOTION_ANXIETY) as avg_anxiety,
                COUNT(*) as article_count
            FROM gkg_emotions
            GROUP BY SUBSTR(DATE, 1, 8)
            ORDER BY date_key DESC
            LIMIT 7
        """).df()
        
        if df.empty:
            st.info("📊 Emotion data is being collected. Check back in ~15 minutes after the next pipeline run.")
            return
        
        today = df.iloc[0]
        mood = today['avg_mood'] if today['avg_mood'] else 0
        emoji = get_emoji_for_mood(mood)
        
        # Mood description
        if mood < -3:
            mood_text = "Very Negative"
            color = "#ef4444"
        elif mood < -1:
            mood_text = "Slightly Negative"
            color = "#f97316"
        elif mood < 1:
            mood_text = "Neutral"
            color = "#eab308"
        elif mood < 3:
            mood_text = "Slightly Positive"
            color = "#84cc16"
        else:
            mood_text = "Positive"
            color = "#22c55e"
        
        # Display current mood
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e3a5f 0%, #0a192f 100%); border-radius: 12px; border: 1px solid #1e3a5f;">
                    <div style="font-size: 3rem;">{emoji}</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{mood_text}</div>
                    <div style="font-size: 0.9rem; color: #94a3b8;">Global Mood Score: {mood:.1f}</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">Based on {today['article_count']:,.0f} articles</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Weekly mood trend
        st.markdown("#### 📅 Weekly Mood Trend")
        
        df_sorted = df.sort_values('date_key')
        emojis = [get_emoji_for_mood(m if m else 0) for m in df_sorted['avg_mood']]
        
        cols = st.columns(min(7, len(df_sorted)))
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            if i < len(cols):
                with cols[i]:
                    date_str = row['date_key']
                    # Format: YYYYMMDD -> day name
                    try:
                        day_name = datetime.strptime(date_str, '%Y%m%d').strftime('%a')
                    except:
                        day_name = date_str[-2:]
                    mood_val = row['avg_mood'] if row['avg_mood'] else 0
                    st.markdown(f"""
                        <div style="text-align: center; padding: 0.5rem; background: #0f2744; border-radius: 8px;">
                            <div style="font-size: 0.7rem; color: #64748b;">{day_name}</div>
                            <div style="font-size: 1.5rem;">{emojis[i]}</div>
                            <div style="font-size: 0.7rem; color: #94a3b8;">{mood_val:.1f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
    except Exception as e:
        st.info("📊 Emotion pulse: Waiting for GKG data collection. Check back in ~15 minutes.")


def render_emotion_breakdown(conn):
    """Render emotion breakdown chart."""
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
        # Define emotions with their colors
        emotion_data = [
            ('😨 Fear', row['fear'] if row['fear'] else 0, '#ef4444'),
            ('😡 Anger', row['anger'] if row['anger'] else 0, '#f97316'),
            ('😢 Sadness', row['sadness'] if row['sadness'] else 0, '#3b82f6'),
            ('😊 Joy', row['joy'] if row['joy'] else 0, '#22c55e'),
            ('🤝 Trust', row['trust'] if row['trust'] else 0, '#06b6d4'),
            ('😰 Anxiety', row['anxiety'] if row['anxiety'] else 0, '#eab308'),
            ('🎯 Anticipation', row['anticipation'] if row['anticipation'] else 0, '#8b5cf6'),
        ]
        
        # Sort by value while keeping colors matched
        emotion_data = sorted(emotion_data, key=lambda x: x[1], reverse=True)
        
        names = [e[0] for e in emotion_data]
        values = [e[1] for e in emotion_data]
        colors = [e[2] for e in emotion_data]
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker_color=colors
        ))
        
        fig.update_layout(
            title=None,
            xaxis_title="Average Score",
            yaxis_title=None,
            height=300,
            margin=dict(l=0, r=20, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='#1e3a5f', zeroline=False),
            yaxis=dict(gridcolor='#1e3a5f'),
        )
        
        st.plotly_chart(fig, width='stretch')
        
    except Exception as e:
        st.info("📊 Emotion breakdown: Waiting for data...")


# Map common GDELT theme codes to human-readable names
THEME_TRANSLATIONS = {
    'TAX_FNCACT': 'Tax Policy',
    'EPU_POLICY': 'Economic Policy',
    'TAX_ETHNICITY': 'Ethnic Issues',
    'TAX_WORLDLANGUAGES': 'Languages',
    'CRISISLEX_CRISISLEXREC': 'Crisis Events',
    'UNGP_FORESTS_RIVERS': 'Environment',
    'USPEC_POLITICS_GENERAL1': 'Politics',
    'TAX_ECON_PRICE': 'Economic Prices',
    'GENERAL_GOVERNMENT': 'Government',
    'MANMADE_DISASTER_IMPLIED': 'Disasters',
    'EPU_ECONOMY_HISTORIC': 'Economic History',
    'EDUCATION': 'Education',
    'SOC_POINTSOFINTEREST': 'Social Issues',
    'GENERAL_HEALTH': 'Health',
    'LEADER': 'Leadership',
    'TERROR': 'Terrorism',
    'PROTEST': 'Protests',
    'MILITARY': 'Military',
    'ENV_': 'Environment',
    'WB_': 'World Bank Topics',
}

def humanize_theme(theme):
    """Convert GDELT theme code to human-readable name."""
    if not theme:
        return None
    theme_upper = theme.upper()
    # Check for exact matches
    if theme_upper in THEME_TRANSLATIONS:
        return THEME_TRANSLATIONS[theme_upper]
    # Check for prefix matches
    for prefix, name in THEME_TRANSLATIONS.items():
        if theme_upper.startswith(prefix):
            return name
    # Default: clean up the name
    return theme.replace('_', ' ').title()


def render_trending_themes(conn):
    """Render trending themes."""
    try:
        df = conn.execute("""
            SELECT 
                TRIM(theme.value) as theme,
                COUNT(*) as mention_count
            FROM gkg_emotions,
            LATERAL UNNEST(STRING_SPLIT(TOP_THEMES, ',')) AS theme(value)
            WHERE TOP_THEMES IS NOT NULL AND TOP_THEMES != ''
            GROUP BY TRIM(theme.value)
            HAVING COUNT(*) >= 5
            ORDER BY mention_count DESC
            LIMIT 15
        """).df()
        
        if df.empty:
            st.info("📊 Theme data is being collected...")
            return
        
        for _, row in df.iterrows():
            theme = row['theme']
            count = row['mention_count']
            
            # Make theme display-friendly
            display_theme = humanize_theme(theme)
            if not display_theme:
                continue
            if len(display_theme) > 25:
                display_theme = display_theme[:22] + "..."
            
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; margin-bottom: 0.25rem; background: #0f2744; border-radius: 6px; border-left: 3px solid #00d4ff;">
                    <span style="color: #e2e8f0; font-weight: 500;">#{display_theme}</span>
                    <span style="color: #94a3b8; font-size: 0.85rem;">{count:,} mentions</span>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.info("📊 Theme data: Waiting for collection...")


def render_emotion_timeline(conn):
    """Render emotion timeline chart."""
    try:
        df = conn.execute("""
            SELECT 
                SUBSTR(DATE, 1, 8) as date_key,
                AVG(AVG_TONE) as avg_mood,
                AVG(EMOTION_FEAR) as avg_fear,
                AVG(EMOTION_JOY) as avg_joy,
                AVG(EMOTION_ANGER) as avg_anger
            FROM gkg_emotions
            GROUP BY SUBSTR(DATE, 1, 8)
            ORDER BY date_key
        """).df()
        
        if df.empty or len(df) < 2:
            st.info("📊 Building emotion timeline... Need at least 2 days of data.")
            return
        
        # Convert date_key to proper dates
        df['event_date'] = df['date_key'].apply(
            lambda x: datetime.strptime(x, '%Y%m%d') if x else None
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['event_date'], y=df['avg_mood'],
            name='Overall Mood', line=dict(color='#00d4ff', width=3),
            fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['event_date'], y=df['avg_joy'],
            name='Joy', line=dict(color='#22c55e', width=2, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['event_date'], y=df['avg_fear'],
            name='Fear', line=dict(color='#ef4444', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title=None,
            xaxis_title="Date",
            yaxis_title="Score",
            height=350,
            margin=dict(l=0, r=20, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(gridcolor='#1e3a5f', zeroline=False),
            yaxis=dict(gridcolor='#1e3a5f', zeroline=True, zerolinecolor='#475569'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)'
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, width='stretch')
        
    except Exception as e:
        st.info("📊 Emotion timeline: Waiting for data...")


def render_emotions_tab(conn):
    """Main render function for Emotions & Themes tab."""
    
    # Check if GKG table exists
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
    
    st.markdown('<div class="card-hdr"><span>🧠</span><span class="card-title">Global Emotion Pulse</span></div>', unsafe_allow_html=True)
    render_emotions_pulse(conn)
    
    st.markdown("---")
    
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.markdown('<div class="card-hdr"><span>📊</span><span class="card-title">Emotion Breakdown</span><span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem;">(All Data)</span></div>', unsafe_allow_html=True)
        render_emotion_breakdown(conn)
    
    with col2:
        st.markdown('<div class="card-hdr"><span>🔥</span><span class="card-title">Trending Themes</span></div>', unsafe_allow_html=True)
        render_trending_themes(conn)
    
    st.markdown("---")
    
    st.markdown('<div class="card-hdr"><span>📈</span><span class="card-title">Emotion Timeline</span></div>', unsafe_allow_html=True)
    render_emotion_timeline(conn)
