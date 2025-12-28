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


def render_emotions_pulse(conn):
    """Render the global emotion pulse meter."""
    try:
        # Get latest day's mood
        df = conn.execute("""
            SELECT 
                event_date,
                avg_mood,
                avg_fear,
                avg_anger,
                avg_joy,
                avg_trust,
                avg_anxiety,
                article_count
            FROM fct_daily_emotions
            ORDER BY event_date DESC
            LIMIT 7
        """).df()
        
        if df.empty:
            st.info("📊 Emotion data is being collected. Check back soon!")
            return
        
        today = df.iloc[0]
        mood = today['avg_mood']
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
        
        df_sorted = df.sort_values('event_date')
        emojis = [get_emoji_for_mood(m) for m in df_sorted['avg_mood']]
        
        cols = st.columns(min(7, len(df_sorted)))
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            if i < len(cols):
                with cols[i]:
                    day_name = row['event_date'].strftime('%a') if hasattr(row['event_date'], 'strftime') else str(row['event_date'])[-2:]
                    st.markdown(f"""
                        <div style="text-align: center; padding: 0.5rem; background: #0f2744; border-radius: 8px;">
                            <div style="font-size: 0.7rem; color: #64748b;">{day_name}</div>
                            <div style="font-size: 1.5rem;">{emojis[i]}</div>
                            <div style="font-size: 0.7rem; color: #94a3b8;">{row['avg_mood']:.1f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
    except Exception as e:
        st.warning(f"Emotion pulse data not yet available. Run the pipeline first.")


def render_emotion_breakdown(conn):
    """Render emotion breakdown chart."""
    try:
        df = conn.execute("""
            SELECT 
                AVG(avg_fear) as fear,
                AVG(avg_anger) as anger,
                AVG(avg_sadness) as sadness,
                AVG(avg_joy) as joy,
                AVG(avg_trust) as trust,
                AVG(avg_anxiety) as anxiety,
                AVG(avg_anticipation) as anticipation
            FROM fct_daily_emotions
            WHERE event_date >= CURRENT_DATE - INTERVAL '7 days'
        """).df()
        
        if df.empty:
            st.info("📊 Collecting emotion data...")
            return
        
        row = df.iloc[0]
        emotions = {
            '😨 Fear': row['fear'] if row['fear'] else 0,
            '😡 Anger': row['anger'] if row['anger'] else 0,
            '😢 Sadness': row['sadness'] if row['sadness'] else 0,
            '😊 Joy': row['joy'] if row['joy'] else 0,
            '🤝 Trust': row['trust'] if row['trust'] else 0,
            '😰 Anxiety': row['anxiety'] if row['anxiety'] else 0,
            '🎯 Anticipation': row['anticipation'] if row['anticipation'] else 0,
        }
        
        # Sort by value
        emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=list(emotions.values()),
            y=list(emotions.keys()),
            orientation='h',
            marker_color=['#ef4444', '#f97316', '#3b82f6', '#22c55e', '#06b6d4', '#eab308', '#8b5cf6']
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
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.info("Emotion breakdown loading...")


def render_trending_themes(conn):
    """Render trending themes."""
    try:
        df = conn.execute("""
            SELECT theme, mention_count, avg_daily_mentions
            FROM fct_trending_themes
            LIMIT 15
        """).df()
        
        if df.empty:
            st.info("📊 Theme data is being collected...")
            return
        
        for _, row in df.iterrows():
            theme = row['theme']
            count = row['mention_count']
            daily = row['avg_daily_mentions']
            
            # Make theme display-friendly
            display_theme = theme.replace('_', ' ').title()
            if len(display_theme) > 25:
                display_theme = display_theme[:22] + "..."
            
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; margin-bottom: 0.25rem; background: #0f2744; border-radius: 6px; border-left: 3px solid #00d4ff;">
                    <span style="color: #e2e8f0; font-weight: 500;">#{display_theme}</span>
                    <span style="color: #94a3b8; font-size: 0.85rem;">{count:,} mentions</span>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.info("Theme data loading...")


def render_emotion_timeline(conn):
    """Render emotion timeline chart."""
    try:
        df = conn.execute("""
            SELECT 
                event_date,
                avg_mood,
                avg_fear,
                avg_joy,
                avg_anger,
                avg_trust
            FROM fct_daily_emotions
            WHERE event_date >= CURRENT_DATE - INTERVAL '30 days'
            ORDER BY event_date
        """).df()
        
        if df.empty or len(df) < 2:
            st.info("📊 Building emotion timeline...")
            return
        
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
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.info("Emotion timeline loading...")


def render_emotions_tab(conn):
    """Main render function for Emotions & Themes tab."""
    st.markdown('<div class="card-hdr"><span>🧠</span><span class="card-title">Global Emotion Pulse</span></div>', unsafe_allow_html=True)
    render_emotions_pulse(conn)
    
    st.markdown("---")
    
    col1, col2 = st.columns([6, 4])
    
    with col1:
        st.markdown('<div class="card-hdr"><span>📊</span><span class="card-title">Emotion Breakdown</span><span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem;">(7 Days)</span></div>', unsafe_allow_html=True)
        render_emotion_breakdown(conn)
    
    with col2:
        st.markdown('<div class="card-hdr"><span>🔥</span><span class="card-title">Trending Themes</span></div>', unsafe_allow_html=True)
        render_trending_themes(conn)
    
    st.markdown("---")
    
    st.markdown('<div class="card-hdr"><span>📈</span><span class="card-title">30-Day Emotion Timeline</span></div>', unsafe_allow_html=True)
    render_emotion_timeline(conn)
