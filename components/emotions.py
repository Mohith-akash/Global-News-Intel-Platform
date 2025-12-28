"""
Emotions & Themes Dashboard Component
Visualizes GDELT GKG emotion data and trending themes.
Premium design with animations and modern UI.
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
        
        # Determine mood label and color
        if mood < -3:
            mood_label, mood_color = "Very Negative", "#ef4444"
        elif mood < -1:
            mood_label, mood_color = "Negative", "#f97316"
        elif mood < 1:
            mood_label, mood_color = "Neutral", "#eab308"
        elif mood < 3:
            mood_label, mood_color = "Positive", "#84cc16"
        else:
            mood_label, mood_color = "Very Positive", "#22c55e"
        
        # Create a gauge chart for global mood
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mood,
            number={'suffix': '', 'font': {'size': 48, 'color': mood_color}, 'valueformat': '.2f'},
            gauge={
                'axis': {'range': [-10, 10], 'tickcolor': '#64748b', 
                         'tickfont': {'color': '#64748b', 'size': 11},
                         'tickwidth': 1, 'dtick': 5},
                'bar': {'color': mood_color, 'thickness': 0.3},
                'bgcolor': '#0f2744',
                'borderwidth': 0,
                'steps': [
                    {'range': [-10, -3], 'color': 'rgba(239,68,68,0.15)'},
                    {'range': [-3, -1], 'color': 'rgba(249,115,22,0.15)'},
                    {'range': [-1, 1], 'color': 'rgba(234,179,8,0.15)'},
                    {'range': [1, 3], 'color': 'rgba(132,204,22,0.15)'},
                    {'range': [3, 10], 'color': 'rgba(34,197,94,0.15)'},
                ],
                'threshold': {
                    'line': {'color': '#ffffff', 'width': 3},
                    'thickness': 0.8,
                    'value': mood
                }
            }
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mood label with pulsing animation
        st.markdown(f"""
            <style>
                @keyframes pulse {{
                    0% {{ opacity: 0.7; }}
                    50% {{ opacity: 1; }}
                    100% {{ opacity: 0.7; }}
                }}
                .mood-pulse {{ animation: pulse 2s infinite; }}
            </style>
            <div style="text-align: center; margin-top: -15px;">
                <span class="mood-pulse" style="display: inline-block; padding: 0.4rem 1.2rem; background: {mood_color}22; border: 1px solid {mood_color}; border-radius: 20px; color: {mood_color}; font-weight: 600; font-size: 0.9rem;">
                    {mood_label}
                </span>
                <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">
                    Analyzed from <b style="color: #00d4ff;">{articles:,}</b> news articles
                </div>
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
        
        # Prepare data for radar chart with emojis
        emotions = ['😨 Fear', '😡 Anger', '😢 Sadness', '😊 Joy', '🤝 Trust', '😰 Anxiety', '🎯 Anticipation']
        values = [
            row['fear'] if row['fear'] else 0,
            row['anger'] if row['anger'] else 0,
            row['sadness'] if row['sadness'] else 0,
            row['joy'] if row['joy'] else 0,
            row['trust'] if row['trust'] else 0,
            row['anxiety'] if row['anxiety'] else 0,
            row['anticipation'] if row['anticipation'] else 0,
        ]
        
        # Colors for each emotion
        colors = ['#ef4444', '#f97316', '#3b82f6', '#22c55e', '#06b6d4', '#eab308', '#8b5cf6']
        
        # Close the radar chart
        emotions_closed = emotions + [emotions[0]]
        values_closed = values + [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=emotions_closed,
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.15)',
            line=dict(color='#00d4ff', width=2),
            hovertemplate='%{theta}: %{r:.1f}<extra></extra>'
        ))
        
        # Add markers at each point
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=emotions,
            mode='markers',
            marker=dict(size=10, color=colors, line=dict(color='#0a192f', width=2)),
            hovertemplate='%{theta}: %{r:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.3 if max(values) > 0 else 20],
                    gridcolor='#1e3a5f',
                    tickfont=dict(color='#64748b', size=9),
                    linecolor='#1e3a5f',
                ),
                angularaxis=dict(
                    gridcolor='#1e3a5f',
                    tickfont=dict(color='#e2e8f0', size=11),
                    linecolor='#1e3a5f',
                ),
                bgcolor='rgba(0,0,0,0)',
            ),
            showlegend=False,
            height=300,
            margin=dict(l=50, r=50, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show dominant emotion
        emotion_labels = ['Fear', 'Anger', 'Sadness', 'Joy', 'Trust', 'Anxiety', 'Anticipation']
        max_idx = values.index(max(values))
        dominant = emotion_labels[max_idx]
        dominant_val = values[max_idx]
        
        st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; background: linear-gradient(90deg, rgba(0,212,255,0.1) 0%, rgba(139,92,246,0.1) 100%); border-radius: 8px; margin-top: -10px;">
                <span style="color: #94a3b8;">Dominant emotion:</span>
                <span style="color: #00d4ff; font-weight: bold; margin-left: 0.5rem;">{dominant}</span>
                <span style="color: #64748b; margin-left: 0.3rem;">({dominant_val:.1f})</span>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.info("📊 Emotion breakdown loading...")


# Map common GDELT theme codes to human-readable names
THEME_TRANSLATIONS = {
    'TAX_FNCACT': 'Finance',
    'EPU_POLICY': 'Policy',
    'TAX_ETHNICITY': 'Ethnicity',
    'TAX_WORLDLANGUAGES': 'Languages',
    'CRISISLEX_CRISISLEXREC': 'Crisis',
    'UNGP_FORESTS_RIVERS': 'Environment',
    'USPEC_POLITICS_GENERAL1': 'Politics',
    'TAX_ECON_PRICE': 'Economy',
    'GENERAL_GOVERNMENT': 'Government',
    'MANMADE_DISASTER_IMPLIED': 'Disaster',
    'EPU_ECONOMY_HISTORIC': 'History',
    'EDUCATION': 'Education',
    'SOC_POINTSOFINTEREST': 'Society',
    'GENERAL_HEALTH': 'Health',
    'LEADER': 'Leadership',
    'TERROR': 'Security',
    'PROTEST': 'Protests',
    'MILITARY': 'Military',
    'ARREST': 'Law',
    'KILL': 'Conflict',
}


def humanize_theme(theme):
    """Convert GDELT theme code to human-readable name."""
    if not theme:
        return None
    theme_upper = theme.upper().strip()
    if theme_upper in THEME_TRANSLATIONS:
        return THEME_TRANSLATIONS[theme_upper]
    for prefix, name in THEME_TRANSLATIONS.items():
        if theme_upper.startswith(prefix):
            return name
    cleaned = theme.replace('_', ' ').title()
    for prefix in ['Tax ', 'Epu ', 'Soc ', 'Wb ', 'Ungp ', 'Uspec ', 'Crisislex ']:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned if 2 < len(cleaned) < 20 else None


def render_trending_themes(conn):
    """Render trending themes with beautiful horizontal bars."""
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
            LIMIT 20
        """).df()
        
        if df.empty:
            st.info("📊 Theme data is being collected...")
            return
        
        # Filter and humanize themes
        themes_data = []
        for _, row in df.iterrows():
            name = humanize_theme(row['theme'])
            if name and name not in [t[0] for t in themes_data]:
                themes_data.append((name, row['mention_count']))
            if len(themes_data) >= 8:
                break
        
        if not themes_data:
            return
            
        max_count = max(t[1] for t in themes_data)
        
        # Create bar chart
        names = [t[0] for t in themes_data]
        values = [t[1] for t in themes_data]
        
        # Gradient colors
        colors = ['#00d4ff', '#00c4ef', '#00b4df', '#00a4cf', '#0094bf', '#0084af', '#00749f', '#00648f']
        
        fig = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker=dict(
                color=colors[:len(names)],
                line=dict(color='rgba(0,0,0,0)', width=0),
            ),
            text=[f'{v:,}' for v in values],
            textposition='inside',
            textfont=dict(color='white', size=11, family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>Mentions: %{x:,}<extra></extra>'
        ))
        
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=20, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(
                gridcolor='#1e3a5f', 
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                gridcolor='transparent',
                tickfont=dict(size=12),
                autorange='reversed',
            ),
            bargap=0.3,
        )
        
        st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.info("📊 Theme data loading...")


def render_emotion_stats(conn):
    """Render emotion statistics cards with icons."""
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
        pos = row['avg_positive'] if row['avg_positive'] else 0
        neg = row['avg_negative'] if row['avg_negative'] else 0
        
        # Sentiment ratio visual
        total = pos + neg if (pos + neg) > 0 else 1
        pos_pct = (pos / total) * 100
        neg_pct = (neg / total) * 100
        
        st.markdown(f"""
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                <!-- Articles Card -->
                <div style="flex: 1; padding: 1rem; background: linear-gradient(135deg, #1e3a5f 0%, #0a192f 100%); border-radius: 12px; border: 1px solid #1e3a5f; text-align: center;">
                    <div style="font-size: 2rem;">📰</div>
                    <div style="color: #00d4ff; font-size: 1.8rem; font-weight: bold;">{int(row['total_articles']):,}</div>
                    <div style="color: #64748b; font-size: 0.75rem;">Articles Analyzed</div>
                </div>
                
                <!-- Sentiment Balance Card -->
                <div style="flex: 2; padding: 1rem; background: linear-gradient(135deg, #1e3a5f 0%, #0a192f 100%); border-radius: 12px; border: 1px solid #1e3a5f;">
                    <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.5rem; text-align: center;">Sentiment Balance</div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: #22c55e; font-size: 0.85rem; width: 60px; text-align: right;">{pos:.1f}%</span>
                        <div style="flex: 1; height: 12px; background: #0f2744; border-radius: 6px; overflow: hidden; display: flex;">
                            <div style="width: {pos_pct}%; background: linear-gradient(90deg, #22c55e, #84cc16); border-radius: 6px 0 0 6px;"></div>
                            <div style="width: {neg_pct}%; background: linear-gradient(90deg, #f97316, #ef4444); border-radius: 0 6px 6px 0;"></div>
                        </div>
                        <span style="color: #ef4444; font-size: 0.85rem; width: 60px;">{neg:.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.3rem;">
                        <span style="color: #64748b; font-size: 0.7rem;">👍 Positive</span>
                        <span style="color: #64748b; font-size: 0.7rem;">Negative 👎</span>
                    </div>
                </div>
                
                <!-- Tone Card -->
                <div style="flex: 1; padding: 1rem; background: linear-gradient(135deg, #1e3a5f 0%, #0a192f 100%); border-radius: 12px; border: 1px solid #1e3a5f; text-align: center;">
                    <div style="font-size: 2rem;">📊</div>
                    <div style="color: #8b5cf6; font-size: 1.8rem; font-weight: bold;">{row['avg_tone']:.2f}</div>
                    <div style="color: #64748b; font-size: 0.75rem;">Average Tone</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
                
    except Exception as e:
        pass


def render_emotion_insights(conn):
    """Render AI-style emotion insights."""
    try:
        df = conn.execute("""
            SELECT 
                AVG(AVG_TONE) as tone,
                AVG(EMOTION_FEAR) as fear,
                AVG(EMOTION_JOY) as joy,
                AVG(EMOTION_ANGER) as anger,
                COUNT(*) as cnt
            FROM gkg_emotions
        """).df()
        
        if df.empty:
            return
        
        row = df.iloc[0]
        tone = row['tone'] if row['tone'] else 0
        fear = row['fear'] if row['fear'] else 0
        joy = row['joy'] if row['joy'] else 0
        
        # Generate insight based on data
        if fear > joy * 1.5:
            insight = "Global news coverage is dominated by <b style='color:#ef4444'>fear and anxiety</b>, indicating heightened concerns."
        elif joy > fear * 1.5:
            insight = "News sentiment leans <b style='color:#22c55e'>positive</b>, with optimism outweighing concerns."
        elif tone < -2:
            insight = "Media tone is <b style='color:#f97316'>notably negative</b>, reflecting challenging global conditions."
        elif tone > 2:
            insight = "Coverage reflects <b style='color:#84cc16'>positive developments</b> across major news sources."
        else:
            insight = "News sentiment is <b style='color:#eab308'>balanced</b>, with mixed emotions across global coverage."
        
        st.markdown(f"""
            <div style="padding: 1rem; background: linear-gradient(90deg, rgba(139,92,246,0.1) 0%, rgba(0,212,255,0.1) 100%); border-radius: 10px; border-left: 4px solid #8b5cf6; margin-top: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem;">
                    <span style="font-size: 1rem;">🤖</span>
                    <span style="color: #8b5cf6; font-weight: 600; font-size: 0.85rem;">AI Insight</span>
                </div>
                <div style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.5;">
                    {insight}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    except:
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
    
    # AI Insight
    render_emotion_insights(conn)
    
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    
    # Main content - 2 columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card-hdr"><span>🎯</span><span class="card-title">Global Mood Index</span></div>', unsafe_allow_html=True)
        render_emotions_pulse(conn)
    
    with col2:
        st.markdown('<div class="card-hdr"><span>📊</span><span class="card-title">Emotion Radar</span></div>', unsafe_allow_html=True)
        render_emotion_breakdown(conn)
    
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
    
    # Trending themes full width
    st.markdown('<div class="card-hdr"><span>🔥</span><span class="card-title">Trending Topics</span></div>', unsafe_allow_html=True)
    render_trending_themes(conn)
