"""
CSS styles for GDELT dashboard.
"""
import streamlit as st


def inject_css():
    """Inject custom CSS."""
    st.markdown("""
    <style>
        /* Import modern fonts from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
        
        /* Define color palette as variables for easy reuse */
        :root { 
            --bg:#0a0e17;
            --card:#111827;
            --card-inner:#1a2332;
            --border:#1e3a5f;
            --text:#e2e8f0;
            --muted:#94a3b8;
            --dim:#64748b;
            --cyan:#06b6d4;
            --green:#10b981;
            --red:#ef4444;
            --amber:#f59e0b;
            --purple:#8b5cf6;
        }
        
        /* Main app background */
        .stApp { background: var(--bg); }
        
        /* Hide Streamlit's default header, menu, and footer */
        header[data-testid="stHeader"], #MainMenu, footer, .stDeployButton { 
            display: none !important; 
        }
        
        /* Set default fonts for different elements */
        html, body, p, span, div { 
            font-family: 'Inter', sans-serif; 
            color: var(--text); 
        }
        h1, h2, h3, code { 
            font-family: 'JetBrains Mono', monospace; 
        }
        
        /* Main content area spacing */
        .block-container { 
            padding: 1.5rem 2rem; 
            max-width: 100%; 
        }
        
        /* Header section styling */
        .header { 
            border-bottom: 1px solid var(--border); 
            padding: 1rem 0 1.5rem; 
            margin-bottom: 1.5rem; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
        }
        
        /* Logo and branding */
        .logo { 
            display: flex; 
            align-items: center; 
            gap: 0.75rem; 
        }
        .logo-icon { font-size: 2.5rem; }
        .logo-title { 
            font-family: 'JetBrains Mono'; 
            font-size: 1.4rem; 
            font-weight: 700; 
            text-transform: uppercase; 
        }
        .logo-sub { 
            font-size: 0.7rem; 
            color: var(--cyan); 
        }
        
        /* "LIVE DATA" badge in header */
        .live-badge { 
            display: flex; 
            align-items: center; 
            gap: 0.5rem; 
            background: rgba(16,185,129,0.15); 
            border: 1px solid rgba(16,185,129,0.4); 
            padding: 0.4rem 0.8rem; 
            border-radius: 20px; 
            font-size: 0.75rem; 
        }
        
        /* Pulsing green dot animation */
        .live-dot { 
            width: 8px; 
            height: 8px; 
            background: var(--green); 
            border-radius: 50%; 
            animation: pulse 2s infinite; 
        }
        @keyframes pulse { 
            0%,100% { opacity:1; } 
            50% { opacity:0.5; } 
        }
        
        /* Metric cards */
        div[data-testid="stMetric"] { 
            background: var(--card); 
            border: 1px solid var(--border); 
            border-radius: 12px; 
            padding: 1rem; 
        }
        div[data-testid="stMetric"] label { 
            color: var(--muted); 
            font-size: 0.7rem; 
            font-family: 'JetBrains Mono'; 
            text-transform: uppercase; 
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { 
            font-size: 1.5rem; 
            font-weight: 700; 
            font-family: 'JetBrains Mono'; 
        }
        
        /* Card headers */
        .card-hdr { 
            display: flex; 
            align-items: center; 
            gap: 0.75rem; 
            margin-bottom: 1rem; 
            padding-bottom: 0.75rem; 
            border-bottom: 1px solid var(--border); 
        }
        .card-title { 
            font-family: 'JetBrains Mono'; 
            font-size: 0.85rem; 
            font-weight: 600; 
            text-transform: uppercase; 
        }
        
        /* Tab navigation styling */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 0; 
            background: #0d1320; 
            border-radius: 8px; 
            padding: 4px; 
            border: 1px solid var(--border); 
            overflow-x: auto; 
        }
        .stTabs [data-baseweb="tab"] { 
            font-family: 'JetBrains Mono'; 
            font-size: 0.75rem; 
            color: var(--muted); 
            padding: 0.5rem 0.9rem; 
            white-space: nowrap; 
        }
        .stTabs [aria-selected="true"] { 
            background: var(--card-inner); 
            color: var(--cyan); 
            border-radius: 6px; 
        }
        .stTabs [data-baseweb="tab-highlight"], 
        .stTabs [data-baseweb="tab-border"] { 
            display: none; 
        }
        
        /* Data table styling */
        div[data-testid="stDataFrame"] { 
            background: var(--card); 
            border: 1px solid var(--border); 
            border-radius: 12px; 
        }
        div[data-testid="stDataFrame"] th { 
            background: var(--card-inner) !important; 
            color: var(--muted) !important; 
            font-size: 0.75rem; 
            text-transform: uppercase; 
        }
        
        /* Live ticker */
        .ticker { 
            background: linear-gradient(90deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05)); 
            border-left: 4px solid var(--red); 
            border-radius: 0 8px 8px 0; 
            padding: 0.6rem 0; 
            overflow: hidden; 
            position: relative; 
            margin: 0.5rem 0; 
        }
        .ticker-label { 
            position: absolute; 
            left: 0; top: 0; bottom: 0; 
            background: linear-gradient(90deg, rgba(127,29,29,0.98), transparent); 
            padding: 0.6rem 1.25rem 0.6rem 0.75rem; 
            font-size: 0.7rem; 
            font-weight: 600; 
            color: var(--red); 
            display: flex; 
            align-items: center; 
            gap: 0.5rem; 
            z-index: 2; 
        }
        .ticker-dot { 
            width: 7px; height: 7px; 
            background: var(--red); 
            border-radius: 50%; 
            animation: blink 1s infinite; 
        }
        @keyframes blink { 
            0%,100% { opacity:1; } 
            50% { opacity:0.3; } 
        }
        .ticker-text { 
            display: inline-block; 
            white-space: nowrap; 
            padding-left: 95px; 
            animation: scroll 40s linear infinite; 
            font-size: 0.8rem; 
            color: #fca5a5; 
        }
        @keyframes scroll { 
            0% { transform: translateX(0); } 
            100% { transform: translateX(-50%); } 
        }
        
        /* Technology badges */
        .tech-badge { 
            display: inline-flex; 
            background: var(--card-inner); 
            border: 1px solid var(--border); 
            border-radius: 20px; 
            padding: 0.4rem 0.8rem; 
            font-size: 0.75rem; 
            color: var(--muted); 
            margin: 0.25rem; 
        }
        
        /* Horizontal divider line */
        hr { 
            border: none; 
            border-top: 1px solid var(--border); 
            margin: 1.5rem 0; 
        }
        
        /* ============================================
           CARD STYLES
           ============================================ */
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
        }
        .card-2rem { padding: 2rem; }
        .card-mb { margin-bottom: 1rem; }
        .card-my { margin: 2rem 0; }
        .card-h280 { min-height: 280px; }
        .card-h200 { min-height: 200px; }
        .card-h100 { height: 100%; }
        .card-center { text-align: center; }
        
        /* ============================================
           PIPELINE FLOW DIAGRAM
           ============================================ */
        .pipeline-step {
            background: var(--card-inner);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            display: inline-block;
            margin: 0.5rem;
        }
        .pipeline-arrow {
            color: var(--cyan);
            margin: 0 0.5rem;
        }
        
        /* ============================================
           SECTION HEADERS
           ============================================ */
        .section-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-title {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text);
        }
        .section-subtitle {
            color: var(--dim);
        }
        .section-desc {
            color: var(--muted);
            max-width: 750px;
            margin: 0 auto 1.5rem;
            font-size: 1.1rem;
        }
        .section-desc-sm {
            color: var(--dim);
            max-width: 700px;
            margin: 0 auto 2rem;
        }
        
        /* ============================================
           TITLE COLORS
           ============================================ */
        .title-cyan { color: var(--cyan); font-size: 0.9rem; }
        .title-green { color: var(--green); font-size: 0.9rem; }
        .title-amber { color: var(--amber); font-size: 0.9rem; }
        .title-purple { color: var(--purple); font-size: 0.9rem; }
        .title-white { color: var(--text); }
        .title-center { text-align: center; margin-bottom: 1rem; }
        .title-center-mb { text-align: center; margin-bottom: 1.5rem; }
        
        /* ============================================
           TEXT STYLES
           ============================================ */
        .text-muted { color: var(--muted); font-size: 0.85rem; }
        .text-muted-lg { color: var(--muted); font-size: 0.9rem; }
        .text-dim { color: var(--dim); }
        .text-white { color: var(--text); }
        .text-cyan { color: var(--cyan); }
        .text-green { color: var(--green); }
        .text-amber { color: var(--amber); }
        .text-center { text-align: center; }
        .text-sm { font-size: 0.75rem; }
        .text-xs { font-size: 0.7rem; }
        
        .muted-list {
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.8;
        }
        .muted-list-pl { padding-left: 1.2rem; }
        
        /* ============================================
           HIGHLIGHT BOXES (stats grid)
           ============================================ */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        .highlight-box {
            text-align: center;
            padding: 1rem;
            border-radius: 8px;
        }
        .highlight-cyan { background: rgba(6,182,212,0.1); }
        .highlight-green { background: rgba(16,185,129,0.1); }
        .highlight-amber { background: rgba(245,158,11,0.1); }
        .highlight-purple { background: rgba(139,92,246,0.1); }
        
        .highlight-value {
            font-size: 2rem;
            font-weight: 700;
        }
        .highlight-label {
            font-size: 0.75rem;
            color: var(--muted);
        }
        .value-cyan { color: var(--cyan); }
        .value-green { color: var(--green); }
        .value-amber { color: var(--amber); }
        .value-purple { color: var(--purple); }
        
        /* ============================================
           CALLOUT BOXES
           ============================================ */
        .callout-success {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: rgba(16,185,129,0.1);
            border-radius: 6px;
            border-left: 3px solid var(--green);
        }
        .callout-success-text {
            color: var(--green);
            font-size: 0.75rem;
        }
        
        .callout-big {
            margin-top: 1rem;
            padding: 1.25rem;
            background: rgba(16,185,129,0.1);
            border-radius: 12px;
            border-left: 4px solid var(--green);
            text-align: center;
        }
        .callout-big-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--green);
            margin-bottom: 0.5rem;
        }
        .callout-big-label {
            font-size: 0.85rem;
            color: var(--muted);
        }
        
        /* ============================================
           INFO BOX
           ============================================ */
        .info-box {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .info-box-text {
            font-size: 0.7rem;
            color: var(--dim);
        }
        
        /* ============================================
           DEPRECATED BADGES
           ============================================ */
        .badge-deprecated {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 0.4rem 0.8rem;
            display: inline-block;
            margin: 0.25rem;
            font-size: 0.75rem;
            color: var(--dim);
        }
        
        /* ============================================
           INTENSITY GUIDE
           ============================================ */
        .intensity-item {
            padding: 0.4rem 0.75rem;
            margin: 0.4rem 0;
            border-radius: 4px;
            border-left: 3px solid;
        }
        .intensity-red {
            background: rgba(239,68,68,0.1);
            border-left-color: var(--red);
        }
        .intensity-red .intensity-title { color: var(--red); }
        
        .intensity-amber {
            background: rgba(245,158,11,0.1);
            border-left-color: var(--amber);
        }
        .intensity-amber .intensity-title { color: var(--amber); }
        
        .intensity-yellow {
            background: rgba(234,179,8,0.1);
            border-left-color: #eab308;
        }
        .intensity-yellow .intensity-title { color: #eab308; }
        
        .intensity-gray {
            background: rgba(148,163,184,0.1);
            border-left-color: var(--muted);
        }
        .intensity-gray .intensity-title { color: var(--muted); }
        
        .intensity-green {
            background: rgba(16,185,129,0.1);
            border-left-color: var(--green);
        }
        .intensity-green .intensity-title { color: var(--green); }
        
        .intensity-cyan {
            background: rgba(6,182,212,0.1);
            border-left-color: var(--cyan);
        }
        .intensity-cyan .intensity-title { color: var(--cyan); }
        
        .intensity-title { font-weight: 600; }
        .intensity-score { font-size: 0.7rem; color: var(--muted); }
        
        /* ============================================
           CONTACT BUTTONS
           ============================================ */
        .contact-section {
            text-align: center;
        }
        .contact-links {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
        }
        .contact-link {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 1.25rem;
            color: var(--text);
            text-decoration: none;
            display: inline-block;
        }
        
        /* ============================================
           METRIC HINT
           ============================================ */
        .metric-hint {
            text-align: center;
            margin-top: -0.5rem;
            font-size: 0.7rem;
            color: var(--dim);
        }
        
        /* ============================================
           FOOTER
           ============================================ */
        .footer {
            text-align: center;
            padding: 2rem 0 1rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }
        .footer-main {
            color: var(--dim);
            font-size: 0.8rem;
        }
        .footer-sub {
            color: #475569;
            font-size: 0.75rem;
        }
        .footer-link {
            color: var(--cyan);
        }
        
        /* ============================================
           SENTIMENT DISPLAY
           ============================================ */
        .sentiment-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sentiment-label {
            font-size: 0.7rem;
            color: var(--dim);
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .sentiment-status {
            font-size: 1.75rem;
            font-weight: 700;
        }
        .sentiment-avg {
            font-size: 0.8rem;
            color: var(--muted);
        }
        
        .sentiment-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
        }
        .sentiment-box {
            text-align: center;
            padding: 0.75rem;
            border-radius: 8px;
        }
        .sentiment-value {
            font-size: 1.5rem;
            font-weight: 700;
        }
        .sentiment-box-label {
            font-size: 0.65rem;
            color: var(--muted);
            text-transform: uppercase;
        }
        
        /* ============================================
           AI CHAT
           ============================================ */
        .ai-info-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 1rem;
        }
        .ai-example-label {
            color: var(--dim);
            font-size: 0.7rem;
            margin-bottom: 0.5rem;
        }
        .ai-examples {
            color: var(--muted);
            font-size: 0.75rem;
            line-height: 1.8;
        }
        
        .prev-convo-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 0.5rem;
        }
        .prev-convo-label {
            font-size: 0.75rem;
            color: var(--dim);
            margin-bottom: 0.5rem;
        }
        .prev-convo-q {
            font-size: 0.75rem;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }
        .prev-convo-text {
            font-size: 0.8rem;
            color: var(--text);
        }
    </style>
    """, unsafe_allow_html=True)